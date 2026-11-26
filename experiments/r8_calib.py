"""R8 calibration (mode coupled = same pipeline on coupled records, i.e. calibrated power): false-positive rate of the R8 LRT and of the R6 path statistic under their
pipeline nulls, on INDEPENDENT traps.

exp      independent partners from results/r8/records.npz (exponential dwells, K = 2 and 3)
erlang   independent traps whose up-state dwell is Erlang-2 (a hidden internal phase, same mean):
         dwell times are non-exponential, but there is no coupling.  Tests whether model
         misspecification of dwells produces false coupling claims.

    python experiments/r8_calib.py exp|erlang [n_per_K=20] [n_sim=40] [workers=2]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim.cfhmm import lrt_ms, simulate_null, lrt_drive, simulate_from_chain  # noqa: E402
from rtnsim.robustk import bits  # noqa: E402
from rtnsim.noise import add_flicker  # noqa: E402

OUT = ROOT / "results" / "r8"


def erlang_trap(T, c, e, rng):
    """States 0 -> 1a -> 1b -> 0; rates c, 2e, 2e (mean up dwell 1/e, Erlang-2)."""
    Q = np.array([[-c, c, 0], [0, -2 * e, 2 * e], [2 * e, 0, -2 * e]])
    P = expm(Q)
    C = P.cumsum(1)
    x = np.empty(T, np.int64)
    s = rng.choice(3, p=[e / (e + c), c / (e + c) / 2, c / (e + c) / 2])
    u = rng.random(T)
    for t in range(T):
        s = int(np.searchsorted(C[s], u[t]))
        x[t] = s
    return (x > 0).astype(float)


def erlang_record(K, T, rng):
    a = np.sort(0.15 + 0.85 * rng.random(K))[::-1]
    a[0] = 1.0
    y = np.zeros(T)
    for k in range(K):
        tau = np.exp(np.log(8) + rng.random() * np.log(200 / 8))
        p = 0.15 + 0.7 * rng.random()
        kt = 2 / tau
        y += a[k] * erlang_trap(T, kt * p, kt * (1 - p), rng)
    sig = a.min() * (0.15 + 0.6 * rng.random())
    y += sig * rng.standard_normal(T)
    y = add_flicker(y[None], 0.6 * sig * rng.random(), fmin=1 / T, fmax=0.5, n_terms=8, dt=1.0, rng=rng)[0]
    return y


_NET = None


def cnn_score(y):
    global _NET
    import torch
    from rtnsim.net_r7 import CouplingNet
    from rtnsim.sim2_torch import preprocess
    if _NET is None:
        torch.set_num_threads(1)
        ck = torch.load(ROOT / "results/r7_gpu/model.pt", map_location="cpu")
        _NET = CouplingNet(C=ck["config"]["C"]).eval()
        _NET.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        x = preprocess(torch.tensor(np.asarray(y)[None, :4096], dtype=torch.float32))
        return float(_NET(x)["coupled"][0])


def _rank_p(obs, null):
    return (1 + np.sum(null >= obs)) / (len(null) + 1)


DRIVE_MODEL = "--drive-model" in sys.argv
R9 = "--r9" in sys.argv
import os
MS = int(os.environ.get("RTN_MS", "1"))
_STEP = None


def _r9_stat(z, K, rng):
    global _STEP
    from rtnsim.r9 import load_stepnet, lrt_cnn3
    if _STEP is None:
        _STEP = load_stepnet(ROOT / "results/r9a_gpu/stepnet.pt")
    if MS > 1:
        from rtnsim.r9 import lrt_cnn3_ms
        return lrt_cnn3_ms(z, K, _STEP, ar1=True, rng=rng, n_start=MS)
    return lrt_cnn3(z, K, _STEP, ar1=True, rng=rng)


def _checklist12(f1, K):
    """Checklist quantities of the fitted coupled model: shortest mean dwell of any joint state (step 1)
    and fewest switches of any trap on the Viterbi path (step 2)."""
    import torch
    from rtnsim.sim2_torch import generator
    J = np.asarray(f1["J"], float)
    if J.ndim == 1:
        M = np.zeros((K, K)); iu = np.triu_indices(K, 1); M[iu] = J; J = M + M.T
    t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
    Q = generator(t(f1["c"])[None], t(f1["e"])[None], t(J)[None], torch.tensor([0.5], dtype=torch.float64))[0].numpy()
    out_rate = (Q - np.diag(np.diag(Q))).sum(1)               # total exit rate of each joint state
    path = np.asarray(f1["path"])
    bits_ = (path[:, None] >> np.arange(K)) & 1
    return dict(dwell_min=float(1.0 / out_rate.max()), trans_min=int(np.abs(np.diff(bits_, axis=0)).sum(0).min()))


def job(args):
    """Pipeline null shared by all statistics: simulate the fitted J = 0 model (record's own
    baseline + block-bootstrapped residual), rerun the full analysis on each copy.
    Statistics: R8 LRT (AR(1) noise + Kalman baseline + global decomposition search), path
    statistic on the same decomposition, CNN score, and CNN+R8 Fisher combination."""
    i, y, K, n_sim = args
    rng = np.random.default_rng(500 + i)
    if DRIVE_MODEL:
        stat = lambda z: lrt_drive(z, K, rng)
    elif R9:
        stat = lambda z: _r9_stat(z, K, rng)
    else:
        stat = lambda z: lrt_ms(z, K, rng, ar1=True, kalman=True)
    try:
        s_obs, f0, f1, beta, p_obs = stat(y)
    except ValueError:                       # every start failed on the record: reported as a fit failure
        return i, dict(K=K, failed=True, p_r8=float("nan"), p_path=float("nan"), p_cnn=float("nan"),
                       p_comb=float("nan"))
    c_obs = cnn_score(y)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    n8, npth, ncnn = [], [], []
    for _ in range(n_sim):
        ys = simulate_from_chain(f0, beta, resid, len(y), rng) if DRIVE_MODEL else simulate_null(f0, beta, resid, len(y), rng)
        try:
            s, *_, p = stat(ys)
        except ValueError:                   # failed null fit: skipped (the effective B is stored)
            continue
        n8.append(s); npth.append(p); ncnn.append(cnn_score(ys))
    n8, npth, ncnn = map(np.array, (n8, npth, ncnn))
    # Fisher combination calibrated on the same null copies (leave-one-out ranks)
    def comb(s8, sc, excl=None):
        m = np.ones(len(n8), bool)
        if excl is not None:
            m[excl] = False
        return -2 * (np.log(_rank_p(s8, n8[m])) + np.log(_rank_p(sc, ncnn[m])))
    c_null = np.array([comb(n8[j], ncnn[j], j) for j in range(len(n8))])
    return i, dict(K=K, n_null=len(n8), **_checklist12(f1, K), stat_r8=s_obs, stat_path=p_obs, cnn=c_obs, J=f1["J"].tolist(), rho=f0.get("rho", 0.0),
                   p_r8=float(_rank_p(s_obs, n8)), p_path=float(_rank_p(p_obs, npth)),
                   p_cnn=float(_rank_p(c_obs, ncnn)), p_comb=float(_rank_p(comb(s_obs, c_obs), c_null)))


def main():
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_sim = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    tasks = []
    if mode in ("exp", "coupled"):
        z = np.load(OUT / "records.npz", allow_pickle=True)
        rows = list(z["rows"])
        for K in ((3,) if "--k3" in sys.argv else (2, 3)):
            want = 1 if mode == "coupled" else 0
            idx = [i for i, r in enumerate(rows) if r["K"] == K and r["coupled"] == want][:n]
            tasks += [(i, z["y"][i], K, n_sim) for i in idx]
    elif mode in ("exp_new", "white_coupled"):
        # fresh records of the same generator: independent (1/f, seed 291) or coupled (white, seed 191)
        sys.path.insert(0, str(ROOT / "experiments"))
        import r8_oracle as O
        white = mode == "white_coupled"
        ys, _, rows = O.make(n, seed=191 if white else 291, white=white)
        want = 1 if white else 0
        for i, (y, r) in enumerate(zip(ys, rows)):
            if r["coupled"] == want:
                tasks.append((i, y, r["K"], n_sim))
    elif mode == "white":
        # white-noise independent records of the R10 benchmark generator (r8_oracle.make, white=True)
        sys.path.insert(0, str(ROOT / "experiments"))
        import r8_oracle as O
        ys, _, rows = O.make(n, seed=191, white=True)
        for i, (y, r) in enumerate(zip(ys, rows)):
            if r["coupled"] == 0:
                tasks.append((i, y, r["K"], n_sim))
    elif mode == "drive":
        # independent traps (I-projection partners) sharing a slow telegraph drive, dwell 10-50x the slowest trap
        import torch
        from rtnsim import sim2_torch as S
        for K in (2, 3):
            g = torch.Generator().manual_seed(700 + K)
            got = 0
            while got < n:
                y, lab = S.make_batch(20, 4096, g, "cpu", p_drive=1.0, p_box=0)
                if int(lab["K"][0]) != K:
                    continue
                for r in range(1, y.shape[0], 2):                # odd rows = independent partners
                    if got < n:
                        tasks.append((len(tasks), y[r].double().numpy(), K, n_sim)); got += 1
    else:
        rng = np.random.default_rng(88)
        for K in (2, 3):
            for _ in range(n):
                tasks.append((len(tasks), erlang_record(K, 4096, rng), K, n_sim))
    path = OUT / (f"r8_calib_{mode}" + ("_dm" if DRIVE_MODEL else "") + ("_r9" if R9 else "") + (f"_ms{MS}" if MS > 1 else "") + os.environ.get("RTN_TAG", "") + ".json")
    res = {}
    if path.exists() and "--resume" in sys.argv:
        res = {int(k): v for k, v in json.load(open(path)).items() if k != "summary"}
        tasks = [t for t in tasks if t[0] not in res]
    with Pool(workers) as pool:
        for i, r in pool.imap_unordered(job, tasks):
            res[i] = r
            json.dump(res, open(path, "w"), indent=1)
            print(i, {k: v for k, v in r.items() if k.startswith("p_") or k == "K"}, flush=True)
    summ = {}
    for K in (2, 3):
        rr = [r for r in res.values() if r["K"] == K and not r.get("failed")]
        summ[f"K={K}"] = {m: float(np.mean([r[m] <= 0.05 for r in rr])) for m in ("p_r8", "p_path", "p_cnn", "p_comb")}
        summ[f"K={K}"]["n"] = len(rr)
        summ[f"K={K}"]["n_failed"] = sum(1 for r in res.values() if r["K"] == K and r.get("failed"))
    res["summary"] = summ
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
