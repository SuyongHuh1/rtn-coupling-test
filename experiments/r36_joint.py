"""R36: J-only LRT + co-flip score test, both calibrated on one set of pipeline nulls (rtnsim.score).
    python experiments/r36_joint.py rtninja <nsrc> [nsim=39] [workers=2]       tested records, integrating acquisition
    python experiments/r36_joint.py sim <scen> <noise> <K> <n> [nsim=39] [workers=2] [acq=point]
        scen  coflip (kappa only) | J (energy coupling) ; each record is paired with its independent projection
        noise white | flicker
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
os.environ["RTN_JOINT"] = "0"; os.environ.setdefault("RTN_RESTART", "3")
import torch
torch.set_num_threads(1)
OUT = ROOT / "results/r36"; OUT.mkdir(parents=True, exist_ok=True)
_STEP = None


def pipe_plain(K):
    from rtnsim.cfhmm import lrt_ms
    return lambda y, rng: lrt_ms(y, K, rng)


def pipe_cnn(K):
    def f(y, rng):
        global _STEP
        from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
        if _STEP is None:
            _STEP = load_stepnet(ROOT / "results/r9a_gpu/stepnet.pt")
        return lrt_cnn3_ms(y, K, _STEP, ar1=True, rng=rng, n_start=3)
    return f


# ---------------------------------------------------------------- simulation
def _gen(c, e, K, kappa=None, J=None):
    """Generator on 2^K states (bit k = trap k); coupling on pair (0, 1) only."""
    S = 2 ** K
    Q = np.zeros((S, S))
    for s in range(S):
        for k in range(K):
            up = not (s >> k) & 1
            r = c[k] if up else e[k]
            if J is not None and k < 2 and (s >> (1 - k)) & 1:      # partner occupied
                r *= np.exp(-0.5 * J) if up else np.exp(0.5 * J)
            Q[s, s ^ (1 << k)] = r
        if kappa and (s & 3) in (0, 3):
            Q[s, s ^ 3] += kappa
    Q[np.diag_indices(S)] = -Q.sum(1)
    return Q


def _stat(Q):
    w, v = np.linalg.eig(Q.T)
    p = np.real(v[:, np.argmin(np.abs(w))])
    return p / p.sum()


def _iproj(Q, K):
    """Independent traps with the same occupancy and switching flux per trap."""
    pi = _stat(Q); S = 2 ** K
    c, e = np.zeros(K), np.zeros(K)
    for k in range(K):
        occ = sum(pi[s] for s in range(S) if (s >> k) & 1)
        flux = sum(pi[s] * Q[s, t] for s in range(S) for t in range(S) if s != t and ((s ^ t) >> k) & 1 and not (s >> k) & 1)
        c[k], e[k] = flux / (1 - occ), flux / occ
    return c, e


def _sample(Q, T, rng, box, sub=20):
    from scipy.linalg import expm
    S = Q.shape[0]; n = T * (sub if box else 1)
    cum = np.cumsum(expm(Q / (sub if box else 1)), 1)
    s = np.empty(n, np.int64); s[0] = rng.choice(S, p=np.abs(_stat(Q)) / np.abs(_stat(Q)).sum())
    u = rng.random(n)
    for t in range(1, n):
        s[t] = min(np.searchsorted(cum[s[t - 1]], u[t]), S - 1)
    K = int(np.log2(S))
    occ = ((s[:, None] >> np.arange(K)) & 1).astype(float)
    return occ.reshape(T, -1, K).mean(1)


def make(i, scen, noise, K, T=4096, box=False):
    from rtnsim.noise import add_flicker
    rng = np.random.default_rng(36000 + i)
    tau = np.exp(np.log(8) + rng.random(K) * (np.log(200) - np.log(8)))
    p = 0.15 + 0.7 * rng.random(K)
    c, e = 2 / tau * p, 2 / tau * (1 - p)
    kap = (0.2 + 0.8 * rng.random()) * np.mean(np.r_[c[:2], e[:2]])
    J = (0.7 + 2.3 * rng.random()) * rng.choice([-1, 1])
    a = np.sort(0.15 + 0.85 * rng.random(K))[::-1]; a[0] = 1.0
    a = a[rng.permutation(K)]
    sig = a.min() * (0.15 + 0.6 * rng.random())
    Q1 = _gen(c, e, K, kappa=kap) if scen == "coflip" else _gen(c, e, K, J=J)
    c0, e0 = _iproj(Q1, K); Q0 = _gen(c0, e0, K)
    nz = sig * rng.standard_normal(T)
    fl = 0.6 * sig * rng.random()
    out = []
    for lab, Q in (("coup", Q1), ("indep", Q0)):
        y = _sample(Q, T, rng, box) @ a + nz
        if noise == "flicker":
            y = add_flicker(y[None], fl, fmin=1 / T, fmax=0.5, n_terms=8, dt=1.0, rng=np.random.default_rng(37000 + i))[0]
        out.append((f"{lab}{i}", y, dict(true_kappa=float(kap / np.mean(np.r_[c[:2], e[:2]])), true_J=float(J))))
    return out


def job(a):
    from rtnsim.score import joint_test
    tag, y, K, seed, nsim, acq, cnn, meta = a[:8]
    t0 = time.time()
    try:
        r = joint_test(y, K, pipe_cnn(K) if cnn else pipe_plain(K), np.random.default_rng(seed), nsim, acq,
                       null_seed=(seed + 10 ** 6) if len(a) > 8 else None)
        return dict(tag=tag, sec=time.time() - t0, **meta, **r)
    except Exception as ex:                                # noqa: BLE001
        import traceback
        return dict(tag=tag, error=repr(ex)[:200], tb=traceback.format_exc()[-500:])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rtninja":
        DATA = Path(os.environ.get("RTN_RTNINJA", "data/rtninja"))
        nsrc = int(sys.argv[2]); nsim = int(sys.argv[3]) if len(sys.argv) > 3 else 39; W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        recs = open(ROOT / f"results/r35/tested{nsrc}.txt").read().strip().split(",")
        jobs = [(r, np.load(DATA / f"n{nsrc}" / f"noise05_{nsrc}source" / r / "MeasurementData.npy").astype(float), nsrc, 3600 + i, nsim, "integ", False, {})
                for i, r in enumerate(recs)]
        outp = OUT / f"rtninja_{nsrc}src.json"
    elif mode == "realneg":                      # r12 real-noise control: independent by construction, real 60 Hz boxcar
        import scipy.io as sio
        n = int(sys.argv[2]); nsim = int(sys.argv[3]) if len(sys.argv) > 3 else 39; W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        D = Path(os.environ.get("RTN_NICHOL", "data/nichol"))
        d = sio.loadmat(D / "Gate_P.mat")["savedat"][0, 0]["data"]
        jobs = []
        for k in range(n):
            a, b, g = k % 20, (k * 7 + 3) % 20, (1.0, 1.5)[(k // 20) % 2]
            ya, yb = d[9, a].astype(float), d[10, b].astype(float)
            jobs.append((f"neg{k}", ya + g * (yb - np.median(yb)), 2, 3800 + k, nsim, "integ", True, {}))
        outp = OUT / "realneg.json"
    elif mode == "conf":                         # confounder controls (no coupling): drive | erlang | onoff | threestate
        kind, n = sys.argv[2], int(sys.argv[3]); nsim = int(sys.argv[4]) if len(sys.argv) > 4 else 39; W = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        jobs = []
        if kind == "drive":
            from rtnsim import sim2_torch as S
            g = torch.Generator().manual_seed(702)
            while len(jobs) < n:
                y, lab = S.make_batch(20, 4096, g, "cpu", p_drive=1.0, p_box=0)
                if int(lab["K"][0]) != 2:
                    continue
                for r in range(1, y.shape[0], 2):
                    if len(jobs) < n:
                        jobs.append((f"drive{len(jobs)}", y[r].double().numpy(), 2, 4000 + len(jobs), nsim, "point", True, {}))
        elif kind == "erlang":
            import r8_calib as C
            rng = np.random.default_rng(88)
            jobs = [(f"erlang{i}", C.erlang_record(2, 4096, rng), 2, 4100 + i, nsim, "point", True, {}) for i in range(n)]
        else:
            import r26_metastable as M
            jobs = [(f"{kind}{i}", np.asarray(M.simulate(kind, i), float), 2, 4200 + i, nsim, "point", True, {}) for i in range(n)]
        outp = OUT / f"conf_{kind}.json"
    elif mode == "south":                        # Southampton records, K = 2 and 3 (verdict at the BIC-chosen K as in r20)
        import r5_real as R
        from rtnsim.realdata import load_b1500
        nsim = int(sys.argv[2]) if len(sys.argv) > 2 else 39; W = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        keep = {r["file"] for r in json.load(open(ROOT / "results/r6/r6_south.json")) if r["K"] >= 2}
        jobs = []
        for i, r in enumerate(R.records()):
            if r["file"] in keep:
                t, I, _ = load_b1500(r["path"])
                y = R.preprocess(t, I, r["trim_s"])[1]
                jobs += [(f"{r['file']}|K{K}", y, K, 3900 + 10 * i + K, nsim, "integ", True, {}) for K in (2, 3)]
        jobs.sort(key=lambda j: len(j[1]) * 4 ** j[2])
        outp = OUT / "south.json"
    else:
        scen, noise, K, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        nsim = int(sys.argv[6]) if len(sys.argv) > 6 else 39; W = int(sys.argv[7]) if len(sys.argv) > 7 else 2
        acq = sys.argv[8] if len(sys.argv) > 8 else "point"
        jobs = [(tag, y, K, 3700 + i, nsim, acq, True, meta) for i in range(n) for tag, y, meta in make(i, scen, noise, K, box=acq == "integ")]
        outp = OUT / f"sim_{scen}_{noise}_K{K}_{acq}.json"
    EXT = int(os.environ.get("RTN_EXTEND", "0") or 0)
    if EXT:                                      # add EXT nulls to every record whose split-level verdict can change
        from rtnsim.score import merge_nulls, split_verdict
        res = json.load(open(outp))
        by = {r["tag"]: r for r in res if "error" not in r}
        need = [t for t, r in by.items() if not r.get("extended") and any(split_verdict(r).values())]
        print("extend", len(need), "of", len(by), flush=True)
        jx = [j[:4] + (EXT,) + j[5:8] + (True,) for j in jobs if j[0] in need]
        with Pool(W) as pool:
            for x in pool.imap_unordered(job, jx):
                if "error" in x:
                    print(x["tag"][:14], "ERR", x["error"], flush=True); continue
                r = merge_nulls(by[x["tag"]], x)
                json.dump(res, open(outp, "w"))
                print(r["tag"][:14], "L", round(r["stat"], 1), "rerun", round(x["stat"], 1), "pJ", r["p_J"], "pk", r["p_kappa"],
                      "sec", round(x.get("sec", 0)), flush=True)
        for r in res:                            # records not extended: split verdict from the first nulls
            if "error" not in r and not r.get("extended"):
                r.update(flag_J=False, flag_kappa=False, flag=False, split_decided=True)
        json.dump(res, open(outp, "w"))
        sys.exit(0)
    res = [r for r in (json.load(open(outp)) if outp.exists() else []) if "error" not in r]
    done = {r["tag"] for r in res}
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, [j for j in jobs if j[0] not in done]):
            res.append(r); json.dump(res, open(outp, "w"))
            print(r["tag"][:14], "L", round(r.get("stat", -1), 1), "pJ", r.get("p_J"), "pk", r.get("p_kappa"), "sec", round(r.get("sec", 0)),
                  r.get("n_fail"), r.get("error"), flush=True)
