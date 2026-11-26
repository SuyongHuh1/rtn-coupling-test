"""Simulation-based power at one Nichol voltage (replaces the Hessian prediction, which is optimistic
under coloured noise).  Generator = the fitted joint model at that voltage (rates, steps, AR(1) noise,
each repetition's baseline, 60 Hz boxcar via 16x oversampling) with a coupling J added; analysis =
the same joint QN fit (J = 0 vs J free, oracle start).  The threshold is the 95% point of the J = 0
replicates, so power is at a calibrated 5% level.   python experiments/r13_power.py <vi> <K> <n> [file]
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from rtnsim.cfhmm import Kinetics
from rtnsim.multiqn import fit_multi_qn
from rtnsim.robustk import bits

OS = 16
JS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def sim(res, J, betas, rng):
    K = res["K"]
    kin = Kinetics(K, np.array(res["c"]), np.array(res["e"]), J=np.full(K * (K - 1) // 2, J), coupled=True)
    with torch.no_grad():
        P = torch.linalg.matrix_exp(kin.Q() / OS).numpy()
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True); cum = np.cumsum(P, 1)
    w, V = np.linalg.eig(P.T); pi = np.abs(np.real(V[:, np.argmax(np.real(w))])); pi /= pi.sum()
    lev = bits(K) @ np.array(res["coef"][1:]) + res["coef"][0]
    out = []
    for b in betas:
        T = b.size
        s = rng.choice(len(pi), p=pi); u = rng.random(T * OS); st = np.empty(T * OS, int)
        for k in range(T * OS):
            s = min(int(np.searchsorted(cum[s], u[k])), len(pi) - 1); st[k] = s
        x = lev[st].reshape(T, OS).mean(1)
        e = rng.normal(0, res["sig"], T); n = np.empty(T); n[0] = e[0] / np.sqrt(1 - res["rho"] ** 2)
        for k in range(1, T):
            n[k] = res["rho"] * n[k - 1] + e[k]
        out.append(x + n)          # baseline removed by construction (oracle baseline)
    return out


def lrt(zs, K, res):
    ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
    f0 = fit_multi_qn(zs, K, False, ini)
    base = dict(ini, coef=f0["coef"], c=f0["c"], e=f0["e"], sig=f0["sig"], rho=f0["rho"], off=f0["off"])
    nJ = K * (K - 1) // 2
    f1 = max((fit_multi_qn(zs, K, True, dict(base, J=np.full(nJ, j))) for j in (0.0, 0.5)), key=lambda f: f["ll"])
    f0b = fit_multi_qn(zs, K, False, dict(base, coef=f1["coef"], c=f1["c"], e=f1["e"], sig=f1["sig"], rho=f1["rho"], off=f1["off"]))
    return 2 * max(f1["ll"] - max(f0["ll"], f0b["ll"]), 0.0), f1["J"].tolist()


def job(a):
    J, j, res, betas = a
    rng = np.random.default_rng(int(1000 * J) * 100 + j)
    zs = sim(res, J, betas, rng)
    s, Jh = lrt(zs, res["K"], res)
    return J, j, s, Jh


def main():
    vi, K, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fname = sys.argv[4] if len(sys.argv) > 4 else "Gate_P.mat"
    res = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"))
    betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
    path = N.OUT / f"{fname[:-4]}_v{vi}_K{K}_power.json"
    done = json.load(open(path)) if path.exists() else {}
    todo = [(J, j, res, betas) for J in JS for j in range(n) if f"{J}_{j}" not in done]
    with Pool(2) as p:
        for J, j, s, Jh in p.imap_unordered(job, todo):
            done[f"{J}_{j}"] = dict(J=J, stat=s, Jhat=Jh)
            json.dump(done, open(path, "w"))
    import scipy.io as sio
    from rtnsim.multiqn import fit_multi_qn as _f  # noqa
    d = sio.loadmat(N.DATA / fname)["savedat"][0, 0]["data"]
    zs = [d[vi, r].astype(float) - betas[r] for r in range(len(betas))]
    obs, Jobs = lrt(zs, K, res)
    null = np.array([v["stat"] for v in done.values() if v["J"] == 0.0])
    thr = float(np.quantile(null, 0.95))
    summ = dict(V=res["V"], obs=obs, J_obs=Jobs, p=float((1 + (null >= obs).sum()) / (1 + len(null))), thr=thr,
                power={str(J): float(np.mean([v["stat"] > thr for v in done.values() if v["J"] == J])) for J in JS},
                Jhat_mean={str(J): float(np.mean([v["Jhat"] for v in done.values() if v["J"] == J])) for J in JS})
    print(json.dumps(summ), flush=True)
    json.dump(dict(done, summary=summ), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
