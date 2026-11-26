"""Calibrate 'coupling given drive' at one voltage: simulate the fitted M2 (independent traps + slow
drive, AR(1) noise, each repetition's baseline), refit M2 and M3 from the generating values
(oracle start), LRT = 2 (ll3 - ll2).   python experiments/r11_null_qn.py <vi> <K> <n_sim> [file] [nrep]
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from rtnsim.cfhmm import KineticsDrive
from rtnsim.multiqn import fit_multi_qn
from rtnsim.robustk import bits


def simulate(d, K, betas, T, rng):
    kin = KineticsDrive(K, np.array(d["c"]), np.array(d["e"]), coupled=False, gamma=np.array(d["gamma"]),
                        drive=np.array(d["drive"]))
    with torch.no_grad():
        P = torch.linalg.matrix_exp(kin.Q()).clamp_min(0).numpy()
    P /= P.sum(1, keepdims=True)
    cum = np.cumsum(P, 1)
    w, V = np.linalg.eig(P.T); pi = np.abs(np.real(V[:, np.argmax(np.real(w))])); pi /= pi.sum()
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)]); Xd = np.vstack([X, X])
    mu = Xd @ np.array(d["coef"])
    out = []
    for b in betas:
        s = rng.choice(len(pi), p=pi)
        u = rng.random(T); st = np.empty(T, int)
        for k in range(T):
            s = min(int(np.searchsorted(cum[s], u[k])), len(pi) - 1); st[k] = s
        e = rng.normal(0, d["sig"], T); n = np.empty(T); n[0] = e[0] / np.sqrt(1 - d["rho"] ** 2)
        for k in range(1, T):
            n[k] = d["rho"] * n[k - 1] + e[k]
        out.append(mu[st] + n + (b - np.median(b)))
    return out


def job(a):
    j, d, K, betas, nrep = a
    rng = np.random.default_rng(4242 + j)
    T = betas.shape[1]
    ys = simulate(d, K, betas[:nrep], T, rng)
    zs = [y - (b - np.median(b)) for y, b in zip(ys, betas[:nrep])]
    slow = 1.0 / np.min(np.array(d["c"]) + np.array(d["e"]))
    ini = dict(coef=np.array(d["coef"]), sig=d["sig"], rho=d["rho"], c=np.array(d["c"]), e=np.array(d["e"]),
               gamma=np.array(d["gamma"]), drive=np.array(d["drive"]), max_rate=1.0 / (10 * slow))
    m2 = fit_multi_qn(zs, K, False, ini, drive=True)
    base = dict(ini, coef=m2["coef"], c=m2["c"], e=m2["e"], sig=m2["sig"], rho=m2["rho"], off=m2["off"],
                gamma=m2["gamma"], drive=m2["drive"])
    nJ = K * (K - 1) // 2
    m3 = max((fit_multi_qn(zs, K, True, dict(base, J=np.full(nJ, x)), drive=True) for x in (0.0, 0.5)), key=lambda f: f["ll"])
    m2b = fit_multi_qn(zs, K, False, dict(base, coef=m3["coef"], c=m3["c"], e=m3["e"], sig=m3["sig"], rho=m3["rho"],
                                          off=m3["off"], gamma=m3["gamma"], drive=m3["drive"]), drive=True)
    return j, 2 * max(m3["ll"] - max(m2["ll"], m2b["ll"]), 0.0)


def main():
    vi, K, n_sim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fname = sys.argv[4] if len(sys.argv) > 4 else "LH_cond_Sgate.mat"
    nrep = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    d = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_altqn.json"))
    # the M2 parameters are not all stored by r11_alt_qn; refit M2 once on the data to get them
    path = N.OUT / f"{fname[:-4]}_v{vi}_K{K}_nullqn.json"
    state = json.load(open(path)) if path.exists() else {}
    if "m2" not in state:
        data = sio.loadmat(N.DATA / fname)["savedat"][0, 0]["data"]
        betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
        zs = [data[vi, r].astype(float) - betas[r] for r in range(nrep)]
        slow = 1.0 / np.min(np.array(d["c"]) + np.array(d["e"]))
        best = None
        for m in (10, 30, 100):
            g = fit_multi_qn(zs, K, False, dict(coef=np.array(d["coef"]), sig=d["sig"], rho=d["rho"], c=np.array(d["c"]),
                             e=np.array(d["e"]), gamma=np.array(d["gamma"]), drive=(1 / (m * slow),) * 2,
                             max_rate=1.0 / (10 * slow)), drive=True)
            if best is None or g["ll"] > best["ll"]:
                best = g
        state["m2"] = dict(coef=best["coef"].tolist(), sig=best["sig"], rho=best["rho"], c=best["c"].tolist(),
                           e=best["e"].tolist(), gamma=best["gamma"].tolist(), drive=best["drive"].tolist(), ll=best["ll"])
        state["null"] = {}
        json.dump(state, open(path, "w"), indent=1)
    betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
    todo = [(j, state["m2"], K, betas, nrep) for j in range(n_sim) if str(j) not in state["null"]]
    with Pool(2) as p:
        for j, s in p.imap_unordered(job, todo):
            state["null"][str(j)] = s
            json.dump(state, open(path, "w"), indent=1)
            print(j, round(s, 2), flush=True)
    obs = d["lrt_coupling_given_drive"]
    null = np.array(list(state["null"].values()))
    print(json.dumps(dict(V=d["V"], obs=obs, n=len(null), p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                          q95=float(np.quantile(null, 0.95)), max=float(null.max()))), flush=True)


if __name__ == "__main__":
    main()
