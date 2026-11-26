"""Power and J uncertainty for the LH coupling candidates (model: K traps + slow drive + coupling).
Generator = the coupled + drive fit (M3) on reps 0-9 with its J scaled by s in SCALES (s = 1: the
estimate itself = parametric bootstrap of J-hat); analysis = M2 vs M3 joint QN fits (oracle start).
Threshold = the stored drive-null replicates (r11_null_qn).   python experiments/r13_power_drive.py <vi> <n> [nrep]
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

FN = "LH_cond_Sgate.mat"
K = 3
SCALES = (0.25, 0.5, 1.0)


def simulate(m, Jv, betas, rng):
    kin = KineticsDrive(K, np.array(m["c"]), np.array(m["e"]), J=np.array(Jv), coupled=True,
                        gamma=np.array(m["gamma"]), drive=np.array(m["drive"]))
    with torch.no_grad():
        P = torch.linalg.matrix_exp(kin.Q()).clamp_min(0).numpy()
    P /= P.sum(1, keepdims=True); cum = np.cumsum(P, 1)
    w, V = np.linalg.eig(P.T); pi = np.abs(np.real(V[:, np.argmax(np.real(w))])); pi /= pi.sum()
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)]); mu = np.vstack([X, X]) @ np.array(m["coef"])
    out = []
    for b in betas:
        T = b.size
        s = rng.choice(len(pi), p=pi); u = rng.random(T); st = np.empty(T, int)
        for k in range(T):
            s = min(int(np.searchsorted(cum[s], u[k])), len(pi) - 1); st[k] = s
        e = rng.normal(0, m["sig"], T); n = np.empty(T); n[0] = e[0] / np.sqrt(1 - m["rho"] ** 2)
        for k in range(1, T):
            n[k] = m["rho"] * n[k - 1] + e[k]
        out.append(mu[st] + n)
    return out


def analyse(zs, m, Jv):
    slow = 1.0 / np.min(np.array(m["c"]) + np.array(m["e"]))
    ini = dict(coef=np.array(m["coef"]), sig=m["sig"], rho=m["rho"], c=np.array(m["c"]), e=np.array(m["e"]),
               gamma=np.array(m["gamma"]), drive=np.array(m["drive"]), max_rate=1.0 / (10 * slow))
    m2 = fit_multi_qn(zs, K, False, ini, drive=True)
    base = dict(ini, coef=m2["coef"], c=m2["c"], e=m2["e"], sig=m2["sig"], rho=m2["rho"], off=m2["off"],
                gamma=m2["gamma"], drive=m2["drive"])
    m3 = max((fit_multi_qn(zs, K, True, dict(base, J=j0), drive=True) for j0 in (np.zeros(3), np.array(Jv))),
             key=lambda f: f["ll"])
    m2b = fit_multi_qn(zs, K, False, dict(base, coef=m3["coef"], c=m3["c"], e=m3["e"], sig=m3["sig"], rho=m3["rho"],
                                          off=m3["off"], gamma=m3["gamma"], drive=m3["drive"]), drive=True)
    return 2 * max(m3["ll"] - max(m2["ll"], m2b["ll"]), 0.0), m3


def job(a):
    s, j, m, Jhat, betas = a
    rng = np.random.default_rng(int(100 * s) * 1000 + j)
    Jv = (s * np.array(Jhat)).tolist()
    zs = simulate(m, Jv, betas, rng)
    stat, m3 = analyse(zs, m, Jv)
    return s, j, stat, m3["J"].tolist()


def main():
    vi, n = int(sys.argv[1]), int(sys.argv[2])
    nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    path = N.OUT / f"LH_cond_Sgate_v{vi}_K3_powerdrive.json"
    state = json.load(open(path)) if path.exists() else {}
    betas = np.load(N.OUT / f"LH_cond_Sgate_v{vi}_K3_beta.npy")[:nrep]
    if "m3" not in state:
        d = sio.loadmat(N.DATA / FN)["savedat"][0, 0]["data"]
        zs = [d[vi, r].astype(float) - betas[r] for r in range(nrep)]
        m2 = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["m2"]
        Jg = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_altqn.json"))["J_given_drive"]
        stat, m3 = analyse(zs, m2, Jg)
        state["m3"] = dict(coef=m3["coef"].tolist(), sig=m3["sig"], rho=m3["rho"], c=m3["c"].tolist(),
                           e=m3["e"].tolist(), gamma=m3["gamma"].tolist(), drive=m3["drive"].tolist(),
                           J=m3["J"].tolist(), obs=stat)
        state["runs"] = {}
        json.dump(state, open(path, "w"), indent=1)
        print("obs", round(stat, 1), np.round(m3["J"], 2), flush=True)
    m = state["m3"]
    todo = [(s, j, m, m["J"], betas) for s in SCALES for j in range(n) if f"{s}_{j}" not in state["runs"]]
    with Pool(2) as p:
        for s, j, stat, Jh in p.imap_unordered(job, todo):
            state["runs"][f"{s}_{j}"] = dict(s=s, stat=stat, J=Jh)
            json.dump(state, open(path, "w"), indent=1)
            print(s, j, round(stat, 1), np.round(Jh, 2), flush=True)
    null = np.array(list(json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["null"].values()))
    thr = float(np.quantile(null, 0.95))
    summ = dict(obs=m["obs"], J=m["J"], thr=thr)
    for s in SCALES:
        rr = [v for v in state["runs"].values() if v["s"] == s]
        Js = np.array([v["J"] for v in rr])
        summ[str(s)] = dict(n=len(rr), power=float(np.mean([v["stat"] > thr for v in rr])),
                            J_true=(s * np.array(m["J"])).round(3).tolist(), J_mean=Js.mean(0).round(3).tolist(),
                            J_sd=Js.std(0).round(3).tolist())
    state["summary"] = summ
    json.dump(state, open(path, "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
