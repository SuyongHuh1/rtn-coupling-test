"""Pipeline-null calibration for one Nichol voltage: simulate the joint J = 0 fit with 60 Hz boxcar
integration (16x oversampled telegraph, block-averaged), AR(1) noise with the fitted (rho, sigma),
plus each repetition's estimated baseline; rerun the full pipeline (per-rep R9a + joint fit).
Resumable.   python experiments/r11_nichol_null.py <vi> <K> <n_sim> [file]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N  # noqa: E402
from rtnsim.cfhmm import Kinetics  # noqa: E402
from rtnsim.robustk import bits  # noqa: E402

OS = 16


def sim_rep(res, beta, T, rng):
    K = res["K"]
    kin = Kinetics(K, np.array(res["c"]), np.array(res["e"]), coupled=False)
    with torch.no_grad():
        P = torch.linalg.matrix_exp(kin.Q() / OS).numpy()
    P = np.clip(P, 0, None); P /= P.sum(1, keepdims=True)
    cum = np.cumsum(P, 1)
    w, V = np.linalg.eig(P.T); pi = np.real(V[:, np.argmax(np.real(w))]); pi = np.abs(pi) / np.abs(pi).sum()
    s = rng.choice(len(pi), p=pi)
    u = rng.random(T * OS)
    st = np.empty(T * OS, np.int64)
    for k in range(T * OS):
        s = int(np.searchsorted(cum[s], u[k])); s = min(s, len(pi) - 1); st[k] = s
    lev = bits(K)[st] @ np.array(res["coef"][1:]) + res["coef"][0]
    x = lev.reshape(T, OS).mean(1)
    rho, sig = res["rho"], res["sig"]
    e = rng.normal(0, sig, T)
    n = np.empty(T); n[0] = e[0] / np.sqrt(1 - rho ** 2)
    for k in range(1, T):
        n[k] = rho * n[k - 1] + e[k]
    return x + n + beta


def one(vi, K, j, fname):
    res = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"))
    betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
    rng = np.random.default_rng(10_000 + 100 * vi + j)
    T = betas.shape[1]
    ys = [sim_rep(res, betas[r] - np.median(betas[r]), T, rng) for r in range(len(betas))]
    reps = [N.per_rep((r, ys[r], K))[1] for r in range(len(ys))]
    zs = [ys[r] - np.asarray(reps[r]["beta"]) for r in range(len(ys))]
    f0, f1 = N.joint(zs, K, reps)
    return j, 2 * max(f1["ll"] - f0["ll"], 0.0)


def job(a):
    return one(*a)


def main():
    vi, K, n_sim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    fname = sys.argv[4] if len(sys.argv) > 4 else "Gate_P.mat"
    path = N.OUT / f"{fname[:-4]}_v{vi}_K{K}_null.json"
    done = json.load(open(path)) if path.exists() else {}
    todo = [(vi, K, j, fname) for j in range(n_sim) if str(j) not in done]
    with Pool(2) as p:
        for j, s in p.imap_unordered(job, todo):
            done[str(j)] = s
            json.dump(done, open(path, "w"))
    obs = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"))["lrt_joint"]
    null = np.array(list(done.values()))
    print(json.dumps(dict(vi=vi, K=K, obs=obs, n=len(null), p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                          null_q95=float(np.quantile(null, 0.95)))))


if __name__ == "__main__":
    main()
