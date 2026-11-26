"""Nichol Si/SiGe charge-sensor series (Gate_P: RP1 0.563-0.568 V, 20 reps x 8192 @ 60 Hz).

Per voltage and K: per-repetition R9a fits (CNN baseline + decomposition search), then a joint
multi-segment fit (shared kinetics/amplitudes/AR(1) noise, per-rep offset) with and without J.
Saves per-voltage json (resumable).   python experiments/r11_nichol.py [file] [K ...]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim.r9 import load_stepnet, lrt_cnn3  # noqa: E402
from rtnsim.cfhmm import fit_ar1_multi  # noqa: E402
from rtnsim.stepnet import denoise  # noqa: E402

DATA = Path(__import__("os").environ.get("RTN_NICHOL", "data/nichol"))
OUT = ROOT / "results/nichol"
_net = None


def per_rep(a):
    global _net
    r, y, K = a
    if _net is None:
        _net = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    s, f0, f1, beta, _ = lrt_cnn3(y, K, _net, ar1=True, rng=np.random.default_rng(r))
    lev, edge, step = denoise(_net, y)
    eh = np.abs(step[edge > 0.5]) if np.any(edge > 0.5) else np.array([])
    return r, dict(stat=float(s), coef=f0["coef"].tolist(), sig=f0["sig"], rho=f0["rho"],
                   c=f0["c"].tolist(), e=f0["e"].tolist(), J1=f1["J"].tolist(), beta=beta, ll=f0["ll"],
                   edges=eh.tolist())


def joint(zs, K, reps):
    """Joint J = 0 fit from several starts (median of rep fits + individual reps), then J free."""
    co = np.array([np.r_[r["coef"][0], np.sort(np.abs(r["coef"][1:]))[::-1]] for r in reps])
    starts = [dict(coef=np.median(co, 0), sig=float(np.median([r["sig"] for r in reps])),
                   rho=float(np.median([r["rho"] for r in reps])),
                   c=np.median([r["c"] for r in reps], 0), e=np.median([r["e"] for r in reps], 0))]
    order = np.argsort([-r["ll"] for r in reps])[:4]
    for i in order:
        r = reps[i]
        starts.append(dict(coef=np.array(r["coef"]), sig=r["sig"], rho=r["rho"], c=np.array(r["c"]), e=np.array(r["e"])))
    best = None
    for st in starts:
        st = dict(st, off=np.zeros(len(zs)))
        f0 = fit_ar1_multi(zs, K, False, st, n_iter=60)
        if best is None or f0["ll"] > best["ll"]:
            best = f0
    f0 = best
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0["rho"], c=f0["c"], e=f0["e"], off=f0["off"])
    f1 = fit_ar1_multi(zs, K, True, ini, n_iter=60)
    f0b = fit_ar1_multi(zs, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1["rho"], c=f1["c"],
                                           e=f1["e"], off=f1["off"]), n_iter=60)
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    return f0, f1


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "Gate_P.mat"
    Ks = [int(k) for k in sys.argv[2:]] or [2, 3]
    s = sio.loadmat(DATA / fname)["savedat"][0, 0]
    d = s["data"]
    V = s["Vgate"].ravel() if "Vgate" in s.dtype.names else np.arange(d.shape[0])
    OUT.mkdir(parents=True, exist_ok=True)
    for vi in range(d.shape[0]):
        for K in Ks:
            path = OUT / f"{fname[:-4]}_v{vi}_K{K}.json"
            if path.exists():
                continue
            ys = [d[vi, r].astype(float) for r in range(d.shape[1])]
            with Pool(2) as p:
                reps = dict(p.imap_unordered(per_rep, [(r, ys[r], K) for r in range(len(ys))]))
            reps = [reps[r] for r in range(len(ys))]
            zs = [ys[r] - np.asarray(reps[r]["beta"]) for r in range(len(ys))]
            f0, f1 = joint(zs, K, reps)
            res = dict(V=float(V[vi]), K=K, lrt_joint=2 * max(f1["ll"] - f0["ll"], 0.0), J=f1["J"].tolist(),
                       bic0=f0["bic"], coef=f0["coef"].tolist(), sig=f0["sig"], rho=f0["rho"],
                       c=f0["c"].tolist(), e=f0["e"].tolist(), c1=f1["c"].tolist(), e1=f1["e"].tolist(),
                       rep_stats=[r["stat"] for r in reps], rep_J=[r["J1"] for r in reps],
                       edges=np.concatenate([r["edges"] for r in reps]).tolist(),
                       switches=[int(np.abs(np.diff(((np.asarray(p)[:, None] >> np.arange(K)[::-1]) & 1), axis=0)).sum())
                                 for p in f0["paths"]])
            json.dump(res, open(path, "w"))
            np.save(OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy", np.array([r["beta"] for r in reps]))
            print(f"V={V[vi]:.4f} K={K} LRT={res['lrt_joint']:.2f} J={np.round(res['J'], 2)} "
                  f"bic0={res['bic0']:.1f} coef={np.round(res['coef'], 5)} "
                  f"rep_sig={np.mean(np.array(res['rep_stats']) > 3.84):.2f}", flush=True)


if __name__ == "__main__":
    main()
