"""R32: checklist item (6) -- does the boxcar (measurement-chain) null keep power for real co-flips?

Two traps; the chain runs on a sub-sample grid (SUB steps per sample) and each sample is the boxcar average,
as in sampled hardware and in RTNinja.  Per record pair:
  coop   independent traps plus 00 <-> 11 co-flips at rate kappa (R31 'coop' draw)
  indep  the I-projection partner (same per-trap occupancy and switching flux, no coupling)
Each record: pipeline LRT (mode from RTN_JOINT), then p from the instant null and from the boxcar null
(NSIM each).  Reports FPR (indep) and power (coop) at p < 0.05 for both nulls.
    python experiments/r32_box_power.py <n_pairs> <workers> [T] [scen]
scen = coop (default: 00<->11 co-flips at rate kappa) or J (state-dependent rates, |J| ~ U(0.7, 3) kT, phi = 0.5).
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
T = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
SCEN = sys.argv[4] if len(sys.argv) > 4 else "coop"
SUB, NSIM = 20, 39
MODE = int(os.environ.get("RTN_JOINT", "0") or 0)


def chain_Q(c, e, kappa=0.0, J=0.0, phi=0.5):
    Q = np.zeros((4, 4))
    for s in range(4):
        o0, o1 = (s >> 1) & 1, s & 1                  # occupancy of the other trap
        Q[s, s ^ 1] += c[0] * np.exp(-phi * J * o0) if not s & 1 else e[0] * np.exp((1 - phi) * J * o0)
        Q[s, s ^ 2] += c[1] * np.exp(-phi * J * o1) if not s & 2 else e[1] * np.exp((1 - phi) * J * o1)
    Q[0, 3] += kappa; Q[3, 0] += kappa
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def stationary(Q):
    w, v = np.linalg.eig(Q.T)
    p = np.real(v[:, np.argmin(np.abs(w))]); return p / p.sum()


def iproj(Q):
    p = stationary(Q)
    c, e = np.zeros(2), np.zeros(2)
    for k, b in enumerate((1, 2)):
        occ = sum(p[s] for s in range(4) if s & b)
        flux = sum(p[s] * Q[s, t] for s in range(4) for t in range(4) if s != t and ((s ^ t) & b) and not (s & b))
        c[k], e[k] = flux / (1 - occ), flux / occ
    return c, e


def sample_box(Q, rng):
    """Boxcar-averaged occupancies (T, 2) of the chain with generator Q per sample."""
    cum = np.cumsum(expm(Q / SUB), 1)
    n = T * SUB
    s = np.empty(n, np.int64); s[0] = rng.choice(4, p=stationary(Q))
    u = rng.random(n)
    for t in range(1, n):
        s[t] = min(np.searchsorted(cum[s[t - 1]], u[t]), 3)
    occ = np.stack([s & 1, (s >> 1) & 1], 1).astype(float)
    return occ.reshape(T, SUB, 2).mean(1)


def make(i):
    rng = np.random.default_rng(32000 + i)
    tau = np.exp(np.log(8) + rng.random(2) * (np.log(200) - np.log(8)))
    p = 0.15 + 0.7 * rng.random(2)
    c, e = 2 / tau * p, 2 / tau * (1 - p)
    kap = (0.2 + 0.8 * rng.random()) * np.mean(np.r_[c, e])
    rj = np.random.default_rng(33000 + i)            # separate stream: coop draws stay as before
    J = (0.7 + 2.3 * rj.random()) * rj.choice([-1, 1])
    Q1 = chain_Q(c, e, kap) if SCEN == "coop" else chain_Q(c, e, J=J)
    c0, e0 = iproj(Q1); Q0 = chain_Q(c0, e0)
    a = np.array([1.0, 0.15 + 0.85 * rng.random()])
    sig = a.min() * (0.1 + 0.3 * rng.random())
    noise = sig * rng.standard_normal(T)
    return [(lab, sample_box(Q, rng) @ a + noise, 900 + i) for lab, Q in (("coop", Q1), ("indep", Q0))]


def simulate_null_box(f0, beta, resid, T, rng, sub=20):
    """J = 0 null that reproduces the measurement chain: chain simulated on a sub-sample grid
    (rates / sub per step) and boxcar-averaged over `sub` steps, as in the imec simulator."""
    from rtnsim.robustk import _block_boot
    K = len(f0["c"])
    occ = np.zeros((T, K))
    for k in range(K):
        c, e = f0["c"][k] / sub, f0["e"][k] / sub
        kt = c + e
        q = 1 - np.exp(-kt)
        p01, p10 = c / kt * q, e / kt * q
        st = int(rng.random() < c / kt)
        u = rng.random(T * sub)
        x = np.empty(T * sub)
        for t in range(T * sub):
            st = st ^ int(u[t] < (p01 if st == 0 else p10))
            x[t] = st
        occ[:, k] = x.reshape(T, sub).mean(1)
    # residual noise: remove boxcar-induced transition outliers from the bootstrap pool
    r = resid - np.median(resid)
    mad = np.median(np.abs(r)) / 0.6745
    r = np.where(np.abs(r) > 5 * mad, rng.normal(0, mad, r.size), r)
    return beta + f0["coef"][0] + occ @ f0["coef"][1:] + _block_boot(r, T, rng)


def job(a):
    from rtnsim.cfhmm import lrt_ms, simulate_null
    from rtnsim.robustk import bits
    lab, y, seed = a
    try:
        rng = np.random.default_rng(seed)
        stat, f0, f1, beta, _ = lrt_ms(y, 2, rng)
        X = np.hstack([np.ones((4, 1)), bits(2)])
        resid = y - beta - (X @ f0["coef"])[f0["path"]]
        out = dict(lab=lab, seed=seed, stat=float(stat), kappa=[float(x) for x in np.atleast_1d(f1.get("kappa", []))])
        for name, sim in (("inst", simulate_null), ("box", simulate_null_box)):
            null = np.array([lrt_ms(sim(f0, beta, resid, len(y), rng), 2, rng)[0] for _ in range(NSIM)])
            out["p_" + name] = float((1 + np.sum(null >= stat)) / (1 + NSIM))
            out["null_" + name] = [float(x) for x in null]
        return out
    except Exception as ex:
        return dict(lab=lab, seed=seed, error=repr(ex)[:200])


if __name__ == "__main__":
    outp = ROOT / (f"results/r32_box_power_j{MODE}.json" if SCEN == "coop" else f"results/r32_box_power_{SCEN}_j{MODE}.json")
    res = json.load(open(outp)) if outp.exists() else []
    done = {(r["lab"], r["seed"]) for r in res}
    todo = [x for i in range(N) for x in make(i) if (x[0], x[2]) not in done]
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, todo):
            res.append(r); json.dump(res, open(outp, "w"))
            print(r["lab"], r.get("stat"), r.get("p_inst"), r.get("p_box"), r.get("error"), flush=True)
    summ = {}
    for lab in ("coop", "indep"):
        rr = [r for r in res if r["lab"] == lab and "error" not in r]
        summ[lab] = dict(n=len(rr), n_err=sum(1 for r in res if r["lab"] == lab and "error" in r),
                         rej_inst=float(np.mean([r["p_inst"] < 0.05 for r in rr])) if rr else None,
                         rej_box=float(np.mean([r["p_box"] < 0.05 for r in rr])) if rr else None)
    summ["mode"], summ["scen"] = MODE, SCEN
    print(json.dumps(summ))
    json.dump(dict(summary=summ, rows=res), open(outp, "w"))
