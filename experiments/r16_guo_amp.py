"""R16: Guo et al. IRPS 2018 (P-TX.6) amplitude coupling strength on INDEPENDENT records.

Guo 2018 fits the state-based HMM of Zhang IRPS 2017 and defines, per RTN, the amplitude coupling
strength  s = lambda_c / lambda - 1  (amplitude with / without the other trap occupied).  They report
mean s of about -0.1 to -0.25 (n/p FinFET, VG = 0.4 V) and note both signs occur.  No decision
threshold is given.  Here: the same free 2^K-state HMM as r10_baselines (state-based HMM), additive
bit-pattern assignment, and s for every ordered pair (i | j) with the other traps empty.  Records are
the R10 records (independent dynamics AND no amplitude coupling, so the true s is 0).  Also the
time-constant strength d = tau(j occ)/tau(j empty) - 1 (Guo 2018 Fig. 4b) for comparison.

    python experiments/r16_guo_amp.py white|full
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from r10_baselines import assign, free_fit, pku_strength  # noqa: E402
from rtnsim.robustk import bits, idealise_k  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"


def amp_strength(mu, idx, K):
    B = bits(K)
    lev = mu[idx]                                  # level of each bit pattern
    pat = {tuple(b): n for n, b in enumerate(B)}
    s = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            def L(bi, bj):
                v = [0] * K; v[i] = bi; v[j] = bj
                return lev[pat[tuple(v)]]
            lam = L(1, 0) - L(0, 0)
            lamc = L(1, 1) - L(0, 1)
            s.append(lamc / lam - 1.0 if abs(lam) > 1e-12 else np.nan)
    return s


def job(a):
    i, y, K = a
    rng = np.random.default_rng(i)
    f = idealise_k(y, K, baseline=(COND != "white"), rng=rng)
    z = y - f["beta"]
    mu, w, P, sig, ll = free_fit(z, K, rng)
    idx = assign(mu, w, K)
    return i, dict(s=[float(v) for v in amp_strength(mu, idx, K)],
                   d=[float(v) for v in pku_strength(P, w, idx, K)])


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    todo = [i for i in range(len(ys)) if rows[i]["coupled"] == 0]
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in todo], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in todo if rows[i]["K"] == K]
        s = np.array([v for i in idx for v in out[i]["s"]], float)
        s = s[np.isfinite(s)]
        smax = np.array([np.nanmax(np.abs(out[i]["s"])) for i in idx])
        summ[f"K={K}"] = dict(
            n_records=len(idx), n_pairs=int(s.size),
            s_mean=float(s.mean()), s_median=float(np.median(s)), s_sd=float(s.std()),
            s_q05=float(np.quantile(s, 0.05)), s_q95=float(np.quantile(s, 0.95)),
            frac_abs_s_gt_0p1=float(np.mean(np.abs(s) > 0.1)),
            frac_abs_s_gt_0p2=float(np.mean(np.abs(s) > 0.2)),
            frac_record_any_abs_s_gt_0p1=float(np.mean(smax > 0.1)),
            frac_s_le_m0p1=float(np.mean(s <= -0.1)))
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r16_guo_amp_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
