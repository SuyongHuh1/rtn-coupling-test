"""Zhang et al. IRPS 2017 ("New insights into the amplitude of RTN", PKU) as described in the paper:
state-based HMM fitted directly on the drain-current states, the number of states taken from the
data, and coupling diagnosed from the state-transition diagram when fewer than 2^K states appear
(their Fig. 2: 3 traps, 6 states instead of 8).  Reproduced rules on the R10 records:

  zhang_bic   free HMM with M = K+1 ... 2^K states, M by BIC; coupled if M_BIC < 2^K
  zhang_occ   free 2^K-state HMM; coupled if any state has stationary occupancy < 1 %

Same records and preprocessing as r10_baselines.py.    python experiments/r10b_zhang.py white|full
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
from rtnsim.cfhmm import _free_hmm  # noqa: E402
from rtnsim.robustk import idealise_k  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"


def fit(z, M, rng, n_starts=3):
    lo, hi = np.quantile(z, [0.01, 0.99])
    best = None
    for s in range(n_starts):
        mu0 = np.quantile(z, (np.arange(M) + 0.5) / M) if s == 0 else np.sort(lo + (hi - lo) * rng.random(M))
        try:
            r = _free_hmm(z, M, mu0, n_iter=100)
        except ZeroDivisionError:
            continue
        if best is None or r[-1] > best[-1]:
            best = r
    return best


def job(a):
    i, y, K = a
    rng = np.random.default_rng(i)
    f = idealise_k(y, K, baseline=(COND != "white"), rng=rng)
    z = y - f["beta"]
    T = z.size
    bic, occ_min = {}, None
    for M in range(K + 1, 2 ** K + 1):
        r = fit(z, M, rng)
        if r is None:
            continue
        mu, w, P, sig, ll = r
        npar = M * (M - 1) + M + 1
        bic[M] = -2 * ll + npar * np.log(T)
        if M == 2 ** K:
            occ_min = float(np.min(w))
    Mb = min(bic, key=bic.get)
    return i, dict(M_bic=int(Mb), occ_min=occ_min)


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    out = {}
    with Pool(1) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        calls = dict(zhang_bic=np.array([out[i]["M_bic"] < 2 ** K for i in idx]),
                     zhang_occ=np.array([out[i]["occ_min"] is not None and out[i]["occ_min"] < 0.01 for i in idx]))
        d = dict(n_indep=int((lab == 0).sum()), n_coupled=int((lab == 1).sum()))
        for k, c in calls.items():
            d[k] = dict(fpr=float(c[lab == 0].mean()), tpr=float(c[lab == 1].mean()))
        d["M_bic_hist_indep"] = np.bincount([out[i]["M_bic"] for i, l in zip(idx, lab) if l == 0], minlength=2 ** K + 1).tolist()
        summ[f"K={K}"] = d
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r10b_zhang_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
