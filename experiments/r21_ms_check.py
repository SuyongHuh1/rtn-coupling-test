"""R21: how much does multi-start (3 seeds, J = 0 likelihood picks) change the R9a result on simulated
records?  1/f + drift records of R10/R9a (r8_oracle.make, 100 pairs per K); the single-start statistics are
the stored R9a v3 values (r9a_g1_full_v3.json, seed = record index).  For a subset (first n pairs per K)
the 3-start run is computed here and compared: LRT change, J = 0 log-likelihood gain, and verdict changes
at the single-start 5 % threshold (set on the independent partners).

    python experiments/r21_ms_check.py [n_pairs=50]
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
NET = None


def job(a):
    global NET
    from rtnsim.r9 import load_stepnet, lrt_cnn3
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    i, y, K = a
    runs = []
    for s in (i, 10 ** 6 + i, 2 * 10 ** 6 + i):          # first seed = the stored single-start run
        st, f0, f1, *_ = lrt_cnn3(y, K, NET, ar1=True, rng=np.random.default_rng(s))
        runs.append((float(f0["ll"]), float(st)))
    best = max(runs, key=lambda r: r[0])
    return i, dict(single=runs[0], best=best, runs=runs)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    ys, ss, rows = O.make(100, white=False)
    ref = json.load(open(ROOT / "results/r8/r9a_g1_full_v3.json"))["res"]
    idx = []
    for K in (2, 3):
        kk = [i for i in range(len(rows)) if rows[i]["K"] == K]
        idx += [i for i in kk if rows[i]["coupled"] == 0][:n] + [i for i in kk if rows[i]["coupled"] == 1][:n]
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in idx], chunksize=2):
            out[i] = r
            json.dump({str(k): v for k, v in out.items()}, open(ROOT / "results/r8/r21_ms_check.json", "w"), indent=1)
    summ = {}
    for K in (2, 3):
        ii = [i for i in idx if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in ii])
        s1 = np.array([out[i]["single"][1] for i in ii]); s3 = np.array([out[i]["best"][1] for i in ii])
        stored = np.array([ref[str(i)]["lrt_cnn"] for i in ii])
        thr = np.quantile(np.array([ref[str(j)]["lrt_cnn"] for j in range(len(rows)) if rows[j]["K"] == K and rows[j]["coupled"] == 0]), 0.95)
        gain = np.array([out[i]["best"][0] - out[i]["single"][0] for i in ii])
        changed = (s1 > thr) != (s3 > thr)
        summ[f"K={K}"] = dict(n=len(ii), stored_matches_seed=float(np.mean(np.abs(stored - s1) < 1e-6 + 1e-6 * np.abs(s1))),
                             frac_decomp_changed=float(np.mean(gain > 1e-3)), median_ll_gain_when_changed=float(np.median(gain[gain > 1e-3])) if np.any(gain > 1e-3) else 0.0,
                             verdict_changed=float(np.mean(changed)), verdict_changed_indep=float(np.mean(changed[lab == 0])),
                             verdict_changed_coupled=float(np.mean(changed[lab == 1])),
                             fpr_single=float(np.mean(s1[lab == 0] > thr)), fpr_ms=float(np.mean(s3[lab == 0] > thr)),
                             tpr_single=float(np.mean(s1[lab == 1] > thr)), tpr_ms=float(np.mean(s3[lab == 1] > thr)))
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}), open(ROOT / "results/r8/r21_ms_check.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
