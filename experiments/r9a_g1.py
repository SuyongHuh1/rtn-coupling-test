"""G1: does the CNN step detector replace the R8 global decomposition search?

On the R8 oracle records (K = 2, 3; 100 pairs each), compare at a 5% false-positive threshold
(set on the independent partners) the R8 LRT started from the CNN only (lrt_cnn) with the stored
global-search results (r8_oracle_*_v3.json: lrt_ms white / lrt_ar1_ms full), and time both.
Also: K estimated from step heights vs truth.

    python experiments/r9a_g1.py white|full [stepnet path]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from rtnsim.r9 import load_stepnet, lrt_cnn, lrt_hybrid, lrt_hybrid2, lrt_cnn3, k_from_steps  # noqa: E402
V3 = '--v3' in sys.argv
import os
MS = int(os.environ.get("RTN_MS", "1")); WORKERS = int(os.environ.get("RTN_WORKERS", "2"))
HYB = '--hybrid' in sys.argv
HYB2 = '--hybrid2' in sys.argv
from rtnsim.stepnet import denoise  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"
NET = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else str(ROOT / "results/r9a_gpu/stepnet.pt")
_net = None


def job(a):
    global _net
    i, y, K = a
    if _net is None:
        _net = load_stepnet(NET)
    t0 = time.time()
    if MS > 1:
        from rtnsim.r9 import lrt_cnn3_ms
        fn = lambda *a, **k: lrt_cnn3_ms(*a, n_start=MS, **k)
    else:
        fn = None
    fn = fn or (lrt_cnn3 if V3 else (lrt_hybrid2 if HYB2 else (lrt_hybrid if HYB else lrt_cnn)))
    s, f0, f1, beta, p = fn(y, K, _net, ar1=(COND != "white"), rng=np.random.default_rng(i))
    dt = time.time() - t0
    lev, edge, step = denoise(_net, y)
    k_hat, prim = k_from_steps(edge, step)
    return i, dict(lrt_cnn=s, path_cnn=p, sec=dt, K_hat=int(k_hat))


def stats(sc, lab):
    s0, s1 = sc[lab == 0], sc[lab == 1]
    thr = np.quantile(s0, 0.95)
    return dict(auc=float((s1[:, None] > s0[None]).mean()), tpr=float((s1 > thr).mean()))


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    ref = json.load(open(ROOT / f"results/r8/r8_oracle{'_white' if COND == 'white' else ''}_v3.json"))["res"]
    key = "lrt_ms" if COND == "white" else "lrt_ar1_ms"
    out = {}
    with Pool(WORKERS) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        summ[f"K={K}"] = dict(
            cnn=stats(np.array([out[i]["lrt_cnn"] for i in idx]), lab),
            path_cnn=stats(np.array([out[i]["path_cnn"] for i in idx]), lab),
            global_search=stats(np.array([ref[str(i)][key] for i in idx]), lab),
            oracle=stats(np.array([ref[str(i)]["oracle_path"] for i in idx]), lab),
            sec_per_record_cnn=float(np.mean([out[i]["sec"] for i in idx])),
            K_hat_accuracy=float(np.mean([out[i]["K_hat"] == K for i in idx])),
            K_hat_counts={str(k): int(sum(out[i]["K_hat"] == k for i in idx)) for k in (1, 2, 3, 4)})
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r9a_g1_{COND}{'_v3' if V3 else ('_hyb2' if HYB2 else ('_hyb' if HYB else ''))}{f'_ms{MS}' if MS > 1 else ''}{os.environ.get('RTN_TAG', '')}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
