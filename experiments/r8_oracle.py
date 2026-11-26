"""R8 oracle ceiling: how much coupling information do the R7-simulator records carry?

For fresh matched pairs (K = 2, 3; no drive, point sampling) compare, at a 5% false-positive
threshold set on the independent partners:
  oracle_path   coupling statistic on the TRUE state paths (perfect idealisation)
  path / lrt / lrt_ar1   the three methods on the noisy records (K_fit = K_true)
Also splits by noise level (sigma / smallest step) and by number of switches.

    python experiments/r8_oracle.py [n_pairs_per_K=100] [workers=2]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim import sim2_torch as S  # noqa: E402
from rtnsim.robustk import idealise_k, coupling_stat  # noqa: E402
from rtnsim.cfhmm import lrt, lrt_ar1, lrt_ms  # noqa: E402

OUT = ROOT / "results" / "r8"
T = 4096


def make(n, seed=91, white=False):
    ys, ss, rows = [], [], []
    for K in (2, 3):
        g = torch.Generator().manual_seed(seed + K)
        got = 0
        while got < n:
            y, lab = S.make_batch(50, T, g, "cpu", p_drive=0, p_box=0, return_states=True,
                                   **(dict(p_drift=0, p_flicker=0) if white else {}))
            if int(lab["K"][0]) != K:
                continue
            take = min(50, n - got)
            for r in range(2 * take):
                st = lab["states"][r].numpy().astype(np.int64)
                ys.append(y[r].double().numpy()); ss.append(st)
                d = np.diff(y[r].double().numpy())
                noise = np.median(np.abs(d)) / 0.6745 / np.sqrt(2)
                rows.append(dict(K=K, coupled=int(lab["coupled"][r]), Jmax=float(lab["Jvec"][r].abs().max()), Jsum=float(lab["Jvec"][r].sum()),
                                 snr=float(lab["steps"][r].min() / max(noise, 1e-9)),
                                 switches=int(np.abs(np.diff(st, axis=0)).sum(0).min())))
            got += take
    return ys, ss, rows


def job(a):
    i, y, st, K = a
    rng = np.random.default_rng(i)
    f = idealise_k(y, K, rng=rng)
    if KALMAN:
        s_k, *_, p_k = lrt_ms(y, K, rng, kalman=True)
        s_ak, *_, p_ak = lrt_ms(y, K, rng, ar1=True, kalman=True)
        return i, dict(oracle_path=coupling_stat(st)[0], path=coupling_stat(f["S"])[0],
                       path_kal=p_k, lrt_kal=s_k, path_ar1_kal=p_ak, lrt_ar1_kal=s_ak)
    s_ms, *_, p_ms = lrt_ms(y, K, rng)
    s_ar = lrt_ms(y, K, rng, ar1=True)[0]
    return i, dict(oracle_path=coupling_stat(st)[0], path=coupling_stat(f["S"])[0],
                   path_ms=p_ms, lrt_ms=s_ms, lrt_ar1_ms=s_ar)


KALMAN = "kalman" in sys.argv


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    white = len(sys.argv) > 3 and sys.argv[3] == "white"
    ys, ss, rows = make(n, white=white)
    res = {}
    with Pool(w) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], ss[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            res[i] = r
    methods = (("oracle_path", "path", "path_kal", "lrt_kal", "path_ar1_kal", "lrt_ar1_kal") if KALMAN
               else ("oracle_path", "path", "path_ms", "lrt_ms", "lrt_ar1_ms"))
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        snr = np.array([rows[i]["snr"] for i in idx])
        sw = np.array([rows[i]["switches"] for i in idx])
        d = {}
        for m in methods:
            sc = np.array([res[i][m] for i in idx])
            s0, s1 = sc[lab == 0], sc[lab == 1]
            thr = np.quantile(s0, 0.95)
            hit = sc > thr
            d[m] = dict(auc=float((s1[:, None] > s0[None]).mean()), tpr=float(hit[lab == 1].mean()),
                        tpr_snr_lo=float(hit[(lab == 1) & (snr < np.median(snr))].mean()),
                        tpr_snr_hi=float(hit[(lab == 1) & (snr >= np.median(snr))].mean()),
                        tpr_sw_lo=float(hit[(lab == 1) & (sw < np.median(sw))].mean()),
                        tpr_sw_hi=float(hit[(lab == 1) & (sw >= np.median(sw))].mean()))
        d["median_snr"], d["median_min_switches"] = float(np.median(snr)), float(np.median(sw))
        summ[f"K={K}"] = d
    json.dump(dict(summary=summ, rows=rows, res={str(k): v for k, v in res.items()}),
              open(OUT / (("r8_oracle_white" if white else "r8_oracle") + ("_kal" if KALMAN else "_v3") + ".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
