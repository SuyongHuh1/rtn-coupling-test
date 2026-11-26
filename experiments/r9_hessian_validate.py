"""Validate the Hessian diagnostics on synthetic K = 2 records (white noise, where the R8 test is
near its ceiling): (1) predicted detection probability from I_eff (J = 0 fit) and the true J vs
actual detection at the same threshold; (2) coverage of the sandwich 95% interval for J."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from rtnsim.cfhmm import lrt_ms  # noqa: E402
from rtnsim.fisher import diagnostics  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def job(a):
    i, y, J_true = a
    rng = np.random.default_rng(i)
    s, f0, f1, beta, _ = lrt_ms(y, 2, rng)
    yb = y - beta
    d0 = diagnostics(yb, dict(f0, J=[0.0]))
    d1 = diagnostics(yb, f1)
    return i, dict(lrt=s, J_true=J_true, J_hat=float(f1["J"][0]), I_eff=d0["I_eff"][0], conf=d0["conf"][0],
                   J_min=d0["J_min"][0], se=d1["se_J"][0], se_model=d1["se_J_model"][0])


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    ref = json.load(open(ROOT / f"results/r8/r8_oracle{'_white' if COND == 'white' else ''}_v3.json"))
    res0 = ref["res"]
    idx0 = [i for i in range(len(rows)) if rows[i]["K"] == 2 and rows[i]["coupled"] == 0]
    thr = float(np.quantile([res0[str(i)]["lrt_ms"] for i in idx0], 0.95))
    idx1 = [i for i in range(len(rows)) if rows[i]["K"] == 2 and rows[i]["coupled"] == 1][:N]
    # signed true J for the single pair (the simulator stores |J| max; recompute sign-free comparison)
    tasks = [(i, ys[i], rows[i]["Jmax"]) for i in idx1]
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, tasks):
            out[i] = r
    R = list(out.values())
    pred = np.array([ncx2.sf(thr, 1, r["I_eff"] * r["J_true"] ** 2) for r in R])
    hit = np.array([r["lrt"] > thr for r in R])
    # calibration of predicted power in bins
    bins = [0, 0.3, 0.7, 0.95, 1.01]
    calib = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (pred >= a) & (pred < b)
        if m.any():
            calib.append(dict(bin=f"{a}-{b}", n=int(m.sum()), predicted=float(pred[m].mean()), observed=float(hit[m].mean())))
    cover = np.mean([abs(abs(r["J_hat"]) - r["J_true"]) <= 1.96 * r["se"] for r in R if np.isfinite(r["se"])])
    cover_m = np.mean([abs(abs(r["J_hat"]) - r["J_true"]) <= 1.96 * r["se_model"] for r in R if np.isfinite(r["se_model"])])
    summ = dict(cond=COND, threshold=thr, n=len(R), mean_predicted=float(pred.mean()), observed=float(hit.mean()),
                calibration=calib, coverage_sandwich=float(cover), coverage_model=float(cover_m),
                median_J_min=float(np.median([r["J_min"] for r in R])), median_conf=float(np.median([r["conf"] for r in R])))
    json.dump(dict(summary=summ, rows={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r9_hessian_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
