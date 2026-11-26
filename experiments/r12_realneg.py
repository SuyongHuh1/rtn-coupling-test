"""Real-noise negative control: sums of two single-trap Nichol records from different repetitions
and different biases (RP1 0.5675 V: one 0.45 mV trap, dwell ~60-80 samples; 0.568 V: 0.35 mV trap,
dwell ~18 samples), the second scaled by g in {1.0, 1.5}.  The two traps come from different time
windows, so they are independent by construction; noise, drift and the 60 Hz boxcar are real.
Runs the existing baselines (PKU / Albrecht / Son) and R9a with pipeline-null calibration.
    python experiments/r12_realneg.py n_records n_sim
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.argv += ["--r9"]
import r8_calib as C
import r10_baselines as R
R.COND = "full"
import os
DATA = Path(os.environ.get("RTN_NICHOL", "data/nichol"))
OUT = ROOT / ("results/r8/r12_realneg" + (f"_ms{C.MS}" if C.MS > 1 else "") + os.environ.get("RTN_TAG", "") + ".json")


def records(n):
    d = sio.loadmat(DATA / "Gate_P.mat")["savedat"][0, 0]["data"]
    out = []
    for k in range(n):
        a, b, g = k % 20, (k * 7 + 3) % 20, (1.0, 1.5)[(k // 20) % 2]
        ya, yb = d[9, a].astype(float), d[10, b].astype(float)
        out.append(ya + g * (yb - np.median(yb)))
    return out


def job(a):
    i, y, n_sim = a
    base = {} if os.environ.get("RTN_CAL_ONLY") else R.job((i, y, 2))[1]
    _, cal = C.job((i, y, 2, n_sim))
    return i, dict(base, p_r9=cal["p_r8"], stat_r9=cal["stat_r8"], J=cal["J"])


def main():
    n, n_sim = int(sys.argv[1]), int(sys.argv[2])
    ys = records(n)
    res = json.load(open(OUT)) if OUT.exists() else {}
    todo = [(i, ys[i], n_sim) for i in range(n) if str(i) not in res]
    import os
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for i, r in p.imap_unordered(job, todo):
            res[str(i)] = r
            json.dump(res, open(OUT, "w"), indent=1)
            print(i, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "J"}, flush=True)
    rr = list(res.values())
    if os.environ.get("RTN_CAL_ONLY"):
        summ = dict(n=len(rr), ours_calibrated=float(np.mean([r["p_r9"] <= 0.05 for r in rr])),
                    ours_chi2=float(np.mean([chi2.sf(2 * r["stat_r9"], 1) < 0.05 for r in rr])))
    else:
      summ = dict(n=len(rr),
                pku_0p5=float(np.mean([r["pku_max"] > 0.5 for r in rr])),
                albrecht_e2=float(np.mean([r["evid"] > 2 for r in rr])),
                son_bonf=float(np.mean([r["son_minp"] < 0.05 / r["son_m"] for r in rr])),
                ours_chi2=float(np.mean([chi2.sf(2 * r["stat_r9"], 1) < 0.05 for r in rr])),
                ours_calibrated=float(np.mean([r["p_r9"] <= 0.05 for r in rr])))
    print(json.dumps(summ, indent=1))
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
