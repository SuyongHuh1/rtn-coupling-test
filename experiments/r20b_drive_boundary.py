"""R20b: step 4 (common-drive alternative) for the Southampton record that reaches the decision level only
under the paper's rule p <= 0.05 (RTS_vg-0.64vd10.3, K = 3, calibrated p = 0.05 with B = 39).
R20 ran step 4 only for p < 0.05.  Three seeds of lrt_drive; output results/r8/r20b_drive_boundary.json.

    RTN_EXT=<folder with edtm/ and ssdm/> python experiments/r20b_drive_boundary.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import r5_real as R
from rtnsim.realdata import load_b1500
from rtnsim.cfhmm import lrt_drive

REC = "RTS_vg-0.64vd10.3_tsample1_25000_plc10.csv"


def main():
    rec = [r for r in R.records() if r["file"] == REC][0]
    t, I, _ = load_b1500(rec["path"])
    y = R.preprocess(t, I, rec["trim_s"])[1]
    out = []
    for s in (1, 2, 3):
        sd, g0, g1, *_ = lrt_drive(y, 3, np.random.default_rng(9300 + s))
        out.append(dict(seed=s, stat_drive=float(sd), p=float(chi2.sf(sd, 3)),
                        J=[float(x) for x in np.atleast_1d(g1["J"])]))
        print(out[-1], flush=True)
    json.dump(dict(rec=REC, K=3, p_cal=0.05, runs=out), open(ROOT / "results/r8/r20b_drive_boundary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
