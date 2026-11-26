"""More null copies (n_sim) for the one record significant at both K in r8_real."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r5_real as R
from rtnsim.realdata import load_b1500
from r8_real import job
NAME = "RTS_vg-0.645vd10.3_tsample1_25000_plc10.csv"
n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 200
r = [x for x in R.records() if x["file"] == NAME][0]
t, I, _ = load_b1500(r["path"])
t, y, _, dt = R.preprocess(t, I, r["trim_s"])
with Pool(2) as p:
    out = {f"K{K}": o for _, K, o in p.map(job, [(NAME, y, K, n_sim, 9100 + K) for K in (2, 3)])}
json.dump(out, open(ROOT / "results/r8/r8_candidate.json", "w"), indent=1)
print({k: {m: v[m] for m in ("p_r8", "p_path", "p_cnn", "p_comb", "J", "steps", "switches")} for k, v in out.items()})
