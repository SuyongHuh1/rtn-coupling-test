"""R19b: full-text Albrecht procedure on the R12 real-noise negative control (40 Nichol sums, J = 0)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.argv = [sys.argv[0], "full"]
import r19_albrecht_faithful as A
from r12_realneg import records

def job(a):
    i, y = a
    return A.job((i, y, 2))

if __name__ == "__main__":
    ys = records(40)
    with Pool(2) as p:
        out = dict(p.map(job, list(enumerate(ys))))
    keep = [r for r in out.values() if r["dwell_min"] >= 2]
    s = dict(n=len(out), n_excluded=len(out) - len(keep),
             alb_e2=float(np.mean([r["e"] > 2 for r in keep])) if keep else None,
             alb_full=float(np.mean([r["e"] > 2 and r.get("rate_diff", False) for r in keep])) if keep else None,
             alb_e2_no_excl=float(np.mean([r["e"] > 2 for r in out.values()])),
             e_range=[float(min(r["e"] for r in out.values())), float(max(r["e"] for r in out.values()))])
    json.dump(dict(summary=s, res={str(k): v for k, v in out.items()}), open(ROOT / "results/r8/r19b_albrecht_realneg.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
