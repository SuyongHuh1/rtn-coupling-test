"""R17g: checklist item (6), measurement-chain null.  RTNinja (like real sampled data) records a transition
that falls inside a sample as an intermediate value; a K-trap decoder can read that sample as a multi-trap
transition, which the co-flip rate kappa absorbs.  Calibrate against the boxcar pipeline null (chain on a
sub-sample grid, averaged per sample) instead of the instantaneous one.  Records whose instant-null p is
already >= 0.05 are skipped (the boxcar null only adds spurious multi-trap transitions, so p can only grow).
    python experiments/r17g_calbox.py <nsrc>
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r17_rtninja_real import _job_b, DATA
from r17e_checklist import verdict
NSRC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSIM = 39
d = json.load(open(ROOT / f"results/r8/r17e_checklist_{NSRC}src.json"))["rows"]
recs = [r["rec"] for r in d if verdict(r, {}) == "coupling (false positive)"]
fi = ROOT / ("results/r8/r17f_cal3.json" if NSRC == 3 else f"results/r8/r17f_cal{NSRC}.json")
inst = {r["rec"]: r.get("p_cal") for r in (json.load(open(fi)) if fi.exists() else [])}
recs = [x for x in recs if inst.get(x) is None or inst[x] < 0.05]
base = DATA / f"n{NSRC}" / f"noise05_{NSRC}source"
OUTP = ROOT / f"results/r8/r17g_box{NSRC}.json"
out = json.load(open(OUTP)) if OUTP.exists() else []
done = {r.get("rec") for r in out}
todo = [(base / x, NSRC, -NSIM, 7000 + i) for i, x in enumerate(recs) if x not in done]
print("todo", len(todo), flush=True)
with Pool(2) as pool:
    for r in pool.imap_unordered(_job_b, todo):
        r["p_inst"] = inst.get(r["rec"])
        out.append(r); json.dump(out, open(OUTP, "w"), indent=1)
        print(r["rec"], round(r.get("stat", -1), 1), "inst", r["p_inst"], "box", r.get("p_cal"), r.get("error"), flush=True)
