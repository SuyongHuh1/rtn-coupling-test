"""R19d: full-text Son rule (trap2 = smaller step | trap1 = larger step, one chi2(2) test at 0.05) on
genuine RTNinja activity (imec Zenodo 16750729), all records J = 0."""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r17_rtninja_real import records
from r19c_son_faithful import son_p
out = {}
for nsrc in (2, 3):
    rows = []
    for d in records(nsrc):
        fa = d / "RTNinja_Output_Attributes.pkl"
        if not fa.exists():
            continue
        a = pickle.load(open(fa, "rb"))
        act = [np.asarray(x).astype(np.int64) for x in a.get("activity") or []]
        if len(act) < 2:
            continue
        amp = np.abs(np.asarray(a["s_ampd"], float))
        S = np.stack(act, 1)[:, np.argsort(-amp)]
        rows.append(dict(rec=d.name, K=len(act), p=son_p(S)))
    allr = rows; kc = [r for r in rows if r["K"] == nsrc]
    out[f"{nsrc}source"] = dict(all_tested=dict(n=len(allr), fpr=float(np.mean([r["p"] < 0.05 for r in allr]))),
                                K_correct=dict(n=len(kc), fpr=float(np.mean([r["p"] < 0.05 for r in kc]))), rows=rows)
json.dump(out, open(ROOT / "results/r8/r19d_son_rtninja.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}))
