"""R17c: our R15 RTNinja reimplementation vs the genuine RTNinja output on the same imec records.

For records where RTNinja found the true number of sources, run r15 extract_levels + map_sources with
K = true, then compare per-sample activity with RTNinja's (best source permutation), and the number of
levels found.  Also report the coupling-rule outcome of both on the same record.
    python experiments/r17c_reimpl_vs_rtninja.py
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.argv = [sys.argv[0], "white"]
import r15_rtninja as R  # noqa: E402
from r17_rtninja_real import DATA, guo_d, son_minp  # noqa: E402

out = {}
for nsrc in (2, 3):
    rows = []
    base = DATA / f"n{nsrc}" / f"noise05_{nsrc}source"
    for d in sorted(p for p in base.glob("mc*_n05_dataset") if p.is_dir()):
        fa, fm = d / "RTNinja_Output_Attributes.pkl", d / "MeasurementData.npy"
        if not (fa.exists() and fm.exists()):
            continue
        a = pickle.load(open(fa, "rb"))
        act = [np.asarray(x).astype(np.int64) for x in a.get("activity") or []]
        if len(act) != nsrc:
            continue
        S_true = np.stack(act, 1)
        y = np.load(fm).astype(float)
        mus, path, sd = R.extract_levels(y - np.median(y))
        S = R.map_sources(mus, path, sd, nsrc)
        agree = max(np.mean(S[:, list(pm)] == S_true) for pm in permutations(range(nsrc)))
        ls = list(d.glob("LevelsOutput_*_Lstats.xlsx"))
        nlev = int(re.search(r"_(\d+)L_", ls[0].name).group(1)) if ls else None
        pm_o, m_o = son_minp(S_true)
        pm_r, m_r = son_minp(S)
        rows.append(dict(rec=d.name, agree=float(agree), lev_rtninja=nlev, lev_reimpl=int(len(mus)),
                         son_rtninja=bool(pm_o < 0.05 / m_o), son_reimpl=bool(pm_r < 0.05 / m_r),
                         guo_rtninja=bool(guo_d(S_true) > 0.5), guo_reimpl=bool(guo_d(S) > 0.5)))
    ag = np.array([r["agree"] for r in rows])
    out[f"{nsrc}source"] = dict(
        n=len(rows), agree_median=float(np.median(ag)), agree_q10=float(np.quantile(ag, 0.1)),
        frac_agree_ge_0p95=float(np.mean(ag >= 0.95)),
        same_levels=float(np.mean([r["lev_rtninja"] == r["lev_reimpl"] for r in rows])),
        son_fpr_rtninja=float(np.mean([r["son_rtninja"] for r in rows])), son_fpr_reimpl=float(np.mean([r["son_reimpl"] for r in rows])),
        guo_fpr_rtninja=float(np.mean([r["guo_rtninja"] for r in rows])), guo_fpr_reimpl=float(np.mean([r["guo_reimpl"] for r in rows])),
        son_same_call=float(np.mean([r["son_rtninja"] == r["son_reimpl"] for r in rows])),
        rows=rows)
json.dump(out, open(ROOT / "results/r8/r17c_reimpl_vs_rtninja.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}, indent=1))
