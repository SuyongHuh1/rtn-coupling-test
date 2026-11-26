"""R18b: robustness of the R18 boundary to (i) energy window W (fraction of active traps), (ii) tail depth
(1e-4 vs 1e-6, SRAM-like worst-cell statistics), (iii) amplitude distribution (exponential vs equal)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import r18_sensitivity as R

def evaluate(devs, tails=(1e-4, 1e-6)):
    vs, ps, qs = [], [], []
    for a, eps, J in devs:
        v, p, q, m = R.device_dists(a, eps, J); vs.append(v); ps.append(p); qs.append(q)
    v = np.concatenate(vs); p = np.concatenate(ps) / len(devs); q = np.concatenate(qs) / len(devs)
    o = np.argsort(v)[::-1]; v, p, q = v[o], p[o], q[o]; cq = np.cumsum(q); cp = np.cumsum(p)
    out = {}
    for t in tails:
        i = np.searchsorted(cq, t); x = v[min(i, len(v) - 1)]
        j = np.searchsorted(cp, t); xc = v[min(j, len(v) - 1)]
        out[f"tail_ratio_{t:g}"] = float(p[v > x].sum() / max(q[v > x].sum(), 1e-300))
        out[f"q_shift_{t:g}"] = float(xc / x - 1)
    return out

rows = []
rng = np.random.default_rng(181)
for amp in ("exp", "equal"):
    for Wv in (1.0, 4.0):
        for sign in (1, -1):
            for f in (0.3, 1.0):
                for Jm in (0.5, 1.0, 2.0):
                    R.W, R.NDEV = Wv, 150000
                    devs = R.population(4, f, Jm, sign, rng)
                    if amp == "equal":
                        devs = [(np.ones_like(a), e, J) for a, e, J in devs]
                    r = evaluate(devs)
                    first = {t: (r[f"tail_ratio_{t}"] < 0.8 or r[f"tail_ratio_{t}"] > 1.25 or abs(r[f"q_shift_{t}"]) >= 0.05) for t in ("0.0001", "1e-06")}
                    rows.append(dict(amp=amp, W=Wv, sign="repulsive" if sign > 0 else "cooperative", f=f, Jm=Jm,
                                     **{k: round(v, 3) for k, v in r.items()}, first_1e4=first["0.0001"], first_1e6=first["1e-06"]))
                    print(rows[-1], flush=True)
json.dump(rows, open(ROOT / "results/r8/r18b_robust.json", "w"), indent=1)
