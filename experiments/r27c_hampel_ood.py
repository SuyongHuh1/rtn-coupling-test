"""R27c: Hampel pre-filter on the other R25 out-of-distribution sets (lorentz, slow, skewed): same pairs and
seeds as R25, compared with R25's raw R9a statistics."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r25_ood as R
NET = None


def job(a):
    global NET
    c, i = a
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    from rtnsim.tnoise import hampel
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    ys, st = R.make_pair(c, i)
    out = []
    for lab, y in zip((1, 0), ys):
        s, f0, f1, *_ = lrt_cnn3_ms(hampel(y), 2, NET, ar1=True, rng=np.random.default_rng(i), n_start=3)
        out.append(dict(coupled=lab, h=float(s), Jh=float(np.atleast_1d(f1["J"])[0])))
    return c, i, out


if __name__ == "__main__":
    prev = json.load(open(ROOT / "results/r23/r25_ood.json"))["res"]
    res = {}
    with Pool(2) as p:
        for c, i, r in p.imap_unordered(job, [(c, i) for c in ("lorentz", "slow", "skewed") for i in range(30)]):
            res.setdefault(c, {})[str(i)] = r
    summ = {}
    for c, v in res.items():
        d = {}
        for m, src in (("raw", "prev"), ("hampel", "new")):
            rows = [x for k, vv in (prev[c] if src == "prev" else v).items() for x in vv]
            key = "r9a" if src == "prev" else "h"
            neg = np.array([x[key] for x in rows if x["coupled"] == 0 and key in x]); pos = np.array([x[key] for x in rows if x["coupled"] == 1 and key in x])
            thr = np.quantile(neg, 0.95)
            au = (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
            d[m] = dict(power=round(float(np.mean(pos > thr)), 3), auc=round(float(au), 3), fpr_chi2=round(float(np.mean(neg > 3.84)), 3))
        summ[c] = d
    json.dump(dict(summary=summ, res=res), open(ROOT / "results/r23/r27c_hampel_ood.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
