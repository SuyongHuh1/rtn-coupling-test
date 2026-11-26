"""R25b: heavy-tailed noise diagnosis.  Same 'heavy' pairs as R25 (Student-t 3), plus t(5); R9a with and
without a Hampel spike filter (window 7, 3 MAD) applied before the pipeline."""
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r25_ood as R
NET = None


def hampel(y, w=7, k=3.0):
    from scipy.ndimage import median_filter
    m = median_filter(y, size=w, mode="nearest")
    mad = median_filter(np.abs(y - m), size=w, mode="nearest") * 1.4826
    s = np.median(np.abs(np.diff(y))) / 0.6745 / np.sqrt(2)
    out = y.copy(); bad = np.abs(y - m) > k * np.maximum(mad, s)
    out[bad] = m[bad]
    return out


def job(a):
    global NET
    dof, i = a
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    R.TDOF = dof
    ys, st = R.make_pair("heavy", i)
    out = []
    for lab, y in zip((1, 0), ys):
        r = dict(coupled=lab)
        r["raw"] = float(lrt_cnn3_ms(y, 2, NET, ar1=True, rng=np.random.default_rng(i), n_start=3)[0])
        r["hampel"] = float(lrt_cnn3_ms(hampel(y), 2, NET, ar1=True, rng=np.random.default_rng(i), n_start=3)[0])
        out.append(r)
    return dof, i, out


if __name__ == "__main__":
    res = {}
    with Pool(2) as p:
        for dof, i, r in p.imap_unordered(job, [(d, i) for d in (3, 5) for i in range(30)]):
            res.setdefault(str(dof), []).extend(r)
    json.dump(res, open(ROOT / "results/r23/r25b_heavy.json", "w"))
    for dof, rows in res.items():
        for m in ("raw", "hampel"):
            neg = np.array([x[m] for x in rows if x["coupled"] == 0]); pos = np.array([x[m] for x in rows if x["coupled"] == 1])
            thr = np.quantile(neg, .95)
            print(f"t({dof}) {m}: power {np.mean(pos > thr):.2f}  fpr_chi2 {np.mean(neg > 3.84):.2f}")
