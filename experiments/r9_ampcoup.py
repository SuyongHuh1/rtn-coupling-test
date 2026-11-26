"""Robustness: amplitude coupling (non-additive levels) on kinetically INDEPENDENT records.
y' = y + delta * s1 * s2, delta = eta * (smaller step) * sign(that step).  eta in {0, -0.25}.
FPR measured against the 95% threshold of the stored R9a (lrt_cnn3) independent stats.
    python experiments/r9_ampcoup.py white|full eta n
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O
from rtnsim.r9 import load_stepnet, lrt_cnn3
COND = sys.argv[1]; ETA = float(sys.argv[2]); N = int(sys.argv[3])
_net = None
def job(a):
    global _net
    i, y, K = a
    if _net is None: _net = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    s, f0, f1, beta, p = lrt_cnn3(y, K, _net, ar1=(COND != "white"), rng=np.random.default_rng(i))
    J = getattr(f1, "J", None)
    return i, dict(stat=float(s), J=(np.asarray(J).tolist() if J is not None else None))
def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    ref = json.load(open(ROOT / f"results/r8/r9a_g1_{COND}_v3.json"))["res"]
    idx = [i for i in range(len(rows)) if rows[i]["K"] == 2 and rows[i]["coupled"] == 0]
    thr = float(np.quantile([ref[str(i)]["lrt_cnn"] for i in idx], 0.95))
    idx = idx[:N]; tasks = []
    for i in idx:
        y = ys[i].copy(); st = ss[i]
        X = np.c_[np.ones(len(y)), st[:, 0], st[:, 1]]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        k = 1 + int(np.argmin(np.abs(b[1:])))
        y = y + ETA * b[k] * st[:, 0] * st[:, 1]
        tasks.append((i, y, 2))
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, tasks): out[i] = r
    st_ = np.array([out[i]["stat"] for i in idx])
    summ = dict(cond=COND, eta=ETA, n=len(idx), thr=thr, fpr=float((st_ > thr).mean()),
                median_stat=float(np.median(st_)))
    json.dump(dict(summary=summ, res=out), open(ROOT / f"results/r8/r9_ampcoup_{COND}_{ETA}.json", "w"), indent=1)
    print(json.dumps(summ))
if __name__ == "__main__": main()
