"""Fairness check: conventional-rule FPR of the existing methods vs record length.
T in (4096, 16384, 65536); per K: 40 coupled/independent pairs.  python experiments/r10_length.py white|full
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r10_baselines as R
from rtnsim import sim2_torch as S
COND = sys.argv[1]; R.COND = COND
def recs(T, n=40, seed=500):
    out = []
    for K in (2, 3):
        g = torch.Generator().manual_seed(seed + K + T); got = 0
        while got < n:
            y, lab = S.make_batch(20, T, g, "cpu", p_drive=0, p_box=0, return_states=True,
                                  **(dict(p_drift=0, p_flicker=0) if COND == "white" else {}))
            if int(lab["K"][0]) != K: continue
            for r in range(y.shape[0]):
                if got >= n and r % 2 == 0: break
                st = lab["states"][r].numpy()
                out.append(dict(y=y[r].double().numpy(), K=K, coupled=int(lab["coupled"][r]),
                                sw=int(np.abs(np.diff(st, axis=0)).sum(0).min())))
                if r % 2 == 1: got += 1
    return out
def job(a):
    i, y, K = a
    return i, R.job((i, y, K))[1]
def main():
    summ = {}
    for T in (4096, 16384, 65536):
        rs = recs(T)
        with Pool(2) as p:
            out = dict(p.imap_unordered(job, [(i, r["y"], r["K"]) for i, r in enumerate(rs)]))
        for K in (2, 3):
            idx = [i for i, r in enumerate(rs) if r["K"] == K]
            lab = np.array([rs[i]["coupled"] for i in idx])
            c = dict(pku=np.array([out[i]["pku_max"] > 0.5 for i in idx]),
                     albrecht=np.array([out[i]["evid"] > 2 for i in idx]),
                     son=np.array([out[i]["son_minp"] < 0.05 / out[i]["son_m"] for i in idx]),
                     albrecht_f=np.array([out[i]["alb_f_full"] and not out[i]["alb_f_excl"] for i in idx]),
                     son_f=np.array([out[i]["son_f_p"] < 0.05 for i in idx]))
            d = dict(median_switches=float(np.median([rs[i]["sw"] for i in idx])))
            for k, v in c.items():
                d[k] = dict(fpr=float(v[lab == 0].mean()), tpr=float(v[lab == 1].mean()))
            summ[f"T={T},K={K}"] = d
            print(f"T={T},K={K}", json.dumps(d), flush=True)
    json.dump(summ, open(ROOT / f"results/r8/r10_length_{COND}_v2.json", "w"), indent=1)
if __name__ == "__main__": main()
