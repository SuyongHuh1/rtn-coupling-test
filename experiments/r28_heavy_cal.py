"""R28: does the pipeline-null calibration hold under heavy-tailed noise, with and without the Hampel
pre-filter?  Independent members of the R25 'heavy' pairs (t(3) and t(5), 30 each); p_cal from 39 pipeline
nulls (fitted J = 0 model + baseline + block-bootstrapped RAW residual, so the nulls carry the spikes).
    python experiments/r28_heavy_cal.py [n=30] [n_sim=39]
"""
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "results/r23/r28_heavy_cal.json"
NET = None


def job(a):
    global NET
    dof, i, hamp, n_sim = a
    import rtnsim.r9 as R9
    from rtnsim.cfhmm import simulate_null
    from rtnsim.robustk import bits
    import r25_ood as R
    R9.HAMPEL = hamp
    if NET is None:
        NET = R9.load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    R.TDOF = dof
    y = R.make_pair("heavy", i)[0][1]                       # independent member
    rng = np.random.default_rng(2800 + i)
    s, f0, f1, beta, _ = R9.lrt_cnn3_ms(y, 2, NET, ar1=True, rng=rng, n_start=3)
    X = np.hstack([np.ones((4, 1)), bits(2)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    null = [R9.lrt_cnn3_ms(simulate_null(f0, beta, resid, len(y), rng), 2, NET, ar1=True, rng=rng, n_start=3)[0] for _ in range(n_sim)]
    return dof, i, hamp, dict(stat=float(s), p=float((1 + np.sum(np.array(null) >= s)) / (1 + n_sim)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 39
    res = json.load(open(OUT)) if OUT.exists() else {}
    tasks = [(d, i, h, n_sim) for d in (3, 5) for h in (False, True) for i in range(n) if f"{i}" not in res.get(f"t{d}_{'h' if h else 'raw'}", {})]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for d, i, h, r in p.imap_unordered(job, tasks):
            res.setdefault(f"t{d}_{'h' if h else 'raw'}", {})[str(i)] = r
            json.dump(res, open(OUT, "w"), indent=1)
    summ = {k: dict(n=len(v), fpr_cal=float(np.mean([r["p"] < 0.05 for r in v.values()])),
                    fpr_chi2=float(np.mean([r["stat"] > 3.84 for r in v.values()]))) for k, v in res.items() if k != "summary"}
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))
