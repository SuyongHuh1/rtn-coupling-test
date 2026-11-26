"""R27: heavy-tailed noise -- option 1 (kurtosis-gated Hampel pre-filter) vs option 2 (Student-t AR(1)
emission) vs both (Hampel only for the CNN / decomposition stage, t-emission fits on the raw record).
  V0 raw R9a (3 starts)
  V1 gated Hampel: innovation excess kurtosis of V0's J = 0 fit > 3 -> R9a on the filtered record
  V2 t-emission: nu from V0's innovations; nu < 30 -> t refit of J = 0 / J free on the raw record
  V3 Hampel decomposition + t refit on the raw record (gate as V1)
Sets (K = 2 matched pairs): heavy t(3), t(5) (R25 generator), in-distribution 1/f R10 records (no harm check).
    python experiments/r27_robust.py [n=30]
"""
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "results/r23/r27_robust.json"
NET = None
KGATE, NUGATE = 3.0, 30.0


def variants(y, K, seed):
    global NET
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    from rtnsim.tnoise import innovations, estimate_nu, t_refit, hampel
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    s0, f0, f1, beta, _ = lrt_cnn3_ms(y, K, NET, ar1=True, rng=np.random.default_rng(seed), n_start=3)
    z = y - beta
    inn = innovations(z, f0, K)
    kz = float(kurtosis(inn)); nu = estimate_nu(inn)
    r = dict(V0=float(s0), kurt=kz, nu=nu, J0=float(np.atleast_1d(f1["J"])[0]))
    r["V2"] = float(t_refit(z, K, f0, f1, nu)[0]) if nu < NUGATE else r["V0"]
    if kz > KGATE:
        yh = hampel(y)
        sh, g0, g1, bh, _ = lrt_cnn3_ms(yh, K, NET, ar1=True, rng=np.random.default_rng(seed), n_start=3)
        r["V1"] = float(sh)
        zr = y - bh
        nu3 = estimate_nu(innovations(zr, g0, K))
        r["nu3"] = nu3
        r["V3"] = float(t_refit(zr, K, g0, g1, nu3)[0]) if nu3 < NUGATE else float(sh)
    else:
        r["V1"] = r["V3"] = r["V2"] if False else r["V0"]
        if nu < NUGATE:
            r["V3"] = r["V2"]
    return r


def job(a):
    setname, i = a
    try:
        if setname.startswith("heavy"):
            import r25_ood as R
            R.TDOF = int(setname[-1])
            ys, _ = R.make_pair("heavy", i)
            out = [dict(coupled=lab, **variants(y, 2, i)) for lab, y in zip((1, 0), ys)]
        else:
            import r8_oracle as O
            global _REC
            if "_REC" not in globals():
                _REC = O.make(100, white=False)
            ys, ss, rows = _REC
            kk = [j for j in range(len(rows)) if rows[j]["K"] == 2]
            pos = [j for j in kk if rows[j]["coupled"] == 1][i]; neg = [j for j in kk if rows[j]["coupled"] == 0][i]
            out = [dict(coupled=1, **variants(ys[pos], 2, pos)), dict(coupled=0, **variants(ys[neg], 2, neg))]
        return setname, i, out
    except Exception as ex:
        return setname, i, [dict(error=repr(ex)[:200])]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    res = json.load(open(OUT)).get("res", {}) if OUT.exists() else {}
    tasks = [(s, i) for s in ("heavy3", "heavy5", "indist") for i in range(n) if str(i) not in res.get(s, {})]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for s, i, r in p.imap_unordered(job, tasks):
            res.setdefault(s, {})[str(i)] = r
            json.dump(dict(res=res), open(OUT, "w"))
            print(s, i, [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in r], flush=True)
    summ = {}
    for s, v in res.items():
        rows = [x for vv in v.values() for x in vv if "error" not in x]
        d = dict(n=len(rows), errors=sum("error" in x for vv in v.values() for x in vv),
                 gated=float(np.mean([x["kurt"] > KGATE for x in rows])), nu_med=float(np.median([x["nu"] for x in rows])))
        for m in ("V0", "V1", "V2", "V3"):
            neg = np.array([x[m] for x in rows if x["coupled"] == 0]); pos = np.array([x[m] for x in rows if x["coupled"] == 1])
            thr = np.quantile(neg, 0.95)
            au = (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
            d[m] = dict(power=float(np.mean(pos > thr)), auc=float(au), fpr_chi2=float(np.mean(neg > 3.84)))
        summ[s] = d
    json.dump(dict(summary=summ, res=res), open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
