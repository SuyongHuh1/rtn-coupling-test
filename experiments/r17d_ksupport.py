"""R17d: checklist step 2 (trap-number support) on the imec RTNinja records before the coupling test.

For each 2-source record: K = 1 fit vs K = 2 independent fit (same baseline), BIC; K = 2 is supported
only if BIC prefers it AND both fitted steps >= 1.2 x noise sigma.  Records without K = 2 support are
"not tested" (as in the checklist), the rest keep the R17 asymptotic / calibrated p-values.
    python experiments/r17d_ksupport.py <nsrc>
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r17_rtninja_real import records  # noqa: E402


def job(a):
    from rtnsim.cfhmm import best_decomposition, fit, init_from_idealiser
    from rtnsim.robustk import idealise_k
    path, K, seed = a
    try:
        y = np.load(path / "MeasurementData.npy").astype(float)
        rng = np.random.default_rng(seed)
        f, f0 = best_decomposition(y, K, rng)
        yb = y - f["beta"]
        g = idealise_k(y, K - 1, rng=rng) if K - 1 >= 1 else None
        f1 = fit(y - g["beta"], K - 1, False, init_from_idealiser(g))
        T = y.size
        npar = lambda k: 2 * k + k + 1 + 1
        bicK = -2 * f0["ll"] + npar(K) * np.log(T)
        bicKm = -2 * f1["ll"] + npar(K - 1) * np.log(T)
        steps = np.abs(np.asarray(f0["coef"][1:])) / float(f0["sig"])
        return dict(rec=path.name, bic_K=float(bicK), bic_Km1=float(bicKm), min_step_snr=float(steps.min()),
                    supported=bool(bicK < bicKm and steps.min() >= 1.2))
    except Exception as ex:
        return dict(rec=path.name, error=repr(ex)[:200])


def main(nsrc):
    recs = [p for p in records(nsrc) if (p / "MeasurementData.npy").exists()]
    out = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [(p, nsrc, 7000 + i) for i, p in enumerate(recs)]):
            out.append(r)
    B = json.load(open(ROOT / f"results/r8/r17_partB_{nsrc}src.json"))
    sup = {r["rec"]: r.get("supported") for r in out}
    asym = [r for r in B["asym"] if "error" not in r]
    tested = [r for r in asym if sup.get(r["rec"])]
    res = dict(n=len(asym), n_supported=len(tested),
               asym_fpr_all=float(np.mean([r["p_asym"] < 0.05 for r in asym])),
               asym_fpr_supported=float(np.mean([r["p_asym"] < 0.05 for r in tested])) if tested else None,
               asym_fpr_supported_resolvable=float(np.mean([r["p_asym"] < 0.05 for r in tested if r["min_dwell"] >= 3])) if tested else None)
    for key in ("cal", "calbox"):
        c = [r for r in B.get(key, []) if "p_cal" in r]
        ct = [r for r in c if sup.get(r["rec"])]
        res[f"{key}_n"] = len(c); res[f"{key}_fpr"] = float(np.mean([r["p_cal"] < 0.05 for r in c])) if c else None
        res[f"{key}_n_supported"] = len(ct)
        res[f"{key}_fpr_supported"] = float(np.mean([r["p_cal"] < 0.05 for r in ct])) if ct else None
        res[f"{key}_detail"] = [(r["rec"][2:7], round(r["stat"], 1), r["p_cal"], sup.get(r["rec"])) for r in c]
    json.dump(dict(summary=res, rows=out), open(ROOT / f"results/r8/r17d_ksupport_{nsrc}src.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
