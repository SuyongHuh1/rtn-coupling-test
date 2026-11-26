"""R17e: full checklist verdicts on the imec RTNinja records (all J = 0 by construction).

Per record (K = true number of sources):
  (1)/(2) identifiability: every trap on the J = 0 decoded path switches >= 15 times, and the K-trap
          independent model beats the (K-1)-trap model by BIC; otherwise "not tested".
  (3)     LRT; p from the pipeline-null calibration where available (R17 cal), else asymptotic chi2.
  (4)     single-defect alternative: the coupled K-trap model must also beat the (K-1)-trap model by BIC,
          and the traps must not co-switch in >= 50 % of transitions (a locked pair = one defect).
A record counts as a false positive only if it passes (1)-(4) with p < 0.05.
    python experiments/r17e_checklist.py <nsrc>
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r17_rtninja_real import records  # noqa: E402


def job(a):
    from rtnsim.cfhmm import fit, init_from_idealiser, lrt_ms
    from rtnsim.robustk import bits, idealise_k
    path, K, seed = a
    try:
        y = np.load(path / "MeasurementData.npy").astype(float)
        rng = np.random.default_rng(seed)
        stat, f0, f1, beta, _ = lrt_ms(y, K, rng)
        B = bits(K)
        S0 = B[f0["path"]]; S1 = B[f1["path"]]
        trans0 = [int(np.count_nonzero(np.diff(S0[:, k]))) for k in range(K)]
        d1 = np.abs(np.diff(S1, axis=0)).sum(1)
        cosw = float(np.mean(d1[d1 > 0] >= 2)) if np.any(d1 > 0) else 0.0
        def state_dwell(path):
            runs = {}
            ch = np.flatnonzero(np.diff(path)) + 1
            st = np.r_[0, ch]; en = np.r_[ch, path.size]
            for a_, b_ in zip(st, en):
                runs.setdefault(int(path[a_]), []).append(b_ - a_)
            return {str(k_): float(np.mean(v_)) for k_, v_ in runs.items() if len(v_) >= 5}
        dw1 = state_dwell(np.asarray(f1["path"])); dw0 = state_dwell(np.asarray(f0["path"]))
        g = idealise_k(y, K - 1, rng=rng)
        fm = fit(y - g["beta"], K - 1, False, init_from_idealiser(g))
        T = y.size
        npar = lambda k, c: 3 * k + 2 + (k * (k - 1) // 2 if c else 0)
        bic0 = -2 * f0["ll"] + npar(K, False) * np.log(T)
        bic1 = -2 * f1["ll"] + npar(K, True) * np.log(T)
        bicm = -2 * fm["ll"] + npar(K - 1, False) * np.log(T)
        return dict(rec=path.name, stat=float(stat), p_asym=float(chi2.sf(stat, K * (K - 1) // 2)),
                    trans=trans0, coswitch=cosw, dwell1=dw1, dwell0=dw0, bic0=float(bic0), bic1=float(bic1), bicm=float(bicm),
                    J=[float(x) for x in np.atleast_1d(f1.get("J", []))])
    except Exception as ex:
        return dict(rec=path.name, error=repr(ex)[:200])


def verdict(r, pcal):
    if "error" in r:
        return "error"
    if min(r["trans"]) < 15 or not r["bic0"] < r["bicm"]:
        return "not tested"
    if min(list(r.get("dwell1", {}).values()) + list(r.get("dwell0", {}).values()) + [1e9]) < 3:
        return "undecidable (resolution)"
    p = pcal.get(r["rec"], r["p_asym"])
    if p >= 0.05:
        return "no evidence"
    if not r["bic1"] < r["bicm"] or r["coswitch"] >= 0.5:
        return "single-defect alternative"
    return "coupling (false positive)"


def main(nsrc):
    recs = [p for p in records(nsrc) if (p / "MeasurementData.npy").exists()]
    fn = ROOT / f"results/r8/r17e_checklist_{nsrc}src.json"
    out = []
    if fn.exists():
        prev = json.load(open(fn))
        out = prev if isinstance(prev, list) else prev.get("rows", [])
        out = [r for r in out if "error" not in r or "dwell1" in r]
    done = {r["rec"] for r in out}
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as pool:
        for r in pool.imap_unordered(job, [(p, nsrc, 9000 + i) for i, p in enumerate(recs) if p.name not in done]):
            out.append(r)
            json.dump(out, open(ROOT / f"results/r8/r17e_checklist_{nsrc}src.json", "w"), indent=1)
    fnB = ROOT / f"results/r8/r17_partB_{nsrc}src.json"
    B = json.load(open(fnB)) if fnB.exists() else {}
    pcal_i = {r["rec"]: r["p_cal"] for r in B.get("cal", []) if "p_cal" in r}
    pcal_b = {r["rec"]: r["p_cal"] for r in B.get("calbox", []) if "p_cal" in r}
    summ = {}
    for name, pc in (("asymptotic", {}), ("instant_null", pcal_i), ("boxcar_null", pcal_b)):
        rows = out if name == "asymptotic" else [r for r in out if r["rec"] in pc]
        v = [verdict(r, pc) for r in rows]
        tested = [x for x in v if x not in ("not tested", "error", "undecidable (resolution)")]
        summ[name] = dict(n=len(v), counts={k: v.count(k) for k in sorted(set(v))},
                          fpr_of_tested=(v.count("coupling (false positive)") / len(tested)) if tested else None,
                          fpr_of_all=v.count("coupling (false positive)") / max(len(v), 1),
                          naive_fpr_ignoring_checklist=float(np.mean([pc.get(r["rec"], r.get("p_asym", 1)) < 0.05 for r in rows if "error" not in r])) if rows else None)
    json.dump(dict(summary=summ, rows=out), open(ROOT / f"results/r8/r17e_checklist_{nsrc}src.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
