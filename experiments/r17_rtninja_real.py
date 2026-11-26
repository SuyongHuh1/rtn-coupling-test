"""R17: coupling rules on GENUINE RTNinja decompositions (imec output files, Zenodo 10.5281/zenodo.16750729).

Data: noise05_{2,3}source, 50,000-sample synthetic records from the authors' own simulator
(codes/RTN_Simulator.py): sources are simulated one by one and summed, so every record is J = 0
(independent).  Each sample is a 20-step boxcar average (Keithley-style integration), Gaussian
noise sigma = 0.05 with unit-exponential amplitudes, lognormal time constants (10^N(1, 1.5)).
RTNinja's own per-source activity (RTNinja_Output_Attributes.pkl -> 'activity') is used as the
decomposition.  The RTNinja code itself is patent-restricted and not public.

Part A (all records with an RTNinja solution): rules applied to RTNinja activity, K = sources RTNinja found
  guo_d     Guo'18-style dTau/tau, |d| > 0.5 (threshold ours)
  son_bonf  Son'17-style conditional comparison as chi2(2) + Bonferroni (our formalisation)
  zhang_lv  RTNinja's number of levels < 2^K_found
Part B (ours, raw MeasurementData, K = true number of sources): asymptotic LRT for all records,
  pipeline-null calibration (own decomposition, n_sim) for a subset.  RTNinja cannot be put inside the
  calibration loop because its code is closed.

    python experiments/r17_rtninja_real.py A
    python experiments/r17_rtninja_real.py B <nsrc> <n_calib> <n_sim>
"""
from __future__ import annotations

import glob
import os
import json
import pickle
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.surrogate import _ll_binary_transitions  # noqa: E402

DATA = Path(os.environ.get("RTN_RTNINJA", "data/rtninja"))


def guo_d(S):
    K = S.shape[1]
    d = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            a0, a1, c = S[:-1, i], S[1:, i], S[:-1, j]
            for a in (0, 1):
                r = []
                for cc in (0, 1):
                    m = (a0 == a) & (c == cc)
                    r.append(((a1[m] != a).sum() + 0.5) / (m.sum() + 1.0))
                d.append(r[0] / max(r[1], 1e-12) - 1.0)
    return float(np.max(np.abs(d)))


def son_minp(S):
    K = S.shape[1]
    ps = []
    for p in range(K):
        for q in range(K):
            if p != q:
                g = _ll_binary_transitions(S[:-1, p], S[1:, p], S[:-1, q]) - _ll_binary_transitions(S[:-1, p], S[1:, p])
                ps.append(chi2.sf(2 * max(g, 0), 2))
    return float(min(ps)), len(ps)


def records(nsrc):
    base = DATA / f"n{nsrc}" / f"noise05_{nsrc}source"
    return sorted(p for p in base.glob("mc*_n05_dataset") if p.is_dir())


def part_a():
    out = {}
    for nsrc in (2, 3):
        rows = []
        for d in records(nsrc):
            fa = d / "RTNinja_Output_Attributes.pkl"
            if not fa.exists():
                continue
            a = pickle.load(open(fa, "rb"))
            act = [np.asarray(x).astype(np.int64) for x in a.get("activity") or []]
            ls = list(d.glob("LevelsOutput_*_Lstats.xlsx"))
            nlev = int(re.search(r"_(\d+)L_", ls[0].name).group(1)) if ls else None
            row = dict(rec=d.name, K_found=len(act), n_levels=nlev)
            if len(act) >= 2:
                S = np.stack(act, 1)
                row["guo"] = guo_d(S)
                row["son_minp"], row["son_m"] = son_minp(S)
                row["switches"] = [int(np.count_nonzero(np.diff(S[:, k]))) for k in range(S.shape[1])]
            rows.append(row)
        tested = [r for r in rows if r["K_found"] >= 2]
        exact = [r for r in tested if r["K_found"] == nsrc]
        def rates(rs):
            if not rs:
                return None
            return dict(n=len(rs),
                        guo_d0p5=float(np.mean([r["guo"] > 0.5 for r in rs])),
                        son_bonf=float(np.mean([r["son_minp"] < 0.05 / r["son_m"] for r in rs])),
                        zhang_lv=float(np.mean([r["n_levels"] is not None and r["n_levels"] < 2 ** r["K_found"] for r in rs])))
        out[f"{nsrc}source"] = dict(
            n_with_solution=len(rows),
            K_found_hist={str(k): int(sum(r["K_found"] == k for r in rows)) for k in range(0, 9)},
            all_tested=rates(tested), K_correct=rates(exact),
            K_correct_min_switch_ge_50=rates([r for r in exact if min(r["switches"]) >= 50]),
            rows=rows)
    json.dump(out, open(ROOT / "results/r8/r17_partA.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()}, indent=1))


def _job_b(a):
    try:
        return _job_b_inner(a)
    except Exception as ex:
        return dict(rec=a[0].name, error=repr(ex)[:200])


def _job_b_inner(a):
    from rtnsim.cfhmm import lrt_ms, simulate_null
    from rtnsim.robustk import bits
    path, K, n_sim, seed = a
    y = np.load(path / "MeasurementData.npy").astype(float)
    rng = np.random.default_rng(seed)
    stat, f0, f1, beta, _ = lrt_ms(y, K, rng)
    dwell = [float(1 / max(f0["c"][k], 1e-12)) for k in range(K)] + [float(1 / max(f0["e"][k], 1e-12)) for k in range(K)]
    res = dict(rec=path.name, stat=float(stat), p_asym=float(chi2.sf(stat, K * (K - 1) // 2)),
               min_dwell=float(min(dwell)), J=[float(x) for x in np.atleast_1d(f1.get("J", []))])
    if n_sim:
        box = n_sim < 0
        n_sim = abs(n_sim)
        X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
        resid = y - beta - (X @ f0["coef"])[f0["path"]]
        sim = simulate_null_box if box else simulate_null
        null = [lrt_ms(sim(f0, beta, resid, len(y), rng), K, rng)[0] for _ in range(n_sim)]
        res["null_type"] = "boxcar" if box else "instant"
        res["null"] = [float(x) for x in null]
        res["p_cal"] = float((1 + np.sum(np.array(null) >= stat)) / (1 + n_sim))
    return res


def simulate_null_box(f0, beta, resid, T, rng, sub=20):
    """J = 0 null that reproduces the measurement chain: chain simulated on a sub-sample grid
    (rates / sub per step) and boxcar-averaged over `sub` steps, as in the imec simulator."""
    from rtnsim.robustk import _block_boot
    K = len(f0["c"])
    occ = np.zeros((T, K))
    for k in range(K):
        c, e = f0["c"][k] / sub, f0["e"][k] / sub
        kt = c + e
        q = 1 - np.exp(-kt)
        p01, p10 = c / kt * q, e / kt * q
        st = int(rng.random() < c / kt)
        u = rng.random(T * sub)
        x = np.empty(T * sub)
        for t in range(T * sub):
            st = st ^ int(u[t] < (p01 if st == 0 else p10))
            x[t] = st
        occ[:, k] = x.reshape(T, sub).mean(1)
    # residual noise: remove boxcar-induced transition outliers from the bootstrap pool
    r = resid - np.median(resid)
    mad = np.median(np.abs(r)) / 0.6745
    r = np.where(np.abs(r) > 5 * mad, rng.normal(0, mad, r.size), r)
    return beta + f0["coef"][0] + occ @ f0["coef"][1:] + _block_boot(r, T, rng)


def part_b(nsrc, n_calib, n_sim, mode="asym", lo=0):
    recs = [p for p in records(nsrc) if (p / "MeasurementData.npy").exists()]
    fn = ROOT / f"results/r8/r17_partB_{nsrc}src.json"
    d = json.load(open(fn)) if fn.exists() else {}
    if not isinstance(d, dict):
        d = {}
    asym, cal = d.get("asym", []), d.get("cal", [])
    calbox = d.get("calbox", [])
    with Pool(2) as pool:
        if mode == "asym":
            done = {r["rec"] for r in asym}
            jobs = [(p, nsrc, 0, 1000 + i) for i, p in enumerate(recs) if p.name not in done]
            for r in pool.imap_unordered(_job_b, jobs):
                asym.append(r)
                json.dump(dict(asym=asym, cal=cal, calbox=calbox), open(fn, "w"), indent=1)
        elif mode == "calbox":
            done = {r["rec"] for r in calbox}
            ok = {r["rec"] for r in asym if "error" not in r and r["min_dwell"] >= 3}
            pool_recs = [p for p in recs if p.name in ok][lo:lo + n_calib]
            for r in pool.imap_unordered(_job_b, [(p, nsrc, -n_sim, 5000 + i) for i, p in enumerate(pool_recs) if p.name not in done]):
                calbox.append(r)
                json.dump(dict(asym=asym, cal=cal, calbox=calbox), open(fn, "w"), indent=1)
        else:
            done = {r["rec"] for r in cal}
            ok = {r["rec"] for r in asym if "error" not in r and r["min_dwell"] >= 3}
            pool_recs = [p for p in recs if p.name in ok][lo:lo + n_calib]
            jobs = [(p, nsrc, n_sim, 5000 + i) for i, p in enumerate(pool_recs) if p.name not in done]
            for r in pool.imap_unordered(_job_b, jobs):
                cal.append(r)
                json.dump(dict(asym=asym, cal=cal, calbox=calbox), open(fn, "w"), indent=1)
    asym_ok = [r for r in asym if "error" not in r]
    resolv = [r for r in asym_ok if r["min_dwell"] >= 3]
    summ = dict(n=len(asym), n_error=len(asym) - len(asym_ok), asym_fpr=float(np.mean([r["p_asym"] < 0.05 for r in asym_ok])) if asym_ok else None,
                n_resolvable=len(resolv),
                asym_fpr_resolvable=float(np.mean([r["p_asym"] < 0.05 for r in resolv])) if resolv else None,
                n_calibrated=len(cal), n_sim=n_sim,
                cal_fpr=float(np.mean([r["p_cal"] < 0.05 for r in cal if "p_cal" in r])) if cal else None,
                cal_p=[round(r["p_cal"], 3) for r in cal if "p_cal" in r],
                n_calbox=len(calbox),
                calbox_fpr=float(np.mean([r["p_cal"] < 0.05 for r in calbox if "p_cal" in r])) if calbox else None,
                calbox_p=[round(r["p_cal"], 3) for r in calbox if "p_cal" in r])
    json.dump(dict(summary=summ, asym=asym, cal=cal, calbox=calbox), open(fn, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "A":
        part_a()
    else:
        part_b(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "asym", int(sys.argv[6]) if len(sys.argv) > 6 else 0)
