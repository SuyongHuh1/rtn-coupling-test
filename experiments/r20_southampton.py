"""R20: Southampton EDTM/SSDM records re-analysed with the final pipeline (R9a CNN-HMM, AR(1) noise)
and the full checklist, replacing the R8 (HMM-only) results.

Per record and K in (2, 3):
  (1) resolution   every joint state of the J = 0 and coupled decoded paths has mean dwell >= 3 samples
  (2) trap number  every trap switches >= 15 times; BIC(K) < BIC(K-1) (K-1 = 1 via an independent fit)
  (3) LRT          asymptotic chi2; when p_asym < 0.2 the pipeline null (n_sim full re-runs of R9a on
                   copies of the fitted J = 0 model + the record's baseline + block-bootstrapped residual)
                   gives p_cal.  (asymptotic p >= 0.2 -> calibrated p is larger still; not re-run)
  (4) alternative  if p_cal < 0.05: coupling LRT with a shared slow drive in both hypotheses (lrt_drive)
EDTM records are B1500 with PLC10 integration (boxcar), so (1) matters.

    python experiments/r20_southampton.py [n_sim=39]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "results" / "r8" / f"r20_southampton_v2{__import__('os').environ.get('RTN_TAG', '')}.json"
NRESTART = 3
NET = None


def net():
    global NET
    if NET is None:
        from rtnsim.r9 import load_stepnet
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    return NET


def dwell(path):
    path = np.asarray(path)
    ch = np.flatnonzero(np.diff(path)) + 1
    st = np.r_[0, ch]; en = np.r_[ch, path.size]
    runs = {}
    for a, b in zip(st, en):
        runs.setdefault(int(path[a]), []).append(b - a)
    return {str(k): float(np.mean(v)) for k, v in runs.items() if len(v) >= 5}


def bic1(y, rng):
    from rtnsim.robustk import idealise_k
    from rtnsim.cfhmm import fit_ar1, init_from_idealiser
    g = idealise_k(y, 1, baseline=True, rng=rng)
    ini = init_from_idealiser(g); ini.setdefault("rho", 0.0)
    f = fit_ar1(y - g["beta"], 1, False, ini)
    return float(f["bic"])


def fit_best(y, K, rng, n=None):
    """R9a run from n random seeds; the J = 0 likelihood (the pipeline's own criterion) picks the run."""
    from rtnsim.r9 import lrt_cnn3
    runs = [lrt_cnn3(y, K, net(), ar1=True, rng=np.random.default_rng(rng.integers(1 << 31))) for _ in range(n or NRESTART)]
    best = max(runs, key=lambda r: r[1]["ll"])
    return best, [(float(r[1]["ll"]), float(r[0])) for r in runs]


def job(a):
    name, y, K, n_sim, seed = a
    from rtnsim.r9 import lrt_cnn3
    from rtnsim.cfhmm import simulate_null, lrt_drive
    from rtnsim.robustk import bits
    rng = np.random.default_rng(seed)
    try:
        (s, f0, f1, beta, _), seeds = fit_best(y, K, rng)
        B = bits(K)
        trans = [int(np.count_nonzero(np.diff(B[f0["path"]][:, k]))) for k in range(K)]
        d0, d1 = dwell(f0["path"]), dwell(f1["path"])
        df = K * (K - 1) // 2
        r = dict(rec=name, K=K, T=len(y), stat=float(s), p_asym=float(chi2.sf(s, df)), J=[float(x) for x in np.atleast_1d(f1["J"])],
                 steps=[float(x) for x in f0["coef"][1:]], sig=float(f0["sig"]), rho=float(f0["rho"]),
                 seeds=seeds, trans=trans, dwell_min=float(min(list(d0.values()) + list(d1.values()) + [1e9])), bic=float(f0["bic"]))
        if r["p_asym"] < 0.2:
            X = np.hstack([np.ones((2 ** K, 1)), B])
            resid = y - beta - (X @ f0["coef"])[f0["path"]]
            null = [fit_best(simulate_null(f0, beta, resid, len(y), rng), K, rng)[0][0] for _ in range(n_sim)]
            r["null"] = [float(x) for x in null]
            r["p_cal"] = float((1 + np.sum(np.array(null) >= s)) / (1 + n_sim))
            if r["p_cal"] < 0.05:
                sd, g0, g1, *_ = lrt_drive(y, K, rng)
                r["stat_drive"] = float(sd); r["J_drive"] = [float(x) for x in np.atleast_1d(g1["J"])]
        return r
    except Exception as ex:
        return dict(rec=name, K=K, error=repr(ex)[:200])


def verdict(r, bicK1):
    if "error" in r:
        return "error"
    if r["dwell_min"] < 3:
        return "판정 불가 (분해능)"
    if min(r["trans"]) < 15 or not r["bic"] < bicK1:
        return "검정 대상 아님 (트랩 식별)"
    p = r.get("p_cal", r["p_asym"])
    if p >= 0.05:
        return "근거 없음"
    if "stat_drive" in r and chi2.sf(r["stat_drive"], r["K"] * (r["K"] - 1) // 2) >= 0.05:
        return "대안으로 설명 (공통 구동)"
    return "후보 (반복 측정 없어 재현 확인 불가)"


def main():
    import r5_real as R
    from rtnsim.realdata import load_b1500
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 39
    keep = {r["file"] for r in json.load(open(ROOT / "results/r6/r6_south.json")) if r["K"] >= 2}
    data = {}
    for r in R.records():
        if r["file"] in keep:
            t, I, _ = load_b1500(r["path"])
            data[r["file"]] = R.preprocess(t, I, r["trim_s"])[1]
    res = json.load(open(OUT)) if OUT.exists() else {"rows": [], "bic1": {}}
    for k, y in data.items():
        if k not in res["bic1"]:
            res["bic1"][k] = bic1(y, np.random.default_rng(1))
    done = {(r["rec"], r["K"]) for r in res["rows"] if "error" not in r}
    res["rows"] = [r for r in res["rows"] if "error" not in r]
    tasks = [(k, y, K, n_sim, 9100 + 10 * i + K) for i, (k, y) in enumerate(sorted(data.items(), key=lambda kv: len(kv[1]))) for K in (2, 3) if (k, K) not in done]
    print(len(tasks), "tasks", flush=True)
    import os
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as pool:
        for r in pool.imap_unordered(job, tasks):
            res["rows"].append(r)
            json.dump(res, open(OUT, "w"), indent=1)
            print(r.get("rec"), r.get("K"), {k: r.get(k) for k in ("stat", "p_asym", "p_cal", "J", "trans", "dwell_min", "error")}, flush=True)
    # verdict per record at the BIC-chosen K
    summ = []
    for k in data:
        rows = {r["K"]: r for r in res["rows"] if r["rec"] == k and "error" not in r}
        if not rows:
            continue
        Kb = min(rows, key=lambda K: rows[K]["bic"])
        bic_prev = {2: res["bic1"][k], 3: rows[2]["bic"] if 2 in rows else np.inf}
        summ.append(dict(rec=k, K_bic=Kb, verdicts={K: verdict(rows[K], bic_prev[K]) for K in rows},
                         p={K: rows[K].get("p_cal", rows[K]["p_asym"]) for K in rows}, J={K: rows[K]["J"] for K in rows}))
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
