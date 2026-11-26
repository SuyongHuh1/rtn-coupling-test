"""R5g: is the two-trap model adequate for each real record?

Checks after baseline removal (rtnsim.robust idealisation):
 1. number of current levels: BIC over Gaussian HMMs with 1-6 states (tied variance) on the
    baseline-corrected signal; two independent traps give at most 4 levels;
 2. level additivity: free-level 4-state HMM (general kinetics) started from the idealisation;
    non-additivity = (mu11 - mu10 - mu01 + mu00) / sigma;
 3. dwell times: per trap and state, KS test of the dwell-time distribution against the
    geometric distribution with the same mean (Markov two-state trap);
 4. multiple testing: Holm and Benjamini-Hochberg over the pipeline-null p values.
Output: results/r5/r5_adequacy.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.realdata import load_b1500  # noqa: E402
from rtnsim.robust import idealise  # noqa: E402
from rtnsim import detectors as det  # noqa: E402
import r5_real as R  # noqa: E402

OUT = ROOT / "results" / "r5"


def n_levels(y, kmax=6):
    from hmmlearn.hmm import GaussianHMM
    best, bics = None, {}
    for k in range(1, kmax + 1):
        ll = -np.inf
        for seed in range(3):
            m = GaussianHMM(n_components=k, covariance_type="tied", n_iter=300, random_state=seed, tol=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(y[:, None])
            ll = max(ll, m.score(y[:, None]))
        bics[k] = -2 * ll + (k * (k - 1) + k + 1 + (k - 1)) * np.log(len(y))
        if best is None or bics[k] < bics[best]:
            best = k
    return best, {str(k): float(v) for k, v in bics.items()}


def additivity(y, idl):
    yb = (y - idl["beta"])[None]
    mu0 = np.array([0.0, idl["a1"], idl["a2"], idl["a1"] + idl["a2"]])
    mu0 = mu0 + np.median(yb[0] - idl["a1"] * idl["s1"] - idl["a2"] * idl["s2"])
    start = det._fit_product_from(yb, mu0[None], np.array([idl["sig"]]), 40, 1e-8)
    gen = det.fit_general(yb, start, n_iter=40)
    mu, sig = gen["mu"][0], float(gen["sig"][0])
    return float((mu[3] - mu[1] - mu[2] + mu[0]) / sig), mu.tolist()


def dwell_ks(s):
    s = np.asarray(s)
    edges = np.flatnonzero(np.diff(s)) + 1
    runs = np.diff(np.r_[0, edges, len(s)])
    states = s[np.r_[0, edges]]
    out = {}
    for a in (0, 1):
        d = runs[1:-1][states[1:-1] == a]          # drop censored first/last runs
        if len(d) < 10:
            out[str(a)] = None
            continue
        p = 1.0 / d.mean()
        cdf = lambda x: 1 - (1 - p) ** np.floor(x)
        out[str(a)] = dict(n=int(len(d)), mean=float(d.mean()), p_ks=float(stats.kstest(d + 0.5, cdf).pvalue),
                           cv=float(d.std() / d.mean()))
    return out


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)[::-1]
    m = len(p)
    adj = np.empty_like(p)
    run = 1.0
    for i, j in enumerate(o):
        run = min(run, p[j] * m / (m - i))
        adj[j] = run
    return adj


def main():
    res = []
    nul = {r["file"]: r for r in json.load(open(OUT / "r5_pipeline_null_southampton.json"))}
    for r in R.records():
        if r["file"] not in nul:
            continue
        t, I, _ = load_b1500(r["path"])
        t, y, _, dt = R.preprocess(t, I, r["trim_s"])
        idl = idealise(y, baseline=True)
        k, bics = n_levels(y - idl["beta"])
        na, mu = additivity(y, idl)
        row = dict(set=r["set"], file=r["file"], n_levels_bic=k, bic=bics, nonadditivity_sd=na, levels=mu,
                   dwell=dict(trap1=dwell_ks(idl["s1"]), trap2=dwell_ks(idl["s2"])),
                   p_pipeline=nul[r["file"]]["p_pipeline"])
        print(r["file"][:44], k, round(na, 2),
              {t_: {a: (None if v is None else round(v["p_ks"], 3)) for a, v in d.items()} for t_, d in row["dwell"].items()},
              flush=True)
        res.append(row)
    p = [r["p_pipeline"] for r in res]
    for r, h, b in zip(res, holm(p), bh(p)):
        r["p_holm"], r["p_bh"] = float(h), float(b)
    # Nichol: adequacy on the first repetition of each plunger bias
    from r5_nichol import D
    s = sio.loadmat(D / "Gate_P.mat", squeeze_me=True, struct_as_record=False)["savedat"]
    nichol = []
    for i, v in enumerate(s.Vgate):
        y = s.data[i][0]
        y = (y - np.median(y)) / (np.median(np.abs(np.diff(y))) / 0.6745 / np.sqrt(2))
        idl = idealise(y, baseline=True, min_win=61)
        k, bics = n_levels(y - idl["beta"])
        na, mu = additivity(y, idl)
        row = dict(dataset="Gate_P", value=float(v), n_levels_bic=k, nonadditivity_sd=na, a1=idl["a1"], a2=idl["a2"],
                   sig=idl["sig"], dwell=dict(trap1=dwell_ks(idl["s1"]), trap2=dwell_ks(idl["s2"])))
        print("Gate_P", v, k, round(na, 2), round(idl["a1"], 2), round(idl["a2"], 2), flush=True)
        nichol.append(row)
    json.dump(dict(southampton=res, nichol=nichol), open(OUT / "r5_adequacy.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
