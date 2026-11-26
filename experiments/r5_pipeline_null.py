"""R5e: idealisation-aware null for the real-data coupling statistic.

Shift surrogates test dependence between the IDEALISED paths.  Idealisation errors can be
correlated between the two traps (one large step split into two, a level mis-assigned), and
such errors are instantaneous, so they look like coupling to every shift test.  The honest
null therefore simulates the whole pipeline: independent two-trap chains with the fitted
rates and step heights, the fitted slow baseline, and the record's own residual noise
(block bootstrap, keeps its colour), idealised with the same code; the observed statistic is
compared with that distribution.

Usage: r5_pipeline_null.py southampton | nichol
Output: results/r5/r5_pipeline_null_<set>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.realdata import load_b1500  # noqa: E402
from rtnsim.robust import idealise  # noqa: E402
from rtnsim.surrogate import cond_lrt  # noqa: E402
import r5_real as R  # noqa: E402

OUT = ROOT / "results" / "r5"


def chain(P, T, rng, s0):
    s = np.empty(T, np.int64)
    s[0] = s0
    u = rng.random(T)
    for k in range(1, T):
        s[k] = s[k - 1] ^ (u[k] < P[s[k - 1], 1 - s[k - 1]])
    return s


def trans_P(s):
    s = np.asarray(s)
    P = np.zeros((2, 2))
    for a in (0, 1):
        m = s[:-1] == a
        n = m.sum()
        P[a, 1 - a] = np.sum(s[1:][m] != a) / n if n else 0.0
        P[a, a] = 1 - P[a, 1 - a]
    return P


def block_boot(r, T, rng, L=200):
    out = np.empty(T)
    k = 0
    while k < T:
        i = rng.integers(0, len(r) - L)
        n = min(L, T - k)
        out[k:k + n] = r[i:i + n]
        k += n
    return out


def null_stats(segments, rng, n_sim, min_win):
    """segments: list of (y, idl).  Returns observed statistic and null samples (summed over segments)."""
    obs = sum(cond_lrt(idl["s1"], idl["s2"]) for _, idl in segments)
    null = np.empty(n_sim)
    for j in range(n_sim):
        tot = 0.0
        for y, idl in segments:
            T = len(y)
            fit = idl["beta"] + idl["a1"] * idl["s1"] + idl["a2"] * idl["s2"]
            off = np.median(y - fit)
            resid = y - fit - off
            s1 = chain(trans_P(idl["s1"]), T, rng, idl["s1"][0])
            s2 = chain(trans_P(idl["s2"]), T, rng, idl["s2"][0])
            ys = off + idl["beta"] + idl["a1"] * s1 + idl["a2"] * s2 + block_boot(resid, T, rng)
            r = idealise(ys, baseline=True, min_win=min_win)
            tot += cond_lrt(r["s1"], r["s2"])
        null[j] = tot
    return float(obs), null


def southampton(n_sim=100):
    rng = np.random.default_rng(21)
    rob = {r["file"]: r for r in json.load(open(OUT / "r5_robust.json"))}
    res = []
    for r in R.records():
        if not rob[r["file"]]["baseline"]["resolvable"]:
            continue
        t, I, _ = load_b1500(r["path"])
        t, y, _, dt = R.preprocess(t, I, r["trim_s"])
        idl = idealise(y, baseline=True)
        obs, null = null_stats([(y, idl)], rng, n_sim, 101)
        p = float((1 + np.sum(null >= obs)) / (n_sim + 1))
        row = dict(file=r["file"], set=r["set"], obs=obs, null_median=float(np.median(null)),
                   null_q95=float(np.quantile(null, 0.95)), p_pipeline=p,
                   p_local_shift=rob[r["file"]]["baseline"]["p_local"])
        print(row, flush=True)
        res.append(row)
    json.dump(res, open(OUT / "r5_pipeline_null_southampton.json", "w"), indent=1)


def nichol(n_sim=40, which="Gate_P", reps=10, idx=(8, 2, 7)):
    from r5_nichol import D
    rng = np.random.default_rng(22)
    s = sio.loadmat(D / f"{which}.mat", squeeze_me=True, struct_as_record=False)["savedat"]
    vals = s.MCtemp if which == "MCtemp" else s.Vgate
    res = []
    for i in idx:
        v = vals[i]
        block = s.data[i][:reps]
        z = (block - np.median(block)) / (np.median(np.abs(np.diff(block, axis=1))) / 0.6745 / np.sqrt(2))
        segs = [(y, idealise(y, baseline=True, min_win=61)) for y in z]
        obs, null = null_stats(segs, rng, n_sim, 61)
        row = dict(dataset=which, value=float(v), reps=reps, obs=obs, null_median=float(np.median(null)),
                   null_q95=float(np.quantile(null, 0.95)), p_pipeline=float((1 + np.sum(null >= obs)) / (n_sim + 1)))
        print(row, flush=True)
        res.append(row)
        json.dump(res, open(OUT / f"r5_pipeline_null_nichol_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    {"southampton": southampton, "nichol": nichol}[sys.argv[1]]()
