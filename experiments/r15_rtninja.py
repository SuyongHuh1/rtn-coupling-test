"""R15: does a modern ML decomposition front-end remove false coupling?

Front-end: RTNinja-style reimplementation (Varanasi, Degraeve, Roussel, Merckling, arXiv:2507.08424,
J. Appl. Phys./AIP 2025).  No code is public, so this follows the algorithm as described:

  LevelsExtractor  sequential Bayesian level building (new level when the point is improbable under
                   every existing level), continuity rule c = 3 (a transition is real only if the
                   new level persists c samples), sigma swept from the median 2x2 moving-range
                   estimate to the 80th percentile of 2x2 ranges; model with the lowest BIC
                   (Gaussian residual log-lik, d = number of levels).
  SourcesMapper    candidate amplitudes = pairwise level differences (merged within 1 sigma);
                   for each K-combination, 2^K on/off sums with the lowest level as baseline;
                   cost = mean level mismatch (in sigma, capped at 1) + fraction of transitions in
                   which more than one source switches (both in [0, 1]); minimum cost wins.
                   K is given (as for every other baseline in R10).

Coupling rules then applied to the RTNinja per-trap binary paths:
  guo_d      conditional dwell ratio d = tau_i(j occ)/tau_i(j empty) - 1, coupled if max|d| > 0.5
  son_bonf   conditional-transition chi2(2), Bonferroni over ordered pairs
  zhang_lv   fewer extracted levels than 2^K

Same records (r8_oracle.make, T = 4096) and baseline preprocessing as r10_baselines.py.
    python experiments/r15_rtninja.py white|full
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from rtnsim.robustk import bits, idealise_k  # noqa: E402
from rtnsim.surrogate import _ll_binary_transitions  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"
Z_NEW = 4.0      # point is "improbable" under a level beyond 4 sigma (two-sided p ~ 6e-5)
C_RUN = 3


def levels_at(z, sig):
    mus, ns = [z[0]], [1]
    for x in z[1:]:
        d = np.abs(np.array(mus) - x) / sig
        k = int(np.argmin(d))
        if d[k] > Z_NEW:
            mus.append(x); ns.append(1)
        else:
            ns[k] += 1
            mus[k] += (x - mus[k]) / ns[k]
    mus = np.array(mus)
    for _ in range(3):                       # refine and drop levels without a c-run
        lab = np.argmin(np.abs(z[:, None] - mus[None]), 1)
        path = continuity(lab)
        keep = [k for k in range(len(mus)) if np.any(path == k)]
        mus = np.array([z[path == k].mean() for k in keep])
        mus = np.sort(mus)
        # merge levels closer than 1 sigma
        m = [mus[0]]
        for u in mus[1:]:
            if u - m[-1] < sig:
                m[-1] = 0.5 * (m[-1] + u)
            else:
                m.append(u)
        mus = np.array(m)
    lab = np.argmin(np.abs(z[:, None] - mus[None]), 1)
    return mus, continuity(lab)


def continuity(lab):
    out = lab.copy()
    cur = lab[0]
    T = len(lab)
    t = 0
    while t < T:
        if lab[t] != cur and t + C_RUN <= T and np.all(lab[t:t + C_RUN] == lab[t]):
            cur = lab[t]
        out[t] = cur
        t += 1
    return out


def extract_levels(z):
    r = np.abs(z[2:] - z[:-2])                  # 2x2 moving range
    s0 = np.median(r) / 0.954 / np.sqrt(2)      # median-based sigma estimate
    s1 = max(np.quantile(r, 0.8), 1.5 * s0)
    best = None
    for sig in np.linspace(s0, s1, 10):
        mus, path = levels_at(z, sig)
        res = z - mus[path]
        sd = max(res.std(), 1e-9)
        ll = -0.5 * len(z) * (np.log(2 * np.pi * sd ** 2) + 1)
        bic = -2 * ll + len(mus) * np.log(len(z))
        if best is None or bic < best[0]:
            best = (bic, mus, path, sd)
    return best[1], best[2], best[3]


def map_sources(mus, path, sd, K):
    B = bits(K)
    diffs = np.array([b - a for a, b in combinations(mus, 2)] or [2 * sd])
    diffs = np.sort(diffs)
    cand = [diffs[0]]
    for d in diffs[1:]:
        if d - cand[-1] > sd:
            cand.append(d)
        else:
            cand[-1] = 0.5 * (cand[-1] + d)
    cand = np.array(cand)
    if len(cand) < K:
        cand = np.concatenate([cand, np.full(K - len(cand), cand.max())])
    trans = np.flatnonzero(np.diff(path))
    best = None
    for combo in combinations(range(len(cand)), K):
        amp = cand[list(combo)]
        conf = mus[0] + B @ amp                   # 2^K configuration levels
        mism = np.abs(mus[:, None] - conf[None]) / sd
        lv2c = np.argmin(mism, 1)
        g = np.minimum(mism[np.arange(len(mus)), lv2c], 1.0).mean()
        Sc = B[lv2c[path]]
        if len(trans):
            nsw = np.abs(Sc[trans + 1] - Sc[trans]).sum(1)
            v = float(np.mean(nsw > 1))
        else:
            v = 0.0
        cost = g + v
        if best is None or cost < best[0]:
            best = (cost, Sc)
    return best[1]


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
                    r.append(((a1[m] != a).sum() + 0.5) / (m.sum() + 1.0))   # +0.5 pseudocount: unvisited conditions stay finite
                d.append(r[0] / max(r[1], 1e-12) - 1.0)   # tau(j occ)/tau(j empty) - 1
    return float(np.max(np.abs(d)))


def job(a):
    i, y, st, K = a
    rng = np.random.default_rng(i)
    if COND == "white":
        z = y - np.median(y)
    else:
        z = y - idealise_k(y, K, baseline=True, rng=rng)["beta"]
    mus, path, sd = extract_levels(z)
    S = map_sources(mus, path, sd, K)
    ps = []
    for p in range(K):
        for q in range(K):
            if p != q:
                g = _ll_binary_transitions(S[:-1, p], S[1:, p], S[:-1, q]) - _ll_binary_transitions(S[:-1, p], S[1:, p])
                ps.append(chi2.sf(2 * max(g, 0), 2))
    from itertools import permutations
    acc = max(np.mean(S[:, list(pm)] == st) for pm in permutations(range(K)))
    from r19c_son_faithful import son_p
    return i, dict(acc=float(acc), n_levels=int(len(mus)), guo=guo_d(S), son_minp=float(min(ps)), son_m=len(ps),
                   son_faithful=son_p(S[:, ::-1]),   # columns ascending in step -> larger trap first
                   active=[int(np.count_nonzero(np.diff(S[:, k]))) for k in range(K)])


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], ss[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        calls = dict(guo_d0p5=np.array([out[i]["guo"] > 0.5 for i in idx]),
                     son_bonf=np.array([out[i]["son_minp"] < 0.05 / out[i]["son_m"] for i in idx]),
                     son_faithful=np.array([out[i]["son_faithful"] < 0.05 for i in idx]),
                     zhang_lv=np.array([out[i]["n_levels"] < 2 ** K for i in idx]))
        d = dict(n_indep=int((lab == 0).sum()), n_coupled=int((lab == 1).sum()))
        for k, c in calls.items():
            d[k] = dict(fpr=float(c[lab == 0].mean()), tpr=float(c[lab == 1].mean()))
        d["levels_hist_indep"] = np.bincount([out[i]["n_levels"] for i, l in zip(idx, lab) if l == 0]).tolist()
        # decomposition accuracy against ground truth (best trap permutation)
        d["state_acc_median"] = float(np.median([out[i]["acc"] for i in idx]))
        summ[f"K={K}"] = d
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r15_rtninja_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
