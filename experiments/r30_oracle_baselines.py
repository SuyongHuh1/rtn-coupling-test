"""R30: fairness check -- published decision statistics given the ORACLE decomposition.

Question: is the gap between the published procedures and ours due to their test statistic or to the
idealisation (decomposition, levels, baseline) they are fed?  Each baseline statistic is recomputed on the
R10 records with the true state path:
  levels / noise   least squares of y on the true occupancies (+ running-median baseline of the residual
                   in the 1/f condition), i.e. the decomposition is exact;
  Albrecht'26      independent factorial model re-fitted by EM with those levels (rates only), then the
                   4LF model (free 2^K x 2^K rate matrix, same levels) by EM; e = 2 dlnL / dN;
  Son'17           trap(smallest) | trap(largest) conditional-rate LRT on the true path;
  Son, all pairs   max over ordered pairs (Bonferroni-free score, for AUC);
  Guo'18           max |d| of the conditional time-constant ratios from the true path.
Scores only (AUC and power at matched 5 % FPR); no decision rules.
    python experiments/r30_oracle_baselines.py white|full [workers]
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from r19_albrecht_faithful import fit_P  # noqa: E402
from rtnsim.robustk import _fb, bits, product_P  # noqa: E402
from rtnsim.surrogate import _ll_binary_transitions  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def trap_P(s):
    """2x2 per-step transition matrix of one binary path."""
    P = np.zeros((2, 2))
    np.add.at(P, (s[:-1], s[1:]), 1)
    P += 1e-3
    return P / P.sum(1, keepdims=True)


def fit_indep(z, mu, sig, Ps, K, n_iter=60):
    """EM for the factorial model with fixed levels: joint posterior pair counts -> per-trap 2x2 rates."""
    B = bits(K).astype(int)
    M = 2 ** K
    pi0 = np.full(M, 1.0 / M)
    prev = -np.inf
    for _ in range(n_iter):
        P = product_P(Ps)
        ll, N, *_r, first, _ = _fb(z, P, pi0, mu, sig, False)
        Ps = np.empty((K, 2, 2))
        for k in range(K):
            C = np.zeros((2, 2))
            for x in range(M):
                for y in range(M):
                    C[B[x, k], B[y, k]] += N[x, y]
            Ps[k] = (C + 1e-6) / (C + 1e-6).sum(1, keepdims=True)
        pi0 = first
        if ll - prev < 1e-7 * z.size:
            break
        prev = ll
    P = product_P(Ps)
    return _fb(z, P, pi0, mu, sig, False)[0], P


def cond_logratios(S):
    """Guo-style d = tau_i(j occ) / tau_i(j empty) - 1 for every ordered pair and direction (true path)."""
    K = S.shape[1]
    d = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for a in (0, 1):
                r = []
                for c in (0, 1):
                    m = (S[:-1, i] == a) & (S[:-1, j] == c)
                    n = m.sum()
                    sw = (m & (S[1:, i] == 1 - a)).sum()
                    r.append((sw + 0.5) / (n + 1.0))
                d.append(r[0] / r[1] - 1.0)      # tau(j occ)/tau(j empty) - 1 = rate(empty)/rate(occ) - 1
    return np.array(d)


def son_g(S, i, j):
    t, c = S[:, i], S[:, j]
    return 2 * max(_ll_binary_transitions(t[:-1], t[1:], c[:-1]) - _ll_binary_transitions(t[:-1], t[1:]), 0.0)


def job(a):
    i, y, S, K = a
    S = S.astype(int)
    X = np.hstack([np.ones((len(y), 1)), S])
    beta = np.zeros_like(y)
    for _ in range(2 if COND != "white" else 1):
        coef, *_ = np.linalg.lstsq(X, y - beta, rcond=None)
        if COND != "white":
            beta = median_filter(y - X @ coef, size=301, mode="nearest")
    coef, *_ = np.linalg.lstsq(X, y - beta, rcond=None)
    z = y - beta
    order = np.argsort(-np.abs(coef[1:]))               # traps by step size, largest first
    S = S[:, order]; a_ = coef[1:][order]
    sig = float(np.std(z - coef[0] - S @ a_))
    mu = coef[0] + bits(K) @ a_
    Ps0 = np.array([trap_P(S[:, k]) for k in range(K)])
    ll0, P0 = fit_indep(z, mu, sig, Ps0, K)
    _, ll1 = fit_P(z, mu, sig, P0)
    M = 2 ** K
    e = 2 * (ll1 - ll0) / (M * (M - 1) - 2 * K)
    son1 = son_g(S, K - 1, 0)
    son_all = max(son_g(S, p, q) for p, q in permutations(range(K), 2))
    guo = float(np.max(np.abs(cond_logratios(S))))
    return i, dict(alb_e=float(e), son_one=float(son1), son_all=float(son_all), guo_d=guo)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def tpr5(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    return float(np.mean(s[y == 1] > np.quantile(s[y == 0], 0.95)))


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    res = {}
    with Pool(W) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], ss[i], rows[i]["K"]) for i in range(len(ys))], chunksize=4):
            res[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = [rows[i]["coupled"] for i in idx]
        summ[f"K={K}"] = {m: dict(auc=auc([res[i][m] for i in idx], lab), tpr5=tpr5([res[i][m] for i in idx], lab))
                          for m in ("alb_e", "son_one", "son_all", "guo_d")}
    out = ROOT / f"results/r8/r30_oracle_baselines_{COND}.json"
    json.dump(dict(summary=summ, res={str(k): v for k, v in res.items()}), open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
