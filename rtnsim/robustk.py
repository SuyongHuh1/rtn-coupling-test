"""K-trap additive idealisation (K = 1, 2, 3) with a slow baseline, plus K-trap coupling tests.

Model: y(t) = beta(t) + b + sum_k a_k s_k(t) + noise; traps are independent two-state chains
(product kinetics) in the idealiser.  States x = sum_k s_k 2^k (S = 2^K).  The coupling statistic
afterwards sums, over every ordered pair (i, j), the gain from letting trap i's switching depend
on trap j's current state (conditional-rate LRT); its null is built by simulating the whole
pipeline (independent chains + fitted baseline + the record's own residual noise).
"""
from __future__ import annotations

import numba as nb
import numpy as np
from scipy.ndimage import median_filter

from .surrogate import _ll_binary_transitions


@nb.njit(cache=True)
def _fb(y, P, pi0, mu, sig, want_path):
    T = y.size
    S = mu.size
    alpha = np.empty((T, S))
    c = np.empty(T)
    inv = 1.0 / sig
    tot = 0.0
    for s in range(S):
        z = (y[0] - mu[s]) * inv
        alpha[0, s] = pi0[s] * np.exp(-0.5 * z * z)
        tot += alpha[0, s]
    c[0] = tot
    alpha[0] /= tot
    for k in range(1, T):
        tot = 0.0
        for t in range(S):
            acc = 0.0
            for s in range(S):
                acc += alpha[k - 1, s] * P[s, t]
            z = (y[k] - mu[t]) * inv
            alpha[k, t] = acc * np.exp(-0.5 * z * z)
            tot += alpha[k, t]
        c[k] = tot
        for t in range(S):
            alpha[k, t] /= tot
    ll = np.sum(np.log(c)) - T * np.log(sig) - 0.5 * T * np.log(2 * np.pi)
    N = np.zeros((S, S))
    g0 = np.zeros(S)
    gy = np.zeros(S)
    gyy = np.zeros(S)
    path = np.zeros(T, np.int64)
    beta = np.ones(S)
    e = np.empty(S)
    nb_ = np.empty(S)
    for k in range(T - 1, -1, -1):
        g = alpha[k] * beta
        g /= g.sum()
        best = 0
        for s in range(S):
            g0[s] += g[s]
            gy[s] += g[s] * y[k]
            gyy[s] += g[s] * y[k] * y[k]
            if g[s] > g[best]:
                best = s
        path[k] = best
        if k == 0:
            break
        for t in range(S):
            z = (y[k] - mu[t]) * inv
            e[t] = np.exp(-0.5 * z * z) * beta[t] / c[k]
        for s in range(S):
            acc = 0.0
            for t in range(S):
                v = P[s, t] * e[t]
                N[s, t] += alpha[k - 1, s] * v
                acc += v
            nb_[s] = acc
        beta[:] = nb_
    first = alpha[0] * beta
    first /= first.sum()
    return ll, N, g0, gy, gyy, first, path


def bits(K):
    S = 2 ** K
    return np.array([[(x >> k) & 1 for k in range(K)] for x in range(S)])   # (S, K)


def product_P(Ps):
    """Ps: (K, 2, 2) -> (S, S) product transition matrix."""
    K = len(Ps)
    B = bits(K)
    S = 2 ** K
    P = np.ones((S, S))
    for k in range(K):
        P *= Ps[k][B[:, k][:, None], B[:, k][None, :]]
    return P


def _em(y, coef, sig, Ps, n_iter=80, tol=1e-8):
    K = len(Ps)
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    T = y.size
    pi0 = np.full(2 ** K, 1.0 / 2 ** K)
    prev = -np.inf
    for _ in range(n_iter):
        mu = X @ coef
        ll, N, g0, gy, gyy, first, _ = _fb(y, product_P(Ps), pi0, mu, sig, False)
        for k in range(K):
            n = np.zeros((2, 2))
            for x in range(2 ** K):
                for xp in range(2 ** K):
                    n[B[x, k], B[xp, k]] += N[x, xp]
            Ps[k] = (n + 1e-9) / (n.sum(1, keepdims=True) + 2e-9)
        G = (X.T * g0) @ X
        coef = np.linalg.solve(G + 1e-9 * np.eye(K + 1), X.T @ gy)
        mu = X @ coef
        sig = float(np.sqrt(max((gyy - 2 * mu * gy + mu * mu * g0).sum() / T, 1e-12)))
        pi0 = first
        if ll - prev < tol * T:
            break
        prev = ll
    return coef, sig, Ps, float(ll)


def _path(y, coef, sig, Ps):
    K = len(Ps)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    *_, path = _fb(y, product_P(Ps), np.full(2 ** K, 1.0 / 2 ** K), X @ coef, sig, True)
    return path


def idealise_k(y, K, n_outer=4, baseline=True, win_dwells=10.0, min_win=101, starts=4, rng=None):
    """Returns dict(S: (T, K) 0/1 paths ordered by step size, a: steps, sig, ll, bic, beta, Ps)."""
    y = np.asarray(y, float)
    rng = rng or np.random.default_rng(0)
    lo, hi = np.quantile(y, [0.02, 0.98])
    best = None
    for j in range(starts):
        w = np.sort(rng.dirichlet(np.ones(K)))[::-1] if j else 0.5 ** np.arange(K) / np.sum(0.5 ** np.arange(K))
        coef = np.r_[lo, (hi - lo) * w]
        Ps = np.array([[[0.95, 0.05], [0.05, 0.95]]] * K)
        r = _em(y, coef, (hi - lo) / 8, Ps.copy())
        if best is None or r[-1] > best[-1]:
            best = r
    coef, sig, Ps, ll = best
    beta = np.zeros_like(y)
    for _ in range(n_outer if baseline else 0):
        path = _path(y - beta, coef, sig, Ps)
        B = bits(K)[path]
        resid = y - coef[0] - B @ coef[1:]
        sw = max(max(P[0, 1], P[1, 0]) for P in Ps)
        dwell = 1.0 / max(min(max(P[0, 1], P[1, 0]) for P in Ps), 1e-6)
        w = int(max(min_win, win_dwells * dwell)) | 1
        beta = median_filter(resid, size=min(w, (y.size // 2) | 1), mode="nearest")
        coef, sig, Ps, ll = _em(y - beta, coef, sig, Ps)
    path = _path(y - beta, coef, sig, Ps)
    S = bits(K)[path]
    order = np.argsort(-np.abs(coef[1:]))
    n_par = 2 * K + (K + 1) + 1
    return dict(S=S[:, order], a=coef[1:][order], b=coef[0], sig=sig, ll=ll, beta=beta,
                Ps=Ps[order], bic=-2 * ll + n_par * np.log(y.size))


def coupling_stat(S):
    """Sum over ordered pairs (i, j) of the conditional-rate LRT of trap i given trap j."""
    S = np.asarray(S, np.int64)
    K = S.shape[1]
    tot = 0.0
    per = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            a0, a1, c0 = S[:-1, i], S[1:, i], S[:-1, j]
            g = _ll_binary_transitions(a0, a1, c0) - _ll_binary_transitions(a0, a1)
            per[f"{i}|{j}"] = float(g)
            tot += g
    return float(tot), per


def trans_P(s):
    s = np.asarray(s)
    P = np.eye(2)
    for a in (0, 1):
        m = s[:-1] == a
        n = m.sum()
        if n:
            q = np.sum(s[1:][m] != a) / n
            P[a] = [1 - q, q] if a == 0 else [q, 1 - q]
    return P


def _chain(P, T, rng, s0):
    s = np.empty(T, np.int64)
    s[0] = s0
    u = rng.random(T)
    for k in range(1, T):
        s[k] = s[k - 1] ^ (u[k] < P[s[k - 1], 1 - s[k - 1]])
    return s


def _block_boot(r, T, rng, L=None):
    if L is None:                                   # block length of the residual bootstrap (sensitivity runs)
        L = int(__import__("os").environ.get("RTN_BLOCK", "200"))
    out = np.empty(T)
    k = 0
    L = min(L, len(r) - 1)
    while k < T:
        i = rng.integers(0, len(r) - L)
        n = min(L, T - k)
        out[k:k + n] = r[i:i + n]
        k += n
    return out


def pipeline_null(segments, K, rng, n_sim=60, min_win=101):
    """segments: list of (y, fit) with fit from idealise_k(y, K). Returns obs, null (summed)."""
    obs = sum(coupling_stat(f["S"])[0] for _, f in segments)
    null = np.empty(n_sim)
    for j in range(n_sim):
        tot = 0.0
        for y, f in segments:
            T = len(y)
            fit = f["beta"] + f["b"] + f["S"] @ f["a"]
            resid = y - fit
            Ss = np.stack([_chain(trans_P(f["S"][:, k]), T, rng, f["S"][0, k]) for k in range(K)], 1)
            ys = f["beta"] + f["b"] + Ss @ f["a"] + _block_boot(resid, T, rng)
            g = idealise_k(ys, K, min_win=min_win, rng=rng)
            tot += coupling_stat(g["S"])[0]
        null[j] = tot
    return float(obs), null
