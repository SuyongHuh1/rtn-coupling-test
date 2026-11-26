"""R3a: drive-aware HMM test for trap-trap coupling.

Hidden state x = (s1, s2, u), index s1 + 2 s2 + 4 u, where u is a hidden shared
telegraph drive (e.g. a gate-voltage or temperature fluctuation).  The levels
depend on (s1, s2) only and are fitted once on the pooled record.

  null  (drive only):        trap i's 2x2 matrix depends on u           P_i^(u)
  alt   (drive + coupling):  trap i's 2x2 matrix depends on (s_j, u)     P_i^(s_j, u)
  both: drive u has its own 2x2 matrix D (independent of the traps).

GLRT = ll(alt) - ll(null): 8 extra transition numbers.  A shared drive is now
part of the null, so correlated switching caused by a drive is explained away;
only a dependence of one trap's rates on the *other trap's state* counts.
"""
from __future__ import annotations

import numpy as np
import numba as nb

from .detectors import fit_shared_levels

S1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
S2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
U = np.array([0, 0, 0, 0, 1, 1, 1, 1])


@nb.njit(parallel=True, cache=True)
def fb_general(y, P, pi0, mu, sig):
    """Forward-backward for S-state HMMs with per-trace parameters."""
    B, T = y.shape
    S = P.shape[1]
    ll = np.empty(B)
    N = np.zeros((B, S, S))
    first = np.empty((B, S))
    for b in nb.prange(B):
        alpha = np.empty((T, S))
        c = np.empty(T)
        e = np.empty(S)
        inv = 1.0 / sig[b]
        tot = 0.0
        for s in range(S):
            z = (y[b, 0] - mu[b, s]) * inv
            alpha[0, s] = pi0[b, s] * np.exp(-0.5 * z * z)
            tot += alpha[0, s]
        c[0] = tot
        for s in range(S):
            alpha[0, s] /= tot
        for k in range(1, T):
            tot = 0.0
            for t in range(S):
                acc = 0.0
                for s in range(S):
                    acc += alpha[k - 1, s] * P[b, s, t]
                z = (y[b, k] - mu[b, t]) * inv
                acc *= np.exp(-0.5 * z * z)
                alpha[k, t] = acc
                tot += acc
            c[k] = tot
            for t in range(S):
                alpha[k, t] /= tot
        lsum = 0.0
        for k in range(T):
            lsum += np.log(c[k])
        ll[b] = lsum - T * np.log(sig[b]) - 0.5 * T * np.log(2.0 * np.pi)
        beta = np.ones(S)
        nb_ = np.empty(S)
        for k in range(T - 1, 0, -1):
            for t in range(S):
                z = (y[b, k] - mu[b, t]) * inv
                e[t] = np.exp(-0.5 * z * z) * beta[t] / c[k]
            for s in range(S):
                acc = 0.0
                for t in range(S):
                    v = P[b, s, t] * e[t]
                    N[b, s, t] += alpha[k - 1, s] * v
                    acc += v
                nb_[s] = acc
            for s in range(S):
                beta[s] = nb_[s]
        tot = 0.0
        for s in range(S):
            first[b, s] = alpha[0, s] * beta[s]
            tot += first[b, s]
        for s in range(S):
            first[b, s] /= tot
    return ll, N, first


def _build_P(P1, P2, D):
    """P1[b, j_state, u, a, a'] (j_state = s2), P2[b, s1, u, a, a'], D[b, u, u'] -> (B, 8, 8)."""
    B = D.shape[0]
    P = np.empty((B, 8, 8))
    for x in range(8):
        for xp in range(8):
            P[:, x, xp] = (P1[:, S2[x], U[x], S1[x], S1[xp]] * P2[:, S1[x], U[x], S2[x], S2[xp]]
                           * D[:, U[x], U[xp]])
    return P


def _norm(n):
    return (n + 1e-12) / (n.sum(-1, keepdims=True) + 2e-12)


def _clip_drive(D, p_max):
    """Slow-drive constraint: switching probability per sample <= p_max (projected M-step)."""
    D = D.copy()
    for u in range(2):
        off = np.minimum(D[:, u, 1 - u], p_max)
        D[:, u, 1 - u], D[:, u, u] = off, 1.0 - off
    return D


def _fit(y, mu8, sig, P1, P2, D, coupled, n_iter, tol, p_max):
    B, T = y.shape
    pi0 = np.full((B, 8), 1 / 8)
    prev = -np.inf
    for it in range(n_iter):
        P = _build_P(P1, P2, D)
        ll, N, first = fb_general(y, P, pi0, mu8, sig)
        n1 = np.zeros((B, 2, 2, 2, 2))
        n2 = np.zeros((B, 2, 2, 2, 2))
        nd = np.zeros((B, 2, 2))
        for x in range(8):
            for xp in range(8):
                n1[:, S2[x], U[x], S1[x], S1[xp]] += N[:, x, xp]
                n2[:, S1[x], U[x], S2[x], S2[xp]] += N[:, x, xp]
                nd[:, U[x], U[xp]] += N[:, x, xp]
        if not coupled:   # tie over the other trap's state
            n1 = np.repeat(n1.sum(1, keepdims=True), 2, axis=1)
            n2 = np.repeat(n2.sum(1, keepdims=True), 2, axis=1)
        P1, P2, D = _norm(n1), _norm(n2), _clip_drive(_norm(nd), p_max)
        pi0 = first
        if it > 0 and np.max(ll - prev) / T < tol:
            break
        prev = ll
    ll, _, _ = fb_general(y, _build_P(P1, P2, D), pi0, mu8, sig)
    return dict(ll=ll, P1=P1, P2=P2, D=D)


def drive_aware_scores(y, n_iter=50, tol=1e-8, p_max=0.03):
    """GLRT for coupling with a hidden *slow* shared drive in both hypotheses.

    Without a speed limit the drive-only null can mimic coupling (the drive tracks one
    trap's state and the other trap follows the drive), so coupling would not be
    identifiable.  Physical gate / temperature fluctuations are slow, and fast drives
    do not bias independence tests (R2), so the drive's switching probability is capped
    at p_max per sample (dwell >= 1/p_max samples)."""
    B, T = y.shape
    mu4, sig4 = fit_shared_levels(y)
    mu8 = np.repeat(np.tile(mu4, 2)[None], B, 0)
    sig = np.full(B, sig4)
    # start: symmetric single-trap matrices, slow/fast under u = 0/1 (breaks the u-label symmetry)
    def trap(sw):
        return np.array([[1 - sw, sw], [sw, 1 - sw]])
    base = np.stack([trap(0.07), trap(0.13)])           # [u, a, a']
    P1 = np.broadcast_to(base[None, None], (B, 2, 2, 2, 2)).copy()
    P2 = P1.copy()
    D = np.broadcast_to(np.array([[0.98, 0.02], [0.02, 0.98]])[None], (B, 2, 2)).copy()
    null = _fit(y, mu8, sig, P1, P2, D, False, n_iter, tol, p_max)
    alt = _fit(y, mu8, sig, null["P1"], null["P2"], null["D"], True, n_iter, tol, p_max)
    return np.maximum(alt["ll"] - null["ll"], 0.0)
