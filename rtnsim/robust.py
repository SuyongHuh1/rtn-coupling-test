"""Idealisation of two-trap records with additive levels and a slowly varying baseline.

Model: y(t) = beta(t) + b + a1 s1(t) + a2 s2(t) + noise, s1, s2 independent 2-state chains
(product kinetics; the coupling test runs afterwards on the idealised paths, so the
idealiser itself must not assume coupling).  Levels are constrained to be additive, which
is what makes the four states two traps rather than four unrelated levels.  beta(t) is
re-estimated between EM rounds as a running median of the residual after subtracting the
fitted RTN, over a window much longer than the dwell times, so it follows drift but not steps.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from . import detectors as det

S1, S2 = det.S1, det.S2
X = np.stack([np.ones(4), S1, S2], 1)          # level design matrix (4 states x [b, a1, a2])


def _em_additive(y, b, a1, a2, sig, P1, P2, n_iter=60, tol=1e-8):
    yb = y[None]
    T = y.size
    pi0 = np.full((1, 4), 0.25)
    prev = -np.inf
    for _ in range(n_iter):
        mu = (X @ np.array([b, a1, a2]))[None]
        P = det._product_P(P1[None], P2[None])
        ll, N, g0, gy, gyy, first, _ = det.forward_backward(yb, P, pi0, mu, np.array([sig]))
        n1 = np.zeros((2, 2))
        n2 = np.zeros((2, 2))
        for x in range(4):
            for xp in range(4):
                n1[S1[x], S1[xp]] += N[0, x, xp]
                n2[S2[x], S2[xp]] += N[0, x, xp]
        P1 = (n1 + 1e-9) / (n1.sum(1, keepdims=True) + 2e-9)
        P2 = (n2 + 1e-9) / (n2.sum(1, keepdims=True) + 2e-9)
        G = (X.T * g0[0]) @ X
        b, a1, a2 = np.linalg.solve(G + 1e-9 * np.eye(3), X.T @ gy[0])
        mu = X @ np.array([b, a1, a2])
        sig = float(np.sqrt(max((gyy[0] - 2 * mu * gy[0] + mu * mu * g0[0]).sum() / T, 1e-12)))
        pi0 = first
        if ll[0] - prev < tol * T:
            break
        prev = ll[0]
    return b, a1, a2, sig, P1, P2, float(ll[0])


def idealise(y, n_outer=4, baseline=True, win_dwells=10.0, min_win=101):
    """Returns dict(s1, s2, a1, a2, sig, P1, P2, ll, beta)."""
    y = np.asarray(y, float)
    lo, hi = np.quantile(y, [0.02, 0.98])
    best = None
    for f in (0.2, 0.35, 0.5):                     # starts: step heights split f : 1 - f
        b, a1, a2, sig = lo, (hi - lo) * (1 - f), (hi - lo) * f, (hi - lo) / 8
        P = np.array([[0.95, 0.05], [0.05, 0.95]])
        r = _em_additive(y, b, a1, a2, sig, P.copy(), P.copy())
        if best is None or r[-1] > best[-1]:
            best = r
    b, a1, a2, sig, P1, P2, ll = best
    beta = np.zeros_like(y)
    for _ in range(n_outer if baseline else 0):
        mu = X @ np.array([b, a1, a2])
        _, _, g0, *_ = det.forward_backward((y - beta)[None], det._product_P(P1[None], P2[None]),
                                            np.full((1, 4), 0.25), mu[None], np.array([sig]))
        path = _path(y - beta, b, a1, a2, sig, P1, P2)
        resid = y - (b + a1 * S1[path] + a2 * S2[path])
        dwell = max(1.0 / max(P1[0, 1], P1[1, 0], 1e-6), 1.0 / max(P2[0, 1], P2[1, 0], 1e-6))
        w = int(max(min_win, win_dwells * dwell)) | 1
        beta = median_filter(resid, size=min(w, (y.size // 2) | 1), mode="nearest")
        b, a1, a2, sig, P1, P2, ll = _em_additive(y - beta, b, a1, a2, sig, P1, P2)
    path = _path(y - beta, b, a1, a2, sig, P1, P2)
    if abs(a2) > abs(a1):                          # trap 1 = larger step
        a1, a2, P1, P2 = a2, a1, P2, P1
        path = S2[path] + 2 * S1[path]
    return dict(s1=S1[path], s2=S2[path], a1=float(a1), a2=float(a2), sig=float(sig), P1=P1, P2=P2,
                ll=ll, beta=beta)


def _path(y, b, a1, a2, sig, P1, P2):
    mu = X @ np.array([b, a1, a2])
    *_, path = det.forward_backward(y[None], det._product_P(P1[None], P2[None]), np.full((1, 4), 0.25),
                                    mu[None], np.array([sig]), want_path=True)
    return path[0]
