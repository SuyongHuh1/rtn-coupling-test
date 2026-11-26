"""Heavy-tailed (Student-t) innovations for the AR(1) factorial HMM.

y_k - mu(s_k) = rho (y_{k-1} - mu(s_{k-1})) + eps_k,   eps_k ~ sigma * t_nu.
EM with the usual latent scale: for each transition the E-step weight w = (nu + 1) / (nu + z^2) multiplies the
sufficient statistics, so the M-step is the Gaussian one on weighted statistics (ECM, nu fixed).  nu is
estimated once from the innovations of the Gaussian J = 0 fit and shared by the J = 0 and J-free fits.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .cfhmm import Kinetics, _emission_mstep_ar1
from .robustk import bits


@nb.njit(cache=True)
def _fb_ar1_t(y, P, pi0, mu, sig, rho, nu):
    T = y.size
    S = mu.size
    alpha = np.empty((T, S))
    c = np.empty(T)
    inv = 1.0 / sig
    s0 = sig / np.sqrt(1 - rho * rho)
    e = -0.5 * (nu + 1.0)
    tot = 0.0
    for s in range(S):
        z = (y[0] - mu[s]) / s0
        alpha[0, s] = pi0[s] * (1.0 + z * z / nu) ** e * sig / s0
        tot += alpha[0, s]
    c[0] = tot
    alpha[0] /= tot
    for k in range(1, T):
        tot = 0.0
        for t in range(S):
            acc = 0.0
            for s in range(S):
                z = (y[k] - mu[t] - rho * (y[k - 1] - mu[s])) * inv
                acc += alpha[k - 1, s] * P[s, t] * (1.0 + z * z / nu) ** e
            alpha[k, t] = acc
            tot += acc
        c[k] = tot
        for t in range(S):
            alpha[k, t] /= tot
    lnorm = math.lgamma(0.5 * (nu + 1)) - math.lgamma(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    ll = np.sum(np.log(c)) - T * np.log(sig) + T * lnorm
    N = np.zeros((S, S))
    st = np.zeros((6, S, S))
    path = np.zeros(T, np.int64)
    beta = np.ones(S)
    nb_ = np.empty(S)
    for k in range(T - 1, 0, -1):
        g = alpha[k] * beta
        best = 0
        for s in range(S):
            if g[s] > g[best]:
                best = s
        path[k] = best
        for s in range(S):
            acc = 0.0
            for t in range(S):
                z = (y[k] - mu[t] - rho * (y[k - 1] - mu[s])) * inv
                v = P[s, t] * (1.0 + z * z / nu) ** e * beta[t] / c[k]
                xi = alpha[k - 1, s] * v
                N[s, t] += xi
                xw = xi * (nu + 1.0) / (nu + z * z)
                st[0, s, t] += xw
                st[1, s, t] += xw * y[k]
                st[2, s, t] += xw * y[k - 1]
                st[3, s, t] += xw * y[k] * y[k]
                st[4, s, t] += xw * y[k - 1] * y[k - 1]
                st[5, s, t] += xw * y[k] * y[k - 1]
                acc += v
            nb_[s] = acc
        beta[:] = nb_
    first = alpha[0] * beta
    first /= first.sum()
    best = 0
    for s in range(S):
        if first[s] > first[best]:
            best = s
    path[0] = best
    return ll, N, st, first, path


def fit_ar1_t(y, K, coupled, init, nu, n_iter=80, tol=1e-7):
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    coef, sig, rho = np.array(init["coef"], float).copy(), float(init["sig"]), float(init.get("rho", 0.0))
    kin = Kinetics(K, np.asarray(init["c"]), np.asarray(init["e"]), init.get("J") if coupled else None, coupled=coupled)
    T = y.size
    pi0 = np.full(2 ** K, 1.0 / 2 ** K)
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, st, first, _ = _fb_ar1_t(y, P, pi0, X @ coef, sig, rho, nu)
        if ll - prev < tol * T and it > 2:
            break
        prev = ll
        kin.m_step(N)
        coef, rho, sig = _emission_mstep_ar1(st, X, coef, rho, T)
        pi0 = first
    P = kin.P_np()
    ll, *_, path = _fb_ar1_t(y, P, pi0, X @ coef, sig, rho, nu)
    n_par = 2 * K + (K + 1) + 3 + (K * (K - 1) // 2 if coupled else 0)
    return dict(ll=float(ll), coef=coef, sig=sig, rho=rho, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), P=P, path=path, nu=nu,
                bic=-2 * ll + n_par * np.log(T))


def innovations(z, f, K):
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    r = z - (X @ f["coef"])[f["path"]]
    return r[1:] - f.get("rho", 0.0) * r[:-1]


def estimate_nu(eps, lo=2.5, hi=100.0):
    """ML degrees of freedom of a zero-centred Student-t on the innovations (grid + scale profile)."""
    from scipy.special import gammaln
    eps = eps / (np.median(np.abs(eps)) / 0.6745 + 1e-12)
    best = (-np.inf, hi)
    for nu in np.geomspace(lo, hi, 40):
        s = 1.0
        for _ in range(30):                       # EM for the scale at fixed nu
            w = (nu + 1) / (nu + (eps / s) ** 2)
            s = np.sqrt(np.mean(w * eps ** 2))
        x = (eps / s) ** 2
        ll = np.sum(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s)
                    - (nu + 1) / 2 * np.log1p(x / nu))
        if ll > best[0]:
            best = (ll, nu)
    return float(best[1])


def t_refit(z, K, f0g, f1g, nu):
    """t-emission J = 0 and J-free fits started from the Gaussian solutions; nested refit guard."""
    ini0 = dict(coef=f0g["coef"], sig=f0g["sig"], rho=f0g.get("rho", 0.0), c=f0g["c"], e=f0g["e"])
    f0 = fit_ar1_t(z, K, False, ini0, nu)
    starts = [dict(coef=f1g["coef"], sig=f1g["sig"], rho=f1g.get("rho", 0.0), c=f1g["c"], e=f1g["e"], J=f1g["J"]),
              dict(coef=f0["coef"], sig=f0["sig"], rho=f0["rho"], c=f0["c"], e=f0["e"])]
    f1 = None
    for st in starts:
        try:
            g = fit_ar1_t(z, K, True, st, nu)
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(g["ll"]) and (f1 is None or g["ll"] > f1["ll"]):
            f1 = g
    try:
        f0b = fit_ar1_t(z, K, False, dict(coef=f1["coef"], sig=f1["sig"], rho=f1["rho"], c=f1["c"], e=f1["e"]), nu)
        if np.isfinite(f0b["ll"]) and f0b["ll"] > f0["ll"]:
            f0 = f0b
    except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        pass
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1


def hampel(y, w=7, k=3.0):
    from scipy.ndimage import median_filter
    m = median_filter(y, size=w, mode="nearest")
    mad = median_filter(np.abs(y - m), size=w, mode="nearest") * 1.4826
    s = np.median(np.abs(np.diff(y))) / 0.6745 / np.sqrt(2)
    out = y.copy()
    bad = np.abs(y - m) > k * np.maximum(mad, s)
    out[bad] = m[bad]
    return out
