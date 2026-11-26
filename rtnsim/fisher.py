"""Hessian diagnostics for the coupled factorial HMM (R8/R9 engine, AR(1) emission).

For a fitted model (baseline held fixed) this computes, by finite differences of the exact
log-likelihood:
  H        observed information (minus Hessian) in the full parameter vector
  I_eff    efficient information for each coupling J_ij (Schur complement: information about J
           left after the other parameters are profiled out).  At the J = 0 fit this is the
           curvature of the expected LRT: LRT ~ noncentral chi2(1, I_eff J^2), so the minimum
           J detectable at level alpha with power beta is sqrt(lambda(alpha, beta) / I_eff).
           I_eff / 2 per unit J^2 is the record's coupling information, the empirical counterpart
           of D* (R1) times the record length.
  conf     confounding of J with the other parameters, 1 - I_eff / H_JJ (0 = none, 1 = total)
  se_J     sandwich (robust) standard error at the J-free fit, with block scores (record split into
           blocks treated as independent) to allow for misspecified noise.
Parameter vector: [log c (K), log e (K), J (K(K-1)/2), coef (K+1), log sigma, atanh rho].
"""
from __future__ import annotations

import numba as nb
import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq

from .robustk import bits


@nb.njit(cache=True)
def _logc_ar1(y, P, pi0, mu, sig, rho):
    T = y.size
    S = mu.size
    a = np.empty(S)
    an = np.empty(S)
    lc = np.empty(T)
    s0 = sig / np.sqrt(1 - rho * rho)
    tot = 0.0
    for s in range(S):
        z = (y[0] - mu[s]) / s0
        a[s] = pi0[s] * np.exp(-0.5 * z * z) / (s0 * np.sqrt(2 * np.pi))
        tot += a[s]
    lc[0] = np.log(tot)
    a /= tot
    for k in range(1, T):
        tot = 0.0
        for t in range(S):
            acc = 0.0
            for s in range(S):
                z = (y[k] - mu[t] - rho * (y[k - 1] - mu[s])) / sig
                acc += a[s] * P[s, t] * np.exp(-0.5 * z * z)
            an[t] = acc / (sig * np.sqrt(2 * np.pi))
            tot += an[t]
        lc[k] = np.log(tot)
        for t in range(S):
            a[t] = an[t] / tot
    return lc


def _Q(c, e, J, phi=0.5):
    K = len(c)
    B = bits(K).astype(float)
    S = 2 ** K
    Jm = np.zeros((K, K))
    iu = np.triu_indices(K, 1)
    Jm[iu] = J
    Jm[(iu[1], iu[0])] = J
    Q = np.zeros((S, S))
    for x in range(S):
        h = Jm @ B[x]
        for k in range(K):
            xp = x ^ (1 << k)
            Q[x, xp] = c[k] * np.exp(-phi * h[k]) if B[x, k] == 0 else e[k] * np.exp((1 - phi) * h[k])
    Q -= np.diag(Q.sum(1))
    return Q


def _stat(P):
    w, v = np.linalg.eig(P.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    p = np.abs(p)
    return p / p.sum()


class Model:
    def __init__(self, K):
        self.K = K
        self.np_ = K * (K - 1) // 2
        self.X = np.hstack([np.ones((2 ** K, 1)), bits(K)])

    def pack(self, f):
        return np.r_[np.log(f["c"]), np.log(f["e"]), np.asarray(f["J"], float) if len(f["J"]) else np.zeros(self.np_),
                     f["coef"], np.log(f["sig"]), np.arctanh(np.clip(f.get("rho", 0.0), -0.98, 0.98))]

    def idx_J(self):
        return np.arange(2 * self.K, 2 * self.K + self.np_)

    def logc(self, th, y):
        K, n = self.K, self.np_
        c, e, J = np.exp(th[:K]), np.exp(th[K:2 * K]), th[2 * K:2 * K + n]
        coef = th[2 * K + n:3 * K + n + 1]
        sig, rho = np.exp(th[-2]), np.tanh(th[-1])
        P = expm(_Q(c, e, J))
        P = np.clip(P, 1e-300, None)
        P /= P.sum(1, keepdims=True)
        return _logc_ar1(y, P, _stat(P), self.X @ coef, sig, rho)


def _hessian(fun, th, h):
    n = len(th)
    H = np.empty((n, n))
    f0 = fun(th)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                tp, tm = th.copy(), th.copy()
                tp[i] += h[i]; tm[i] -= h[i]
                H[i, i] = (fun(tp) - 2 * f0 + fun(tm)) / h[i] ** 2
            else:
                t = [th.copy() for _ in range(4)]
                t[0][i] += h[i]; t[0][j] += h[j]
                t[1][i] += h[i]; t[1][j] -= h[j]
                t[2][i] -= h[i]; t[2][j] += h[j]
                t[3][i] -= h[i]; t[3][j] -= h[j]
                H[i, j] = H[j, i] = (fun(t[0]) - fun(t[1]) - fun(t[2]) + fun(t[3])) / (4 * h[i] * h[j])
    return H


def lambda_needed(df=1, alpha=0.05, power=0.8):
    crit = chi2.ppf(1 - alpha, df)
    return brentq(lambda lam: ncx2.sf(crit, df, lam) - power, 1e-6, 500)


def diagnostics(y, f, h=1e-3, n_blocks=20):
    """y: record minus baseline; f: fitted model dict (fit / fit_ar1 output; J free or J = 0)."""
    K = len(f["c"])
    m = Model(K)
    th = m.pack(f)
    hs = h * np.maximum(1.0, np.abs(th))
    obsinfo = -_hessian(lambda t: m.logc(t, y).sum(), th, hs)
    iJ = m.idx_J()
    rest = np.setdiff1d(np.arange(len(th)), iJ)
    Hrr = obsinfo[np.ix_(rest, rest)]
    out = dict(I_eff=[], conf=[], J_min=[], se_J=[], cond=float(np.linalg.cond(obsinfo)))
    lam = lambda_needed(1)
    for q, j in enumerate(iJ):
        hJr = obsinfo[j, rest]
        ieff = float(obsinfo[j, j] - hJr @ np.linalg.solve(Hrr, hJr))
        out["I_eff"].append(ieff)
        out["conf"].append(float(1 - ieff / obsinfo[j, j]) if obsinfo[j, j] > 0 else float("nan"))
        out["J_min"].append(float(np.sqrt(lam / ieff)) if ieff > 0 else float("inf"))
    # sandwich with block scores
    T = len(y)
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    G = np.zeros((n_blocks, len(th)))
    for i in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[i] += hs[i]; tm[i] -= hs[i]
        d = (m.logc(tp, y) - m.logc(tm, y)) / (2 * hs[i])
        G[:, i] = np.add.reduceat(d, edges[:-1])
    try:
        Hinv = np.linalg.inv(obsinfo)
        V = Hinv @ (G.T @ G) @ Hinv
        out["se_J"] = [float(np.sqrt(max(V[j, j], 0))) for j in iJ]
        out["se_J_model"] = [float(np.sqrt(max(Hinv[j, j], 0))) for j in iJ]
    except np.linalg.LinAlgError:
        out["se_J"] = out["se_J_model"] = [float("nan")] * len(iJ)
    return out


def predicted_power(I_eff, J, df=1, alpha=0.05):
    crit = chi2.ppf(1 - alpha, df)
    return float(ncx2.sf(crit, df, I_eff * J * J))
