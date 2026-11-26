"""Integrated-sampling emission for RTN records taken with a boxcar / integrating instrument.

Each sample is the time average of the trap signal over one sampling interval (60 Hz boxcar in the
Nichol data; NPLC integration on SMUs).  Given the chain state at the start (i) and end (j) of the
interval, the average occupancy of each trap has a mean w_ij,k and covariance C_ij,kl that follow
exactly from the generator Q (Van Loan block exponentials).  Emission (moment-matched Gaussian):

    y_t | (s_{t-1} = i, s_t = j)  ~  N( b + sum_k a_k w_ijk ,  sigma^2 + sum_kl a_k a_l C_ijkl )

This keeps the pairwise structure of the HMM, so forward-backward is unchanged in cost.  The
M-step maximises the full expected complete-data log-likelihood (transitions + emissions, both of
which depend on Q) jointly over kinetics, J, amplitudes, per-segment offsets and sigma by L-BFGS
through the block matrix exponentials.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import torch

from .robustk import bits
from .sim2_torch import generator


def _blocks(Q, D1, D2=None):
    S = Q.shape[0]
    Z = torch.zeros_like(Q)
    if D2 is None:
        C = torch.cat([torch.cat([Q, D1], 1), torch.cat([Z, Q], 1)], 0)
        return torch.linalg.matrix_exp(C)[:S, S:]
    C = torch.cat([torch.cat([Q, D1, Z], 1), torch.cat([Z, Q, D2], 1), torch.cat([Z, Z, Q], 1)], 0)
    return torch.linalg.matrix_exp(C)[:S, 2 * S:]


def interval_moments(Q, K):
    """P (S,S); W (S,S,K) mean occupancy of trap k over the interval given endpoints;
    C (S,S,K,K) its covariance given endpoints."""
    S = Q.shape[0]
    B = torch.tensor(bits(K), dtype=Q.dtype)
    P = torch.linalg.matrix_exp(Q).clamp_min(1e-300)
    D = [torch.diag(B[:, k]) for k in range(K)]
    W = torch.stack([_blocks(Q, D[k]) for k in range(K)], -1) / P[..., None]
    M2 = torch.zeros(S, S, K, K, dtype=Q.dtype)
    for k in range(K):
        for l in range(k, K):
            m = (_blocks(Q, D[k], D[l]) + _blocks(Q, D[l], D[k])) / P   # E[int f_k int f_l | i, j]
            M2[:, :, k, l] = m
            M2[:, :, l, k] = m
    C = M2 - W[..., :, None] * W[..., None, :]
    return P, W, C


@nb.njit(cache=True)
def _fb_pair(y, P, pi0, m, v):
    """Forward-backward with pairwise Gaussian emission N(m[i,j], v[i,j]); first sample uses the
    diagonal (i = j) emission.  Returns ll, transition/emission stats per pair, first-state post."""
    T = y.size
    S = P.shape[0]
    alpha = np.empty((T, S))
    c = np.empty(T)
    tot = 0.0
    for s in range(S):
        z = y[0] - m[s, s]
        alpha[0, s] = pi0[s] * np.exp(-0.5 * z * z / v[s, s]) / np.sqrt(2 * np.pi * v[s, s])
        tot += alpha[0, s]
    c[0] = tot
    alpha[0] /= tot
    E = np.empty((S, S))
    for k in range(1, T):
        for i in range(S):
            for j in range(S):
                z = y[k] - m[i, j]
                E[i, j] = P[i, j] * np.exp(-0.5 * z * z / v[i, j]) / np.sqrt(2 * np.pi * v[i, j])
        tot = 0.0
        for j in range(S):
            acc = 0.0
            for i in range(S):
                acc += alpha[k - 1, i] * E[i, j]
            alpha[k, j] = acc
            tot += acc
        c[k] = tot
        for j in range(S):
            alpha[k, j] /= tot
    ll = np.sum(np.log(c))
    n = np.zeros((S, S)); sy = np.zeros((S, S)); syy = np.zeros((S, S))
    beta = np.ones(S)
    nb_ = np.empty(S)
    for k in range(T - 1, 0, -1):
        for i in range(S):
            acc = 0.0
            for j in range(S):
                z = y[k] - m[i, j]
                e = P[i, j] * np.exp(-0.5 * z * z / v[i, j]) / np.sqrt(2 * np.pi * v[i, j]) * beta[j] / c[k]
                xi = alpha[k - 1, i] * e
                n[i, j] += xi; sy[i, j] += xi * y[k]; syy[i, j] += xi * y[k] * y[k]
                acc += e
            nb_[i] = acc
        beta[:] = nb_
    first = alpha[0] * beta
    first /= first.sum()
    return ll, n, sy, syy, first


class IntegModel:
    """Parameters: log c, log e, J (if coupled), coef = (b, a_1..a_K), log sigma, offsets."""

    def __init__(self, K, n_seg, init, coupled):
        t = lambda x, g=True: torch.tensor(np.asarray(x, float), dtype=torch.float64, requires_grad=g)
        self.K, self.coupled = K, coupled
        self.lc, self.le = t(np.log(init["c"])), t(np.log(init["e"]))
        self.j = t(init.get("J", np.zeros(K * (K - 1) // 2)) if coupled else np.zeros(K * (K - 1) // 2), coupled)
        self.coef = t(init["coef"])
        self.ls = t(np.log(init["sig"]))
        self.off = t(init.get("off", np.zeros(n_seg)))
        self.iu = torch.triu_indices(K, K, 1)

    def params(self):
        return [self.lc, self.le, self.coef, self.ls, self.off] + ([self.j] if self.coupled else [])

    def Q(self):
        K = self.K
        Jm = torch.zeros(K, K, dtype=torch.float64)
        Jm = Jm.index_put((self.iu[0], self.iu[1]), self.j).index_put((self.iu[1], self.iu[0]), self.j)
        return generator(self.lc.exp()[None], self.le.exp()[None], Jm[None],
                         torch.tensor([0.5], dtype=torch.float64))[0]

    def emission(self):
        P, W, C = interval_moments(self.Q(), self.K)
        a = self.coef[1:]
        m = self.coef[0] + W @ a                                  # (S,S)
        v = self.ls.exp() ** 2 + torch.einsum("ijkl,k,l->ij", C, a, a).clamp_min(0)
        return P, m, v

    def numpy(self):
        with torch.no_grad():
            P, m, v = self.emission()
            P = P.numpy(); P = P / P.sum(1, keepdims=True)
            return P, m.numpy(), v.numpy(), self.off.numpy().copy()


def fit_integ(zs, K, coupled, init, n_iter=40, tol=1e-7, lbfgs_iter=20):
    mod = IntegModel(K, len(zs), init, coupled)
    S = 2 ** K
    pis = [np.full(S, 1.0 / S) for _ in zs]
    Ttot = sum(z.size for z in zs)
    best_ll, best_state, stall = -np.inf, None, 0
    for it in range(n_iter):
        P, m, v, off = mod.numpy()
        ll, stats = 0.0, []
        for r, z in enumerate(zs):
            l, n, sy, syy, first = _fb_pair(z - off[r], P, pis[r], m, v)
            ll += l; pis[r] = first
            stats.append((n, sy, syy))
        if ll > best_ll + tol * Ttot:
            best_ll, stall = ll, 0
            best_state = ([x.detach().clone() for x in mod.params()], [p.copy() for p in pis])
        else:
            stall += 1
            if stall >= 3:
                break
        Nt = torch.tensor(sum(s[0] for s in stats))
        st = [tuple(torch.tensor(x) for x in s) for s in stats]
        opt = torch.optim.LBFGS(mod.params(), max_iter=lbfgs_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            P_, m_, v_ = mod.emission()
            f = (Nt * torch.log(P_)).sum()
            for r, (n, sy, syy) in enumerate(st):
                mm = m_ + (mod.off[r] - float(off[r]))
                f = f - 0.5 * (n * torch.log(2 * np.pi * v_) + (syy - 2 * mm * sy + mm * mm * n) / v_).sum()
            f = -f / Ttot
            if mod.coupled:
                f = f + 1e-6 * (mod.j ** 2).sum()
            f.backward()
            return f

        opt.step(closure)
        with torch.no_grad():
            mod.lc.clamp_(-12, 3); mod.le.clamp_(-12, 3)
            if mod.coupled:
                mod.j.clamp_(-8, 8)
    with torch.no_grad():
        for x, b in zip(mod.params(), best_state[0]):
            x.copy_(b)
    pis = best_state[1]
    P, m, v, off = mod.numpy()
    ll = sum(_fb_pair(z - off[r], P, pis[r], m, v)[0] for r, z in enumerate(zs))
    det = lambda x: x.detach().numpy().copy()
    return dict(ll=float(ll), coef=det(mod.coef), sig=float(mod.ls.exp().detach()), off=det(mod.off),
                c=det(mod.lc.exp()), e=det(mod.le.exp()), J=det(mod.j), rho=0.0)


def fit_integ_qn(zs, K, coupled, init, maxiter=200, em_warm=5):
    """Direct maximisation of the marginal likelihood: value from forward-backward, exact gradient
    via Fisher's identity (gradient of the expected complete-data log-likelihood at the current
    parameters), L-BFGS-B.  A few EM iterations first for a good start."""
    from scipy.optimize import minimize
    sc = float(init["sig"])                         # work in noise units (conditioning)
    zs = [z / sc for z in zs]
    init = dict(init, coef=np.asarray(init["coef"]) / sc, sig=1.0,
                off=np.asarray(init.get("off", np.zeros(len(zs)))) / sc)
    if em_warm:
        w = fit_integ(zs, K, coupled, init, n_iter=em_warm)
        init = dict(init, coef=w["coef"], c=w["c"], e=w["e"], sig=w["sig"], off=w["off"], J=w["J"])
    mod = IntegModel(K, len(zs), init, coupled)
    params = mod.params()
    sizes = [p.numel() for p in params]
    S = 2 ** K
    pis = [np.full(S, 1.0 / S) for _ in zs]

    def setp(x):
        with torch.no_grad():
            i = 0
            for p, n in zip(params, sizes):
                p.copy_(torch.tensor(x[i:i + n]).reshape(p.shape)); i += n

    def fg(x):
        setp(x)
        P, m, v, off = mod.numpy()
        ll, st, Nt = 0.0, [], 0.0
        for r, z in enumerate(zs):
            l, n, sy, syy, first = _fb_pair(z - off[r], P, pis[r], m, v)
            ll += l; st.append((torch.tensor(n), torch.tensor(sy), torch.tensor(syy), torch.tensor(first), float(z[0] - off[r]))); Nt = Nt + n
        for p in params:
            p.grad = None
        P_, m_, v_ = mod.emission()
        q = (torch.tensor(Nt) * torch.log(P_)).sum()
        for r, (n, sy, syy, fst, y0) in enumerate(st):
            mm = m_ + (mod.off[r] - float(off[r]))
            q = q - 0.5 * (n * torch.log(2 * np.pi * v_) + (syy - 2 * mm * sy + mm * mm * n) / v_).sum()
            md, vd = torch.diagonal(mm), torch.diagonal(v_)
            q = q - 0.5 * (fst * (torch.log(2 * np.pi * vd) + (y0 - md) ** 2 / vd)).sum()
        q.backward()
        g = np.concatenate([p.grad.numpy().ravel() for p in params])
        if not np.isfinite(ll):
            return 1e30, np.zeros_like(g)
        return -ll, -g

    x0 = np.concatenate([p.detach().numpy().ravel() for p in params])
    names = ["lc"] * K + ["le"] * K + ["coef"] * (K + 1) + ["ls"] + ["off"] * len(zs) + (["j"] * (K * (K - 1) // 2) if coupled else [])
    bnd = [(-12, 3) if n in ("lc", "le") else (-8, 8) if n == "j" else (None, None) for n in names]
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=maxiter, gtol=1e-7, ftol=1e-14, maxcor=30))
    setp(r.x)
    det = lambda x: x.detach().numpy().copy()
    Ttot = sum(z.size for z in zs)
    return dict(ll=float(-r.fun) - Ttot * np.log(sc), coef=det(mod.coef) * sc, sig=float(mod.ls.exp().detach()) * sc,
                off=det(mod.off) * sc,
                c=det(mod.lc.exp()), e=det(mod.le.exp()), J=det(mod.j), rho=0.0, nit=r.nit, msg=str(r.message))
