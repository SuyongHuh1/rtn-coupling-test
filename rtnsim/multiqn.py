"""Quasi-Newton maximum likelihood for the multi-segment AR(1) cfHMM (optionally with a shared slow
drive).  Value from the numba forward-backward (_fb_ar1), exact gradient by Fisher's identity
(gradient of the expected complete-data log-likelihood at the current parameters), L-BFGS-B in
noise units.  Replaces the multi-segment EM, which stalls / lands in poor local optima on records
of ~10^5 samples.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from .cfhmm import Kinetics, KineticsDrive, _fb_ar1, _kappa_starts
from .robustk import bits


@_kappa_starts
def fit_multi_qn(zs, K, coupled, init, drive=False, maxiter=400):
    sc = float(init["sig"])
    zs = [np.asarray(z, float) / sc for z in zs]
    n_seg = len(zs)
    t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64, requires_grad=True)
    if drive:
        kin = KineticsDrive(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                            gamma=init.get("gamma"), drive=init.get("drive", (0.002, 0.002)), max_rate=init.get("max_rate"),
                            kappa=init.get("kappa"))
    else:
        kin = Kinetics(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                       kappa=init.get("kappa"))
    coef = t(np.asarray(init["coef"]) / sc)
    ls = t(0.0)
    ar = t(np.arctanh(np.clip(init.get("rho", 0.0), -0.95, 0.95)))
    off = t(np.asarray(init.get("off", np.zeros(n_seg))) / sc)
    kp = kin.params()
    params = kp + [coef, ls, ar, off]
    sizes = [p.numel() for p in params]
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    if drive:
        X = np.vstack([X, X])
    Xt = torch.tensor(X)
    S = X.shape[0]
    pis = [np.full(S, 1.0 / S) for _ in zs]

    def setp(x):
        with torch.no_grad():
            i = 0
            for p, n in zip(params, sizes):
                p.copy_(torch.tensor(x[i:i + n]).reshape(p.shape)); i += n

    def fg(x):
        setp(x)
        with torch.no_grad():
            P = torch.linalg.matrix_exp(kin.Q()).clamp_min(1e-300).numpy()
            P = P / P.sum(1, keepdims=True)
            mu = (Xt @ coef).numpy()
            sig, rho, o = float(ls.exp()), float(torch.tanh(ar)), off.numpy().copy()
        ll, stats, Nt = 0.0, [], 0.0
        for r, z in enumerate(zs):
            try:
                l, N, st, first, _ = _fb_ar1(z - o[r], P, pis[r], mu, sig, rho, False)
            except ZeroDivisionError:
                return 1e30, np.zeros(sum(sizes))
            ll += l; Nt = Nt + N
            stats.append((torch.tensor(st), torch.tensor(first), float(z[0] - o[r]), o[r]))
        for p in params:
            p.grad = None
        P_ = torch.linalg.matrix_exp(kin.Q()).clamp_min(1e-300)
        q = (torch.tensor(Nt) * torch.log(P_)).sum()
        s2, rh = (ls.exp()) ** 2, torch.tanh(ar)
        mu_ = Xt @ coef
        for r, (st, fst, y0, o_r) in enumerate(stats):
            m = mu_ + (off[r] - o_r)
            n, Sy, Sw, Syy, Sww, Syw = st
            d = m[None, :] - rh * m[:, None]
            rss = Syy - 2 * rh * Syw + rh ** 2 * Sww - 2 * d * (Sy - rh * Sw) + d ** 2 * n
            q = q - 0.5 * (n.sum() * torch.log(2 * np.pi * s2) + rss.sum() / s2)
            v0 = s2 / (1 - rh ** 2)
            q = q - 0.5 * (fst * (torch.log(2 * np.pi * v0) + (y0 - m) ** 2 / v0)).sum()
        q.backward()
        g = np.concatenate([p.grad.numpy().ravel() if p.grad is not None else np.zeros(p.numel()) for p in params])
        if not np.isfinite(ll):
            return 1e30, np.zeros_like(g)
        return -ll, -g

    x0 = np.concatenate([p.detach().numpy().ravel() for p in params])
    bnd = []
    for p, n in zip(params, sizes):
        if p is kin.lc or p is kin.le:
            bnd += [(-12, 3)] * n
        elif coupled and p is kin.j:
            bnd += [(-8, 8)] * n
        elif p is kin.lk:
            bnd += [(-16, 1)] * n
        elif drive and p is kin.g:
            bnd += [(-8, 8)] * n
        elif drive and p is kin.ld:
            bnd += [(-12, kin.max_ld)] * n
        elif p is ar:
            bnd += [(-2.5, 2.5)]
        elif p is ls:
            bnd += [(-5, 3)]
        else:
            bnd += [(None, None)] * n
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=maxiter, gtol=1e-7, ftol=1e-14, maxcor=30))
    setp(r.x)
    Ttot = sum(z.size for z in zs)
    det = lambda x: x.detach().numpy().copy()
    out = dict(ll=float(-r.fun) - Ttot * np.log(sc), coef=det(coef) * sc, sig=float(ls.exp().detach()) * sc,
               rho=float(torch.tanh(ar).detach()), off=det(off) * sc, c=det(kin.lc.exp()), e=det(kin.le.exp()),
               J=det(kin.j), kappa=kin.kappa(), nit=r.nit, msg=str(r.message), fg=fg, x=r.x)
    if drive:
        out.update(gamma=det(kin.g), drive=det(kin.ld.exp()))
    return out
