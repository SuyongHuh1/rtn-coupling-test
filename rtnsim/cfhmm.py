"""R8: coupled factorial HMM with exact likelihood (minimal version).

Model (time in samples): K two-state traps, joint state x = sum_k s_k 2^k (S = 2^K).
Kinetics: continuous-time generator Q(theta) with capture/emission rates c_k, e_k and pairwise
couplings J_ij that tilt trap rates by the other traps' states,
    c'_k = c_k exp(-phi h_k),  e'_k = e_k exp((1 - phi) h_k),  h_k = sum_j J_kj s_j,
and P = expm(Q) per sample.  Emission: y_t = beta_t + b + sum_k a_k s_k + N(0, sigma^2), with
beta a slow baseline (taken from the idealiser and held fixed, identical under both hypotheses).

Fit: ECM.  E-step = exact forward-backward (numba, shared with robustk); M-step for the
emission is closed form, M-step for the kinetics maximises sum_ij N_ij log P_ij(theta) through
the matrix exponential (torch, L-BFGS).  The J = 0 model is the same code with J frozen.

Test: LRT = 2 (ll_J - ll_0), soft (no state path is fixed), calibrated by a pipeline null that
simulates the fitted J = 0 model with the record's own baseline and block-bootstrapped residual.
"""
from __future__ import annotations

import numpy as np
import torch

from .robustk import _fb, bits, idealise_k, _block_boot
from .sim2_torch import generator

torch.set_num_threads(1)

import os as _os
# Joint (simultaneous) switching.  RTN_JOINT selects the alternative (coupled) model; the null (J = 0,
# kappa = 0) is never changed:
#   0 (default)  J only                      (occupancy-dependent rates)
#   1            J and kappa                 (one omnibus test)
#   2            kappa only, J fixed at 0    (co-flip test, used next to the J-only test)
# kappa_ij is the rate of 00 <-> 11 co-flips of traps i, j with the other traps unchanged.
JOINT_MODE = int(_os.environ.get("RTN_JOINT", "0") or 0)
JOINT = JOINT_MODE in (1, 2)


def _kappa_starts(fn):
    """kappa near 0 has a vanishing log-scale gradient: fit the coupled model from a small and a moderate
    co-flip rate and keep the better likelihood."""
    def wrapped(data, K, coupled, init, *a, **kw):
        if coupled and JOINT and "kappa" not in init:
            rbar = float(np.mean(np.r_[init["c"], init["e"]]))
            n = K * (K - 1) // 2
            runs = []
            for k0 in (1e-4 * rbar, 0.3 * rbar):
                try:
                    r = fn(data, K, True, dict(init, kappa=np.full(n, k0)), *a, **kw)
                    if np.isfinite(r["ll"]):
                        runs.append(r)
                except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
                    pass
            if not runs:
                raise ValueError("joint fit failed")
            return max(runs, key=lambda r: r["ll"])
        return fn(data, K, coupled, init, *a, **kw)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _rates_from_P(Pk):
    """2x2 discrete transition matrix (dt = 1) -> (c, e) of the continuous chain."""
    p01, p10 = float(Pk[0, 1]), float(Pk[1, 0])
    lam = min(max(p01 + p10, 1e-6), 0.999)
    k = -np.log(1 - lam)
    return max(p01 / lam * k, 1e-5), max(p10 / lam * k, 1e-5)


class Kinetics:
    def __init__(self, K, c, e, J=None, phi=0.5, coupled=True, kappa=None, joint=None):
        self.K, self.coupled, self.phi = K, coupled, phi
        self.lc = torch.tensor(np.log(c), dtype=torch.float64, requires_grad=True)
        self.le = torch.tensor(np.log(e), dtype=torch.float64, requires_grad=True)
        n = K * (K - 1) // 2
        self.iu = torch.triu_indices(K, K, 1)
        mode = JOINT_MODE if joint is None else int(joint)
        self.jfree = bool(coupled and mode != 2)
        self.j = torch.tensor(np.zeros(n) if (J is None or mode == 2) else J, dtype=torch.float64,
                              requires_grad=self.jfree)
        self.joint = bool(coupled and mode in (1, 2))
        k0 = np.full(n, 1e-4) if kappa is None else np.maximum(np.asarray(kappa, float), 1e-7)
        self.lk = torch.tensor(np.log(k0), dtype=torch.float64, requires_grad=self.joint)
        if self.joint:                      # co-flip index pairs (src, dst) per trap pair
            S = 2 ** K
            x = np.arange(S)
            self._cf = []
            for i, j in zip(self.iu[0].tolist(), self.iu[1].tolist()):
                same = ((x >> i) & 1) == ((x >> j) & 1)
                src = x[same]
                self._cf.append((torch.tensor(src), torch.tensor(src ^ ((1 << i) | (1 << j)))))

    def params(self):
        return [self.lc, self.le] + ([self.j] if self.jfree else []) + ([self.lk] if self.joint else [])

    def Q(self):
        K = self.K
        Jm = torch.zeros(K, K, dtype=torch.float64)
        Jm = Jm.index_put((self.iu[0], self.iu[1]), self.j).index_put((self.iu[1], self.iu[0]), self.j)
        Q = generator(self.lc.exp()[None].double(), self.le.exp()[None].double(), Jm[None],
                      torch.tensor([self.phi], dtype=torch.float64))[0]
        return self._coflip(Q, 1)

    def _coflip(self, Q, layers):
        """Add the co-flip rates in each of `layers` copies of the trap state space (drive layers)."""
        if not self.joint:
            return Q
        S0 = 2 ** self.K
        off = Q - torch.diag(torch.diagonal(Q))
        kap = self.lk.exp()
        for u in range(layers):
            for p, (src, dst) in enumerate(self._cf):
                add = torch.zeros_like(off)
                add[src + u * S0, dst + u * S0] = kap[p]
                off = off + add
        return off - torch.diag(off.sum(1))

    def kappa(self):
        return self.lk.exp().detach().numpy().copy() if self.joint else np.zeros(len(self.j))

    def P(self):
        return torch.linalg.matrix_exp(self.Q())

    def P_np(self):
        with torch.no_grad():
            P = self.P().clamp_min(1e-300).numpy()
        return P / P.sum(1, keepdims=True)

    def m_step(self, N, iters=15):
        Nt = torch.tensor(N, dtype=torch.float64)
        opt = torch.optim.LBFGS(self.params(), max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            f = -(Nt * torch.log(self.P().clamp_min(1e-300))).sum() / Nt.sum()
            if self.coupled:
                f = f + 1e-4 * (self.j ** 2).sum() if self.jfree else f   # tiny ridge: keeps J finite
            f.backward()
            return f

        opt.step(closure)
        with torch.no_grad():
            self.lc.clamp_(-12, 3); self.le.clamp_(-12, 3)
            if self.jfree:
                self.j.clamp_(-8, 8)
            if self.joint:
                self.lk.clamp_(-16, 1)


@_kappa_starts
def fit(y, K, coupled, init, n_iter=60, tol=1e-7):
    """y: record minus baseline. init: dict(coef, sig, c, e, J?). Returns dict."""
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    coef, sig = init["coef"].copy(), float(init["sig"])
    kin = Kinetics(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                   kappa=init.get("kappa"))
    T = y.size
    pi0 = np.full(2 ** K, 1.0 / 2 ** K)
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, g0, gy, gyy, first, _ = _fb(y, P, pi0, X @ coef, sig, False)
        if ll - prev < tol * T and it > 2:
            break
        prev = ll
        kin.m_step(N)
        G = (X.T * g0) @ X
        coef = np.linalg.solve(G + 1e-9 * np.eye(K + 1), X.T @ gy)
        mu = X @ coef
        sig = float(np.sqrt(max((gyy - 2 * mu * gy + mu * mu * g0).sum() / T, 1e-12)))
        pi0 = first
    P = kin.P_np()
    ll, *_, path = _fb(y, P, pi0, X @ coef, sig, True)
    return dict(ll=float(ll), coef=coef, sig=sig, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), kappa=kin.kappa(), P=P, path=path)


def init_from_idealiser(f):
    """robustk.idealise_k fit -> R8 initial values (traps already ordered by step size)."""
    K = len(f["a"])
    ce = np.array([_rates_from_P(P) for P in f["Ps"]])
    return dict(coef=np.r_[f["b"], f["a"]], sig=f["sig"], c=ce[:, 0], e=ce[:, 1])


def lrt(y, K, rng=None, f=None, min_win=101):
    """Returns (stat, fit0, fit1, idealiser fit).  Baseline from the idealiser, shared by both."""
    rng = rng or np.random.default_rng(0)
    f = f or idealise_k(y, K, min_win=min_win, rng=rng)
    yb = y - f["beta"]
    ini = init_from_idealiser(f)
    f0 = fit(yb, K, False, ini)
    ini1 = dict(ini, coef=f0["coef"], sig=f0["sig"], c=f0["c"], e=f0["e"])
    f1 = fit(yb, K, True, ini1)
    best = f1
    if f1["ll"] < f0["ll"]:                   # nested: J = 0 is inside the alternative
        best = f0
    return 2 * max(best["ll"] - f0["ll"], 0.0), f0, f1, f


def simulate_null(f0, beta, resid, T, rng):
    """One record from the fitted J = 0 model with the record's baseline and residual noise."""
    K = len(f0["c"])
    s = np.empty((T, K), np.int64)
    for k in range(K):
        kt = f0["c"][k] + f0["e"][k]
        q = 1 - np.exp(-kt)
        p01, p10 = f0["c"][k] / kt * q, f0["e"][k] / kt * q
        st = int(rng.random() < f0["c"][k] / kt)
        u = rng.random(T)
        for t in range(T):
            if t:
                st = st ^ int(u[t] < (p01 if st == 0 else p10))
            s[t, k] = st
    return beta + f0["coef"][0] + s @ f0["coef"][1:] + _block_boot(resid, T, rng)


def pipeline_null(y, K, rng, n_sim=40, min_win=101):
    obs, f0, f1, f = lrt(y, K, rng, min_win=min_win)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    resid = y - f["beta"] - (X @ f0["coef"])[f0["path"]]
    null = np.array([lrt(simulate_null(f0, f["beta"], resid, len(y), rng), K, rng, min_win=min_win)[0]
                     for _ in range(n_sim)])
    return obs, null, f0, f1


# ---------------------------------------------------------------------------------------------
# AR(1) residual noise: y_t - beta_t - mu(s_t) = r_t,  r_t = rho r_{t-1} + eps_t.
# The emission then depends on (s_{t-1}, s_t) only, so the forward recursion stays exact on S states.
# ---------------------------------------------------------------------------------------------
import numba as nb  # noqa: E402


@nb.njit(cache=True)
def _fb_ar1(y, P, pi0, mu, sig, rho, want_path):
    T = y.size
    S = mu.size
    alpha = np.empty((T, S))
    c = np.empty(T)
    inv = 1.0 / sig
    s0 = sig / np.sqrt(1 - rho * rho)
    tot = 0.0
    for s in range(S):
        z = (y[0] - mu[s]) / s0
        alpha[0, s] = pi0[s] * np.exp(-0.5 * z * z) * sig / s0
        tot += alpha[0, s]
    c[0] = tot
    alpha[0] /= tot
    for k in range(1, T):
        tot = 0.0
        for t in range(S):
            acc = 0.0
            for s in range(S):
                z = (y[k] - mu[t] - rho * (y[k - 1] - mu[s])) * inv
                acc += alpha[k - 1, s] * P[s, t] * np.exp(-0.5 * z * z)
            alpha[k, t] = acc
            tot += acc
        c[k] = tot
        for t in range(S):
            alpha[k, t] /= tot
    ll = np.sum(np.log(c)) - T * np.log(sig) - 0.5 * T * np.log(2 * np.pi)
    N = np.zeros((S, S))
    st = np.zeros((6, S, S))                     # n, Sy, Sw, Syy, Sww, Syw per (prev, cur)
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
                v = P[s, t] * np.exp(-0.5 * z * z) * beta[t] / c[k]
                xi = alpha[k - 1, s] * v
                N[s, t] += xi
                st[0, s, t] += xi
                st[1, s, t] += xi * y[k]
                st[2, s, t] += xi * y[k - 1]
                st[3, s, t] += xi * y[k] * y[k]
                st[4, s, t] += xi * y[k - 1] * y[k - 1]
                st[5, s, t] += xi * y[k] * y[k - 1]
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


def _emission_mstep_ar1(st, X, coef, rho, T, n_inner=3):
    S = X.shape[0]
    n, Sy, Sw, Syy, Sww, Syw = st
    for _ in range(n_inner):
        # coef | rho : weighted LS on D_ij = X_j - rho X_i
        A = np.zeros((X.shape[1], X.shape[1]))
        bvec = np.zeros(X.shape[1])
        for i in range(S):
            D = X - rho * X[i]                       # rows j
            A += (D.T * n[i]) @ D
            bvec += D.T @ (Sy[i] - rho * Sw[i])
        coef = np.linalg.solve(A + 1e-9 * np.eye(len(bvec)), bvec)
        mu = X @ coef
        # rho | coef
        mi, mj = mu[:, None], mu[None, :]
        num = (Syw - mi * Sy - mj * Sw + n * mi * mj).sum()
        den = (Sww - 2 * mi * Sw + n * mi * mi).sum()
        rho = float(np.clip(num / max(den, 1e-12), -0.98, 0.98))
    mu = X @ coef
    mi, mj = mu[:, None], mu[None, :]
    a = Sy - mj * n                                  # sum xi (y - mu_j)
    b = Sw - mi * n                                  # sum xi (w - mu_i)
    q = (Syy - 2 * mj * Sy + mj * mj * n) - 2 * rho * (Syw - mi * Sy - mj * Sw + n * mi * mj) \
        + rho * rho * (Sww - 2 * mi * Sw + n * mi * mi)
    sig = float(np.sqrt(max(q.sum() / (T - 1), 1e-12)))
    return coef, rho, sig


@_kappa_starts
def fit_ar1(y, K, coupled, init, n_iter=80, tol=1e-7):
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    coef, sig, rho = init["coef"].copy(), float(init["sig"]), float(init.get("rho", 0.0))
    kin = Kinetics(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                   kappa=init.get("kappa"))
    T = y.size
    pi0 = np.full(2 ** K, 1.0 / 2 ** K)
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, st, first, _ = _fb_ar1(y, P, pi0, X @ coef, sig, rho, False)
        if ll - prev < tol * T and it > 2:
            break
        prev = ll
        kin.m_step(N)
        coef, rho, sig = _emission_mstep_ar1(st, X, coef, rho, T)
        pi0 = first
    P = kin.P_np()
    ll, *_, path = _fb_ar1(y, P, pi0, X @ coef, sig, rho, True)
    n_par = 2 * K + (K + 1) + 2 + (K * (K - 1) // 2 if coupled else 0) * (2 if kin.joint else 1)
    return dict(ll=float(ll), coef=coef, sig=sig, rho=rho, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), kappa=kin.kappa(), P=P, path=path,
                bic=-2 * ll + n_par * np.log(T))


def lrt_ar1(y, K, rng=None, f=None, min_win=101):
    """AR(1)-emission version of lrt(); the J = 0 fit's BIC is returned for K selection."""
    rng = rng or np.random.default_rng(0)
    f = f or idealise_k(y, K, min_win=min_win, rng=rng)
    yb = y - f["beta"]
    ini = init_from_idealiser(f)
    f0 = fit_ar1(yb, K, False, ini)
    f1 = fit_ar1(yb, K, True, dict(ini, coef=f0["coef"], sig=f0["sig"], rho=f0["rho"], c=f0["c"], e=f0["e"]))
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1, f


# ---------------------------------------------------------------------------------------------
# Global search over level decompositions.  Spurious coupling in multi-trap records comes mainly
# from a wrong additive decomposition of the levels (a local optimum of the emission), which a
# coupled model then partly patches.  Candidates: several idealiser runs (seeds x baseline on/off,
# optionally an external initialiser such as the R7 network); the J = 0 model picks the best.
# ---------------------------------------------------------------------------------------------
def best_decomposition(y, K, rng, seeds=3, starts=8, min_win=101, extra_inits=(), ar1=False, free_levels=True):
    fitter = fit_ar1 if ar1 else fit
    best = None
    for base in (True, False):
        for s in range(seeds):
            f = idealise_k(y, K, min_win=min_win, rng=np.random.default_rng(int(rng.integers(1 << 30))),
                           starts=starts, baseline=base)
            f0 = fitter(y - f["beta"], K, False, init_from_idealiser(f))
            if best is None or f0["ll"] > best[1]["ll"]:
                best = (f, f0)
    if free_levels:                                  # structurally different initialiser
        for base_f in (best[0], None):
            beta = base_f["beta"] if base_f is not None else np.zeros_like(y)
            for ini in free_level_inits(y - beta, K, rng):
                extra_inits = list(extra_inits) + [dict(ini, beta=beta)]
    for ini in extra_inits:                          # dict(coef, sig, c, e, beta?)
        beta = ini.get("beta", np.zeros_like(y))
        f0 = fitter(y - beta, K, False, ini)
        if f0["ll"] > best[1]["ll"]:
            best = (dict(beta=beta), f0)
    return best


N_RESTART = int(_os.environ.get("RTN_RESTART", "3") or 1)


def _ini_from(f, coupled):
    ini = dict(coef=np.asarray(f["coef"]).copy(), sig=f["sig"], rho=f.get("rho", 0.0), c=f["c"], e=f["e"])
    if coupled:
        ini["J"] = np.asarray(f["J"]).copy()
        if JOINT and f.get("kappa") is not None and np.size(f["kappa"]):
            ini["kappa"] = np.asarray(f["kappa"]).copy()
    return ini


def pool_runs(z, K, runs, ar1):
    """Combine independent restarts of the LRT into one likelihood-ratio statistic.  The restart with the
    best J = 0 likelihood fixes the decomposition and baseline (z = y - beta of that run); then both models
    are re-fitted on that z from every other restart's solutions and each keeps its best likelihood, so the
    statistic is 2 (max ll1 - max ll0) on one record rather than whichever restart happened to win.
    runs: list of (f0, f1); z: record minus the chosen run's baseline.  Returns (f0, f1, stat)."""
    fitter = fit_ar1 if ar1 else fit
    i0 = int(np.argmax([r[0]["ll"] for r in runs]))
    f0, f1 = runs[i0]
    for j, (g0, g1) in enumerate(runs):
        if j == i0:
            continue
        for coupled, g in ((False, g0), (True, g1), (True, g0)):
            try:
                ini = _ini_from(g, coupled) if not (coupled and g is g0) else dict(_ini_from(g, False), J=np.full(K * (K - 1) // 2, 0.01))
                h = fitter(z, K, coupled, ini)
            except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
                continue
            if not np.isfinite(h["ll"]):
                continue
            if coupled and h["ll"] > f1["ll"]:
                f1 = h
            if not coupled and h["ll"] > f0["ll"]:
                f0 = h
    try:                                              # J = 0 from the best coupled optimum (nested models)
        h = fitter(z, K, False, _ini_from(f1, False))
        if np.isfinite(h["ll"]) and h["ll"] > f0["ll"]:
            f0 = h
    except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        pass
    return f0, f1, 2 * max(f1["ll"] - f0["ll"], 0.0)


def lrt_ms(y, K, rng, ar1=False, **kw):
    """Multistart LRT, N_RESTART independent restarts pooled by pool_runs().
    Returns stat, f0, f1, beta, path statistic on the J = 0 decoded path."""
    from .robustk import coupling_stat
    if N_RESTART <= 1:
        return _lrt_ms_once(y, K, rng, ar1=ar1, **kw)
    runs, err = [], None
    for _ in range(N_RESTART):
        try:
            runs.append(_lrt_ms_once(y, K, np.random.default_rng(int(rng.integers(1 << 31))), ar1=ar1, **dict(kw)))
        except Exception as ex:                       # noqa: BLE001 -- a failed restart is just dropped
            err = ex
    if not runs:
        raise err
    best = max(runs, key=lambda r: r[1]["ll"])
    beta = best[3]
    f0, f1, stat = pool_runs(y - beta, K, [(r[1], r[2]) for r in runs], ar1)
    return stat, f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def _lrt_ms_once(y, K, rng, ar1=False, **kw):
    """One multistart LRT run (the pre-R35 lrt_ms)."""
    from .robustk import coupling_stat
    fitter = fit_ar1 if ar1 else fit
    kalman = kw.pop("kalman", False)
    f, f0 = best_decomposition(y, K, rng, ar1=ar1, **kw)
    if kalman:                                       # learned baseline, started from the best decomposition
        beta_k, f0k = refine_baseline(y - f["beta"], K, f0, ar1=ar1)
        f = dict(f, beta=f["beta"] + beta_k)
        f0 = f0k
    yb = y - f["beta"]
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    f1 = fitter(yb, K, True, ini)
    f0b = fitter(yb, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1.get("rho", 0.0), c=f1["c"], e=f1["e"]))
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    stat = 2 * max(f1["ll"] - f0["ll"], 0.0)
    path_stat = coupling_stat(bits(K)[f0["path"]])[0]
    return stat, f0, f1, f["beta"], path_stat


def _free_hmm(y, M, mu, n_iter=60):
    """Plain HMM with M free Gaussian levels (shared sigma), full transition matrix."""
    P = np.full((M, M), 0.02 / (M - 1)) + np.eye(M) * (0.98 - 0.02 / (M - 1))
    sd = float(np.std(y)) + 1e-12
    sig = sd / M
    pi0 = np.full(M, 1.0 / M)
    T = y.size
    prev = -np.inf
    for _ in range(n_iter):
        ll, N, g0, gy, gyy, first, _ = _fb(y, P, pi0, mu, sig, False)
        P = (N + 1e-6) / (N + 1e-6).sum(1, keepdims=True)
        mu = gy / np.maximum(g0, 1e-9)
        sig = float(np.sqrt(max((gyy - 2 * mu * gy + mu * mu * g0).sum() / T, (1e-3 * sd) ** 2)))
        pi0 = first
        if ll - prev < 1e-7 * T:
            break
        prev = ll
    return mu, g0 / T, P, sig, ll


def _additive_fit(levels, w, K):
    """Best (b, a) and assignment of 2^K bit patterns to the given levels (weighted LS, exhaustive)."""
    from itertools import permutations
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    order = np.argsort(levels)
    lv, ww = levels[order], w[order] + 1e-3
    best = (np.inf, None)
    # bit pattern 0 (all empty) -> lowest level and full pattern -> highest (fixes the sign symmetry)
    mid = list(range(1, 2 ** K - 1))
    for perm in permutations(mid):
        idx = np.array([0, *perm, 2 ** K - 1])         # pattern x -> level idx[x]
        A = X * np.sqrt(ww[idx])[:, None]
        bb = lv[idx] * np.sqrt(ww[idx])
        coef, *_ = np.linalg.lstsq(A, bb, rcond=None)
        err = np.sum((A @ coef - bb) ** 2)
        if err < best[0]:
            best = (err, coef)
    return best[1]


def free_level_inits(y, K, rng, n_starts=3):
    """Initialisers for best_decomposition() from free-level HMM fits."""
    M = 2 ** K
    out = []
    lo, hi = np.quantile(y, [0.01, 0.99])
    for s in range(n_starts):
        mu0 = np.quantile(y, (np.arange(M) + 0.5) / M) if s == 0 else np.sort(lo + (hi - lo) * rng.random(M))
        try:
            mu, w, P, sig, ll = _free_hmm(y, M, mu0)
        except (ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(w)) and np.isfinite(sig)):
            continue                                   # degenerate free-level fit: skip this start
        try:
            coef = _additive_fit(mu, w, K)
        except np.linalg.LinAlgError:
            continue
        a = coef[1:]
        flip = a < 0                                   # make steps positive (flip that trap's states)
        coef[0] += a[flip].sum()
        a = np.abs(a)
        order = np.argsort(-a)
        out.append(dict(coef=np.r_[coef[0], a[order]], sig=sig, c=np.full(K, 0.02), e=np.full(K, 0.02)))
    return out


# ---------------------------------------------------------------------------------------------
# Learned baseline: local-level (random walk + white) Kalman/RTS smoother on the trap-free
# residual, with the drift-to-noise ratio chosen by the innovation likelihood.  Replaces the
# dwell-dependent median filter, which either follows slow traps or misses drift.
# ---------------------------------------------------------------------------------------------
@nb.njit(cache=True)
def _rts(r, q, R):
    T = r.size
    m = np.empty(T); P = np.empty(T); mp = np.empty(T); Pp = np.empty(T)
    x, v = r[0], R
    ll = 0.0
    for t in range(T):
        if t:
            x, v = x, v + q
        mp[t], Pp[t] = x, v
        S = v + R
        innov = r[t] - x
        ll += -0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
        k = v / S
        x = x + k * innov
        v = (1 - k) * v
        m[t], P[t] = x, v
    s = m.copy()
    for t in range(T - 2, -1, -1):
        g = P[t] / Pp[t + 1]
        s[t] = m[t] + g * (s[t + 1] - mp[t + 1])
    return s, ll


def kalman_baseline(r, ratios=np.geomspace(1e-7, 1e-1, 25)):
    """r: residual (record minus trap levels). Returns smoothed baseline and the chosen q/R."""
    R = (np.median(np.abs(np.diff(r))) / 0.6745) ** 2 / 2 + 1e-12
    best = None
    for k in ratios:
        s, ll = _rts(r, k * R, R)
        if best is None or ll > best[1]:
            best = (s, ll, k)
    return best[0], best[2]


def refine_baseline(y, K, f0, n_outer=3, ar1=False):
    """Alternate: baseline <- Kalman smoother of (y - levels on the decoded path); refit J = 0."""
    fitter = fit_ar1 if ar1 else fit
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    beta = np.zeros_like(y)
    for _ in range(n_outer):
        lev = (X @ f0["coef"])[f0["path"]]
        beta, ratio = kalman_baseline(y - lev)
        beta = beta - np.median(beta)
        f0 = fitter(y - beta, K, False, dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0),
                                              c=f0["c"], e=f0["e"]))
    return beta, f0


# ---------------------------------------------------------------------------------------------
# Shared slow drive (R2 confound) as part of the model: a hidden two-state telegraph u that
# shifts every trap's energy by gamma_k when u = 1 (rates tilted like a field).  State index =
# trap bits + 2^K u; the emission ignores u.  The coupling test is then J free vs J = 0, BOTH
# with the drive, so slow common modulation is in the null.
# ---------------------------------------------------------------------------------------------
class KineticsDrive(Kinetics):
    def __init__(self, K, c, e, J=None, phi=0.5, coupled=True, gamma=None, drive=(0.002, 0.002), max_rate=None,
                 kappa=None, joint=None):
        super().__init__(K, c, e, J, phi, coupled, kappa=kappa, joint=joint)
        self.max_ld = -1.0 if max_rate is None else float(np.log(max_rate))   # time-scale separation
        self.g = torch.tensor(np.zeros(K) if gamma is None else gamma, dtype=torch.float64, requires_grad=True)
        self.ld = torch.tensor(np.log(drive), dtype=torch.float64, requires_grad=True)

    def params(self):
        return super().params() + [self.g, self.ld]

    def Q(self):
        K = self.K
        Jm = torch.zeros(K, K, dtype=torch.float64)
        Jm = Jm.index_put((self.iu[0], self.iu[1]), self.j).index_put((self.iu[1], self.iu[0]), self.j)
        Q = generator(self.lc.exp()[None], self.le.exp()[None], Jm[None],
                      torch.tensor([self.phi], dtype=torch.float64), self.g[None], self.ld.exp()[None])[0]
        return self._coflip(Q, 2)

    def m_step(self, N, iters=15):
        super().m_step(N, iters)
        with torch.no_grad():
            self.g.clamp_(-8, 8)
            self.ld.clamp_(-12, self.max_ld)


@_kappa_starts
def fit_ar1_drive(y, K, coupled, init, n_iter=80, tol=1e-7):
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    Xd = np.vstack([X, X])
    coef, sig, rho = init["coef"].copy(), float(init["sig"]), float(init.get("rho", 0.0))
    kin = KineticsDrive(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                        gamma=init.get("gamma"), drive=init.get("drive", (0.002, 0.002)), max_rate=init.get("max_rate"),
                        kappa=init.get("kappa"))
    T = y.size
    S = 2 ** (K + 1)
    pi0 = np.full(S, 1.0 / S)
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, st, first, _ = _fb_ar1(y, P, pi0, Xd @ coef, sig, rho, False)
        if ll - prev < tol * T and it > 2:
            break
        prev = ll
        kin.m_step(N)
        coef, rho, sig = _emission_mstep_ar1(st, Xd, coef, rho, T)
        pi0 = first
    P = kin.P_np()
    ll, *_, path = _fb_ar1(y, P, pi0, Xd @ coef, sig, rho, True)
    return dict(ll=float(ll), coef=coef, sig=sig, rho=rho, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), kappa=kin.kappa(), P=P, path=path % (2 ** K),
                path_full=path, gamma=kin.g.detach().numpy().copy(), drive=kin.ld.exp().detach().numpy().copy(),
                pi0=pi0, Xd=Xd)


def lrt_drive(y, K, rng, sep=10.0, **kw):
    """Coupling LRT with a shared slow drive in both hypotheses.  The decomposition comes from the
    global search (as lrt_ms); the drive time scale is started at 10, 30 and 100 x the slowest trap."""
    from .robustk import coupling_stat
    f, f0 = best_decomposition(y, K, rng, ar1=True, **kw)
    beta_k, f0 = refine_baseline(y - f["beta"], K, f0, ar1=True)
    beta = f["beta"] + beta_k
    z = y - beta
    slow = 1.0 / np.min(f0["c"] + f0["e"])
    best0 = None
    for m in (10, 30, 100):
        r = 1.0 / (m * slow)
        ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0["rho"], c=f0["c"], e=f0["e"],
                   gamma=rng.normal(0, 0.5, K), drive=(r, r), max_rate=1.0 / (sep * slow))
        g0 = fit_ar1_drive(z, K, False, ini)
        if best0 is None or g0["ll"] > best0["ll"]:
            best0 = g0
    ini1 = dict(coef=best0["coef"], sig=best0["sig"], rho=best0["rho"], c=best0["c"], e=best0["e"],
                gamma=best0["gamma"], drive=best0["drive"], max_rate=1.0 / (sep * slow))
    g1 = fit_ar1_drive(z, K, True, ini1)
    stat = 2 * max(g1["ll"] - best0["ll"], 0.0)
    return stat, best0, g1, beta, coupling_stat(bits(K)[best0["path"]])[0]


def simulate_from_chain(fm, beta, resid, T, rng):
    """Sample the fitted joint chain (any state space, e.g. with drive) + baseline + residual."""
    P = fm["P"]
    C = P.cumsum(1)
    C[:, -1] = 1.0 + 1e-9
    x = np.empty(T, np.int64)
    s = int(rng.choice(len(P), p=fm["pi0"] / fm["pi0"].sum()))
    u = rng.random(T)
    for t in range(T):
        s = int(np.searchsorted(C[s], u[t]))
        x[t] = s
    return beta + (fm["Xd"] @ fm["coef"])[x] + _block_boot(resid, T, rng)


# ---------------------------------------------------------------------------------------------
# Multi-segment fit: repeated records at one bias (e.g. Nichol's 20-30 repetitions) share the
# kinetics, step amplitudes, AR(1) noise; each segment has its own baseline removed beforehand
# and its own offset (absorbed by fitting a per-segment intercept shift).
# ---------------------------------------------------------------------------------------------
@_kappa_starts
def fit_ar1_multi(zs, K, coupled, init, n_iter=80, tol=1e-7):
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    coef, sig, rho = init["coef"].copy(), float(init["sig"]), float(init.get("rho", 0.0))
    off = np.array(init.get("off", np.zeros(len(zs))), float)
    kin = Kinetics(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                   kappa=init.get("kappa"))
    Ttot = sum(z.size for z in zs)
    pis = [np.full(2 ** K, 1.0 / 2 ** K) for _ in zs]
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, st = 0.0, 0.0, 0.0
        sts = []
        for n, z in enumerate(zs):
            l, Nn, s, first, _ = _fb_ar1(z - off[n], P, pis[n], X @ coef, sig, rho, False)
            ll += l; N = N + Nn; st = st + s; sts.append(s); pis[n] = first
        if ll - prev < tol * Ttot and it > 2:
            break
        prev = ll
        kin.m_step(N)
        coef, rho, sig = _emission_mstep_ar1(st, X, coef, rho, Ttot)
        mu = X @ coef
        for n, s in enumerate(sts):              # per-segment offset: mean AR(1) innovation residual
            cnt, Sy, Sw = s[0], s[1], s[2]
            r = (Sy - mu[None, :] * cnt) - rho * (Sw - mu[:, None] * cnt)
            off[n] += r.sum() / max(cnt.sum(), 1.0) / (1 - rho)
    P = kin.P_np()
    ll, paths = 0.0, []
    for n, z in enumerate(zs):
        l, *_, path = _fb_ar1(z - off[n], P, pis[n], X @ coef, sig, rho, True)
        ll += l; paths.append(path)
    n_par = 2 * K + (K + 1) + 2 + len(zs) - 1 + (K * (K - 1) // 2 if coupled else 0)
    return dict(ll=float(ll), coef=coef, sig=sig, rho=rho, off=off, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), kappa=kin.kappa(), P=P, paths=paths,
                bic=-2 * ll + n_par * np.log(Ttot))


@_kappa_starts
def fit_ar1_multi_drive(zs, K, coupled, init, n_iter=60, tol=1e-7):
    """fit_ar1_multi with a shared hidden slow drive (KineticsDrive; state = bits + 2^K * u)."""
    B = bits(K)
    X = np.hstack([np.ones((2 ** K, 1)), B])
    Xd = np.vstack([X, X])
    coef, sig, rho = init["coef"].copy(), float(init["sig"]), float(init.get("rho", 0.0))
    off = np.array(init.get("off", np.zeros(len(zs))), float)
    kin = KineticsDrive(K, init["c"], init["e"], init.get("J") if coupled else None, coupled=coupled,
                        gamma=init.get("gamma"), drive=init.get("drive", (0.002, 0.002)), max_rate=init.get("max_rate"),
                        kappa=init.get("kappa"))
    S = 2 ** (K + 1)
    Ttot = sum(z.size for z in zs)
    pis = [np.full(S, 1.0 / S) for _ in zs]
    prev = -np.inf
    for it in range(n_iter):
        P = kin.P_np()
        ll, N, st, sts = 0.0, 0.0, 0.0, []
        for n, z in enumerate(zs):
            l, Nn, s, first, _ = _fb_ar1(z - off[n], P, pis[n], Xd @ coef, sig, rho, False)
            ll += l; N = N + Nn; st = st + s; sts.append(s); pis[n] = first
        if ll - prev < tol * Ttot and it > 2:
            break
        prev = ll
        kin.m_step(N)
        coef, rho, sig = _emission_mstep_ar1(st, Xd, coef, rho, Ttot)
        mu = Xd @ coef
        for n, s in enumerate(sts):
            cnt, Sy, Sw = s[0], s[1], s[2]
            r = (Sy - mu[None, :] * cnt) - rho * (Sw - mu[:, None] * cnt)
            off[n] += r.sum() / max(cnt.sum(), 1.0) / (1 - rho)
    ll = 0.0
    P = kin.P_np()
    for n, z in enumerate(zs):
        ll += _fb_ar1(z - off[n], P, pis[n], Xd @ coef, sig, rho, False)[0]
    return dict(ll=float(ll), coef=coef, sig=sig, rho=rho, off=off, c=kin.lc.exp().detach().numpy(),
                e=kin.le.exp().detach().numpy(), J=kin.j.detach().numpy().copy(), kappa=kin.kappa(),
                gamma=kin.g.detach().numpy().copy(), drive=kin.ld.exp().detach().numpy().copy())
