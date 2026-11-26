"""R9b / G2: bias-series joint model for multi-trap RTN.

A series = N records of the same device at gate biases v_n (scaled to [-1, 1]).  Trap k has
  amplitude   a_k(v) = a_k + d_k v                       (shared across records, slowly varying)
  rates       c_k(v) = exp(ac_k + bc_k v),  e_k(v) = exp(ae_k + be_k v)
Each record has its own offset b_n and white-noise sigma_n; its baseline (1/f, drift) is removed
beforehand with the R9a step detector + Kalman smoother, which does not depend on K.

fit_series(zs, vs, K)  joint EM (exact forward-backward per record, closed-form emission M-step,
                       per-trap kinetic M-step over all records); K chosen by joint BIC.
The same code with one record and no slopes is the per-record model (fit_series([z], [0], K,
slopes=False)), so the comparison isolates the effect of pooling.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from .robustk import _fb, bits, product_P, _chain


# ------------------------------------------------------------------------------- simulator
def simulate_series(K, N, T, rng, sig_rel=(0.15, 0.6), flicker=True, drift=True, amp_slope=0.05,
                    tau_range=(10, 300)):
    from .noise import add_flicker
    vs = np.linspace(-1, 1, N)
    a = np.sort(0.15 + 0.85 * rng.random(K))[::-1]
    a[0] = 1.0
    d = a * rng.normal(0, amp_slope, K)
    v0 = rng.uniform(-0.8, 0.8, K)                      # bias where the trap is half occupied
    s = rng.uniform(2.0, 5.0, K)                        # log(c/e) changes by s per unit v
    phi = rng.uniform(0.3, 0.7, K)
    ktot0 = 2.0 / np.exp(rng.uniform(np.log(tau_range[0]), np.log(tau_range[1]), K))
    truth = dict(a=a, d=d, ac=np.log(ktot0 / 2) - phi * s * v0, bc=phi * s,
                 ae=np.log(ktot0 / 2) + (1 - phi) * s * v0, be=-(1 - phi) * s, vs=vs)
    sig = a.min() * rng.uniform(*sig_rel)
    ys, states = [], []
    for n, v in enumerate(vs):
        S = np.zeros((T, K), np.int64)
        for k in range(K):
            c, e = np.exp(truth["ac"][k] + truth["bc"][k] * v), np.exp(truth["ae"][k] + truth["be"][k] * v)
            P = two_state_P(c, e)
            S[:, k] = _chain(P, T, rng, int(rng.random() < c / (c + e)))
        y = S @ (a + d * v) + sig * rng.standard_normal(T)
        if flicker:
            y = add_flicker(y[None], 0.6 * sig * rng.random(), fmin=1 / T, fmax=0.5, n_terms=8, dt=1.0, rng=rng)[0]
        if drift:
            y = y + rng.random() * a[0] * rng.standard_normal(T).cumsum() / np.sqrt(T)
        ys.append(y)
        states.append(S)
    return ys, vs, truth, states


def two_state_P(c, e):
    k = c + e
    q = 1 - np.exp(-k)
    return np.array([[1 - c / k * q, c / k * q], [e / k * q, 1 - e / k * q]])


# ------------------------------------------------------------------------------- joint model
def _kin_negll(theta, counts, vs, slopes):
    ac, ae = theta[0], theta[1]
    bc, be = (theta[2], theta[3]) if slopes else (0.0, 0.0)
    f = 0.0
    for n, v in enumerate(vs):
        P = two_state_P(np.exp(ac + bc * v), np.exp(ae + be * v))
        f -= np.sum(counts[n] * np.log(np.clip(P, 1e-300, None)))
    return f


class SeriesModel:
    def __init__(self, K, vs, slopes=True, amp_slopes=True):
        self.K, self.vs, self.slopes, self.amp_slopes = K, np.asarray(vs, float), slopes, amp_slopes
        self.N = len(vs)
        self.B = bits(K).astype(float)

    def n_par(self):
        K, N = self.K, self.N
        return K * (4 if self.slopes else 2) + K * (2 if self.amp_slopes else 1) + 2 * N

    def P_list(self, kin):
        out = []
        for v in self.vs:
            Ps = np.array([two_state_P(np.exp(kin[k, 0] + kin[k, 2] * v), np.exp(kin[k, 1] + kin[k, 3] * v))
                           for k in range(self.K)])
            out.append(product_P(Ps))
        return out

    def means(self, a, d, b, n):
        return b[n] + self.B @ (a + d * self.vs[n])


def fit_series(zs, vs, K, init, slopes=True, amp_slopes=True, n_iter=60, tol=1e-7, n_kin=1):
    """zs: list of baseline-removed records; init: dict(a, d?, b?, sig?, kin?).  Returns dict."""
    m = SeriesModel(K, vs, slopes, amp_slopes and len(vs) > 1)
    N, B, S = m.N, m.B, 2 ** K
    a = np.array(init["a"], float)
    d = np.array(init.get("d", np.zeros(K)), float)
    b = np.array(init.get("b", [np.quantile(z, 0.02) for z in zs]), float)
    sig = np.array(init.get("sig", [np.median(np.abs(np.diff(z))) / 0.6745 / np.sqrt(2) for z in zs]), float)
    kin = np.array(init.get("kin", np.tile([np.log(0.02), np.log(0.02), 0.0, 0.0], (K, 1))), float)
    pi0 = [np.full(S, 1.0 / S) for _ in range(N)]
    Ttot = sum(len(z) for z in zs)
    prev = -np.inf
    ll = -np.inf
    for it in range(n_iter):
        Pl = m.P_list(kin)
        ll = 0.0
        stats = []
        for n, z in enumerate(zs):
            lln, Nn, g0, gy, gyy, first, _ = _fb(z, Pl[n], pi0[n], m.means(a, d, b, n), sig[n], False)
            ll += lln
            pi0[n] = first
            stats.append((Nn, g0, gy, gyy))
        if ll - prev < tol * Ttot and it > 3:
            break
        prev = ll
        # emission: weighted least squares for (b_1..b_N, a, d) given sigma_n
        p = N + K + (K if m.amp_slopes else 0)
        A = np.zeros((p, p)); r = np.zeros(p)
        for n, (Nn, g0, gy, gyy) in enumerate(stats):
            w = 1.0 / sig[n] ** 2
            X = np.zeros((S, p))
            X[:, n] = 1.0
            X[:, N:N + K] = B
            if m.amp_slopes:
                X[:, N + K:] = B * m.vs[n]
            A += w * (X.T * g0) @ X
            r += w * X.T @ gy
        sol = np.linalg.solve(A + 1e-9 * np.eye(p), r)
        b, a = sol[:N], sol[N:N + K]
        d = sol[N + K:] if m.amp_slopes else np.zeros(K)
        for n, (Nn, g0, gy, gyy) in enumerate(stats):
            mu = m.means(a, d, b, n)
            sig[n] = np.sqrt(max((gyy - 2 * mu * gy + mu * mu * g0).sum() / len(zs[n]), 1e-12))
        # kinetics: per trap, marginal transition counts per record
        for k in range(K):
            counts = []
            for (Nn, *_r) in stats:
                c2 = np.zeros((2, 2))
                for x in range(S):
                    for xp in range(S):
                        c2[int(B[x, k]), int(B[xp, k])] += Nn[x, xp]
                counts.append(c2)
            th0 = kin[k] if slopes else kin[k, :2]
            res = minimize(_kin_negll, th0, args=(counts, m.vs, slopes), method="L-BFGS-B",
                           bounds=[(-12, 2), (-12, 2)] + ([(-15, 15), (-15, 15)] if slopes else []),
                           options=dict(maxiter=30 * n_kin))
            kin[k, :len(res.x)] = res.x
    return dict(ll=float(ll), a=a, d=d, b=b, sig=sig, kin=kin, K=K,
                bic=-2 * ll + m.n_par() * np.log(Ttot), aic=-2 * ll + 2 * m.n_par())


def screen_series(zs, vs, cands, K, top=3, n_iter=5, slopes=True, max_c=12, ar1=False):
    """Every K-subset of candidate steps, short joint EM; best few results."""
    cands = np.asarray(sorted(cands))
    cands = cands[cands > 0.1 * cands.max()]
    merged = []
    for c in cands:
        if not merged or c > merged[-1] * 1.08:
            merged.append(c)
    cands = np.array(merged)
    if len(cands) > max_c:
        cands = cands[np.unique(np.linspace(0, len(cands) - 1, max_c).round().astype(int))]
    out = []
    for comb in combinations(cands, K):
        try:
            f = (fit_series_ar1 if ar1 else fit_series)(zs, vs, K, dict(a=np.sort(comb)[::-1]), slopes=slopes, n_iter=n_iter)
        except (ZeroDivisionError, np.linalg.LinAlgError):
            continue
        out.append(f)
    out.sort(key=lambda f: -f["ll"])
    return out[:top]


def select_K(zs, vs, cands, Ks=(1, 2, 3, 4), slopes=True, top=3, ar1=False):
    """Joint (or single-record) fits for each K from screened candidates; returns fits and BIC choice."""
    fits = {}
    for K in Ks:
        if len(cands) < K:
            continue
        best = None
        for f in screen_series(zs, vs, cands, K, top=top, slopes=slopes, ar1=ar1):
            g = (fit_series_ar1 if ar1 else fit_series)(zs, vs, K, dict(a=f["a"], d=f["d"], b=f["b"], sig=f["sig"], rho=f.get("rho", np.zeros(len(zs))), kin=f["kin"]), slopes=slopes)
            if best is None or g["ll"] > best["ll"]:
                best = g
        if best is not None:
            fits[K] = best
    K_bic = min(fits, key=lambda k: fits[k]["bic"]) if fits else 1
    return K_bic, fits


def init_from_records(rec_fits, vs, K):
    """Joint starting values from per-record fits at the same K: amplitudes = median of the
    per-record (sorted) amplitudes; trap k in record n is the per-record trap with the closest
    amplitude; log rates regressed on v (weights = estimated switches); offsets from the records."""
    A = np.array([np.sort(np.abs(f["a"]))[::-1] for f in rec_fits])
    a = np.median(A, 0)
    kin = np.zeros((K, 4))
    for k in range(K):
        lc, le, w = [], [], []
        for n, f in enumerate(rec_fits):
            j = int(np.argmin(np.abs(np.abs(f["a"]) - a[k])))
            ac, ae = f["kin"][j, 0], f["kin"][j, 1]
            if f["a"][j] < 0:                       # flipped trap: swap roles of capture and emission
                ac, ae = ae, ac
            lc.append(ac); le.append(ae)
            w.append(np.exp(min(ac, ae)))          # ~ switching rate: weak records count less
        X = np.c_[np.ones(len(vs)), vs]
        W = np.sqrt(np.array(w) / max(w))
        bc = np.linalg.lstsq(X * W[:, None], np.array(lc) * W, rcond=None)[0]
        be = np.linalg.lstsq(X * W[:, None], np.array(le) * W, rcond=None)[0]
        kin[k] = [bc[0], be[0], bc[1], be[1]]
    b = np.array([f["b"][0] for f in rec_fits])
    sig = np.array([f["sig"][0] for f in rec_fits])
    rho = np.array([f.get("rho", np.zeros(1))[0] for f in rec_fits])
    return dict(a=a, d=np.zeros(K), b=b, sig=sig, rho=rho, kin=kin)


def select_K_joint(zs, vs, rec_fits_by_K, pooled, Ks=(1, 2, 3, 4), top=2, ar1=False):
    """Joint fits per K started from (i) the aligned per-record fits and (ii) screened pooled
    candidates; best log-likelihood kept; K by joint BIC."""
    fits = {}
    for K in Ks:
        cands_init = []
        rf = rec_fits_by_K.get(K)
        if rf is not None and len(rf) == len(zs):
            cands_init.append(init_from_records(rf, vs, K))
        if len(pooled) >= K:
            for f in screen_series(zs, vs, pooled, K, top=top, slopes=True, ar1=ar1):
                cands_init.append(dict(a=f["a"], d=f["d"], b=f["b"], sig=f["sig"], rho=f.get("rho", np.zeros(len(zs))), kin=f["kin"]))
        best = None
        for ini in cands_init:
            try:
                g = (fit_series_ar1 if ar1 else fit_series)(zs, vs, K, ini, slopes=True)
            except (ZeroDivisionError, np.linalg.LinAlgError):
                continue
            if best is None or g["ll"] > best["ll"]:
                best = g
        if best is not None:
            fits[K] = best
    K_bic = min(fits, key=lambda k: fits[k]["bic"]) if fits else 1
    return K_bic, fits


# ------------------------------------------------------------------------------- AR(1) noise
def fit_series_ar1(zs, vs, K, init, slopes=True, amp_slopes=True, n_iter=60, tol=1e-7):
    """fit_series with AR(1) residual noise per record (exact, pairwise emission as in cfhmm)."""
    from .cfhmm import _fb_ar1
    m = SeriesModel(K, vs, slopes, amp_slopes and len(vs) > 1)
    N, B, S = m.N, m.B, 2 ** K
    a = np.array(init["a"], float)
    d = np.array(init.get("d", np.zeros(K)), float)
    b = np.array(init.get("b", [np.quantile(z, 0.02) for z in zs]), float)
    sig = np.array(init.get("sig", [np.median(np.abs(np.diff(z))) / 0.6745 / np.sqrt(2) for z in zs]), float)
    rho = np.array(init.get("rho", np.zeros(N)), float)
    kin = np.array(init.get("kin", np.tile([np.log(0.02), np.log(0.02), 0.0, 0.0], (K, 1))), float)
    pi0 = [np.full(S, 1.0 / S) for _ in range(N)]
    Ttot = sum(len(z) for z in zs)
    p = N + K + (K if m.amp_slopes else 0)

    def design(n):
        X = np.zeros((S, p))
        X[:, n] = 1.0
        X[:, N:N + K] = B
        if m.amp_slopes:
            X[:, N + K:] = B * m.vs[n]
        return X

    prev, ll = -np.inf, -np.inf
    for it in range(n_iter):
        Pl = m.P_list(kin)
        ll = 0.0
        stats = []
        for n, z in enumerate(zs):
            lln, Nn, st, first, _ = _fb_ar1(z, Pl[n], pi0[n], m.means(a, d, b, n), sig[n], rho[n], False)
            ll += lln
            pi0[n] = first
            stats.append((Nn, st))
        if ll - prev < tol * Ttot and it > 3:
            break
        prev = ll
        for inner in range(2):
            A = np.zeros((p, p)); r = np.zeros(p)
            for n, (Nn, st) in enumerate(stats):
                w = 1.0 / sig[n] ** 2
                X = design(n)
                cnt, Sy, Sw = st[0], st[1], st[2]
                for i in range(S):
                    D = X - rho[n] * X[i]                 # rows j: x_j - rho x_i
                    A += w * (D.T * cnt[i]) @ D
                    r += w * D.T @ (Sy[i] - rho[n] * Sw[i])
            sol = np.linalg.solve(A + 1e-9 * np.eye(p), r)
            b, a = sol[:N], sol[N:N + K]
            d = sol[N + K:] if m.amp_slopes else np.zeros(K)
            for n, (Nn, st) in enumerate(stats):
                cnt, Sy, Sw, Syy, Sww, Syw = st
                mu = m.means(a, d, b, n)
                mi, mj = mu[:, None], mu[None, :]
                num = (Syw - mi * Sy - mj * Sw + cnt * mi * mj).sum()
                den = (Sww - 2 * mi * Sw + cnt * mi * mi).sum()
                rho[n] = float(np.clip(num / max(den, 1e-12), -0.98, 0.98))
                q = (Syy - 2 * mj * Sy + mj * mj * cnt) - 2 * rho[n] * (Syw - mi * Sy - mj * Sw + cnt * mi * mj) \
                    + rho[n] ** 2 * (Sww - 2 * mi * Sw + cnt * mi * mi)
                sig[n] = float(np.sqrt(max(q.sum() / (len(zs[n]) - 1), 1e-12)))
        for k in range(K):
            counts = []
            for (Nn, st) in stats:
                c2 = np.zeros((2, 2))
                for x in range(S):
                    for xp in range(S):
                        c2[int(B[x, k]), int(B[xp, k])] += Nn[x, xp]
                counts.append(c2)
            th0 = kin[k] if slopes else kin[k, :2]
            res = minimize(_kin_negll, th0, args=(counts, m.vs, slopes), method="L-BFGS-B",
                           bounds=[(-12, 2), (-12, 2)] + ([(-15, 15), (-15, 15)] if slopes else []),
                           options=dict(maxiter=30))
            kin[k, :len(res.x)] = res.x
    npar = m.n_par() + N
    return dict(ll=float(ll), a=a, d=d, b=b, sig=sig, rho=rho, kin=kin, K=K,
                bic=-2 * ll + npar * np.log(Ttot), aic=-2 * ll + 2 * npar)


def trap_support(f, edge_heights, n_edges_min=15, frac_min=0.03, tol=0.15):
    """For each trap of a fit: amplitude over the marginal noise std, and the number of detected
    step events (R9a step detector) whose height matches the amplitude within tol."""
    sig = np.asarray(f["sig"]); rho = np.asarray(f.get("rho", np.zeros_like(sig)))
    marg = float(np.median(sig / np.sqrt(1 - rho ** 2)))
    H = np.asarray(edge_heights)
    out = []
    for a in np.abs(f["a"]):
        cnt = int(np.sum(np.abs(H - a) < tol * a))
        out.append(dict(a=float(a), snr=float(a / marg), n_edges=cnt,
                        ok=bool(a / marg >= 1.2 and cnt >= max(n_edges_min, frac_min * len(H)))))
    return out


def choose_K_supported(fits, edge_heights, **kw):
    """Largest K whose BIC beats every smaller K and whose traps all pass trap_support."""
    best = 1 if 1 in fits else min(fits)
    for K in sorted(fits):
        if K == min(fits):
            continue
        better = all(fits[K]["bic"] < fits[k]["bic"] for k in fits if k < K)
        supp = all(t["ok"] for t in trap_support(fits[K], edge_heights, **kw))
        if better and supp:
            best = K
    return best
