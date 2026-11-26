"""R2 practical detectors (parameters unknown), vectorised over a batch of traces.

D1  GLRT by batch EM.  Hidden Markov models with four hidden states (s1, s2)
    and free emission levels (so non-additive readout cannot fake kinetics):
      product  P = P1 (x) P2                         (independent traps, 4 numbers)
      coupled  trap i's 2x2 matrix depends on s_j     (kinetic coupling, 8 numbers)
      general  P unconstrained 4 x 4                 (anything, 12 numbers; "glrt_general")
    Statistic = loglik(alternative) - loglik(product); alternatives start from the
    product fit, so the statistic is >= 0 (EM is monotone).
D2a |corr(s1, s2)| of the posterior-decoded (idealised) trace.
D2b discrete-chain LRT on the idealised trace (general vs product counts):
    the usual "idealise, then analyse dwell statistics" pipeline.
Oracle: exact HMM LLR with the true coupled and independent models.

State index x = s1 + 2 s2; trap 1 is the larger step by convention, which fixes
the level-to-state assignment used for initialisation.
"""
from __future__ import annotations

import numpy as np

S1 = np.array([0, 1, 0, 1])
S2 = np.array([0, 0, 1, 1])
_ORDER = np.array([0, 2, 1, 3])   # sorted levels (A1 > A2): (0,0) < (0,1) < (1,0) < (1,1)


# --------------------------------------------------------------------------- #
# Batch forward-backward
# --------------------------------------------------------------------------- #
def _emis(yk, mu, sig):
    return np.exp(-0.5 * ((yk[:, None] - mu) / sig[:, None]) ** 2)


try:
    import numba as _nb

    @_nb.njit(parallel=True, cache=True, fastmath=False)
    def _fb_kernel(y, P, pi0, mu, sig, want_path):
        B, T = y.shape
        ll = np.empty(B)
        N = np.zeros((B, 4, 4))
        g0 = np.zeros((B, 4))
        gy = np.zeros((B, 4))
        gyy = np.zeros((B, 4))
        first = np.empty((B, 4))
        path = np.zeros((B, T if want_path else 1), dtype=np.int8)
        for b in _nb.prange(B):
            alpha = np.empty((T, 4))
            c = np.empty(T)
            e = np.empty(4)
            inv = 1.0 / sig[b]
            tot = 0.0
            for s in range(4):
                z = (y[b, 0] - mu[b, s]) * inv
                alpha[0, s] = pi0[b, s] * np.exp(-0.5 * z * z)
                tot += alpha[0, s]
            c[0] = tot
            for s in range(4):
                alpha[0, s] /= tot
            for k in range(1, T):
                tot = 0.0
                for t in range(4):
                    acc = 0.0
                    for s in range(4):
                        acc += alpha[k - 1, s] * P[b, s, t]
                    z = (y[b, k] - mu[b, t]) * inv
                    acc *= np.exp(-0.5 * z * z)
                    alpha[k, t] = acc
                    tot += acc
                c[k] = tot
                for t in range(4):
                    alpha[k, t] /= tot
            lsum = 0.0
            for k in range(T):
                lsum += np.log(c[k])
            ll[b] = lsum - T * np.log(sig[b]) - 0.5 * T * np.log(2.0 * np.pi)
            beta = np.ones(4)
            nb_ = np.empty(4)
            g = np.empty(4)
            for k in range(T - 1, -1, -1):
                tot = 0.0
                for s in range(4):
                    g[s] = alpha[k, s] * beta[s]
                    tot += g[s]
                best = 0
                for s in range(4):
                    g[s] /= tot
                    g0[b, s] += g[s]
                    gy[b, s] += g[s] * y[b, k]
                    gyy[b, s] += g[s] * y[b, k] * y[b, k]
                    if g[s] > g[best]:
                        best = s
                if want_path:
                    path[b, k] = best
                if k > 0:
                    for t in range(4):
                        z = (y[b, k] - mu[b, t]) * inv
                        e[t] = np.exp(-0.5 * z * z) * beta[t] / c[k]
                    for s in range(4):
                        acc = 0.0
                        for t in range(4):
                            v = P[b, s, t] * e[t]
                            N[b, s, t] += alpha[k - 1, s] * v
                            acc += v
                        nb_[s] = acc
                    for s in range(4):
                        beta[s] = nb_[s]
                else:
                    for s in range(4):
                        first[b, s] = g[s]
        return ll, N, g0, gy, gyy, first, path

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False


def forward_backward(y, P, pi0, mu, sig, want_path=False):
    """Per-trace parameters. Returns loglik (B,), expected transition counts N (B,4,4),
    occupation sums g0, gy, gyy (B,4), first-step posterior (B,4), optional
    posterior-argmax path (B,T)."""
    if _HAVE_NUMBA:
        ll, N, g0, gy, gyy, first, path = _fb_kernel(
            np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(pi0, dtype=np.float64), np.ascontiguousarray(mu, dtype=np.float64),
            np.ascontiguousarray(sig, dtype=np.float64), bool(want_path))
        return ll, N, g0, gy, gyy, first, (path if want_path else None)
    return forward_backward_numpy(y, P, pi0, mu, sig, want_path)


def forward_backward_numpy(y, P, pi0, mu, sig, want_path=False):
    B, T = y.shape
    alpha = np.empty((T, B, 4))
    c = np.empty((T, B))
    a = pi0 * _emis(y[:, 0], mu, sig)
    c[0] = a.sum(1)
    alpha[0] = a / c[0][:, None]
    for k in range(1, T):
        a = np.einsum("bs,bst->bt", alpha[k - 1], P) * _emis(y[:, k], mu, sig)
        c[k] = a.sum(1)
        alpha[k] = a / c[k][:, None]
    ll = np.log(c).sum(0) - T * np.log(sig) - 0.5 * T * np.log(2 * np.pi)
    N = np.zeros((B, 4, 4))
    g0 = np.zeros((B, 4))
    gy = np.zeros((B, 4))
    gyy = np.zeros((B, 4))
    path = np.empty((B, T), dtype=np.int8) if want_path else None
    beta = np.ones((B, 4))
    for k in range(T - 1, -1, -1):
        g = alpha[k] * beta
        g /= g.sum(1, keepdims=True)
        yk = y[:, k][:, None]
        g0 += g
        gy += g * yk
        gyy += g * yk * yk
        if want_path:
            path[:, k] = g.argmax(1)
        if k > 0:
            eb = _emis(y[:, k], mu, sig) * beta / c[k][:, None]
            N += alpha[k - 1][:, :, None] * P * eb[:, None, :]
            beta = np.einsum("bst,bt->bs", P, eb)
    first = alpha[0] * beta
    first /= first.sum(1, keepdims=True)
    return ll, N, g0, gy, gyy, first, path


# --------------------------------------------------------------------------- #
# EM fits
# --------------------------------------------------------------------------- #
def _init_levels(y, n_iter=25):
    """Shared-sigma 4-component Gaussian mixture per trace; returns levels in state order."""
    B, T = y.shape
    mu = np.quantile(y, [0.125, 0.375, 0.625, 0.875], axis=1).T     # (B,4) sorted
    w = np.full((B, 4), 0.25)
    sig = np.std(y, axis=1) / 3
    for _ in range(n_iter):
        r = w[:, None, :] * np.exp(-0.5 * ((y[:, :, None] - mu[:, None, :]) / sig[:, None, None]) ** 2)
        r /= r.sum(2, keepdims=True) + 1e-300
        n = r.sum(1) + 1e-12
        mu = (r * y[:, :, None]).sum(1) / n
        sig = np.sqrt((r * (y[:, :, None] - mu[:, None, :]) ** 2).sum((1, 2)) / T)
        w = n / T
    mu = np.sort(mu, axis=1)
    out = np.empty_like(mu)
    out[:, _ORDER] = mu
    return out, sig


def _product_P(P1, P2):
    """P[x, x'] = P1[s1, s1'] P2[s2, s2'] for batched 2x2 matrices."""
    return P1[:, S1][:, :, S1] * P2[:, S2][:, :, S2]


def fit_shared_levels(y, n_sub=60, n_iter=60, fracs=(0.15, 0.3, 0.45)):
    """Device-level emission model: one HMM (general kinetics) whose parameters are shared
    by all traces (segments of one record).  Several additive-level starts; best pooled
    likelihood wins.  Returns levels (4,) in state order and sigma."""
    B, T = y.shape
    sub = y[np.linspace(0, B - 1, min(B, n_sub)).astype(int)]
    lo, hi = np.quantile(sub, 0.02), np.quantile(sub, 0.98)
    starts = [_init_levels(sub.reshape(1, -1))]
    for f in fracs:
        A1, A2 = (hi - lo) * (1 - f), (hi - lo) * f
        mu = np.array([[lo, lo + A1, lo + A2, hi]])
        starts.append((mu, np.array([(hi - lo) / 8])))
    best = None
    for mu0, sig0 in starts:
        r = _fit_shared(sub, mu0[0], float(sig0[0]), n_iter)
        if best is None or r[2] > best[2]:
            best = r
    mu, sig, _ = _fit_shared(y, best[0], best[1], 30)
    return mu, sig


def _fit_shared(y, mu, sig, n_iter):
    """Pooled EM with one unconstrained 4x4 transition matrix shared by all traces.

    The kinetics model is deliberately the general one: the levels are device
    properties and must not be biased by assuming independence (a product model
    forced onto coupled data mislabels rarely visited states)."""
    B, T = y.shape
    sw = 0.1
    P1 = np.array([[1 - sw, sw], [sw, 1 - sw]])
    P = P1[S1][:, S1] * P1[S2][:, S2]
    pi0 = np.full(4, 0.25)
    for it in range(n_iter):
        ll, N, g0, gy, gyy, first, _ = forward_backward(
            y, np.repeat(P[None], B, 0), np.repeat(pi0[None], B, 0), np.repeat(mu[None], B, 0), np.full(B, sig))
        N, g0, gy, gyy = N.sum(0), g0.sum(0), gy.sum(0), gyy.sum(0)
        P = (N + 1e-12) / (N.sum(1, keepdims=True) + 4e-12)
        mu = gy / np.maximum(g0, 1e-12)
        sig = float(np.sqrt(max((gyy - 2 * mu * gy + mu * mu * g0).sum() / (B * T), 1e-12)))
        pi0 = first.mean(0)
    return mu, sig, float(ll.sum())


def fit_product(y, n_iter=40, tol=1e-7, restarts=True):
    """Product-model EM.  With restarts, also starts from levels of a mixture fitted to
    the pooled batch (traces = segments of one device record) and keeps, per trace,
    the better optimum."""
    a = _fit_product_from(y, *_init_levels(y), n_iter, tol)
    if not restarts:
        return a
    mu_pool, sig_pool = _init_levels(y.reshape(1, -1)[:, :: max(1, y.size // 200000)])
    B = y.shape[0]
    for freeze in (0, 10):
        b = _fit_product_from(y, np.repeat(mu_pool, B, 0), np.repeat(sig_pool, B), n_iter, tol, freeze=freeze)
        take_b = b["ll"] > a["ll"]
        a = {k: np.where(take_b.reshape((-1,) + (1,) * (a[k].ndim - 1)), b[k], a[k]) for k in a}
    return a


def _fit_product_from(y, mu, sig, n_iter, tol, freeze=0, fixed=False):
    """EM from given levels; the first `freeze` iterations keep the levels fixed
    (kinetics warm-up), which avoids collapsing a weak level onto its neighbour."""
    B, T = y.shape
    sw = 0.1
    P1 = np.tile(np.array([[1 - sw, sw], [sw, 1 - sw]]), (B, 1, 1))
    P2 = P1.copy()
    pi0 = np.full((B, 4), 0.25)
    prev = -np.inf
    for it in range(n_iter):
        P = _product_P(P1, P2)
        ll, N, g0, gy, gyy, first, _ = forward_backward(y, P, pi0, mu, sig)
        n1 = np.zeros((B, 2, 2))
        n2 = np.zeros((B, 2, 2))
        for x in range(4):
            for xp in range(4):
                n1[:, S1[x], S1[xp]] += N[:, x, xp]
                n2[:, S2[x], S2[xp]] += N[:, x, xp]
        P1 = (n1 + 1e-12) / (n1.sum(2, keepdims=True) + 2e-12)
        P2 = (n2 + 1e-12) / (n2.sum(2, keepdims=True) + 2e-12)
        if it >= freeze and not fixed:
            mu = gy / np.maximum(g0, 1e-12)
            sig = np.sqrt(np.maximum((gyy - 2 * mu * gy + mu * mu * g0).sum(1) / T, 1e-12))
        pi0 = first
        if (fixed or it >= freeze) and np.max(ll - prev) / T < tol:
            break
        prev = ll
    P = _product_P(P1, P2)
    ll, *_ = forward_backward(y, P, pi0, mu, sig)
    return dict(P=P, P1=P1, P2=P2, mu=mu, sig=sig, pi0=pi0, ll=ll)


def _cond_P(P1c, P2c):
    """Kinetically coupled update: trap i's 2x2 transition depends on the other trap's
    current state.  P1c[:, b] is trap 1's matrix when s2 = b; P2c[:, a] when s1 = a.
    P[x, x'] = P1c[s2(x)][s1, s1'] * P2c[s1(x)][s2, s2'].  Nests the product model."""
    B = P1c.shape[0]
    P = np.empty((B, 4, 4))
    for x in range(4):
        for xp in range(4):
            P[:, x, xp] = P1c[:, S2[x], S1[x], S1[xp]] * P2c[:, S1[x], S2[x], S2[xp]]
    return P


def fit_coupled(y, start, n_iter=25, tol=1e-7, fixed=False):
    """Physics-structured alternative: 8 transition numbers (vs 4 for the product)."""
    B, T = y.shape
    P1c = np.stack([start["P1"], start["P1"]], axis=1)   # (B, 2[s2], 2, 2)
    P2c = np.stack([start["P2"], start["P2"]], axis=1)   # (B, 2[s1], 2, 2)
    mu, sig, pi0 = start["mu"].copy(), start["sig"].copy(), start["pi0"].copy()
    prev = start["ll"]
    for it in range(n_iter):
        P = _cond_P(P1c, P2c)
        ll, N, g0, gy, gyy, first, _ = forward_backward(y, P, pi0, mu, sig)
        n1 = np.zeros((B, 2, 2, 2))
        n2 = np.zeros((B, 2, 2, 2))
        for x in range(4):
            for xp in range(4):
                n1[:, S2[x], S1[x], S1[xp]] += N[:, x, xp]
                n2[:, S1[x], S2[x], S2[xp]] += N[:, x, xp]
        P1c = (n1 + 1e-12) / (n1.sum(3, keepdims=True) + 2e-12)
        P2c = (n2 + 1e-12) / (n2.sum(3, keepdims=True) + 2e-12)
        if not fixed:
            mu = gy / np.maximum(g0, 1e-12)
            sig = np.sqrt(np.maximum((gyy - 2 * mu * gy + mu * mu * g0).sum(1) / T, 1e-12))
        pi0 = first
        if it > 0 and np.max(ll - prev) / T < tol:
            break
        prev = ll
    P = _cond_P(P1c, P2c)
    ll, *_ = forward_backward(y, P, pi0, mu, sig)
    return dict(P=P, mu=mu, sig=sig, pi0=pi0, ll=np.maximum(ll, start["ll"]))


def fit_general(y, start, n_iter=25, tol=1e-7, fixed=False):
    B, T = y.shape
    P, mu, sig, pi0 = start["P"].copy(), start["mu"].copy(), start["sig"].copy(), start["pi0"].copy()
    prev = start["ll"]
    ll = prev
    for it in range(n_iter):
        ll, N, g0, gy, gyy, first, _ = forward_backward(y, P, pi0, mu, sig)
        P = (N + 1e-12) / (N.sum(2, keepdims=True) + 4e-12)
        if not fixed:
            mu = gy / np.maximum(g0, 1e-12)
            sig = np.sqrt(np.maximum((gyy - 2 * mu * gy + mu * mu * g0).sum(1) / T, 1e-12))
        pi0 = first
        if it > 0 and np.max(ll - prev) / T < tol:
            break
        prev = ll
    ll, *_ = forward_backward(y, P, pi0, mu, sig)
    return dict(P=P, mu=mu, sig=sig, pi0=pi0, ll=np.maximum(ll, start["ll"]))


# --------------------------------------------------------------------------- #
# Statistics on an idealised path
# --------------------------------------------------------------------------- #
def _discrete_lrt(path):
    B, T = path.shape
    N = np.zeros((B, 4, 4))
    idx = path[:, :-1].astype(np.int64) * 4 + path[:, 1:]
    for b in range(B):
        N[b] = np.bincount(idx[b], minlength=16).reshape(4, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        row = N.sum(2, keepdims=True)
        llg = np.nansum(np.where(N > 0, N * np.log(N / row), 0.0), axis=(1, 2))
        n1 = np.zeros((B, 2, 2))
        n2 = np.zeros((B, 2, 2))
        for x in range(4):
            for xp in range(4):
                n1[:, S1[x], S1[xp]] += N[:, x, xp]
                n2[:, S2[x], S2[xp]] += N[:, x, xp]
        l1 = np.nansum(np.where(n1 > 0, n1 * np.log(n1 / n1.sum(2, keepdims=True)), 0.0), axis=(1, 2))
        l2 = np.nansum(np.where(n2 > 0, n2 * np.log(n2 / n2.sum(2, keepdims=True)), 0.0), axis=(1, 2))
    return llg - l1 - l2


def _abs_corr(path):
    a, b = S1[path].astype(float), S2[path].astype(float)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.abs(np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1), 0.0))


def detector_scores(y, batch=1000, shared_levels=True):
    """All practical detector statistics for traces y (B, T).

    shared_levels=True (default): the emission model (levels, sigma) is fitted once on the
    pooled batch (traces = segments of one device record) and held fixed; per trace only
    the kinetics are fitted, so the GLRT compares 8 vs 4 transition numbers.
    shared_levels=False: levels and sigma are also fitted per trace (with restarts).
    """
    out = {k: np.empty(y.shape[0]) for k in ("glrt", "glrt_general", "corr", "ideal_lrt")}
    if shared_levels:
        mu_s, sig_s = fit_shared_levels(y)
    for s in range(0, y.shape[0], batch):
        yb = y[s:s + batch]
        B = yb.shape[0]
        if shared_levels:
            prod = _fit_product_from(yb, np.repeat(mu_s[None], B, 0), np.full(B, sig_s), 60, 1e-8, fixed=True)
            cpl = fit_coupled(yb, prod, n_iter=40, tol=1e-8, fixed=True)
            gen = fit_general(yb, prod, n_iter=40, tol=1e-8, fixed=True)
        else:
            prod = fit_product(yb)
            cpl = fit_coupled(yb, prod)
            gen = fit_general(yb, prod)
        out["glrt"][s:s + batch] = cpl["ll"] - prod["ll"]
        out["glrt_general"][s:s + batch] = gen["ll"] - prod["ll"]
        *_, path = forward_backward(yb, prod["P"], prod["pi0"], prod["mu"], prod["sig"], want_path=True)
        out["corr"][s:s + batch] = _abs_corr(path)
        out["ideal_lrt"][s:s + batch] = _discrete_lrt(path)
    return out


def baseline_correct(y, n_outer=4, window=301):
    """Model-based drift removal.

    Alternates (i) a pooled level fit and per-trace product HMM on the corrected
    trace, (ii) a baseline estimate = moving average of the residual y - mu[x_hat],
    where x_hat is the posterior-argmax state.  Subtracting the *expected level*
    first means trap switching (including correlated switching from a shared
    drive) does not leak into the baseline, unlike a running median of y itself.
    """
    from scipy.ndimage import uniform_filter1d
    B, T = y.shape
    # start: long moving average of y (trap occupancy averages out over the window)
    base = uniform_filter1d(np.asarray(y, np.float64), size=window, axis=1, mode="nearest")
    base -= np.median(base, axis=1, keepdims=True)
    for _ in range(n_outer):
        yc = y - base
        mu, sig = fit_shared_levels(yc)
        prod = _fit_product_from(yc, np.repeat(mu[None], B, 0), np.full(B, sig), 40, 1e-7, fixed=True)
        *_, path = forward_backward(yc, prod["P"], prod["pi0"], prod["mu"], prod["sig"], want_path=True)
        resid = y - np.take_along_axis(prod["mu"], path.astype(np.int64), axis=1)
        base = uniform_filter1d(resid, size=window, axis=1, mode="nearest")
        base -= np.median(base, axis=1, keepdims=True)
    return y - base
