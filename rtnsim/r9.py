"""R9 pipeline pieces: CNN step detector -> decomposition, K and baseline -> exact-likelihood test.

cnn_init(y, K, net)       trap-signal estimate from the step detector; free-level HMM on the
                          denoised trace (well separated levels) -> additive fit -> initial
                          (coef, sig, c, e); baseline = Kalman smoother of (y - trap signal)
lrt_cnn(y, K, net)        the R8 likelihood-ratio test started from cnn_init only (no global search)
k_from_steps(...)         number of distinct step heights (K estimate) from detected edges
"""
from __future__ import annotations

import numpy as np
import torch

from .cfhmm import (_free_hmm, _additive_fit, fit, fit_ar1, kalman_baseline, bits)
from .robustk import coupling_stat
from .stepnet import StepNet, denoise


import os as _os
HAMPEL = _os.environ.get("RTN_HAMPEL", "0") == "1"


def load_stepnet(path):
    ck = torch.load(path, map_location="cpu")
    net = StepNet(C=ck["config"]["C"], n_dil=ck["config"].get("n_dil", 11)).eval()
    net.load_state_dict(ck["state_dict"])
    torch.set_num_threads(1)
    return net


def cnn_init(y, K, net, n_starts=2, rng=None):
    rng = rng or np.random.default_rng(0)
    lev, edge, step = denoise(net, y)
    beta, _ = kalman_baseline(y - lev)
    beta = beta - np.median(beta)
    z = y - beta
    M = 2 ** K
    inits = []
    lo, hi = np.quantile(lev, [0.01, 0.99])
    for s in range(n_starts):
        mu0 = np.quantile(lev, (np.arange(M) + 0.5) / M) if s == 0 else np.sort(lo + (hi - lo) * rng.random(M))
        try:
            mu, w, P, sig_l, ll = _free_hmm(lev - np.median(lev) + np.median(z), M, mu0 - np.median(lev) + np.median(z))
        except ZeroDivisionError:
            continue
        coef = _additive_fit(mu, w, K)
        a = coef[1:]
        flip = a < 0
        coef[0] += a[flip].sum()
        a = np.abs(a)
        a = a[np.argsort(-a)]
        sig = float(np.median(np.abs(np.diff(z - lev))) / 0.6745 / np.sqrt(2))
        inits.append(dict(coef=np.r_[coef[0], a], sig=max(sig, 1e-6), c=np.full(K, 0.02), e=np.full(K, 0.02)))
    return beta, inits, lev, edge, step


def lrt_cnn(y, K, net, ar1=True, rng=None):
    fitter = fit_ar1 if ar1 else fit
    beta, inits, *_ = cnn_init(y, K, net, rng=rng)
    z = y - beta
    best = None
    for ini in inits:
        f0 = fitter(z, K, False, ini)
        if best is None or f0["ll"] > best["ll"]:
            best = f0
    f0 = best
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    f1 = fitter(z, K, True, ini)
    stat = 2 * max(f1["ll"] - f0["ll"], 0.0)
    return stat, f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def k_from_steps(edge, step, max_k=4, min_frac=0.04, prom=0.08):
    """K from the distribution of detected step heights: peaks of a KDE of |step| at edges, then
    keep the peaks that are not (approximately) sums of two smaller ones (simultaneous switches)."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    # one height per edge event: take the sample with the largest edge probability in each run
    on = edge > 0.5
    if on.sum() < 5:
        return 1, np.array([])
    idx = np.flatnonzero(on)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    h = np.array([abs(step[r].sum()) for r in runs])           # boxcar spreads one switch over 2 samples
    h = h[h > 0]
    if h.size < 5:
        return 1, np.array([])
    grid = np.linspace(0, np.quantile(h, 0.995) * 1.1, 400)
    dens = gaussian_kde(h, bw_method=0.08)(grid)
    pk, prop = find_peaks(dens, prominence=prom * dens.max())
    centres = [grid[k] for k in pk if np.mean(np.abs(h - grid[k]) < 0.1 * grid[k] + 1e-9) >= min_frac]
    prim = []
    for c in sorted(centres):
        if not any(abs(c - (a + b)) < 0.12 * c for i, a in enumerate(prim) for b in prim[i + 1:]):
            prim.append(c)
    return max(1, min(len(prim), max_k)), np.array(prim)


def lrt_hybrid(y, K, net, ar1=True, rng=None, n_free=3, ideal_starts=4):
    """CNN baseline + CNN starting values + a cheap search on the CNN-cleaned record
    (free-level inits and one idealiser run without its own baseline)."""
    from .cfhmm import free_level_inits, init_from_idealiser
    from .robustk import idealise_k
    rng = rng or np.random.default_rng(0)
    fitter = fit_ar1 if ar1 else fit
    beta, inits, *_ = cnn_init(y, K, net, rng=rng)
    z = y - beta
    cands = list(inits) + list(free_level_inits(z, K, rng, n_starts=n_free))
    try:
        cands.append(init_from_idealiser(idealise_k(z, K, rng=rng, starts=ideal_starts, baseline=False)))
    except Exception:
        pass
    best = None
    for ini in cands:
        f0 = fitter(z, K, False, ini)
        if best is None or f0["ll"] > best["ll"]:
            best = f0
    f0 = best
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    f1 = fitter(z, K, True, ini)
    f0b = fitter(z, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1.get("rho", 0.0), c=f1["c"], e=f1["e"]))
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    stat = 2 * max(f1["ll"] - f0["ll"], 0.0)
    return stat, f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def lrt_hybrid2(y, K, net, ar1=True, rng=None):
    """Light global search (1 seed per baseline option instead of 3) + CNN candidates (CNN baseline
    and CNN starting values).  The J = 0 likelihood picks the decomposition, as in R8."""
    from .cfhmm import best_decomposition
    rng = rng or np.random.default_rng(0)
    fitter = fit_ar1 if ar1 else fit
    beta_c, inits, *_ = cnn_init(y, K, net, rng=rng)
    extra = [dict(ini, beta=beta_c) for ini in inits]
    f, f0 = best_decomposition(y, K, rng, seeds=1, starts=4, ar1=ar1, extra_inits=extra)
    beta = f["beta"]
    z = y - beta
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    f1 = fitter(z, K, True, ini)
    f0b = fitter(z, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1.get("rho", 0.0), c=f1["c"], e=f1["e"]))
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def step_candidates(edge, step, lev, K, n_max=10):
    """Candidate single-trap step heights: KDE peaks of detected |step| (no sum filtering) plus
    pairwise differences between levels of the denoised trace (free-level HMM)."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    from .cfhmm import _free_hmm
    cands = []
    on = edge > 0.5
    if on.sum() >= 5:
        idx = np.flatnonzero(on)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
        h = np.array([abs(step[r].sum()) for r in runs])
        h = h[h > 0]
        if h.size >= 5:
            grid = np.linspace(0, np.quantile(h, 0.995) * 1.1, 400)
            dens = gaussian_kde(h, bw_method=0.06)(grid)
            pk, prop = find_peaks(dens, prominence=0.03 * dens.max())
            order = np.argsort(-dens[pk])
            cands += list(grid[pk[order]][:n_max])
    M = 2 ** K
    try:
        mu, w, *_ = _free_hmm(lev, M, np.quantile(lev, (np.arange(M) + 0.5) / M))
        mu = np.sort(mu[w > 0.005])
        cands += list(np.abs(np.subtract.outer(mu, mu))[np.triu_indices(len(mu), 1)])
    except ZeroDivisionError:
        pass
    cands = np.sort(np.array([c for c in cands if c > 0]))
    uniq = []
    for c in cands:                                    # merge within 5 %
        if not uniq or c > uniq[-1] * 1.05:
            uniq.append(c)
    return np.array(uniq)


def enumerate_decompositions(lev, cands, K, top=4, n_em=4):
    """Score every K-subset of candidate steps on the denoised trace with a product-HMM
    likelihood (levels fixed by the candidate, per-trap switching probabilities re-estimated
    by a few EM steps, offset re-centred).  Returns the best few (score, offset, steps)."""
    from itertools import combinations
    from .robustk import _fb, product_P
    B = bits(K)
    sub = lev[:: max(1, lev.size // 4096)].astype(float)
    cands = np.asarray(cands)
    cands = cands[cands > 0.1 * cands.max()]
    sig = max(float(np.median(np.abs(np.diff(sub))) / 0.6745 / np.sqrt(2)), 0.02 * float(np.ptp(sub)))
    S = 2 ** K
    pi0 = np.full(S, 1.0 / S)
    scored = []
    for comb in combinations(cands, K):
        a = np.array(sorted(comb, reverse=True))
        L = B @ a
        b = np.quantile(sub, 0.02)
        near = L[np.argmin(np.abs(sub[:, None] - b - L[None]), 1)]
        b = np.mean(sub - near)
        Ps = np.array([[[0.97, 0.03], [0.03, 0.97]]] * K)
        ll = -np.inf
        for _ in range(n_em):
            ll, N, g0, gy, gyy, first, _p = _fb(sub, product_P(Ps), pi0, b + L, sig, False)
            for k in range(K):
                n = np.zeros((2, 2))
                for x in range(S):
                    for xp in range(S):
                        n[B[x, k], B[xp, k]] += N[x, xp]
                Ps[k] = (n + 1e-3) / (n.sum(1, keepdims=True) + 2e-3)
            b += float((gy - (b + L) * g0).sum() / g0.sum())
        scored.append((-ll, b, a))
    scored.sort(key=lambda t: t[0])
    return scored[:top]


def lrt_cnn2(y, K, net, ar1=True, rng=None, top=4):
    """CNN-guided decomposition search: candidate steps from the step detector, enumerated and
    scored on the denoised trace, best few fitted with the exact likelihood (CNN baseline)."""
    rng = rng or np.random.default_rng(0)
    fitter = fit_ar1 if ar1 else fit
    beta, inits, lev, edge, step = cnn_init(y, K, net, rng=rng)
    z = y - beta
    sig = float(np.median(np.abs(np.diff(z - lev))) / 0.6745 / np.sqrt(2)) + 1e-9
    cands = step_candidates(edge, step, lev, K)
    cand_inits = list(inits)
    if len(cands) >= K:
        off = np.median(z - lev)
        for sse, b, a in enumerate_decompositions(lev, cands, K, top=top):
            cand_inits.append(dict(coef=np.r_[b + off, a], sig=sig, c=np.full(K, 0.02), e=np.full(K, 0.02)))
    best = None
    def _try(ini):
        try:
            f = fitter(z, K, False, ini)
            return f if np.isfinite(f["ll"]) else None
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
            return None
    for ini in cand_inits:
        f0 = _try(ini)
        if f0 is not None and (best is None or f0["ll"] > best["ll"]):
            best = f0
    if best is None:   # every CNN-based start failed numerically: fall back to the idealiser start
        from .robustk import idealise_k
        from .cfhmm import init_from_idealiser
        g = idealise_k(z, K, baseline=False, rng=rng)
        ini = init_from_idealiser(g); ini.setdefault("rho", 0.0)
        best = _try(ini)
    f0 = best
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    f1 = fitter(z, K, True, ini)
    f0b = fitter(z, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1.get("rho", 0.0), c=f1["c"], e=f1["e"]))
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def screen_on_record(z, cands, K, sig, top=3, n_iter=6, max_c=12):
    """Score every K-subset of candidate steps on the RECORD (minus baseline) with a few EM
    iterations of the independent product HMM (robustk._em: levels, noise, per-trap switching
    all re-estimated).  Returns the best few fitted (coef, sig, Ps) by log-likelihood."""
    from itertools import combinations
    from .robustk import _em
    cands = np.asarray(cands)
    cands = cands[cands > 0.1 * cands.max()]
    merged = []
    for c in np.sort(cands):
        if not merged or c > merged[-1] * 1.08:
            merged.append(c)
    cands = np.array(merged)
    if len(cands) > max_c:                                  # keep the largest and an even spread of the rest
        keep = np.unique(np.r_[np.linspace(0, len(cands) - 1, max_c).round().astype(int)])
        cands = cands[keep]
    lo = np.quantile(z, 0.01)
    out = []
    for comb in combinations(cands, K):
        a = np.array(sorted(comb, reverse=True))
        coef = np.r_[lo, a]
        Ps = np.array([[[0.97, 0.03], [0.03, 0.97]]] * K)
        try:
            coef, s_, Ps, ll = _em(z, coef, sig, Ps, n_iter=n_iter)
        except ZeroDivisionError:
            continue
        out.append((ll, coef, s_, Ps))
    out.sort(key=lambda t: -t[0])
    return out[:top]


def lrt_cnn3(y, K, net, ar1=True, rng=None, top=3, free=True, grid=True):
    """R9a final: CNN baseline + CNN step candidates, every K-subset screened on the record with a
    short EM, the best few fitted with the exact likelihood; J = 0 likelihood picks."""
    from .cfhmm import _rates_from_P
    rng = rng or np.random.default_rng(0)
    if HAMPEL:                      # isolated-spike filter before everything (R27: heavy-tailed noise)
        from .tnoise import hampel
        y = hampel(y)
    fitter = fit_ar1 if ar1 else fit
    beta, inits, lev, edge, step = cnn_init(y, K, net, rng=rng)
    z = y - beta
    sig = float(np.median(np.abs(np.diff(z - lev))) / 0.6745 / np.sqrt(2)) + 1e-9
    cand_inits = list(inits)
    cands = step_candidates(edge, step, lev, K)
    screened = []
    if len(cands) >= K:
        screened += screen_on_record(z, cands, K, sig, top=top)
    if grid and len(cands) >= 1:
        # rare traps: clear peaks for K-1 traps + a fine grid for the remaining one
        peaks = strong_peaks(edge, step, n=4)
        if len(peaks) >= K - 1:
            g = np.linspace(0.1, 1.05, 24) * max(peaks.max(), cands.max())
            screened += screen_combos(z, peaks, g, K, sig, top=top)
    screened.sort(key=lambda t: -t[0])
    for ll, coef, s_, Ps in screened[:top + 2]:
        ce = np.array([_rates_from_P(P) for P in Ps])
        cand_inits.append(dict(coef=coef, sig=s_, c=ce[:, 0], e=ce[:, 1]))
    if free:
        from .cfhmm import free_level_inits
        cand_inits += list(free_level_inits(z, K, rng, n_starts=3))
    best = None
    def _try(ini):
        try:
            f = fitter(z, K, False, ini)
            return f if np.isfinite(f["ll"]) else None
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
            return None
    for ini in cand_inits:
        f0 = _try(ini)
        if f0 is not None and (best is None or f0["ll"] > best["ll"]):
            best = f0
    if best is None:   # every CNN-based start failed numerically: fall back to the idealiser start
        from .robustk import idealise_k
        from .cfhmm import init_from_idealiser
        g = idealise_k(z, K, baseline=False, rng=rng)
        ini = init_from_idealiser(g); ini.setdefault("rho", 0.0)
        best = _try(ini)
    f0 = best
    ini = dict(coef=f0["coef"], sig=f0["sig"], rho=f0.get("rho", 0.0), c=f0["c"], e=f0["e"])
    try:
        f1 = fitter(z, K, True, ini)
    except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        # coupled fit failed numerically from the J = 0 optimum: restart it with small J
        f1 = fitter(z, K, True, dict(ini, J=np.full(K * (K - 1) // 2, 0.01)))
    try:
        f0b = fitter(z, K, False, dict(ini, coef=f1["coef"], sig=f1["sig"], rho=f1.get("rho", 0.0), c=f1["c"], e=f1["e"]))
        if np.isfinite(f0b["ll"]) and f0b["ll"] > f0["ll"]:
            f0 = f0b
    except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        pass
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1, beta, coupling_stat(bits(K)[f0["path"]])[0]


def strong_peaks(edge, step, n=4):
    """The n most populated step-height peaks (one height per edge event)."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde
    on = edge > 0.5
    if on.sum() < 5:
        return np.array([])
    idx = np.flatnonzero(on)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    h = np.array([abs(step[r].sum()) for r in runs])
    h = h[h > 0.05 * h.max()]
    if h.size < 5:
        return np.array([])
    grid = np.linspace(0, np.quantile(h, 0.995) * 1.1, 400)
    dens = gaussian_kde(h, bw_method=0.06)(grid)
    pk, _ = find_peaks(dens, prominence=0.05 * dens.max())
    return np.sort(grid[pk[np.argsort(-dens[pk])][:n]])


def screen_combos(z, peaks, grid, K, sig, top=3, n_iter=6):
    """(K-1)-subsets of the peaks + one value from the grid, screened by a short EM on the record."""
    from itertools import combinations
    from .robustk import _em
    lo = np.quantile(z, 0.01)
    out = []
    seen = set()
    for comb in combinations(peaks, K - 1):
        for gv in grid:
            if min(abs(gv - c) for c in comb) < 0.05 * max(comb):
                continue
            a = np.array(sorted(list(comb) + [gv], reverse=True))
            key = tuple(np.round(a, 3))
            if key in seen:
                continue
            seen.add(key)
            Ps = np.array([[[0.97, 0.03], [0.03, 0.97]]] * K)
            try:
                coef, s_, Ps, ll = _em(z, np.r_[lo, a], sig, Ps, n_iter=n_iter)
            except ZeroDivisionError:
                continue
            out.append((ll, coef, s_, Ps))
    out.sort(key=lambda t: -t[0])
    return out[:top]


def lrt_cnn3_ms(y, K, net, ar1=True, rng=None, n_start=3, **kw):
    """Multi-start R9a: lrt_cnn3 from n_start independent seeds; the J = 0 likelihood (the pipeline's
    own model-selection criterion) picks the run.  Returns lrt_cnn3's tuple plus the per-start (ll0, stat)."""
    rng = rng or np.random.default_rng(0)
    runs = []
    for _ in range(n_start):
        try:
            runs.append(lrt_cnn3(y, K, net, ar1=ar1, rng=np.random.default_rng(int(rng.integers(1 << 31))), **kw))
        except Exception:
            pass
    best = max(runs, key=lambda r: r[1]["ll"])
    if len(runs) < 2:
        return best
    from .cfhmm import pool_runs, bits as _bits
    from .robustk import coupling_stat
    yy = y
    if HAMPEL:
        from .tnoise import hampel
        yy = hampel(y)
    f0, f1, stat = pool_runs(yy - best[3], K, [(r[1], r[2]) for r in runs], ar1)
    return (stat, f0, f1, best[3], coupling_stat(_bits(K)[f0["path"]])[0]) + tuple(best[5:])
