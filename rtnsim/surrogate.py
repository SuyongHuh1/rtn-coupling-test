"""Model-light coupling tests on idealised two-trap paths, with shift surrogates.

Statistic: conditional-rate log-likelihood ratio. Each trap's discrete-time switching
probabilities are allowed to depend on the other trap's current state (coupled) or not
(independent); the statistic is the gain in log likelihood, summed over both traps.

Surrogates: trap 2's path is circularly shifted relative to trap 1's.  Each trap's own
process (including any non-stationarity of its own) is preserved exactly.
  * global shifts (a large fraction of the record) destroy every cross-dependence:
    kinetic coupling AND correlation induced by a shared slow drive;
  * local shifts (a few dwell times) destroy instantaneous dependence (kinetic coupling)
    but keep a dependence that lives on timescales much longer than the shift (a slow
    shared drive).
So "significant vs global, not vs local" points to a slow common cause, and "significant vs
local" points to dependence on the timescale of the switching itself - the timescale-
separation condition of R3 turned into a test that needs no drive model.
"""
from __future__ import annotations

import numpy as np


def _ll_binary_transitions(cur, nxt, cond=None):
    """Log-likelihood of a 2-state chain's transitions, optionally conditioned on cond (0/1)."""
    if cond is None:
        cond = np.zeros_like(cur)
    idx = cond * 4 + cur * 2 + nxt
    n = np.bincount(idx, minlength=8).reshape(2, 2, 2).astype(float)   # [cond, cur, nxt]
    row = n.sum(2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(n > 0, n * np.log(n / row), 0.0)))


def cond_lrt(s1, s2):
    s1 = np.asarray(s1, np.int64)
    s2 = np.asarray(s2, np.int64)
    a0, a1, b0, b1 = s1[:-1], s1[1:], s2[:-1], s2[1:]
    g1 = _ll_binary_transitions(a0, a1, b0) - _ll_binary_transitions(a0, a1)
    g2 = _ll_binary_transitions(b0, b1, a0) - _ll_binary_transitions(b0, b1)
    return g1 + g2


def mean_dwell(s):
    s = np.asarray(s)
    n_sw = np.count_nonzero(np.diff(s))
    return len(s) / max(n_sw, 1)


def shift_test(s1, s2, rng, n_sur=199, local=(1.0, 4.0)):
    """Returns statistic, p_global, p_local (one-sided, add-one)."""
    T = len(s1)
    stat = cond_lrt(s1, s2)
    tau = max(mean_dwell(s1), mean_dwell(s2))
    glob = rng.integers(int(0.1 * T), int(0.9 * T), n_sur)
    lo, hi = max(1, int(local[0] * tau)), max(2, int(local[1] * tau))
    loc = rng.integers(lo, hi + 1, n_sur) * rng.choice([-1, 1], n_sur)
    sg = np.array([cond_lrt(s1, np.roll(s2, k)) for k in glob])
    sl = np.array([cond_lrt(s1, np.roll(s2, k)) for k in loc])
    return stat, float((1 + np.sum(sg >= stat)) / (n_sur + 1)), float((1 + np.sum(sl >= stat)) / (n_sur + 1))


def cond_lrt_multi(pairs):
    """Statistic summed over independent segments [(s1, s2), ...] (pooled counts per segment)."""
    return float(sum(cond_lrt(a, b) for a, b in pairs))


def shift_test_multi(pairs, rng, n_sur=199, local=(1.0, 4.0)):
    """Shift surrogates applied within each segment independently."""
    stat = cond_lrt_multi(pairs)
    tau = max(max(mean_dwell(a), mean_dwell(b)) for a, b in pairs)
    sg, sl = np.empty(n_sur), np.empty(n_sur)
    for j in range(n_sur):
        g = l = 0.0
        for a, b in pairs:
            T = len(a)
            kg = rng.integers(int(0.1 * T), int(0.9 * T))
            lo, hi = max(1, int(local[0] * tau)), max(2, int(local[1] * tau))
            kl = rng.integers(lo, hi + 1) * rng.choice([-1, 1])
            g += cond_lrt(a, np.roll(b, kg))
            l += cond_lrt(a, np.roll(b, kl))
        sg[j], sl[j] = g, l
    return stat, float((1 + np.sum(sg >= stat)) / (n_sur + 1)), float((1 + np.sum(sl >= stat)) / (n_sur + 1))
