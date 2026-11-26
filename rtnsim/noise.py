"""Vectorised coloured-noise confounds for batches of traces (B, T)."""
from __future__ import annotations

import numpy as np


def add_flicker(y, rms, fmin, fmax, n_terms, dt, rng):
    """Sum of n_terms Lorentzians (exact AR(1)), log-spaced corners, equal power: ~1/f between fmin and fmax."""
    B, T = y.shape
    out = y.copy()
    var_each = rms ** 2 / n_terms
    for f0 in np.geomspace(fmin, fmax, n_terms):
        a = np.exp(-2 * np.pi * f0 * dt)
        x = rng.standard_normal(B) * np.sqrt(var_each)
        z = rng.standard_normal((T, B)) * np.sqrt(var_each * (1 - a * a))
        for k in range(T):
            x = a * x + z[k]
            out[:, k] += x
    return out


def add_drift(y, sd_per_sqrt_time, dt, rng):
    B, T = y.shape
    return y + np.cumsum(rng.standard_normal((B, T)), axis=1) * sd_per_sqrt_time * np.sqrt(dt)
