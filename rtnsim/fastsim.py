"""Fast batched sampling of point-sampled noisy RTN with per-trace parameters (numba)."""
from __future__ import annotations

import numpy as np
import numba as nb

from .config import RTNConfig
from .markov import generator, stationary, mean_signal
from .theory import transition_matrix


@nb.njit(parallel=True, cache=True)
def _sample_kernel(C, mu, sig, x0, T, seeds):
    B, S, _ = C.shape
    y = np.empty((B, T), dtype=np.float32)
    for b in nb.prange(B):
        np.random.seed(seeds[b])
        x = x0[b]
        for k in range(T):
            if k > 0:
                u = np.random.random()
                nx = 0
                while nx < S - 1 and u > C[b, x, nx]:
                    nx += 1
                x = nx
            y[b, k] = mu[b, x] + sig[b] * np.random.standard_normal()
    return y


def sample_configs(cfgs, T, rng, reps=1):
    """One block of `reps` traces per config; returns (len(cfgs) * reps, T) float32."""
    S = 2 ** cfgs[0].traps.K * (2 if cfgs[0].drive.enabled else 1)
    n = len(cfgs) * reps
    C = np.empty((n, S, S))
    mu = np.empty((n, S))
    sig = np.empty(n)
    x0 = np.empty(n, dtype=np.int64)
    for i, cfg in enumerate(cfgs):
        P = transition_matrix(cfg)
        Ci = np.cumsum(P, axis=1)
        Ci[:, -1] = 1.0
        pi = stationary(generator(cfg))
        m = mean_signal(cfg)
        for r in range(reps):
            j = i * reps + r
            C[j], mu[j], sig[j] = Ci, m, cfg.readout.sigma
            x0[j] = rng.choice(S, p=pi)
    seeds = rng.integers(0, 2 ** 31 - 1, n)
    return _sample_kernel(C, mu, sig, x0, T, seeds)
