"""Exact Markov description: rates, state enumeration, generator matrix, stationary law.

A joint state is (s_1..s_K, u); index = sum_i s_i 2^i + u 2^K (u only if the drive is on).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from .config import RTNConfig


def base_rates(cfg: RTNConfig):
    c = 1.0 / np.asarray(cfg.traps.tau_c, dtype=float)
    e = 1.0 / np.asarray(cfg.traps.tau_e, dtype=float)
    return c, e


def flip_rates(s: np.ndarray, u: int, c: np.ndarray, e: np.ndarray, J: np.ndarray,
               gamma: np.ndarray, phi: float) -> np.ndarray:
    """Rate at which each trap flips, given the full state (s, u)."""
    h = J @ s + u * gamma
    cap = c * np.exp(-phi * h)
    emi = e * np.exp((1.0 - phi) * h)
    return np.where(s == 0, cap, emi)


def energy(s: np.ndarray, u: int, cfg: RTNConfig) -> float:
    c, e = base_rates(cfg)
    eps = np.log(e / c)
    J = cfg.J_matrix()
    g = np.asarray(cfg.drive.gamma, dtype=float) if cfg.drive.enabled else np.zeros_like(eps)
    return float(eps @ s + 0.5 * s @ J @ s + u * (g @ s))


def states(cfg: RTNConfig) -> np.ndarray:
    """(n_states, K + 1) array of (s_1..s_K, u)."""
    K = cfg.traps.K
    nu = 2 if cfg.drive.enabled else 1
    out = []
    for u in range(nu):
        for idx in range(2 ** K):
            out.append([(idx >> i) & 1 for i in range(K)] + [u])
    return np.array(out, dtype=int)


def state_index(s: np.ndarray, u: int, K: int) -> int:
    return int(np.sum(s * (1 << np.arange(K)))) + u * (1 << K)


def generator(cfg: RTNConfig) -> np.ndarray:
    K = cfg.traps.K
    c, e = base_rates(cfg)
    J = cfg.J_matrix()
    gamma = np.asarray(cfg.drive.gamma, dtype=float) if cfg.drive.enabled else np.zeros(K)
    S = states(cfg)
    n = len(S)
    Q = np.zeros((n, n))
    for a, row in enumerate(S):
        s, u = row[:K], row[K]
        r = flip_rates(s, u, c, e, J, gamma, cfg.coupling.phi)
        for i in range(K):
            s2 = s.copy()
            s2[i] ^= 1
            Q[a, state_index(s2, u, K)] += r[i]
        if cfg.drive.enabled:
            ku = 1.0 / (cfg.drive.tau_off if u == 0 else cfg.drive.tau_on)
            Q[a, state_index(s, 1 - u, K)] += ku
    Q[np.diag_indices(n)] = -Q.sum(axis=1)
    return Q


def stationary(Q: np.ndarray) -> np.ndarray:
    ns = null_space(Q.T)
    p = np.abs(ns[:, 0])
    return p / p.sum()


def mean_signal(cfg: RTNConfig, S: np.ndarray | None = None) -> np.ndarray:
    """Noise-free readout level of every joint state."""
    if S is None:
        S = states(cfg)
    K = cfg.traps.K
    s = S[:, :K].astype(float)
    A = np.asarray(cfg.traps.amp, dtype=float)
    B = cfg.B_matrix()
    return cfg.readout.baseline + s @ A + np.einsum("ni,ij,nj->n", s, B, s)
