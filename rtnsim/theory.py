"""R1 theory: how much data does it take to tell coupled traps from independent ones?

Hypotheses
----------
H1: coupled traps, generator Q1 (interaction J).
H0: the *closest independent model* Q0* = argmin over product chains of the
    KL rate D(P1 || P0).  For continuous-time chains with single-trap flips the
    minimiser is closed-form: each independent rate is the H1 rate averaged over
    the stationary law conditional on that trap's own state,

        c_i* = E_pi1[ c_i'(s) | s_i = 0 ],     e_i* = E_pi1[ e_i'(s) | s_i = 1 ].

    Q0* matches every trap's occupancy and switching flux, so an analysis that
    fits independent traps (as current RTN tools do) cannot reject it from
    one-trap statistics.  D* = D(P1 || P0*) is the coupling information that
    remains when the independent model's parameters are unknown.

Information hierarchy (data-processing inequality)
--------------------------------------------------
    continuous noise-free path  >=  sampled noise-free states  >=  sampled noisy readout
so  D* = KL rate of the continuous path  bounds every detector from above, and
T_lower = ln(1/alpha) / D*  (Stein) is the asymptotic minimum observation time
for false-positive rate alpha at fixed sensitivity.  The HMM oracle (exact
forward algorithm on the noisy point-sampled readout, parameters known) gives
the achievable value; the gap is the price of noise and finite bandwidth.

Continuous-time KL rate between Markov jump processes on the same state space:
    D(P1||P0) = sum_x pi1(x) sum_{y != x} [ q1 ln(q1/q0) - q1 + q0 ].
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
from scipy.linalg import expm

from .config import RTNConfig
from .markov import generator, stationary, states, mean_signal


# --------------------------------------------------------------------------- #
# Continuous-time information
# --------------------------------------------------------------------------- #
def kl_rate(Q1: np.ndarray, Q0: np.ndarray, pi1: np.ndarray | None = None) -> float:
    if pi1 is None:
        pi1 = stationary(Q1)
    n = len(Q1)
    off = ~np.eye(n, dtype=bool)
    q1, q0 = Q1[off].reshape(n, n - 1), Q0[off].reshape(n, n - 1)
    if np.any((q1 > 0) & (q0 <= 0)):
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(q1 > 0, q1 * np.log(q1 / q0), 0.0) - q1 + q0
    return float(pi1 @ term.sum(axis=1))


def closest_independent(cfg: RTNConfig) -> RTNConfig:
    """I-projection of the coupled chain onto independent traps (no drive)."""
    if cfg.drive.enabled:
        raise NotImplementedError("projection is defined for the undriven chain")
    K = cfg.traps.K
    Q = generator(cfg)
    pi = stationary(Q)
    S = states(cfg)[:, :K]
    idx = (S * (1 << np.arange(K))).sum(1)
    c_star, e_star = np.empty(K), np.empty(K)
    for i in range(K):
        flip = idx ^ (1 << i)
        rate = Q[np.arange(len(S)), flip]
        empty, full = S[:, i] == 0, S[:, i] == 1
        c_star[i] = pi[empty] @ rate[empty] / pi[empty].sum()
        e_star[i] = pi[full] @ rate[full] / pi[full].sum()
    out = copy.deepcopy(cfg)
    out.traps.tau_c = (1.0 / c_star).tolist()
    out.traps.tau_e = (1.0 / e_star).tolist()
    out.coupling.J = None
    return out


def coupling_information(cfg: RTNConfig) -> Tuple[float, RTNConfig]:
    """D* (per unit time) and the closest independent model."""
    cfg0 = closest_independent(cfg)
    Q1, Q0 = generator(cfg), generator(cfg0)
    return kl_rate(Q1, Q0), cfg0


def stein_time(D: float, alpha: float) -> float:
    return np.log(1.0 / alpha) / D


# --------------------------------------------------------------------------- #
# Path log-likelihood ratio (continuous, noise-free observation)
# --------------------------------------------------------------------------- #
def path_llr(t_ev: np.ndarray, s_ev: np.ndarray, t_end: float, Q1: np.ndarray, Q0: np.ndarray) -> float:
    K = s_ev.shape[1]
    idx = (s_ev * (1 << np.arange(K))).sum(1)
    dur = np.diff(np.append(t_ev, t_end))
    out1, out0 = -np.diag(Q1), -np.diag(Q0)
    jumps = np.log(Q1[idx[:-1], idx[1:]] / Q0[idx[:-1], idx[1:]]).sum()
    return float(jumps - np.sum((out1[idx] - out0[idx]) * dur))


# --------------------------------------------------------------------------- #
# Point-sampled chain and HMM oracle
# --------------------------------------------------------------------------- #
def transition_matrix(cfg: RTNConfig) -> np.ndarray:
    return expm(generator(cfg) * cfg.readout.dt)


def sample_states(cfg: RTNConfig, n_traces: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Exact point-sampled state indices (n_traces, n_steps), stationary start."""
    P = transition_matrix(cfg)
    C = np.cumsum(P, axis=1)
    C[:, -1] = 1.0
    pi = stationary(generator(cfg))
    x = rng.choice(len(pi), size=n_traces, p=pi)
    out = np.empty((n_traces, n_steps), dtype=np.int64)
    u = rng.random((n_steps, n_traces))
    for k in range(n_steps):
        x = (u[k][:, None] > C[x]).sum(axis=1)
        out[:, k] = x
    return out


def sample_readout(cfg: RTNConfig, n_traces: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    x = sample_states(cfg, n_traces, n_steps, rng)
    mu = mean_signal(cfg)
    return mu[x] + cfg.readout.sigma * rng.standard_normal(x.shape)


def hmm_loglik(y: np.ndarray, cfg: RTNConfig) -> np.ndarray:
    """Exact forward log-likelihood of point-sampled noisy readout, batch over rows of y."""
    P = transition_matrix(cfg)
    pi = stationary(generator(cfg))
    mu = mean_signal(cfg)
    s = cfg.readout.sigma
    B, T = y.shape
    logc = -0.5 * np.log(2 * np.pi * s * s)

    def emis(k):
        return np.exp(-0.5 * ((y[:, k][:, None] - mu[None, :]) / s) ** 2)

    a = pi[None, :] * emis(0)
    norm = a.sum(1)
    ll = np.log(norm)
    a /= norm[:, None]
    for k in range(1, T):
        a = (a @ P) * emis(k)
        norm = a.sum(1)
        ll += np.log(norm)
        a /= norm[:, None]
    return ll + T * logc


def hmm_loglik_checkpoints(y: np.ndarray, cfg: RTNConfig, checkpoints) -> np.ndarray:
    """Forward log-likelihood of the first n samples for every n in checkpoints: (len(checkpoints), B)."""
    P = transition_matrix(cfg)
    pi = stationary(generator(cfg))
    mu = mean_signal(cfg)
    s = cfg.readout.sigma
    B, T = y.shape
    logc = -0.5 * np.log(2 * np.pi * s * s)
    cps = sorted(set(int(c) for c in checkpoints if 1 <= c <= T))
    out, j = np.empty((len(cps), B)), 0
    a = pi[None, :] * np.exp(-0.5 * ((y[:, 0][:, None] - mu) / s) ** 2)
    nrm = a.sum(1)
    ll = np.log(nrm)
    a /= nrm[:, None]
    for k in range(1, T + 1):
        if j < len(cps) and k == cps[j]:
            out[j] = ll + k * logc
            j += 1
            if j == len(cps):
                break
        if k == T:
            break
        a = (a @ P) * np.exp(-0.5 * ((y[:, k][:, None] - mu) / s) ** 2)
        nrm = a.sum(1)
        ll += np.log(nrm)
        a /= nrm[:, None]
    return out, cps


def streaming_llr(cfg_true: RTNConfig, cfg1: RTNConfig, cfg0: RTNConfig, n_traces: int, n_steps: int,
                  checkpoints, rng: np.random.Generator) -> Tuple[np.ndarray, list]:
    """Simulate point-sampled noisy readout under cfg_true and accumulate the exact
    HMM log-likelihood ratio log p1 - log p0 on the fly (memory O(n_traces)).

    Returns LLR at each checkpoint (len(cps), n_traces) and the checkpoint list.
    All three configs must share the readout (dt, sigma, levels).
    """
    Pt = transition_matrix(cfg_true)
    Ct = np.cumsum(Pt, axis=1)
    Ct[:, -1] = 1.0
    pit = stationary(generator(cfg_true))
    mu = mean_signal(cfg_true)
    s = cfg_true.readout.sigma
    P1, P0 = transition_matrix(cfg1), transition_matrix(cfg0)
    pi1, pi0 = stationary(generator(cfg1)), stationary(generator(cfg0))
    cps = sorted(set(int(c) for c in checkpoints if 1 <= c <= n_steps))
    out, j = np.empty((len(cps), n_traces)), 0

    x = rng.choice(len(pit), size=n_traces, p=pit)
    y = mu[x] + s * rng.standard_normal(n_traces)
    em = np.exp(-0.5 * ((y[:, None] - mu[None, :]) / s) ** 2)
    a1, a0 = pi1[None, :] * em, pi0[None, :] * em
    n1, n0 = a1.sum(1), a0.sum(1)
    llr = np.log(n1) - np.log(n0)
    a1 /= n1[:, None]
    a0 /= n0[:, None]
    for k in range(1, n_steps + 1):
        if j < len(cps) and k == cps[j]:
            out[j] = llr
            j += 1
            if j == len(cps):
                break
        x = (rng.random(n_traces)[:, None] > Ct[x]).sum(axis=1)
        y = mu[x] + s * rng.standard_normal(n_traces)
        em = np.exp(-0.5 * ((y[:, None] - mu[None, :]) / s) ** 2)
        a1 = (a1 @ P1) * em
        a0 = (a0 @ P0) * em
        n1, n0 = a1.sum(1), a0.sum(1)
        llr += np.log(n1) - np.log(n0)
        a1 /= n1[:, None]
        a0 /= n0[:, None]
    return out, cps


def small_coupling_D(cfg: RTNConfig) -> float:
    """Closed-form leading term of D* for two traps (sympy-derived):

        D* ~ (J^2 / 2) [phi^2 + (1 - phi)^2] p1 (1 - p1) p2 (1 - p2) (k1 + k2),

    p_i = c_i / (c_i + e_i) occupancy, k_i = c_i + e_i switching rate.
    """
    if cfg.traps.K != 2:
        raise ValueError("two traps only")
    c = 1.0 / np.asarray(cfg.traps.tau_c)
    e = 1.0 / np.asarray(cfg.traps.tau_e)
    p, k = c / (c + e), c + e
    J = cfg.J_matrix()[0, 1]
    phi = cfg.coupling.phi
    return 0.5 * J ** 2 * (phi ** 2 + (1 - phi) ** 2) * np.prod(p * (1 - p)) * k.sum()


def sampled_kl_rate(cfg1: RTNConfig, cfg0: RTNConfig) -> float:
    """KL rate of the noise-free point-sampled chains (pairs of consecutive states)."""
    P1, P0 = transition_matrix(cfg1), transition_matrix(cfg0)
    pi1 = stationary(generator(cfg1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P1 > 0, P1 * np.log(P1 / P0), 0.0)
    return float(pi1 @ t.sum(axis=1)) / cfg1.readout.dt
