"""Exact stochastic simulation of coupled traps and the sampled readout.

Kinetics: Anderson's modified next-reaction method with one unit-rate Poisson
stream per channel (each trap, plus the drive).  A channel's internal clock is
only touched when its own propensity changes, so with J = 0 and no drive the
path of trap i depends only on trap i's stream: it is bit-identical whether or
not other traps exist (unit test).

Readout: exact box averages of the piecewise-constant noise-free signal (or
point samples), plus white noise, flicker (sum of Lorentzians, exact AR(1)
updates) and a random-walk drift, each from its own stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RTNConfig
from .markov import base_rates, flip_rates, generator, stationary, states, mean_signal

_P_CLOCK, _P_INIT = 0, 1
_G_INIT, _G_WHITE, _G_FLICK, _G_DRIFT, _G_DRIVE = 10, 11, 12, 13, 14
_EXP_BLOCK = 4096


def _rng(seed, *key) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


class _ExpStream:
    def __init__(self, g):
        self.g, self.buf, self.i = g, None, _EXP_BLOCK

    def next(self) -> float:
        if self.i == _EXP_BLOCK:
            self.buf, self.i = self.g.standard_exponential(_EXP_BLOCK), 0
        v = self.buf[self.i]
        self.i += 1
        return v


@dataclass
class RTNTrace:
    t_events: np.ndarray     # (E + 1,) time of each state change; t_events[0] = 0
    s_events: np.ndarray     # (E + 1, K) trap states from t_events[k] on
    u_events: np.ndarray     # (E + 1,) drive state from t_events[k] on
    t: np.ndarray            # (N,) sample times
    s_sampled: np.ndarray    # (N, K) trap states at the sample instants
    clean: np.ndarray        # (N,) noise-free readout (point or box average)
    signal: np.ndarray       # (N,) measured readout
    flicker: np.ndarray      # (N,)
    drift: np.ndarray        # (N,)


def _initial_state(cfg: RTNConfig):
    K = cfg.traps.K
    J = cfg.J_matrix()
    c, e = base_rates(cfg)
    if not np.any(J) and not cfg.drive.enabled:
        p = c / (c + e)
        s = np.array([int(_rng(cfg.seed, i, _P_INIT).random() < p[i]) for i in range(K)])
        return s, 0
    pi = stationary(generator(cfg))
    S = states(cfg)
    k = _rng(cfg.seed, _G_INIT).choice(len(S), p=pi)
    return S[k, :K].copy(), int(S[k, K])


def simulate_kinetics(cfg: RTNConfig, t_end: float):
    K = cfg.traps.K
    c, e = base_rates(cfg)
    J = cfg.J_matrix()
    phi = cfg.coupling.phi
    drive = cfg.drive.enabled
    gamma = np.asarray(cfg.drive.gamma, dtype=float) if drive else np.zeros(K)
    ku = (1.0 / cfg.drive.tau_off, 1.0 / cfg.drive.tau_on) if drive else (0.0, 0.0)

    s, u = _initial_state(cfg)
    nch = K + (1 if drive else 0)
    streams = [_ExpStream(_rng(cfg.seed, i, _P_CLOCK)) for i in range(K)]
    if drive:
        streams.append(_ExpStream(_rng(cfg.seed, _G_DRIVE)))

    def props(s, u):
        a = flip_rates(s, u, c, e, J, gamma, phi)
        return np.append(a, ku[u]) if drive else a

    a = props(s, u)
    T = np.zeros(nch)                         # internal clocks
    P = np.array([st.next() for st in streams])
    t_last = np.zeros(nch)                    # time each internal clock was last synced
    tau = t_last + (P - T) / a                # absolute next firing times

    t_ev, s_ev, u_ev = [0.0], [s.copy()], [u]
    while True:
        mu = int(np.argmin(tau))
        t = tau[mu]
        if t > t_end:
            break
        # fire channel mu
        T[mu], t_last[mu] = P[mu], t
        P[mu] += streams[mu].next()
        if mu < K:
            s[mu] ^= 1
        else:
            u = 1 - u
        a_new = props(s, u)
        changed = np.flatnonzero(a_new != a)
        for k in changed:
            if k != mu:
                T[k] += a[k] * (t - t_last[k])
                t_last[k] = t
        a = a_new
        for k in set(changed.tolist()) | {mu}:
            tau[k] = t_last[k] + (P[k] - T[k]) / a[k]
        t_ev.append(t)
        s_ev.append(s.copy())
        u_ev.append(u)
    return np.array(t_ev), np.array(s_ev, dtype=int), np.array(u_ev, dtype=int)


def _levels(cfg, s_ev):
    K = cfg.traps.K
    S = np.concatenate([s_ev, np.zeros((len(s_ev), 1), int)], axis=1)
    return mean_signal(cfg, S)


def readout(cfg: RTNConfig, t_ev, s_ev):
    r = cfg.readout
    N = cfg.n_samples
    t = (np.arange(N) + 1) * r.dt
    lev = _levels(cfg, s_ev)
    idx = np.searchsorted(t_ev, t, side="right") - 1
    s_sampled = s_ev[idx]
    if r.integrate <= 0:
        clean = lev[idx]
    else:
        # cumulative integral of the piecewise-constant level at event times
        cum = np.concatenate([[0.0], np.cumsum(lev[:-1] * np.diff(t_ev))])

        def integral(x):
            k = np.searchsorted(t_ev, x, side="right") - 1
            return cum[k] + lev[k] * (x - t_ev[k])

        w = r.integrate * r.dt
        clean = (integral(t) - integral(t - w)) / w

    white = _rng(cfg.seed, _G_WHITE).standard_normal(N) * r.sigma
    flick = np.zeros(N)
    if r.flicker_rms > 0:
        g = _rng(cfg.seed, _G_FLICK)
        fc = np.geomspace(r.flicker_fmin, r.flicker_fmax, r.flicker_terms)
        var_each = r.flicker_rms ** 2 / r.flicker_terms
        for f0 in fc:
            a = np.exp(-2 * np.pi * f0 * r.dt)
            x = np.empty(N)
            prev = g.standard_normal() * np.sqrt(var_each)
            z = g.standard_normal(N) * np.sqrt(var_each * (1 - a * a))
            for k in range(N):
                prev = a * prev + z[k]
                x[k] = prev
            flick += x
    drift = np.zeros(N)
    if r.drift_sd > 0:
        drift = np.cumsum(_rng(cfg.seed, _G_DRIFT).standard_normal(N)) * r.drift_sd * np.sqrt(r.dt)
    return t, s_sampled, clean, clean + white + flick + drift, flick, drift


def simulate(cfg: RTNConfig) -> RTNTrace:
    t_ev, s_ev, u_ev = simulate_kinetics(cfg, cfg.duration)
    t, s_sampled, clean, signal, flick, drift = readout(cfg, t_ev, s_ev)
    return RTNTrace(t_ev, s_ev, u_ev, t, s_sampled, clean, signal, flick, drift)
