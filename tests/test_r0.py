"""R0 unit tests for the coupled-trap RTN simulator.

T1 J = 0, no drive: trap 0 is bit-identical with 1 or 3 traps present
T2 stationary law = Boltzmann exp(-E) (3 traps, couplings), and = null vector of Q
T3 detailed balance: forward / backward transition counts agree; flux = pi q
T4 conditional rates: captures of trap 0 per time with trap 1 in state b = c0 exp(-phi J b)
T5 shared drive with J = 0: joint law matches the generator, traps become correlated
T6 readout: box average exact, point sample exact, white / flicker / drift variances
T7 the stationary law does not depend on the Bronsted splitting phi
"""
import copy

import numpy as np

from rtnsim import RTNConfig, simulate, simulate_kinetics, generator, stationary, states, energy
from rtnsim.simulate import readout


def occupancy_time_weighted(t_ev, s_ev, u_ev, t_end, K, drive=False):
    idx = (s_ev * (1 << np.arange(K))).sum(1) + (u_ev * (1 << K) if drive else 0)
    dur = np.diff(np.append(t_ev, t_end))
    n = 2 ** K * (2 if drive else 1)
    return np.bincount(idx, weights=dur, minlength=n) / t_end, idx, dur


def cfg3(J=None, seed=0):
    cfg = RTNConfig(seed=seed)
    cfg.traps.tau_c = [1e-3, 2e-3, 0.5e-3]
    cfg.traps.tau_e = [1e-3, 0.7e-3, 1.5e-3]
    cfg.traps.amp = [1.0, 0.6, 0.3]
    cfg.coupling.J = J
    return cfg


# --------------------------------------------------------------------------- T1
def test_t1_independent_traps_bitwise():
    one = RTNConfig(seed=4)
    one.traps.tau_c, one.traps.tau_e, one.traps.amp = [1e-3], [2e-3], [1.0]
    three = cfg3(seed=4)
    three.traps.tau_c[0], three.traps.tau_e[0] = 1e-3, 2e-3
    t1, s1, _ = simulate_kinetics(one, 2.0)
    t3, s3, _ = simulate_kinetics(three, 2.0)
    flips1 = t1[1:][np.diff(s1[:, 0]) != 0]
    flips3 = t3[1:][np.diff(s3[:, 0]) != 0]
    assert flips1.size > 500
    assert np.array_equal(flips1, flips3)
    assert s1[0, 0] == s3[0, 0]


# --------------------------------------------------------------------------- T2
def test_t2_boltzmann_three_traps():
    J = [[0, 1.5, -1.0], [1.5, 0, 0.7], [-1.0, 0.7, 0]]
    cfg = cfg3(J, seed=1)
    T = 60.0
    t_ev, s_ev, u_ev = simulate_kinetics(cfg, T)
    occ, _, _ = occupancy_time_weighted(t_ev, s_ev, u_ev, T, 3)
    S = states(cfg)
    w = np.exp([-energy(r[:3], 0, cfg) for r in S])
    pB = w / w.sum()
    pQ = stationary(generator(cfg))
    np.testing.assert_allclose(pQ, pB, atol=1e-12)
    np.testing.assert_allclose(occ, pB, atol=0.006)


# --------------------------------------------------------------------------- T3
def test_t3_detailed_balance_and_flux():
    cfg = cfg3([[0, 2.0, 0], [2.0, 0, -1.0], [0, -1.0, 0]], seed=2)
    T = 40.0
    t_ev, s_ev, u_ev = simulate_kinetics(cfg, T)
    _, idx, _ = occupancy_time_weighted(t_ev, s_ev, u_ev, T, 3)
    Q = generator(cfg)
    pi = stationary(Q)
    n = np.zeros((8, 8))
    np.add.at(n, (idx[:-1], idx[1:]), 1)
    for x in range(8):
        for y in range(x + 1, 8):
            if Q[x, y] > 0:
                expect = pi[x] * Q[x, y] * T
                se = np.sqrt(expect)
                assert abs(n[x, y] - expect) < 4.5 * se
                assert abs(n[x, y] - n[y, x]) < 4.5 * np.sqrt(2 * expect)


# --------------------------------------------------------------------------- T4
def test_t4_conditional_capture_rates():
    J12, phi = 2.0, 0.3
    cfg = RTNConfig(seed=3)
    cfg.coupling.J, cfg.coupling.phi = [[0, J12], [J12, 0]], phi
    T = 60.0
    t_ev, s_ev, _ = simulate_kinetics(cfg, T)
    dur = np.diff(np.append(t_ev, T))
    c0 = 1.0 / cfg.traps.tau_c[0]
    for b in (0, 1):
        in_state = (s_ev[:, 0] == 0) & (s_ev[:, 1] == b)
        time_b = dur[in_state].sum()
        captures = np.sum(in_state[:-1] & (s_ev[1:, 0] == 1))
        rate = captures / time_b
        expect = c0 * np.exp(-phi * J12 * b)
        assert abs(rate - expect) < 4 * expect / np.sqrt(captures)


# --------------------------------------------------------------------------- T5
def test_t5_shared_drive():
    cfg = RTNConfig(seed=5)
    cfg.drive.enabled = True
    cfg.drive.tau_off = cfg.drive.tau_on = 5e-3
    cfg.drive.gamma = [2.0, 2.0]
    T = 60.0
    t_ev, s_ev, u_ev = simulate_kinetics(cfg, T)
    occ, _, _ = occupancy_time_weighted(t_ev, s_ev, u_ev, T, 2, drive=True)
    pQ = stationary(generator(cfg))
    np.testing.assert_allclose(occ, pQ, atol=0.008)
    # traps correlate through the drive although J = 0
    p = pQ.reshape(2, 2, 2)                # [u, s1, s0]
    p01 = p.sum(axis=0)                    # [s1, s0]
    cov = p01[1, 1] - p01[1, :].sum() * p01[:, 1].sum()
    assert cov > 0.01


# --------------------------------------------------------------------------- T6
def test_t6_readout():
    cfg = RTNConfig(seed=6, n_samples=4000)
    cfg.coupling.B = [[0, 0.25], [0, 0]]
    cfg.readout.integrate = 0.7
    cfg.readout.sigma = 0.0
    t_ev, s_ev, _ = simulate_kinetics(cfg, cfg.duration)
    t, s_samp, clean, sig, _, _ = readout(cfg, t_ev, s_ev)
    A = np.array(cfg.traps.amp)
    lev = s_ev @ A + 0.25 * s_ev[:, 0] * s_ev[:, 1]
    # brute-force box average on a fine grid for a few samples
    w = 0.7 * cfg.readout.dt
    for k in (10, 500, 2999):
        grid = np.linspace(t[k] - w, t[k], 20001)
        vals = lev[np.searchsorted(t_ev, grid, side="right") - 1]
        assert abs(np.mean(vals) - clean[k]) < 2e-3
    # point sampling equals the level at the sample instant
    cfg2 = copy.deepcopy(cfg)
    cfg2.readout.integrate = 0.0
    _, s2, clean2, _, _, _ = readout(cfg2, t_ev, s_ev)
    np.testing.assert_allclose(clean2, s2 @ A + 0.25 * s2[:, 0] * s2[:, 1])
    # noise components
    cfg3_ = RTNConfig(seed=7, n_samples=200000)
    cfg3_.readout.sigma, cfg3_.readout.flicker_rms, cfg3_.readout.drift_sd = 0.3, 0.2, 0.5
    cfg3_.readout.flicker_fmin = 100.0
    tr = simulate(cfg3_)
    white = tr.signal - tr.clean - tr.flicker - tr.drift
    np.testing.assert_allclose(white.std(), 0.3, rtol=0.01)
    np.testing.assert_allclose(tr.flicker.std(), 0.2, rtol=0.15)
    inc = np.diff(tr.drift)
    np.testing.assert_allclose(inc.std(), 0.5 * np.sqrt(cfg3_.readout.dt), rtol=0.01)


# --------------------------------------------------------------------------- T7
def test_t7_stationary_independent_of_phi():
    ps = []
    for phi in (0.0, 0.5, 1.0):
        cfg = cfg3([[0, 1.2, 0.4], [1.2, 0, -0.8], [0.4, -0.8, 0]])
        cfg.coupling.phi = phi
        ps.append(stationary(generator(cfg)))
    np.testing.assert_allclose(ps[0], ps[1], atol=1e-12)
    np.testing.assert_allclose(ps[0], ps[2], atol=1e-12)
