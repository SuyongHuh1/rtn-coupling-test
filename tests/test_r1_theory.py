"""R1 tests: the information quantities and oracle machinery are exact."""
import copy
import itertools

import numpy as np

from rtnsim import RTNConfig, generator, stationary, simulate, mean_signal
from rtnsim import theory as th


def base(J=1.0, phi=0.5, tau=(1.0, 1.0, 1.0, 1.0), sigma=0.5, dt=0.1):
    c = RTNConfig()
    c.traps.tau_c, c.traps.tau_e = [tau[0], tau[2]], [tau[1], tau[3]]
    c.coupling.J, c.coupling.phi = [[0, J], [J, 0]], phi
    c.readout.sigma, c.readout.dt = sigma, dt
    return c


def test_projection_is_the_kl_minimiser():
    cfg = base(1.5, 0.3, tau=(1.0, 2.0, 0.5, 0.8))
    D, cfg0 = th.coupling_information(cfg)
    Q1 = generator(cfg)
    rng = np.random.default_rng(0)
    for _ in range(20):
        c2 = copy.deepcopy(cfg0)
        c2.traps.tau_c = list(np.array(cfg0.traps.tau_c) * np.exp(rng.normal(0, 0.03, 2)))
        c2.traps.tau_e = list(np.array(cfg0.traps.tau_e) * np.exp(rng.normal(0, 0.03, 2)))
        assert th.kl_rate(Q1, generator(c2)) > D


def test_projection_matches_occupancy_and_flux():
    cfg = base(2.0, 0.5, tau=(1.0, 2.0, 0.5, 0.8))
    _, cfg0 = th.coupling_information(cfg)
    p1 = stationary(generator(cfg)).reshape(2, 2)    # [s2, s1]
    p0 = stationary(generator(cfg0)).reshape(2, 2)
    np.testing.assert_allclose(p1.sum(0), p0.sum(0), atol=1e-12)   # trap 1 occupancy
    np.testing.assert_allclose(p1.sum(1), p0.sum(1), atol=1e-12)   # trap 2 occupancy


def test_zero_coupling_has_zero_information():
    D, cfg0 = th.coupling_information(base(0.0))
    assert abs(D) < 1e-14
    np.testing.assert_allclose(cfg0.traps.tau_c, [1.0, 1.0])


def test_small_coupling_closed_form():
    for tau in [(1.0, 1.0, 1.0, 1.0), (0.3, 2.0, 5.0, 0.7)]:
        for phi in (0.0, 0.3, 0.5, 1.0):
            cfg = base(0.01, phi, tau=tau)
            D, _ = th.coupling_information(cfg)
            np.testing.assert_allclose(D, th.small_coupling_D(cfg), rtol=0.01)  # O(J) correction


def test_sampling_loses_information_and_converges():
    cfg = base(1.0)
    D, cfg0 = th.coupling_information(cfg)
    prev = 0.0
    for dt in (1.0, 0.3, 0.1, 0.01):
        a, b = copy.deepcopy(cfg), copy.deepcopy(cfg0)
        a.readout.dt = b.readout.dt = dt
        r = th.sampled_kl_rate(a, b)
        assert prev <= r <= D * (1 + 1e-9)
        prev = r
    assert prev > 0.99 * D


def test_fast_sampler_matches_exact_simulator():
    cfg = base(2.0, dt=0.2)
    cfg.n_samples = 60000
    tr = simulate(cfg)
    idx = (tr.s_sampled * np.array([1, 2])).sum(1)
    n_exact = np.zeros((4, 4))
    np.add.at(n_exact, (idx[:-1], idx[1:]), 1)
    x = th.sample_states(cfg, 1, 60000, np.random.default_rng(1))[0]
    n_fast = np.zeros((4, 4))
    np.add.at(n_fast, (x[:-1], x[1:]), 1)
    P = th.transition_matrix(cfg)
    for n in (n_exact, n_fast):
        emp = n / n.sum(1, keepdims=True)
        se = np.sqrt(P * (1 - P) / n.sum(1, keepdims=True))
        assert np.all(np.abs(emp - P) < 5 * se + 1e-3)


def test_hmm_forward_equals_brute_force():
    cfg = base(1.0, sigma=0.7, dt=0.4)
    rng = np.random.default_rng(2)
    y = th.sample_readout(cfg, 3, 4, rng)
    P, pi, mu, s = th.transition_matrix(cfg), stationary(generator(cfg)), mean_signal(cfg), 0.7
    for b in range(3):
        tot = 0.0
        for path in itertools.product(range(4), repeat=4):
            p = pi[path[0]]
            for k in range(1, 4):
                p *= P[path[k - 1], path[k]]
            for k in range(4):
                p *= np.exp(-0.5 * ((y[b, k] - mu[path[k]]) / s) ** 2) / np.sqrt(2 * np.pi * s * s)
            tot += p
        np.testing.assert_allclose(th.hmm_loglik(y[b:b + 1], cfg)[0], np.log(tot), rtol=1e-10)
        ll, _ = th.hmm_loglik_checkpoints(y[b:b + 1], cfg, [4])
        np.testing.assert_allclose(ll[0, 0], np.log(tot), rtol=1e-10)


def test_streaming_llr_matches_stored_forward():
    cfg1 = base(1.0, sigma=0.6, dt=0.2)
    _, cfg0 = th.coupling_information(cfg1)
    l_stream, cps = th.streaming_llr(cfg1, cfg1, cfg0, 4000, 60, [60], np.random.default_rng(3))
    y = th.sample_readout(cfg1, 4000, 60, np.random.default_rng(4))
    l_store = th.hmm_loglik(y, cfg1) - th.hmm_loglik(y, cfg0)
    se = np.sqrt(l_stream[0].var() / 4000 + l_store.var() / 4000)
    assert abs(l_stream[0].mean() - l_store.mean()) < 4 * se
    np.testing.assert_allclose(l_stream[0].var(), l_store.var(), rtol=0.1)
