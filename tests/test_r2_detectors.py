"""R2 detector tests."""
import numpy as np

from rtnsim import RTNConfig, mean_signal
from rtnsim import theory as th
from rtnsim import detectors as dd


def base(J, sigma, dt=0.1):
    c = RTNConfig()
    c.traps.tau_c, c.traps.tau_e, c.traps.amp = [1.0, 1.0], [1.0, 1.0], [1.0, 0.6]
    c.coupling.J = [[0, J], [J, 0]]
    c.readout.sigma, c.readout.dt = sigma, dt
    return c


def test_numba_forward_backward_matches_numpy():
    cfg = base(1.0, 0.5)
    y = th.sample_readout(cfg, 20, 300, np.random.default_rng(0))
    B = 20
    P = np.repeat(th.transition_matrix(cfg)[None], B, 0)
    mu = np.repeat(mean_signal(cfg)[None], B, 0)
    a = dd.forward_backward(y, P, np.full((B, 4), 0.25), mu, np.full(B, 0.5), True)
    b = dd.forward_backward_numpy(y, P, np.full((B, 4), 0.25), mu, np.full(B, 0.5), True)
    for u, v in zip(a, b):
        np.testing.assert_allclose(np.asarray(u, float), np.asarray(v, float), atol=1e-9)
    # and the oracle log-likelihood agrees with the theory module
    np.testing.assert_allclose(dd.forward_backward(y, P, np.repeat(th.stationary(th.generator(cfg))[None], B, 0),
                                                   mu, np.full(B, 0.5))[0], th.hmm_loglik(y, cfg), atol=1e-8)


def test_shared_levels_recovered_on_coupled_data():
    cfg = base(2.0, 0.5)
    y = th.sample_readout(cfg, 120, 2000, np.random.default_rng(1))
    mu, sig = dd.fit_shared_levels(y)
    np.testing.assert_allclose(mu, mean_signal(cfg), atol=0.1)
    np.testing.assert_allclose(sig, 0.5, rtol=0.03)


def test_glrt_null_is_chi2_like_and_power_on_coupling():
    cfg1 = base(2.0, 0.25)
    _, cfg0 = th.coupling_information(cfg1)
    rng = np.random.default_rng(2)
    s0 = dd.detector_scores(th.sample_readout(cfg0, 200, 1500, rng))["glrt"]
    s1 = dd.detector_scores(th.sample_readout(cfg1, 100, 1500, rng))["glrt"]
    # 4 extra transition numbers -> statistic ~ chi2_4 / 2 under the null (mean 2)
    assert 1.2 < s0.mean() < 3.0
    assert np.all(s0 >= -1e-9)
    assert np.mean(s1 > np.quantile(s0, 0.99)) > 0.9
