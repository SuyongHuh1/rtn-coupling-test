"""R3 tests: causal network, confound-aware pairs, integer datapath, HLS C simulation."""
import shutil
import subprocess

import jax
import numpy as np
import pytest

from rtnsim import learned_causal as LC
from rtnsim import quant as Q
from rtnsim import hls_gen as H
from rtnsim import drive_hmm as DH
from rtnsim import theory as th
from rtnsim import RTNConfig


def test_network_is_causal():
    p = LC.init_params(jax.random.PRNGKey(0))
    y, _ = LC.make_dataset(2, 900, np.random.default_rng(0))
    x = LC.normalise(y)
    a = LC.features(p, x)
    b = LC.features(p, x.at[:, 700:, :].set(0.0))
    assert float(np.abs(a[:, :700] - b[:, :700]).max()) == 0.0


def test_confound_pairs_share_confound_and_are_balanced():
    rng = np.random.default_rng(1)
    y, lab = LC.make_dataset(20, 600, rng)
    assert y.shape == (40, 600) and lab.sum() == 20
    assert np.all(lab[0::2] == 1) and np.all(lab[1::2] == 0)


def test_integer_model_tracks_float_model():
    p = LC.init_params(jax.random.PRNGKey(2))
    y, _ = LC.make_dataset(30, 800, np.random.default_rng(2))
    q = Q.quantize(p, y)
    assert np.corrcoef(LC.scores(p, y), Q.scores_int(q, y))[0, 1] > 0.95


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_hls_c_simulation_is_bit_exact(tmp_path):
    p = LC.init_params(jax.random.PRNGKey(3))
    y, _ = LC.make_dataset(4, 500, np.random.default_rng(3))
    q = Q.quantize(p, y)
    d = H.emit(q, tmp_path)
    xq = Q.quantize_input(q, y)
    (d / "in.txt").write_text(f"{xq.shape[0]} {xq.shape[1]}\n" + " ".join(map(str, xq.ravel())))
    r = subprocess.run(f"g++ -O2 -std=c++17 -Wno-unknown-pragmas -o {d}/tb {d}/tb.cpp {d}/ccnn.cpp && "
                       f"{d}/tb {d}/in.txt {d}/out.txt", shell=True, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = np.loadtxt(d / "out.txt", dtype=np.int64)
    S1, S2 = Q.run_int(q, xq)
    assert np.array_equal(out[:, :32], S1) and np.array_equal(out[:, 32:], S2)


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")
def test_verilog_rtl_is_bit_exact(tmp_path):
    from rtnsim import rtl_gen as RG
    p = LC.init_params(jax.random.PRNGKey(6))
    y, _ = LC.make_dataset(1, 60, np.random.default_rng(6))
    q = Q.quantize(p, y)
    xq = Q.quantize_input(q, y)
    d = RG.emit(q, tmp_path, xq)
    r = subprocess.run(f"cd {d} && iverilog -g2005 -o tb tb_core.v ccnn_core.v && vvp -n tb",
                       shell=True, capture_output=True, text=True)
    assert "PASS" in r.stdout, r.stdout + r.stderr


def test_drive_hmm_forward_matches_4state_on_degenerate_drive():
    """With identical drive states and no drive switching, the 8-state likelihood equals the 4-state one."""
    c = RTNConfig()
    c.traps.tau_c, c.traps.tau_e, c.traps.amp = [1.0, 1.0], [1.0, 1.0], [1.0, 0.6]
    c.coupling.J = [[0, 1.0], [1.0, 0]]
    c.readout.sigma, c.readout.dt = 0.4, 0.1
    y = th.sample_readout(c, 5, 300, np.random.default_rng(4))
    P4 = th.transition_matrix(c)
    P8 = np.zeros((8, 8))
    P8[:4, :4] = P4
    P8[4:, 4:] = P4
    mu = np.tile(np.array([0.0, 1.0, 0.6, 1.6]), 2)
    pi4 = th.stationary(th.generator(c))
    pi8 = np.r_[pi4, np.zeros(4)]
    ll8, _, _ = DH.fb_general(y, np.repeat(P8[None], 5, 0), np.repeat(pi8[None], 5, 0), np.repeat(mu[None], 5, 0),
                              np.full(5, 0.4))
    np.testing.assert_allclose(ll8, th.hmm_loglik(y, c), atol=1e-8)


def test_difference_input_is_offset_and_ramp_invariant_up_to_scale():
    p = LC.init_params(jax.random.PRNGKey(5))
    y, _ = LC.make_dataset(3, 900, np.random.default_rng(5))
    try:
        LC.INPUT = "diff"
        a = LC.scores(p, y)
        b = LC.scores(p, y + 3.7)                       # constant offset: exactly invisible
        np.testing.assert_allclose(a, b, atol=1e-5)
    finally:
        LC.INPUT = "level"
