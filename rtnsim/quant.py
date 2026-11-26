"""R3c: integer (int8) streaming implementation of the causal CNN.

Scheme (hardware-friendly, bit-exact between this numpy reference and the C++/HLS code):
* input: normalised sample (calibration-prefix offset/scale, done once per record on the
  host) quantised to int8 with scale s_in;
* every convolution: int8 weights with per-output-channel scale, int32 accumulator,
  int32 bias; ReLU; requantisation to the next activation scale by an integer
  multiplier and right shift (round half up);
* all activations are non-negative after the first ReLU, so they live in [0, 127];
* residual blocks requantise both branches to the block-output scale and add with
  saturation;
* time pooling keeps per-channel running sums S1 = sum h and S2 = sum h^2 (int64);
  the final mean/std and the two small dense layers run once per record in float
  (host / soft core), so the streaming datapath is integer only.
"""
from __future__ import annotations

import numpy as np

from . import learned_causal as LC

QMAX = 127


def _mult_shift(real, shift=24):
    """Represent a positive real multiplier as integer m with a right shift."""
    m = np.round(np.asarray(real, np.float64) * (1 << shift)).astype(np.int64)
    if np.any(m >= (1 << 31)):
        raise ValueError("multiplier overflow; reduce shift")
    return m, shift


def _requant(acc, m, shift):
    """round-half-up of acc * m / 2^shift with int64 arithmetic."""
    return (acc.astype(np.int64) * m + (1 << (shift - 1))) >> shift


def _layers():
    out = [("c0", "b0", 1)]
    out += [(f"r{i}", f"rb{i}", d) for i, d in enumerate(LC.DIL)]
    out += [("h", "hb", 1)]
    return out


def calibrate(p, y_cal):
    """Float activation ranges on calibration traces -> per-layer scales."""
    import jax.numpy as jnp
    x = LC.normalise(y_cal)
    acts = {"in": np.asarray(x)}
    h = np.maximum(np.asarray(LC._cconv(x, p["c0"], p["b0"])), 0)
    acts["c0"] = h
    for i, d in enumerate(LC.DIL):
        r = np.maximum(np.asarray(LC._cconv(jnp.asarray(h), p[f"r{i}"], p[f"rb{i}"], d)), 0)
        h = h + r
        acts[f"r{i}"] = h
    acts["h"] = np.maximum(np.asarray(LC._cconv(jnp.asarray(h), p["h"], p["hb"])), 0)
    return {k: float(np.quantile(np.abs(v), 0.9999)) / QMAX for k, v in acts.items()}


def quantize(p, y_cal):
    sc = calibrate(p, y_cal)
    q = {"s_in": sc["in"], "layers": []}
    s_prev = sc["in"]
    for wname, bname, dil in _layers():
        w = np.asarray(p[wname], np.float64)                   # (K, Cin, Cout)
        b = np.asarray(p[bname], np.float64)
        s_w = np.maximum(np.abs(w).max(axis=(0, 1)), 1e-12) / QMAX
        wq = np.clip(np.round(w / s_w), -QMAX, QMAX).astype(np.int8)
        bq = np.round(b / (s_prev * s_w)).astype(np.int64)
        s_out = sc[wname]
        m_acc, sh = _mult_shift(s_prev * s_w / s_out)
        layer = dict(name=wname, dil=dil, w=wq, b=bq, m=m_acc, shift=sh, s_in=s_prev, s_out=s_out,
                     residual=wname.startswith("r"))
        if layer["residual"]:
            layer["m_skip"], _ = _mult_shift(np.full(w.shape[2], s_prev / s_out))
        q["layers"].append(layer)
        s_prev = s_out
    q["s_feat"] = s_prev
    q["head"] = {k: np.asarray(p[k], np.float64) for k in ("d1", "d1b", "d2", "d2b")}
    return q


def quantize_input(q, y):
    x = np.asarray(LC.normalise(y))[..., 0]
    return np.clip(np.round(x / q["s_in"]), -QMAX, QMAX).astype(np.int32)


def run_int(q, xq):
    """Integer datapath over whole traces (B, T) -> int64 running sums S1, S2 (B, C)."""
    B, T = xq.shape
    h = xq[:, :, None].astype(np.int64)                        # (B, T, Cin)
    for L in q["layers"]:
        K, Cin, Cout = L["w"].shape
        pad = (K - 1) * L["dil"]
        hp = np.concatenate([np.zeros((B, pad, Cin), np.int64), h], axis=1)
        acc = np.zeros((B, T, Cout), np.int64)
        for k in range(K):
            acc += hp[:, k * L["dil"]:k * L["dil"] + T, :] @ L["w"][k].astype(np.int64)
        acc += L["b"][None, None, :]
        r = np.clip(_requant(np.maximum(acc, 0), L["m"][None, None, :], L["shift"]), 0, QMAX)
        if L["residual"]:
            skip = np.clip(_requant(h, L["m_skip"][None, None, :], L["shift"]), 0, QMAX)
            h = np.minimum(skip + r, QMAX)
        else:
            h = r
    S1 = h.sum(axis=1)
    S2 = (h * h).sum(axis=1)
    return S1, S2


def head_from_sums(q, S1, S2, T):
    s = q["s_feat"]
    mean = S1 / T * s
    var = np.maximum(S2 / T - (S1 / T) ** 2, 0.0) * s * s
    feat = np.concatenate([mean, np.sqrt(var)], axis=-1)
    hd = q["head"]
    z = np.maximum(feat @ hd["d1"] + hd["d1b"], 0.0)
    return (z @ hd["d2"] + hd["d2b"])[..., 0]


def scores_int(q, y, chunk=100):
    out = []
    for i in range(0, len(y), chunk):
        xq = quantize_input(q, y[i:i + chunk])
        S1, S2 = run_int(q, xq)
        out.append(head_from_sums(q, S1, S2, xq.shape[1]))
    return np.concatenate(out)


def cost_model(fs_hz=1e5):
    """Analytical per-sample cost and energy (Horowitz 2014, 45 nm): int8 mult 0.2 pJ,
    int32 add 0.1 pJ; fp32 mult 3.7 pJ, fp32 add 0.9 pJ.  Compute energy only (no memory)."""
    macs = LC.macs_per_sample()
    e_int8 = macs * (0.2 + 0.1) * 1e-12
    e_fp32 = macs * (3.7 + 0.9) * 1e-12
    return dict(macs_per_sample=macs, energy_int8_J=e_int8, energy_fp32_J=e_fp32,
                power_int8_W_at_fs=e_int8 * fs_hz, fs_hz=fs_hz,
                dsp_needed_at_200MHz=macs * fs_hz / 200e6)
