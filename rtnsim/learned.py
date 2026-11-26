"""R2 learned detector: a small dilated 1-D CNN trained on matched pairs.

Training pairs: every coupled configuration (random J, rates, step heights, SNR,
Bronsted phi) is paired with its *closest independent model* (same occupancies
and switching fluxes), so the only cue that separates the classes is the
coupling itself -- the network cannot win on marginal statistics.

The network is fully convolutional with mean/std pooling over time, so it scores
traces of any length; ~25k parameters (small enough for an FPGA / SNN port).
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import RTNConfig
from .theory import coupling_information
from .fastsim import sample_configs

CH = 32
DIL = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Random matched configurations (time unit = dt * 10, matching the evaluation grid)
# --------------------------------------------------------------------------- #
def random_pair(rng, dt=0.1):
    cfg = RTNConfig()
    cfg.traps.tau_c = list(np.exp(rng.uniform(np.log(0.4), np.log(4.0), 2)))
    cfg.traps.tau_e = list(np.exp(rng.uniform(np.log(0.4), np.log(4.0), 2)))
    a2 = rng.uniform(0.3, 0.95)
    cfg.traps.amp = [1.0, a2]
    J = rng.uniform(0.3, 3.0) * rng.choice([-1, 1])
    cfg.coupling.J = [[0, J], [J, 0]]
    cfg.coupling.phi = rng.uniform(0.0, 1.0)
    cfg.readout.dt = dt
    cfg.readout.sigma = 1.0 / rng.uniform(1.5, 6.0)
    _, cfg0 = coupling_information(cfg)
    return cfg, cfg0


def make_dataset(n_pairs, T, rng, dt=0.1):
    c1, c0 = zip(*[random_pair(rng, dt) for _ in range(n_pairs)])
    y = np.concatenate([sample_configs(list(c1), T, rng), sample_configs(list(c0), T, rng)])
    lab = np.r_[np.ones(n_pairs), np.zeros(n_pairs)].astype(np.float32)
    return y, lab


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def init_params(key):
    ks = jax.random.split(key, 8)

    def w(k, shape):
        fan_in = np.prod(shape[:-1])
        return jax.random.normal(k, shape) * np.sqrt(2.0 / fan_in)

    p = {"c0": w(ks[0], (7, 1, CH)), "b0": jnp.zeros(CH)}
    for i, _ in enumerate(DIL):
        p[f"r{i}"] = w(ks[1 + i], (5, CH, CH))
        p[f"rb{i}"] = jnp.zeros(CH)
    p["h"] = w(ks[5], (1, CH, CH))
    p["hb"] = jnp.zeros(CH)
    p["d1"] = w(ks[6], (2 * CH, 32))
    p["d1b"] = jnp.zeros(32)
    p["d2"] = w(ks[7], (32, 1)) * 0.1
    p["d2b"] = jnp.zeros(1)
    return p


def _conv(x, w, b, dil=1):
    y = jax.lax.conv_general_dilated(x, w, window_strides=(1,), padding="SAME", rhs_dilation=(dil,),
                                     dimension_numbers=("NWC", "WIO", "NWC"))
    return y + b


def normalise(y):
    y = jnp.asarray(y, jnp.float32)
    m = jnp.median(y, axis=1, keepdims=True)
    s = jnp.std(y, axis=1, keepdims=True) + 1e-6
    return ((y - m) / s)[..., None]


def forward(p, x):
    h = jax.nn.relu(_conv(x, p["c0"], p["b0"]))
    for i, d in enumerate(DIL):
        h = h + jax.nn.relu(_conv(h, p[f"r{i}"], p[f"rb{i}"], d))
    h = jax.nn.relu(_conv(h, p["h"], p["hb"]))
    feat = jnp.concatenate([h.mean(axis=1), h.std(axis=1)], axis=-1)
    z = jax.nn.relu(feat @ p["d1"] + p["d1b"])
    return (z @ p["d2"] + p["d2b"])[:, 0]


def loss_fn(p, x, lab):
    return optax.sigmoid_binary_cross_entropy(forward(p, x), lab).mean()


def train(n_pairs=30000, T=1024, epochs=8, batch=64, lr=1e-3, seed=0, val_pairs=2000, log=print):
    rng = np.random.default_rng(seed)
    y, lab = make_dataset(n_pairs, T, rng)
    yv, labv = make_dataset(val_pairs, T, rng)
    xv = normalise(yv)
    p = init_params(jax.random.PRNGKey(seed))
    steps = epochs * (len(y) // batch)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, l):
        L, g = jax.value_and_grad(loss_fn)(p, x, l)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, L

    fwd = jax.jit(forward)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(y))
        tot = 0.0
        for i in range(len(y) // batch):
            idx = perm[i * batch:(i + 1) * batch]
            p, st, L = step(p, st, normalise(y[idx]), jnp.asarray(lab[idx]))
            tot += float(L)
        sv = np.concatenate([np.asarray(fwd(p, xv[j:j + 500])) for j in range(0, len(xv), 500)])
        acc = float(np.mean((sv > 0) == (labv > 0.5)))
        hist.append(dict(epoch=ep, train_loss=tot / (len(y) // batch), val_acc=acc))
        log(f"epoch {ep}: loss {tot / (len(y) // batch):.4f} val acc {acc:.3f}")
    return p, hist


def scores(p, y, chunk=200):
    fwd = jax.jit(forward)
    return np.concatenate([np.asarray(fwd(p, normalise(y[i:i + chunk]))) for i in range(0, len(y), chunk)])


def save(p, path):
    with open(path, "wb") as fh:
        pickle.dump({k: np.asarray(v) for k, v in p.items()}, fh)


def load(path):
    with open(path, "rb") as fh:
        return {k: jnp.asarray(v) for k, v in pickle.load(fh).items()}
