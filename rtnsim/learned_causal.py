"""R3 confound-aware, streaming (causal) CNN.

Differences from rtnsim.learned (R2):
* causal dilated convolutions (left padding only) and running mean / std pooling,
  so the score can be produced sample by sample on hardware;
* fixed normalisation from a calibration prefix (first CAL samples), not from
  the whole trace;
* training pairs carry the *same* confounds on both members (shared telegraph
  drive, random-walk drift, 1/f noise, readout coupling), drawn at random, so a
  confound alone carries no label information and the network must learn to
  ignore it.
"""
from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .noise import add_drift, add_flicker
from .fastsim import sample_configs
from .learned import random_pair

CH = 32
DIL = (1, 2, 4, 8)
K0, KR = 7, 5
CAL = 512
INPUT = "level"   # "level": calibrated level; "diff": first difference (offset/drift-free by construction)


# --------------------------------------------------------------------------- #
# Confound augmentation
# --------------------------------------------------------------------------- #
def random_confound(rng):
    c = {}
    if rng.random() < 0.5:
        c["drive"] = dict(tau=float(np.exp(rng.uniform(np.log(0.3), np.log(10.0)))),
                          gamma=[float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-1.5, 1.5))])
    if rng.random() < 0.4:
        c["drift"] = float(rng.uniform(0.0, 1.0))
    if rng.random() < 0.4:
        c["flicker"] = float(rng.uniform(0.0, 0.6))
    if rng.random() < 0.3:
        c["B"] = float(rng.uniform(-0.3, 0.3))
    return c


def apply_config_confound(cfg, conf):
    cfg = copy.deepcopy(cfg)
    if "drive" in conf:
        cfg.drive.enabled = True
        cfg.drive.tau_on = cfg.drive.tau_off = conf["drive"]["tau"]
        cfg.drive.gamma = list(conf["drive"]["gamma"])
    if "B" in conf:
        cfg.coupling.B = [[0, conf["B"]], [0, 0]]
    return cfg


def apply_trace_confound(y, conf, dt, sigma, rng):
    T = y.shape[1]
    y = y.astype(np.float64)
    if "flicker" in conf and conf["flicker"] > 0:
        y = add_flicker(y, conf["flicker"] * sigma, fmin=0.3 / (T * dt), fmax=0.5 / dt, n_terms=6, dt=dt, rng=rng)
    if "drift" in conf and conf["drift"] > 0:
        y = add_drift(y, sd_per_sqrt_time=conf["drift"] / np.sqrt(T * dt), dt=dt, rng=rng)
    return y.astype(np.float32)


def make_dataset(n_pairs, T, rng, dt=0.1, confound=True):
    ys, labs = [], []
    for _ in range(n_pairs):
        c1, c0 = random_pair(rng, dt)
        conf = random_confound(rng) if confound else {}
        pair = []
        for cfg in (c1, c0):
            y = sample_configs([apply_config_confound(cfg, conf)], T, rng)
            pair.append(apply_trace_confound(y, conf, dt, cfg.readout.sigma, rng)[0])
        ys += pair
        labs += [1.0, 0.0]
    return np.stack(ys), np.array(labs, np.float32)


# --------------------------------------------------------------------------- #
# Causal network
# --------------------------------------------------------------------------- #
def init_params(key):
    ks = jax.random.split(key, 8)

    def w(k, shape):
        return jax.random.normal(k, shape) * np.sqrt(2.0 / np.prod(shape[:-1]))

    p = {"c0": w(ks[0], (K0, 1, CH)), "b0": jnp.zeros(CH)}
    for i, _ in enumerate(DIL):
        p[f"r{i}"] = w(ks[1 + i], (KR, CH, CH))
        p[f"rb{i}"] = jnp.zeros(CH)
    p["h"] = w(ks[5], (1, CH, CH))
    p["hb"] = jnp.zeros(CH)
    p["d1"] = w(ks[6], (2 * CH, 32))
    p["d1b"] = jnp.zeros(32)
    p["d2"] = w(ks[7], (32, 1)) * 0.1
    p["d2b"] = jnp.zeros(1)
    return p


def _cconv(x, w, b, dil=1):
    k = w.shape[0]
    pad = (k - 1) * dil
    y = jax.lax.conv_general_dilated(x, w, window_strides=(1,), padding=[(pad, 0)], rhs_dilation=(dil,),
                                     dimension_numbers=("NWC", "WIO", "NWC"))
    return y + b


def normalise(y):
    """Offset and scale from the calibration prefix only (streaming-compatible).
    With INPUT == "diff" the network sees y[t] - y[t-1], scaled by the prefix sd of differences."""
    y = jnp.asarray(y, jnp.float32)
    if INPUT == "diff":
        d = jnp.diff(y, axis=1, prepend=y[:, :1])
        s = jnp.std(d[:, 1:CAL], axis=1, keepdims=True) + 1e-6
        return (d / s)[..., None]
    cal = y[:, :CAL]
    m = jnp.median(cal, axis=1, keepdims=True)
    s = jnp.std(cal, axis=1, keepdims=True) + 1e-6
    return ((y - m) / s)[..., None]


def features(p, x):
    h = jax.nn.relu(_cconv(x, p["c0"], p["b0"]))
    for i, d in enumerate(DIL):
        h = h + jax.nn.relu(_cconv(h, p[f"r{i}"], p[f"rb{i}"], d))
    return jax.nn.relu(_cconv(h, p["h"], p["hb"]))


def head(p, mean, std):
    z = jax.nn.relu(jnp.concatenate([mean, std], -1) @ p["d1"] + p["d1b"])
    return (z @ p["d2"] + p["d2b"])[..., 0]


def forward(p, x):
    h = features(p, x)
    return head(p, h.mean(axis=1), h.std(axis=1))


def receptive_field():
    return (K0 - 1) + sum((KR - 1) * d for d in DIL) + 1


def macs_per_sample():
    return K0 * CH + len(DIL) * KR * CH * CH + CH * CH


def n_params(p):
    return int(sum(np.prod(v.shape) for v in p.values()))


def loss_fn(p, x, lab):
    return optax.sigmoid_binary_cross_entropy(forward(p, x), lab).mean()


def train(n_pairs=40000, T=1024, epochs=20, batch=64, lr=1e-3, seed=0, val_pairs=2000, log=print,
          cache_dir=None):
    """Train with an on-disk data cache and per-epoch checkpoints (resumes after interruption)."""
    from pathlib import Path
    cache = Path(cache_dir) if cache_dir else None
    rng = np.random.default_rng(seed)
    if cache and (cache / "data.npz").exists():
        z = np.load(cache / "data.npz")
        y, lab, yv, labv = z["y"], z["lab"], z["yv"], z["labv"]
        log("loaded cached dataset")
    else:
        y, lab = make_dataset(n_pairs, T, rng)
        yv, labv = make_dataset(val_pairs, T, rng)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.savez(cache / "data.npz", y=y, lab=lab, yv=yv, labv=labv)
    xv = normalise(yv)
    steps = epochs * (len(y) // batch)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    p = init_params(jax.random.PRNGKey(seed))
    st = opt.init(p)
    hist, start = [], 0
    if cache and (cache / "ckpt.pkl").exists():
        with open(cache / "ckpt.pkl", "rb") as fh:
            ck = pickle.load(fh)
        p = {k: jnp.asarray(v) for k, v in ck["p"].items()}
        st = jax.tree_util.tree_map(jnp.asarray, ck["st"])
        hist, start = ck["hist"], ck["epoch"] + 1
        log(f"resumed after epoch {ck['epoch']}")

    @jax.jit
    def step(p, st, x, l):
        L, g = jax.value_and_grad(loss_fn)(p, x, l)
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, L

    fwd = jax.jit(forward)
    erng = np.random.default_rng(seed + 1)
    for ep in range(start, epochs):
        perm = np.random.default_rng(seed * 1000 + ep).permutation(len(y))
        tot = 0.0
        nb = len(y) // batch
        for i in range(nb):
            idx = perm[i * batch:(i + 1) * batch]
            p, st, L = step(p, st, normalise(y[idx]), jnp.asarray(lab[idx]))
            tot += float(L)
        sv = np.concatenate([np.asarray(fwd(p, xv[j:j + 500])) for j in range(0, len(xv), 500)])
        acc = float(np.mean((sv > 0) == (labv > 0.5)))
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_acc=acc))
        log(f"epoch {ep}: loss {tot / nb:.4f} val acc {acc:.3f}")
        if cache:
            with open(cache / "ckpt.pkl", "wb") as fh:
                pickle.dump(dict(p={k: np.asarray(v) for k, v in p.items()},
                                 st=jax.tree_util.tree_map(np.asarray, st), hist=hist, epoch=ep), fh)
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
