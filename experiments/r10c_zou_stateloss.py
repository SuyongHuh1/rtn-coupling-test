"""Zou et al. IEDM 2014 ("New understanding of state-loss in complex RTN"): over 270 devices, the
number of current states found by GMM + EM is far below the 2^n expected from a Poisson number of
independent traps ("over 50% of states lost"), attributed to trap interaction.

Test of the inference: simulate a device population with a Poisson number of INDEPENDENT traps
(no coupling), realistic spread of amplitudes and time constants, white noise, and count states the
same way (GMM on the current histogram, number of components by BIC).  If independent traps already
lose > 50% of states, state-loss is not evidence of interaction.

    python experiments/r10c_zou_stateloss.py [n_dev=300]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
T = 20000                 # samples per trace


def device(args):
    seed, lam, am = args
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam)
    while n == 0:                         # Zou's histogram counts devices with RTN; 1-state devices handled separately
        n = rng.poisson(lam)
    n = min(n, 5)
    y = np.zeros(T)
    for _ in range(n):
        amp = rng.exponential(1.0) * am           # step / noise sd: exponential amplitudes (RTN literature), mean am
        tc, te = 10 ** rng.uniform(0.5, 3.5, 2)   # mean dwell 3 - 3000 samples (log-uniform), independent
        # alternating geometric dwells (discrete-time telegraph), vectorised
        s0 = int(rng.random() < tc / (tc + te)) ^ 1
        nseg = int(3 * T / min(tc, te)) + 10
        d0 = rng.geometric(1 / tc, nseg); d1 = rng.geometric(1 / te, nseg)
        dw = np.empty(2 * nseg, int)
        if s0 == 0:
            dw[0::2], dw[1::2] = d0, d1
        else:
            dw[0::2], dw[1::2] = d1, d0
        lv = (np.arange(2 * nseg) + s0) % 2
        x = np.repeat(lv, dw)[:T].astype(float)
        y += amp * x
    y += rng.normal(0, 1.0, T)
    z = y[::4].reshape(-1, 1)                     # GMM on a thinned histogram (speed)
    bic = []
    for m in range(1, 2 ** n + 3):
        g = GaussianMixture(m, covariance_type="tied", n_init=2, random_state=0).fit(z)
        bic.append(g.bic(z))
    return dict(n=int(n), states=int(np.argmin(bic) + 1), expected=int(2 ** n))


def main():
    n_dev = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = {}
    with Pool(1) as p:  # noqa
        for lam, am in ((1.4, 4.0), (3.5, 4.0), (1.4, 10.0), (3.5, 10.0)):   # Zou's fitted trap numbers; mean step 4 or 10 noise sd
            res = p.map(device, [(1000 * int(lam * 10) + i, lam, am) for i in range(n_dev)])
            st = np.array([r["states"] for r in res]); ex = np.array([r["expected"] for r in res])
            loss = 1 - np.minimum(st, ex).sum() / ex.sum()
            out[f"lambda={lam},amp={am}"] = dict(n_dev=n_dev, state_loss=float(loss),
                                        frac_devices_missing=float(np.mean(st < ex)),
                                        hist_states=np.bincount(st, minlength=12).tolist(),
                                        hist_traps=np.bincount([r["n"] for r in res], minlength=6).tolist())
            print(lam, am, json.dumps(out[f"lambda={lam},amp={am}"]), flush=True)
    json.dump(out, open(ROOT / "results/r8/r10c_zou_stateloss.json", "w"), indent=1)


if __name__ == "__main__":
    main()
