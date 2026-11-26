"""R18: when is trap-trap coupling a first-order effect on industry RTN metrics?

Population model (standard RTN statistics, e.g. Zou IEDM'14, Wirth SSE'21):
  traps per device  N ~ Poisson(lam)        (capped at NMAX)
  step amplitude    a ~ Exponential(mean 1) (unit = mean single-trap dVth, eta)
  trap energy       eps ~ Uniform(-W, W) kT -> occupancy set by eps (active traps have |eps| small)
  coupling          each pair coupled with probability f; J = +|J| (repulsive) or -|J|,
                    |J| ~ Exponential(mean Jm)
  equilibrium       p(s) ~ exp(-sum eps_k s_k - sum_{k<l} J_kl s_k s_l)   (detailed balance of our model)

Comparator = what an analyst with an independent-trap model would use: the SAME device with each trap's
measured (marginal) occupancy but no correlation (product of the coupled marginals).

Metrics (population of devices x random instant):
  tail_p   P(dVth > x*) where x* is the 99.99 % instantaneous quantile of the independent model
  q9999    99.99 % quantile of instantaneous dVth
  var      mean per-device dVth variance (RTN noise power)
  ptp      peak-to-peak over a long window = sum of amplitudes of traps that switch at all
           (identical in both models unless a state is never visited -> J does not change it)
First-order := tail probability changes by >= 25 % (ratio outside 0.8-1.25) or q9999 shifts by >= 5 %.

    python experiments/r18_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NMAX, W, NDEV = 8, 4.0, 60000
BITS = {n: np.array(list(product([0, 1], repeat=n)), float) for n in range(1, NMAX + 1)}


def population(lam, f, Jm, sign, rng):
    devs = []
    N = np.minimum(rng.poisson(lam, NDEV), NMAX)
    for n in N:
        if n == 0:
            continue
        a = rng.exponential(1.0, n)
        eps = rng.uniform(-W, W, n)
        J = np.zeros((n, n))
        for k in range(n):
            for l in range(k + 1, n):
                if rng.random() < f:
                    J[k, l] = sign * rng.exponential(Jm)
        devs.append((a, eps, J))
    return devs


def device_dists(a, eps, J):
    B = BITS[len(a)]
    E = B @ eps + np.einsum("si,ij,sj->s", B, J, B)
    p = np.exp(-(E - E.min())); p /= p.sum()
    v = B @ a
    m = p @ B                                        # marginal occupancies
    q = np.prod(np.where(B > 0, m, 1 - m), axis=1)   # independent comparator
    return v, p, q, m


def evaluate(devs):
    vs, ps, qs, var_c, var_i, ptp = [], [], [], [], [], []
    for a, eps, J in devs:
        v, p, q, m = device_dists(a, eps, J)
        vs.append(v); ps.append(p); qs.append(q)
        var_c.append(p @ v ** 2 - (p @ v) ** 2); var_i.append(q @ v ** 2 - (q @ v) ** 2)
        ptp.append(a[(m > 1e-4) & (m < 1 - 1e-4)].sum())
    v = np.concatenate(vs); p = np.concatenate(ps) / len(devs); q = np.concatenate(qs) / len(devs)
    o = np.argsort(v)[::-1]; v, p, q = v[o], p[o], q[o]
    cp, cq = np.cumsum(p), np.cumsum(q)               # P(X >= v) in descending order
    def quant(c, tail):
        i = np.searchsorted(c, tail); return float(v[min(i, len(v) - 1)])
    x_star = quant(cq, 1e-4)
    tail_c = float(p[v > x_star].sum()); tail_i = float(q[v > x_star].sum())
    return dict(tail_ratio=tail_c / max(tail_i, 1e-300), q9999_c=quant(cp, 1e-4), q9999_i=x_star,
                q999_c=quant(cp, 1e-3), q999_i=quant(cq, 1e-3),
                var_ratio=float(np.mean(var_c) / np.mean(var_i)), ptp_mean=float(np.mean(ptp)))


def main():
    rng = np.random.default_rng(18)
    rows = []
    for lam in (2, 4):
        for sign in (+1, -1):
            for f in (0.1, 0.3, 0.5, 1.0):
                for Jm in (0.1, 0.25, 0.5, 1.0, 2.0):
                    r = evaluate(population(lam, f, Jm, sign, rng))
                    shift = r["q9999_c"] / r["q9999_i"] - 1
                    first = (r["tail_ratio"] < 0.8 or r["tail_ratio"] > 1.25 or abs(shift) >= 0.05)
                    rows.append(dict(lam=lam, sign="repulsive" if sign > 0 else "cooperative", f=f, Jm=Jm,
                                     q_shift=round(shift, 4), first_order=bool(first),
                                     **{k: round(v_, 4) for k, v_ in r.items()}))
                    print(rows[-1], flush=True)
    json.dump(rows, open(ROOT / "results/r8/r18_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
