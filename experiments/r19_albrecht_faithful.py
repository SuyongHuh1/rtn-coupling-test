"""R19: Albrecht et al. 2026 (arXiv:2601.08088) re-implemented as in the full text, on the R10 records.

Full-text procedure vs. the R10 version:
  R10 (deviation)   free 2^K-level, fully connected HMM vs independent FHMM -> the free levels can absorb
                    noise structure that has nothing to do with coupling.
  paper             '4LF' = joint 2^K-state Markov chain whose LEVELS are those of the independent model
                    (middle levels tied); only the rate matrix is freed.  Evidence ratio
                    e = 2 (lnL_i - lnL_j) / (N_i - N_j); e > 2 = strong evidence for the larger model.
                    Coupling is then read from the rate matrix: under independence Gamma_01 = Gamma_23
                    (switching of one fluctuator does not depend on the other); rates get 30 bootstrap
                    samples; points whose rates are comparable to the sampling rate are excluded (aliasing).
Rules reported:
  alb_e2        e > 2 only
  alb_full      e > 2 AND, for some ordered pair, the 30-sample bootstrap 95 % interval of
                log(rate | other empty) - log(rate | other full) excludes 0   (paper's two-part reading)
  excluded      any independent-model mean dwell < 2 samples (aliasing exclusion)

    python experiments/r19_albrecht_faithful.py white|full
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from rtnsim.robustk import _fb, bits, idealise_k, product_P  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"
NBOOT = 30


def fit_P(z, mu, sig, P0, n_iter=60):
    M = mu.size
    P, pi0 = P0.copy(), np.full(M, 1.0 / M)
    prev = -np.inf
    for _ in range(n_iter):
        ll, N, *_r, first, _ = _fb(z, P, pi0, mu, sig, False)
        P = (N + 1e-6) / (N + 1e-6).sum(1, keepdims=True)
        pi0 = first
        if ll - prev < 1e-7 * z.size:
            break
        prev = ll
    return P, ll


def cond_rates(P, K):
    """log rate(i switches | j empty) - log rate(i switches | j full), per ordered pair and direction."""
    B = bits(K)
    ev, V = np.linalg.eig(P.T)
    w = np.abs(np.real(V[:, np.argmin(np.abs(ev - 1))])); w = w / w.sum() + 1e-12
    out = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for a in (0, 1):
                r = []
                for c in (0, 1):
                    src = np.where((B[:, i] == a) & (B[:, j] == c))[0]
                    rate = 0.0
                    for x in src:
                        tgt = np.where((B[:, i] == 1 - a) & np.all(np.delete(B, i, 1) == np.delete(B[x], i), 1))[0]
                        rate += w[x] * P[x, tgt].sum()
                    r.append(rate / w[src].sum())
                out.append(np.log(max(r[0], 1e-9)) - np.log(max(r[1], 1e-9)))
    return np.array(out)


def simulate(P, mu, sig, T, rng):
    M = mu.size
    cum = np.cumsum(P, 1)
    s = np.empty(T, int); s[0] = rng.integers(M)
    u = rng.random(T)
    for t in range(1, T):
        s[t] = min(np.searchsorted(cum[s[t - 1]], u[t]), M - 1)
    return mu[s] + rng.normal(0, sig, T)


def job(a):
    i, y, K = a
    rng = np.random.default_rng(i)
    f = idealise_k(y, K, baseline=(COND != "white"), rng=rng)
    z = y - f["beta"]
    B = bits(K)
    mu = f["b"] + B @ f["a"]
    P0 = product_P(f["Ps"])
    pi = np.full(2 ** K, 1.0 / 2 ** K)
    ll0 = _fb(z, P0, pi, mu, f["sig"], False)[0]
    P1, ll1 = fit_P(z, mu, f["sig"], P0)
    M = 2 ** K
    e = 2 * (ll1 - ll0) / (M * (M - 1) - 2 * K)
    dwell_min = float(min(1 / max(P[0, 1], 1e-12) for P in f["Ps"]) if True else 0)
    dwell_min = float(min(min(1 / max(P[0, 1], 1e-12), 1 / max(P[1, 0], 1e-12)) for P in f["Ps"]))
    res = dict(e=float(e), dwell_min=dwell_min)
    if e > 2:
        d0 = cond_rates(P1, K)
        boot = []
        for _ in range(NBOOT):
            zb = simulate(P1, mu, f["sig"], z.size, rng)
            Pb, _ = fit_P(zb, mu, f["sig"], P1, n_iter=30)
            boot.append(cond_rates(Pb, K))
        boot = np.array(boot)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        res["rate_diff"] = bool(np.any((lo > 0) | (hi < 0)))
        res["max_logratio"] = float(np.max(np.abs(d0)))
    return i, res


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        excl = np.array([out[i]["dwell_min"] < 2 for i in idx])
        calls = dict(alb_e2=np.array([out[i]["e"] > 2 for i in idx]),
                     alb_full=np.array([out[i]["e"] > 2 and out[i].get("rate_diff", False) for i in idx]))
        d = dict(n_indep=int((lab == 0).sum()), n_coupled=int((lab == 1).sum()), n_excluded=int(excl.sum()))
        for k, c in calls.items():
            keep = ~excl
            d[k] = dict(fpr=float(c[(lab == 0) & keep].mean()), tpr=float(c[(lab == 1) & keep].mean()),
                        fpr_no_exclusion=float(c[lab == 0].mean()))
        summ[f"K={K}"] = d
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r19_albrecht_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
