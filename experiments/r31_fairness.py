"""R31: fairness checks requested after the oracle-decomposition comparison (R30).

Modes
  misspec   Coupling OUTSIDE our model family, two traps, 100 coupled + 100 matched independent records per
            mechanism and noise condition (white, 1/f + drift), T = 4096:
              oneway  only trap 2's rates depend on trap 1 (c2' = c2 e^{-phi J}, e2' = e2 e^{(1-phi) J} when
                      trap 1 is occupied; trap 1 unaffected; violates the symmetric-J model),
              coop    simultaneous switching 00 <-> 11 at rate kappa on top of independent traps (no J
                      structure at all; single flips are independent).
            Independent partner: per-trap occupancy and switching flux of the coupled chain (I-projection),
            same levels and the same noise realisation.
  ablation  The R10 benchmark records (r8_oracle.make(100)), our pipeline run once more to keep its
            decomposition; the Albrecht evidence ratio and the all-pairs conditional-rate statistic are
            recomputed ON OUR DECOMPOSITION (component analysis: decomposition vs statistic).
  long      T = 65536 records of the length study (r10_length.recs, 40 pairs per K, white and 1/f):
            power at matched 5 % FPR and AUC for every method with its own decomposition.
Every mode reports per record: ours (pipeline LRT), Albrecht e (own decomposition), Son one-pair and
all-pairs (own decomposition), Guo max|d| (own), and the same statistics on the true path (oracle).
    python experiments/r31_fairness.py misspec|ablation|long [workers]
"""
from __future__ import annotations

import json
import os
import sys
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "misspec"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sys.argv = [sys.argv[0], "white"]             # imported scripts read sys.argv[1] at import time
from r19_albrecht_faithful import fit_P  # noqa: E402
from r30_oracle_baselines import fit_indep, trap_P, son_g, cond_logratios  # noqa: E402
from rtnsim.robustk import bits, idealise_k, product_P  # noqa: E402

MS = int(os.environ.get("RTN_MS", "1"))
_NET = None


def net():
    global _NET
    if _NET is None:
        from rtnsim.r9 import load_stepnet
        _NET = load_stepnet(ROOT / "models/stepnet.pt") if (ROOT / "models/stepnet.pt").exists() \
            else load_stepnet(ROOT / "results/r9a_gpu/stepnet.pt")
    return _NET


# ------------------------------------------------------------------ records outside the model family
def chain_Q(c, e, mech, J=0.0, phi=0.5, kappa=0.0):
    """4-state generator, state = n1 + 2 n2."""
    Q = np.zeros((4, 4))
    for s in range(4):
        n1, n2 = s & 1, (s >> 1) & 1
        r1 = c[0] if n1 == 0 else e[0]
        r2 = c[1] if n2 == 0 else e[1]
        if mech == "oneway" and n1 == 1:
            r2 = r2 * (np.exp(-phi * J) if n2 == 0 else np.exp((1 - phi) * J))
        Q[s, s ^ 1] += r1
        Q[s, s ^ 2] += r2
    if mech == "coop":
        Q[0, 3] += kappa; Q[3, 0] += kappa
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def stationary(Q):
    w, v = np.linalg.eig(Q.T)
    p = np.real(v[:, np.argmin(np.abs(w))]); return p / p.sum()


def iproj(Q):
    """Independent rates with each trap's occupancy and switching flux of Q."""
    p = stationary(Q)
    c, e = np.zeros(2), np.zeros(2)
    for k, b in enumerate((1, 2)):
        occ = sum(p[s] for s in range(4) if s & b)
        flux = sum(p[s] * Q[s, t] for s in range(4) for t in range(4) if s != t and ((s ^ t) & b) and not (s & b))
        c[k], e[k] = flux / (1 - occ), flux / occ
    return c, e


def sample(Q, T, rng):
    P = expm(Q); cum = np.cumsum(P, 1)
    s = np.empty(T, int); s[0] = rng.choice(4, p=stationary(Q))
    u = rng.random(T)
    for t in range(1, T):
        s[t] = min(np.searchsorted(cum[s[t - 1]], u[t]), 3)
    return np.stack([s & 1, (s >> 1) & 1], 1)


def flick(T, rng):
    W_ = np.fft.rfft(rng.standard_normal(T)); f = np.fft.rfftfreq(T)
    x = np.fft.irfft(W_ * np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1 / T)), 0), n=T)
    return x / x.std()


def make_misspec(mech, cond, n=100, T=4096, seed=0):
    rng = np.random.default_rng(seed + (0 if mech == "oneway" else 1000) + (0 if cond == "white" else 7))
    ys, ss, rows = [], [], []
    while len(rows) < 2 * n:
        tau = np.exp(np.log(8) + rng.random(2) * (np.log(200) - np.log(8)))
        p = 0.15 + 0.7 * rng.random(2)
        c, e = 2 / tau * p, 2 / tau * (1 - p)
        if mech == "oneway":
            J = (0.7 + 2.3 * rng.random()) * rng.choice([-1, 1]); Q1 = chain_Q(c, e, "oneway", J=J, phi=0.2 + 0.6 * rng.random())
        else:
            kap = (0.2 + 0.8 * rng.random()) * np.mean(np.r_[c, e]); Q1 = chain_Q(c, e, "coop", kappa=kap)
        c0, e0 = iproj(Q1); Q0 = chain_Q(c0, e0, "none")
        a = np.sort(0.15 + 0.85 * rng.random(2))[::-1]; a[0] = 1.0
        sig = a.min() * (0.15 + 0.6 * rng.random())
        noise = sig * rng.standard_normal(T)
        if cond != "white":
            noise = noise + 0.6 * sig * rng.random() * flick(T, rng) + rng.random() * np.cumsum(rng.standard_normal(T)) / np.sqrt(T)
        for lab, Q in ((1, Q1), (0, Q0)):
            S = sample(Q, T, rng)
            ys.append(S @ a + noise); ss.append(S); rows.append(dict(K=2, coupled=lab))
    return ys, ss, rows


# ------------------------------------------------------------------ statistics
def albrecht_e(z, mu, sig, S0, K):
    Ps0 = np.array([trap_P(S0[:, k]) for k in range(K)])
    ll0, P0 = fit_indep(z, mu, sig, Ps0, K)
    _, ll1 = fit_P(z, mu, sig, P0)
    M = 2 ** K
    return float(2 * (ll1 - ll0) / (M * (M - 1) - 2 * K))


def stats_on(z, mu, sig, S, K):
    """Albrecht e, Son (smallest | largest) and all-pairs, Guo max|d| for a given decomposition.
    S columns follow the bit order of mu (state index = sum_k n_k 2^k)."""
    alb = albrecht_e(z, mu, sig, S, K)
    steps = np.abs(np.array([mu[2 ** k] - mu[0] for k in range(K)]))
    Sd = S[:, np.argsort(-steps)]                              # largest step first
    return dict(alb=alb, son1=son_g(Sd, K - 1, 0), son_all=max(son_g(Sd, p, q) for p, q in permutations(range(K), 2)),
                guo=float(np.max(np.abs(cond_logratios(Sd)))))


def job(a):
    i, y, S_true, K = a
    rng = np.random.default_rng(1000 + i)
    S_true = np.asarray(S_true).astype(int)
    out = {}
    # ours (also gives our decomposition)
    from rtnsim.r9 import lrt_cnn3_ms
    try:
        s, f0, f1, beta, _ = lrt_cnn3_ms(y, K, net(), ar1=True, rng=rng, n_start=MS)
        out["ours"] = float(s)
        X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
        mu = X @ f0["coef"]; z = y - beta
        S_ours = bits(K)[f0["path"]].astype(int)
        sig = float(np.std(z - mu[f0["path"]]))
        so = stats_on(z, mu, sig, S_ours, K)
        out.update({k + "_ourdec": v for k, v in so.items()})
    except Exception as ex:                                   # pipeline failure: recorded, excluded from AUC
        out["ours_failed"] = repr(ex)[:80]
    # own decomposition (practitioner idealisation, as in R10/R19)
    f = idealise_k(y, K, baseline=(COND != "white"), rng=rng)
    z = y - f["beta"]; mu = f["b"] + bits(K) @ f["a"]
    so = stats_on(z, mu, f["sig"], f["S"].astype(int), K)
    out.update({k + "_own": v for k, v in so.items()})
    # oracle decomposition
    Xs = np.hstack([np.ones((len(y), 1)), S_true])
    from scipy.ndimage import median_filter
    bta = np.zeros_like(y)
    for _ in range(2 if COND != "white" else 1):
        coef, *_ = np.linalg.lstsq(Xs, y - bta, rcond=None)
        if COND != "white":
            bta = median_filter(y - Xs @ coef, size=301, mode="nearest")
    coef, *_ = np.linalg.lstsq(Xs, y - bta, rcond=None)
    z = y - bta; mu = np.hstack([np.ones((2 ** K, 1)), bits(K)]) @ coef
    sig = float(np.std(z - Xs @ coef))
    so = stats_on(z, mu, sig, S_true, K)
    out.update({k + "_oracle": v for k, v in so.items()})
    return i, out


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def tpr5(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    return float(np.mean(s[y == 1] > np.quantile(s[y == 0], 0.95)))


def summarise(res, rows, idx):
    keys = sorted({k for i in idx for k in res[i] if not k.endswith("failed")})
    out = {}
    for k in keys:
        ok = [i for i in idx if k in res[i] and np.isfinite(res[i][k])]
        lab = [rows[i]["coupled"] for i in ok]
        if len(set(lab)) == 2:
            out[k] = dict(auc=round(auc([res[i][k] for i in ok], lab), 3), tpr5=round(tpr5([res[i][k] for i in ok], lab), 3), n=len(ok))
    out["n_failed"] = sum(1 for i in idx if "ours_failed" in res[i])
    return out


COND = "white"


def run(tag, ys, ss, rows, cond):
    global COND
    COND = cond
    path = ROOT / f"results/r8/r31_{tag}{os.environ.get('RTN_TAG', '')}.json"
    res = {}
    if path.exists():
        res = {int(k): v for k, v in json.load(open(path)).get("res", {}).items()}
    todo = [(i, ys[i], ss[i], rows[i]["K"]) for i in range(len(ys)) if i not in res]
    with Pool(W, initializer=_set_cond, initargs=(cond,)) as p:
        for i, r in p.imap_unordered(job, todo, chunksize=1):
            res[i] = r
            json.dump(dict(res={str(k): v for k, v in res.items()}), open(path, "w"))
            print(tag, i, flush=True)
    summ = {f"K={K}": summarise(res, rows, [i for i in range(len(rows)) if rows[i]["K"] == K])
            for K in sorted({r["K"] for r in rows})}
    json.dump(dict(summary=summ, res={str(k): v for k, v in res.items()}), open(path, "w"), indent=1)
    print(tag, json.dumps(summ, indent=1), flush=True)


def _set_cond(c):
    global COND
    COND = c


def main():
    if MODE == "misspec":
        for mech in ("oneway", "coop"):
            for cond in ("white", "full"):
                ys, ss, rows = make_misspec(mech, cond)
                run(f"misspec_{mech}_{cond}", ys, ss, rows, cond)
    elif MODE == "ablation":
        import r8_oracle as O
        for cond in ("white", "full"):
            ys, ss, rows = O.make(100, white=(cond == "white"))
            run(f"ablation_{cond}", ys, ss, rows, cond)
    elif MODE == "long":
        import torch
        from rtnsim import sim2_torch as S_
        T = 65536
        for cond in ("white", "full"):
            ys, ss, rows = [], [], []
            for K in (2, 3):                                      # same generator and seeds as r10_length.recs
                g = torch.Generator().manual_seed(500 + K + T); got = 0
                while got < 40:
                    y, lab = S_.make_batch(20, T, g, "cpu", p_drive=0, p_box=0, return_states=True,
                                           **(dict(p_drift=0, p_flicker=0) if cond == "white" else {}))
                    if int(lab["K"][0]) != K:
                        continue
                    for r in range(y.shape[0]):
                        if got >= 40 and r % 2 == 0:
                            break
                        ys.append(y[r].double().numpy()); ss.append(lab["states"][r].numpy().astype(int))
                        rows.append(dict(K=K, coupled=int(lab["coupled"][r])))
                        if r % 2 == 1:
                            got += 1
            run(f"long_{cond}", ys, ss, rows, cond)


if __name__ == "__main__":
    main()
