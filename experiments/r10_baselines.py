"""Head-to-head: how often do existing multi-trap RTN coupling analyses call INDEPENDENT records coupled?

Same records as R8/R9a (r8_oracle.make, T = 4096, K = 2/3, ~50 independent + ~50 coupled per K).
All baselines get the same practitioner preprocessing: running-median baseline from idealise_k
(full condition only) and the independent factorial-HMM idealisation (idealise_k, 4 starts).

  A  PKU state-based HMM (Zhang IRPS'17 / Guo IRPS'18 style): free 2^K-state HMM, full transition
     matrix, free levels; bit patterns assigned by the additive level fit; coupling strength
     d_ij = tau_i(j occupied) / tau_i(j empty) - 1 for capture and emission.  Practitioner rule:
     coupled if max |d| > 0.2 (also reported: > 0.5).
  B  Albrecht et al. 2025 evidence ratio: e = 2 dlnL / dN between the free 2^K-level fully
     connected HMM and the independent factorial HMM.  Rule: e > 2 (AIC-equivalent).
  C  Son et al. 2017 conditional-dwell comparison: on the idealised path, rates of trap i
     conditioned on trap j vs unconditioned; chi2(2) per ordered pair, Bonferroni over pairs.
  D  Ours (R9a lrt_cnn3) with (i) the asymptotic chi2 threshold and (ii) the pipeline-null
     calibration (FPR taken from r8_calib_*_r9.json, 1/f only; reported separately).

    python experiments/r10_baselines.py white|full
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r8_oracle as O  # noqa: E402
from rtnsim.cfhmm import _free_hmm  # noqa: E402
from rtnsim.robustk import bits, idealise_k, _fb  # noqa: E402
from rtnsim.surrogate import _ll_binary_transitions  # noqa: E402

COND = sys.argv[1] if len(sys.argv) > 1 else "white"


def assign(levels, w, K):
    """Additive assignment: idx[x] = level index of bit pattern x (exhaustive weighted LS)."""
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    order = np.argsort(levels)
    best = (np.inf, None)
    for perm in permutations(range(1, 2 ** K - 1)):
        idx = order[np.array([0, *perm, 2 ** K - 1])]
        sw = np.sqrt(w[idx] + 1e-3)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], levels[idx] * sw, rcond=None)
        err = np.sum(((X @ coef) - levels[idx]) ** 2 * sw ** 2)
        if err < best[0]:
            best = (err, idx)
    return best[1]


def free_fit(z, K, rng, n_starts=3):
    M = 2 ** K
    lo, hi = np.quantile(z, [0.01, 0.99])
    best = None
    for s in range(n_starts):
        mu0 = np.quantile(z, (np.arange(M) + 0.5) / M) if s == 0 else np.sort(lo + (hi - lo) * rng.random(M))
        try:
            r = _free_hmm(z, M, mu0, n_iter=100)
        except ZeroDivisionError:
            continue
        if best is None or r[-1] > best[-1]:
            best = r
    return best   # mu, w, P, sig, ll


def pku_strength(P, w, idx, K):
    """Per ordered pair (i|j): capture/emission rate ratio from the state transition matrix."""
    B = bits(K)
    Pb = P[np.ix_(idx, idx)]          # pattern-indexed transition matrix
    wb = w[idx] + 1e-12
    d = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for a in (0, 1):          # a = 0: capture (0 -> 1), a = 1: emission
                r = []
                for c in (0, 1):
                    src = np.where((B[:, i] == a) & (B[:, j] == c))[0]
                    rate = 0.0
                    for x in src:
                        tgt = np.where((B[:, i] == 1 - a) & np.all(np.delete(B, i, 1) == np.delete(B[x], i), 1))[0]
                        rate += wb[x] * Pb[x, tgt].sum()
                    r.append(rate / wb[src].sum())
                d.append(r[0] / max(r[1], 1e-12) - 1.0)   # tau(j occ)/tau(j empty) - 1
    return np.array(d)


def job(a):
    i, y, K = a
    rng = np.random.default_rng(i)
    f = idealise_k(y, K, baseline=(COND != "white"), rng=rng)
    z = y - f["beta"]
    mu, w, P, sig, ll_free = free_fit(z, K, rng)
    idx = assign(mu, w, K)
    d = pku_strength(P, w, idx, K)
    M = 2 ** K
    dN = (M * (M - 1) + M + 1) - (2 * K + K + 1 + 1)
    e = 2 * (ll_free - f["ll"]) / dN
    S = f["S"]
    ps = []
    for p in range(K):
        for q in range(K):
            if p != q:
                g = _ll_binary_transitions(S[:-1, p], S[1:, p], S[:-1, q]) - _ll_binary_transitions(S[:-1, p], S[1:, p])
                ps.append(chi2.sf(2 * max(g, 0), 2))
    # full-text versions (R19): Albrecht level-tied 4LF + bootstrap rate test; Son single test trap2 | trap1
    import r19_albrecht_faithful as AF
    from r19c_son_faithful import son_p
    AF.COND = COND
    af = AF.job((i, y, K))[1]
    return i, dict(pku_max=float(np.max(np.abs(d))), evid=float(e), son_minp=float(min(ps)),
                   son_m=len(ps), alb_f_e=af["e"], alb_f_full=bool(af["e"] > 2 and af.get("rate_diff", False)),
                   alb_f_excl=bool(af["dwell_min"] < 2), son_f_p=son_p(S))


def rate(x):
    return float(np.mean(x))


def main():
    ys, ss, rows = O.make(100, white=(COND == "white"))
    ours = json.load(open(ROOT / f"results/r8/r9a_g1_{COND}_v3.json"))["res"]
    out = {}
    with Pool(2) as p:
        for i, r in p.imap_unordered(job, [(i, ys[i], rows[i]["K"]) for i in range(len(ys))], chunksize=2):
            out[i] = r
    summ = {}
    for K in (2, 3):
        idx = [i for i in range(len(rows)) if rows[i]["K"] == K]
        lab = np.array([rows[i]["coupled"] for i in idx])
        dfJ = K * (K - 1) // 2
        calls = dict(
            pku_0p2=np.array([out[i]["pku_max"] > 0.2 for i in idx]),
            pku_0p5=np.array([out[i]["pku_max"] > 0.5 for i in idx]),
            albrecht_e2=np.array([out[i]["evid"] > 2 for i in idx]),
            son_bonf=np.array([out[i]["son_minp"] < 0.05 / out[i]["son_m"] for i in idx]),
            ours_chi2=np.array([chi2.sf(2 * ours[str(i)]["lrt_cnn"], dfJ) < 0.05 for i in idx]))
        scores = dict(pku=np.array([out[i]["pku_max"] for i in idx]),
                      albrecht=np.array([out[i]["evid"] for i in idx]),
                      son=-np.log10(np.array([out[i]["son_minp"] for i in idx]) + 1e-300),
                      ours=np.array([ours[str(i)]["lrt_cnn"] for i in idx]))
        d = dict(n_indep=int((lab == 0).sum()), n_coupled=int((lab == 1).sum()))
        for k, c in calls.items():
            d[k] = dict(fpr=rate(c[lab == 0]), tpr=rate(c[lab == 1]))
        for k, s in scores.items():
            s0, s1 = s[lab == 0], s[lab == 1]
            thr = np.quantile(s0, 0.95)
            d[f"auc_{k}"] = float((s1[:, None] > s0[None]).mean() + 0.5 * (s1[:, None] == s0[None]).mean())
            d[f"tpr5_{k}"] = float((s1 > thr).mean())
        summ[f"K={K}"] = d
    json.dump(dict(summary=summ, res={str(k): v for k, v in out.items()}),
              open(ROOT / f"results/r8/r10_baselines_{COND}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
