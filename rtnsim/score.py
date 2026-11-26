"""Co-flip score test next to the J-only likelihood-ratio test (R36).

The J-only pipeline is left untouched.  A second statistic asks whether the fitted J = 0 model leaves an
excess of *simultaneous* two-trap transitions: for each pair p the generator gets a co-flip rate kappa_p
(00 <-> 11 of that pair) and U_p = d loglik / d kappa_p at kappa = 0 is evaluated by Fisher's identity.
kappa is never fitted, so the J estimate, its interval and its null distribution do not change.

Observation model = acquisition:
  acq = "point"  each sample is the instantaneous state (simulations); score from the pipeline's own J = 0 fit
  acq = "integ"  each sample is the average over the sampling interval (every measured record here);
                 the J = 0 model is refitted with the exact integrated-sampling emission (rtnsim.integ) and
                 the score is taken there, so a transition inside a sample is explained by the emission
                 instead of being read as a visit to another state.
Null records are simulated through the same acquisition and pass through the whole pipeline; one set of
nulls calibrates both statistics.  Verdict: Bonferroni over the two calibrated p-values.
"""
from __future__ import annotations

import numpy as np
import torch

from .robustk import _fb, bits, _block_boot


def _pairs(K):
    out = []
    for k in range(K):
        for l in range(k + 1, K):
            m = (1 << k) | (1 << l)
            src = [s for s in range(2 ** K) if not s & m]
            out.append((src, [s | m for s in src]))
    return out


def _add_kappa(Q0, kap, K):
    off = Q0 - torch.diag(torch.diagonal(Q0))
    for p, (src, dst) in enumerate(_pairs(K)):
        add = torch.zeros_like(off)
        add[src, dst] = 1.0
        add[dst, src] = 1.0
        off = off + kap[p] * add
    return off - torch.diag(off.sum(1))


def _Q0(c, e, K):
    from .sim2_torch import generator
    t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
    Q = generator(t(c)[None], t(e)[None], torch.zeros(1, K, K, dtype=torch.float64),
                  torch.tensor([0.5], dtype=torch.float64))[0]
    off = Q - torch.diag(torch.diagonal(Q))
    return off - torch.diag(off.sum(1))


def score_point(z, K, f):
    """U for the point-sampling model; f = J = 0 fit (white or AR(1) emission)."""
    Q0 = _Q0(f["c"], f["e"], K)
    P = torch.linalg.matrix_exp(Q0).numpy()
    P = P / P.sum(1, keepdims=True)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    pi0 = np.full(2 ** K, 1.0 / 2 ** K)
    mu = X @ np.asarray(f["coef"], float)
    rho = float(f.get("rho", 0.0) or 0.0)
    if rho != 0.0:
        from .cfhmm import _fb_ar1
        N = _fb_ar1(z, P, pi0, mu, float(f["sig"]), rho, False)[1]
    else:
        N = _fb(z, P, pi0, mu, float(f["sig"]), False)[1]
    kap = torch.zeros(K * (K - 1) // 2, dtype=torch.float64, requires_grad=True)
    q = (torch.tensor(N) * torch.log(torch.linalg.matrix_exp(_add_kappa(Q0, kap, K)).clamp_min(1e-300))).sum()
    q.backward()
    return kap.grad.numpy()


def score_integ(z, K, f):
    """U for the integrated-sampling model; f = fit_integ_qn J = 0 fit."""
    from .integ import interval_moments, _fb_pair
    Q0 = _Q0(f["c"], f["e"], K)
    coef = torch.tensor(np.asarray(f["coef"], float))
    sig = float(f["sig"])
    off = float(np.atleast_1d(f.get("off", 0.0))[0])

    def em(Q):
        P, W, C = interval_moments(Q, K)
        a = coef[1:]
        return P, coef[0] + W @ a, sig ** 2 + torch.einsum("ijkl,k,l->ij", C, a, a).clamp_min(0)
    with torch.no_grad():
        P, m, v = em(Q0)
        P = P.numpy()
        P = P / P.sum(1, keepdims=True)
    n, sy, syy = (torch.tensor(x) for x in _fb_pair(z - off, P, np.full(2 ** K, 1.0 / 2 ** K), m.numpy(), v.numpy())[1:4])
    kap = torch.zeros(K * (K - 1) // 2, dtype=torch.float64, requires_grad=True)
    P_, m_, v_ = em(_add_kappa(Q0, kap, K))
    q = (n * torch.log(P_)).sum() - 0.5 * (n * torch.log(2 * np.pi * v_) + (syy - 2 * m_ * sy + m_ * m_ * n) / v_).sum()
    q.backward()
    return kap.grad.numpy()


def coflip_score(z, K, f0, acq):
    """Score vector and the J = 0 fit it was taken at.  z = record minus pipeline baseline."""
    if acq == "point":
        return score_point(z, K, f0), f0
    from .integ import fit_integ_qn
    ini = dict(coef=np.asarray(f0["coef"], float), sig=float(f0["sig"]), c=np.asarray(f0["c"], float), e=np.asarray(f0["e"], float))
    fi = fit_integ_qn([z], K, False, ini)
    return score_integ(z, K, fi), fi


def boxcar_occ(c, e, T, rng):
    """Independent traps, exact average occupancy over each sampling interval (event-driven)."""
    K = len(c)
    occ = np.empty((T, K))
    edges = np.arange(T + 1.0)
    for k in range(K):
        st = int(rng.random() < c[k] / (c[k] + e[k]))
        n = int(3 * T * max(c[k], e[k]) + 50)
        while True:
            d = rng.exponential(1.0, 2 * n)
            d[0::2] /= (c[k] if st == 0 else e[k])
            d[1::2] /= (e[k] if st == 0 else c[k])
            t = np.cumsum(d)
            if t[-1] > T:
                break
            n *= 2
        tt = np.r_[0.0, t]
        s_i = (st ^ (np.arange(tt.size - 1) & 1)).astype(float)
        I = np.r_[0.0, np.cumsum(s_i * np.diff(tt))]
        occ[:, k] = np.diff(np.interp(edges, tt, I))
    return occ


def simulate_null_integ(fi, beta, resid, T, rng):
    """J = 0 record through an integrating acquisition: boxcar-averaged chain from the integrated-model
    fit, the record's baseline, block-resampled residuals (samples that hold an in-sample transition are
    outliers of the point-model residual and are replaced before resampling)."""
    r = resid - np.median(resid)
    mad = np.median(np.abs(r)) / 0.6745 + 1e-300
    r = np.where(np.abs(r) > 5 * mad, rng.normal(0, mad, r.size), r)
    coef = np.asarray(fi["coef"], float)
    return beta + coef[0] + boxcar_occ(fi["c"], fi["e"], T, rng) @ coef[1:] + _block_boot(r, T, rng)


def _T(U, s):
    return float(np.sum(np.maximum(np.asarray(U) / s, 0.0) ** 2))


def score_p(U_obs, U_null):
    """One-sided (kappa >= 0) statistic sum_p max(U_p / s_p, 0)^2 with s_p the spread of U_p over the
    observed record and its nulls together (exchangeable under H0, so the rank p-value stays exact)."""
    A = np.vstack([np.atleast_1d(U_obs)] + [np.atleast_1d(u) for u in U_null])
    s = A.std(0) + 1e-300
    t = np.array([_T(a, s) for a in A])
    return float((1 + np.sum(t[1:] >= t[0])) / len(t)), float(t[0]), t[1:].tolist()


def joint_test(y, K, pipe, rng, n_sim=39, acq="point", alpha=0.05, prefilter=None, null_seed=None):
    """pipe(y, rng) -> (stat, f0, f1, beta, ...) is the unchanged J-only pipeline.
    Returns both calibrated p-values from one set of pipeline nulls and the Bonferroni verdict."""
    stat, f0, f1, beta = pipe(y, rng)[:4]
    z = y - beta
    U, fs = coflip_score(z, K, f0, acq)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    resid = z - (X @ np.asarray(f0["coef"]))[f0["path"]]
    out = dict(stat=float(stat), J=[float(x) for x in np.atleast_1d(f1["J"])], U=U.tolist(), acq=acq,
               c=np.asarray(fs["c"]).tolist(), e=np.asarray(fs["e"]).tolist(), coef=np.asarray(fs["coef"]).tolist(),
               sig=float(fs["sig"]), c_point=np.asarray(f0["c"]).tolist(), e_point=np.asarray(f0["e"]).tolist())
    if acq == "integ":
        out["ll_integ"], out["ll_point"] = float(fs["ll"]), float(f0["ll"])
    if not n_sim:
        return out
    if acq == "point":
        from .cfhmm import simulate_null
        sim = lambda: simulate_null(f0, beta, resid, len(y), rng)
    else:
        sim = lambda: simulate_null_integ(fs, beta, resid, len(y), rng)
    if null_seed is not None:                              # extra nulls drawn independently of the first batch
        rng = np.random.default_rng(null_seed)
    nJ, nU, fail = [], [], 0
    for _ in range(n_sim):
        yb = sim()
        try:
            sb, g0, _, bb = pipe(yb, rng)[:4]
            ub, _ = coflip_score(yb - bb, K, g0, acq)
        except Exception:                                  # noqa: BLE001 -- failed null: dropped, B stored
            fail += 1
            continue
        nJ.append(float(sb)); nU.append(ub.tolist())
    pk, T, Tn = score_p(U, nU)
    pJ = float((1 + np.sum(np.array(nJ) >= stat)) / (1 + len(nJ)))
    out.update(p_J=pJ, p_kappa=pk, T=T, null_J=nJ, null_U=nU, null_T=Tn, n_null=len(nJ), n_fail=fail,
               flag_J=bool(pJ <= alpha / 2), flag_kappa=bool(pk <= alpha / 2), flag=bool(min(pJ, pk) <= alpha / 2))
    return out


ALPHA_J, ALPHA_K = 0.04, 0.01          # split of the 0.05 family-wise level (B = 99 nulls)


def merge_nulls(r, extra):
    """Append the extra nulls of `extra` to record r (same observed statistics), recompute both p-values on
    all nulls and the split-level verdict.  Exact: the nulls are i.i.d. given the record's fit."""
    nJ = list(r["null_J"]) + list(extra["null_J"])
    nU = list(r["null_U"]) + list(extra["null_U"])
    pk, T, Tn = score_p(r["U"], nU)
    pJ = float((1 + np.sum(np.array(nJ) >= r["stat"])) / (1 + len(nJ)))
    r.update(null_J=nJ, null_U=nU, null_T=Tn, T=T, p_J=pJ, p_kappa=pk, n_null=len(nJ),
             n_fail=r.get("n_fail", 0) + extra.get("n_fail", 0), stat_rerun=extra["stat"], U_rerun=extra["U"],
             flag_J=bool(pJ <= ALPHA_J), flag_kappa=bool(pk <= ALPHA_K), flag=bool(pJ <= ALPHA_J or pk <= ALPHA_K),
             extended=True)
    return r


def split_verdict(r, B=99):
    """Verdict at the split levels for a record that was not extended: decided from the 39 nulls alone.
    With n_ex exceedances among the first 39, p on 99 nulls is at least (1 + n_ex) / (B + 1)."""
    exJ = int(round(r["p_J"] * (1 + r["n_null"]))) - 1
    exK = int(round(r["p_kappa"] * (1 + r["n_null"]))) - 1
    return dict(need_J=(1 + exJ) / (B + 1) <= ALPHA_J, need_K=(1 + exK) / (B + 1) <= ALPHA_K)
