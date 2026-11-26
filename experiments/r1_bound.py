"""R1: coupling-identification bound for two RTN traps.

A. D* (closest-independent KL rate): closed form vs exact, phi / occupancy / speed dependence
B. Continuous noise-free path oracle (Monte Carlo on exact simulator paths):
   E[LLR]/T -> D*, FPR(T) at 90 % sensitivity, T_min vs Stein ln(1/alpha)/D*
C. HMM oracle on point-sampled noisy readout: T_min over SNR x dt x J,
   efficiency T_lower / T_min (1 = no loss), level-degeneracy slice (equal step heights)

Time unit: tau_c = tau_e = 1 for both traps unless stated (k_i = 2).  alpha = 1 %, eps = 10 %.
Outputs results/r1/r1.json.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim import RTNConfig, generator, simulate_kinetics  # noqa: E402
from rtnsim import theory as th  # noqa: E402

OUT = ROOT / "results" / "r1"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA, EPS = 0.01, 0.1


def base(J=1.0, phi=0.5, amp2=0.6, sigma=0.5, dt=0.1, tau=(1.0, 1.0, 1.0, 1.0)):
    c = RTNConfig()
    c.traps.tau_c = [tau[0], tau[2]]
    c.traps.tau_e = [tau[1], tau[3]]
    c.traps.amp = [1.0, amp2]
    c.coupling.J, c.coupling.phi = [[0, J], [J, 0]], phi
    c.readout.sigma, c.readout.dt = sigma, dt
    return c


def tmin_from_curve(T, fpr, alpha=ALPHA):
    T, fpr = np.asarray(T, float), np.asarray(fpr, float)
    ok = np.flatnonzero(fpr <= alpha)
    if ok.size == 0:
        return float("inf")
    k = ok[0]
    if k == 0:
        return float(T[0])
    a, b = np.log(max(fpr[k - 1], 1e-9)), np.log(max(fpr[k], 1e-9))
    return float(T[k - 1] + (np.log(alpha) - a) / (b - a) * (T[k] - T[k - 1]))


# --------------------------------------------------------------------------- A
def part_a():
    rows = []
    for J in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
        for phi in (0.0, 0.5, 1.0):
            cfg = base(J, phi)
            D, _ = th.coupling_information(cfg)
            rows.append(dict(J=J, phi=phi, D=D, D_small=th.small_coupling_D(cfg)))
    occ = []
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        # trap 1 occupancy p at fixed switching rate k = 2
        tc, te = 1 / (2 * p), 1 / (2 * (1 - p))
        cfg = base(1.0, 0.5, tau=(tc, te, 1.0, 1.0))
        D, _ = th.coupling_information(cfg)
        occ.append(dict(p1=p, D=D, D_small=th.small_coupling_D(cfg)))
    speed = []
    for r in (0.01, 0.1, 1.0, 10.0, 100.0):
        cfg = base(1.0, 0.5, tau=(1.0, 1.0, 1.0 / r, 1.0 / r))
        D, _ = th.coupling_information(cfg)
        speed.append(dict(k2_over_k1=r, D=D, D_small=th.small_coupling_D(cfg)))
    return dict(J_phi=rows, occupancy=occ, speed=speed)


# --------------------------------------------------------------------------- B
def _path_job(args):
    J, hyp, n, T_end, seed = args
    cfg1 = base(J)
    _, cfg0 = th.coupling_information(cfg1)
    Q1, Q0 = generator(cfg1), generator(cfg0)
    cfg = cfg1 if hyp == 1 else cfg0
    Ts = np.geomspace(T_end / 30, T_end, 14)
    out = np.empty((len(Ts), n))
    for r in range(n):
        c = copy.deepcopy(cfg)
        c.seed = seed + r
        t_ev, s_ev, _ = simulate_kinetics(c, T_end)
        for j, Tc in enumerate(Ts):
            m = np.searchsorted(t_ev, Tc, side="right")
            out[j, r] = th.path_llr(t_ev[:m], s_ev[:m], Tc, Q1, Q0)
    return Ts, out


def part_b(pool):
    res = []
    for J in (0.5, 1.0, 2.0):
        D, _ = th.coupling_information(base(J))
        T_end = 2.5 * th.stein_time(D, ALPHA)
        jobs = [(J, 1, 250, T_end, 10_000 * k) for k in range(4)] + \
               [(J, 0, 750, T_end, 10_000_000 + 10_000 * k) for k in range(4)]
        outs = list(pool.map(_path_job, jobs))
        Ts = outs[0][0]
        l1 = np.concatenate([o[1] for o in outs[:4]], axis=1)
        l0 = np.concatenate([o[1] for o in outs[4:]], axis=1)
        thr = np.quantile(l1, EPS, axis=1)
        fpr = (l0 > thr[:, None]).mean(axis=1)
        tmin = tmin_from_curve(Ts, fpr)
        res.append(dict(J=J, D=D, stein=th.stein_time(D, ALPHA), T=Ts.tolist(), fpr=fpr.tolist(),
                        llr1_mean_over_T=(l1.mean(axis=1) / Ts).tolist(), tmin_path=tmin))
        print(f"B  J={J}: D*={D:.4f}  Stein={th.stein_time(D, ALPHA):.1f}  path-oracle T_min={tmin:.1f}  "
              f"E[LLR]/T at T_end={l1[-1].mean() / Ts[-1]:.4f}", flush=True)
    return res


# --------------------------------------------------------------------------- C
def _hmm_job(args):
    J, snr, dt, amp2, seed = args
    cfg1 = base(J, amp2=amp2, sigma=1.0 / snr, dt=dt)
    D, cfg0 = th.coupling_information(cfg1)
    t_low = th.stein_time(D, ALPHA)
    T_end = 6.0 * t_low
    n_steps = int(np.ceil(T_end / dt))
    cps = np.unique(np.geomspace(max(2, n_steps / 60), n_steps, 48).astype(int))
    rng = np.random.default_rng(seed)
    l1, cps_ = th.streaming_llr(cfg1, cfg1, cfg0, 2000, n_steps, cps, rng)
    l0, _ = th.streaming_llr(cfg0, cfg1, cfg0, 6000, n_steps, cps, rng)
    thr = np.quantile(l1, EPS, axis=1)
    fpr = (l0 > thr[:, None]).mean(axis=1)
    T = np.array(cps_) * dt
    tmin = tmin_from_curve(T, fpr)
    s_noise_free = th.sampled_kl_rate(cfg1, cfg0)
    return dict(J=J, snr=snr, dt=dt, amp2=amp2, D=D, t_lower=t_low, tmin_hmm=tmin,
                efficiency=t_low / tmin if np.isfinite(tmin) else 0.0,
                sampled_noise_free_ratio=s_noise_free / D, T=T.tolist(), fpr=fpr.tolist())


def part_c(pool):
    jobs = []
    for J in (0.5, 1.0, 2.0):
        for snr in (0.5, 1.0, 2.0, 4.0):
            for dt in (0.02, 0.1, 0.3, 1.0):
                jobs.append((J, snr, dt, 0.6, len(jobs) + 1))
    for snr in (1.0, 2.0, 4.0):
        for amp2 in (0.3, 0.9, 1.0):
            jobs.append((1.0, snr, 0.1, amp2, len(jobs) + 1))
    # noise-free reference through the identical pipeline: SNR 1000, dt = 0.005
    for J in (0.5, 1.0, 2.0):
        jobs.append((J, 1000.0, 0.005, 0.6, len(jobs) + 1))
    t0 = time.time()
    out = list(pool.map(_hmm_job, jobs))
    print(f"C  {len(jobs)} HMM conditions in {time.time() - t0:.0f}s", flush=True)
    for r in out[:48:4]:
        print(f"   J={r['J']} snr={r['snr']} dt={r['dt']}  T_lower={r['t_lower']:.0f}  T_hmm={r['tmin_hmm']:.0f}  "
              f"eff={r['efficiency']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    a = part_a()
    part = OUT / "r1_partial.json"
    saved = json.load(open(part)) if part.exists() else {}
    with ProcessPoolExecutor() as pool:
        b = saved.get("part_b") or part_b(pool)
        json.dump(dict(part_b=b), open(part, "w"), default=float)
        c = part_c(pool)
    with open(OUT / "r1.json", "w") as fh:
        json.dump(dict(alpha=ALPHA, eps=EPS, part_a=a, part_b=b, part_c=c), fh, indent=1, default=float)
