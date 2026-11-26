"""R23: does the 6-chapter positive control (FPGA cross-feedback between two single-trap devices) produce a
signal our model can read?  Discrete-time simulation of the real set-up:

  traps A, B (own rates, own phi_A, phi_B in 0.2-0.8: the gate split is device physics, not 0.5)
  FPGA estimate of A:  a_hat(t) = a(t - d) xor (misclassification, prob eps per sample)   (and b_hat for B)
  B's rates  c_B exp(-phi_B J_B a_hat), e_B exp((1-phi_B) J_B a_hat)      (J_B: A -> B)
  A's rates  c_A exp(-phi_A J_A b_hat), e_A exp((1-phi_A) J_A b_hat)      (J_A: B -> A)
  analysed signal = summed currents:  y = sA a + sB (1 + kappa a_hat) b + delta_B a_hat + delta_A b_hat
                                        + white + 1/f + drift
  (delta = the gate shift seen directly in the shifted device's current; kappa = its step-size change)
  'reg' analysis: the recorded feedback outputs a_hat, b_hat are regressed out of y before the pipeline.

Fit: the paper's pipeline (R9a CNN-HMM, AR(1), 3 starts, symmetric J, phi = 0.5).  Threshold = 95 % quantile
of the J = 0 condition (feedback off).
    python experiments/r23_feedback.py [n=20]
"""
from __future__ import annotations
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
T = 8192
OUT = ROOT / "results/r23/r23_feedback.json"
NET = None

# condition: J_A, J_B, delay, eps, delta(in units of sB), kappa, analysis, shuffled
BASE = dict(JA=0.0, JB=0.0, d=0, eps=0.0, delta=0.0, kappa=0.0, ana="reg", shuf=False)
CONDS = {
    "J0_off":            dict(),
    "ideal_J1":          dict(JA=1, JB=1),
    "real_J0.5":         dict(JA=.5, JB=.5, d=2, eps=.01, delta=5, kappa=.2),
    "real_J1":           dict(JA=1, JB=1, d=2, eps=.01, delta=5, kappa=.2),
    "real_J2":           dict(JA=2, JB=2, d=2, eps=.01, delta=5, kappa=.2),
    "asym_J1_0.5":       dict(JA=.5, JB=1, d=2, eps=.01, delta=5, kappa=.2),
    "oneway_J1":         dict(JA=0, JB=1, d=2, eps=.01, delta=5, kappa=.2),
    "delay10_J1":        dict(JA=1, JB=1, d=10, eps=.01, delta=5, kappa=.2),
    "eps5_J1":           dict(JA=1, JB=1, d=2, eps=.05, delta=5, kappa=.2),
    "raw_J1":            dict(JA=1, JB=1, d=2, eps=.01, delta=5, kappa=.2, ana="raw"),
    "shuf_J1_reg":       dict(JA=1, JB=1, d=2, eps=.01, delta=5, kappa=.2, shuf=True),
    "shuf_J1_raw":       dict(JA=1, JB=1, d=2, eps=.01, delta=5, kappa=.2, shuf=True, ana="raw"),
    # R23b: gate shift as large as a strong-inversion operating point gives (dI/I ~ 40 % for dV ~ 0.1 V,
    # RTN step ~ 1-5 % of I -> delta ~ 8-40 steps, kappa ~ 0.4)
    "big_shuf_reg":      dict(JA=1, JB=1, d=3, eps=.01, delta=40, kappa=.4, shuf=True),
    "big_J0.5":          dict(JA=.5, JB=.5, d=3, eps=.01, delta=40, kappa=.4),
    "big_J1":            dict(JA=1, JB=1, d=3, eps=.01, delta=40, kappa=.4),
}


def simulate(cfg, rng):
    tau = np.exp(rng.uniform(np.log(25), np.log(80), 2))           # mean switching time scale (samples)
    p = rng.uniform(0.3, 0.7, 2)                                     # occupancy
    c, e = 2 / tau * p, 2 / tau * (1 - p)
    phi = rng.uniform(0.2, 0.8, 2)
    J = np.array([cfg["JA"], cfg["JB"]])                             # J[k]: field on trap k from the other
    d, eps = cfg["d"], cfg["eps"]
    s = np.zeros((T + d, 2), int); s[0] = rng.random(2) < p
    hat = np.zeros((T + d, 2), int)
    shuf_src = None
    if cfg["shuf"]:                                                  # feedback driven by an unrelated telegraph
        shuf_src = np.zeros((T + d, 2), int); shuf_src[0] = rng.random(2) < p
    u = rng.random((T + d, 2)); m = rng.random((T + d, 2)) < eps
    for t in range(1, T + d):
        src = shuf_src if shuf_src is not None else s
        if shuf_src is not None:
            for k in range(2):
                r = c[k] if shuf_src[t - 1, k] == 0 else e[k]
                shuf_src[t, k] = shuf_src[t - 1, k] ^ (u[t, k] < r)
            u2 = rng.random(2)
        else:
            u2 = u[t]
        for k in range(2):
            o = 1 - k
            h = J[k] * hat[t - 1, o]
            r = c[k] * np.exp(-phi[k] * h) if s[t - 1, k] == 0 else e[k] * np.exp((1 - phi[k]) * h)
            s[t, k] = s[t - 1, k] ^ (u2[k] < min(r, 0.5))
        hat[t] = src[max(t - d, 0)] ^ m[t]      # FPGA output after sample t (d = 0: the current state)
    s, hat = s[d:], hat[d:]
    sA, sB = 1.0, 0.6
    sig = sB * rng.uniform(0.15, 0.35)
    def chan(step, own, other_hat, kap, dl):
        """one device's current: own RTN (step changes with the feedback gate shift) + the gate shift itself"""
        return step * (1 + kap * other_hat) * own + dl * step * other_hat
    yA = chan(sA, s[:, 0], hat[:, 1], cfg["kappa"] * (cfg["JA"] > 0), cfg["delta"] * (cfg["JA"] > 0))
    yB = chan(sB, s[:, 1], hat[:, 0], cfg["kappa"], cfg["delta"] * (cfg["JB"] > 0))
    nz = lambda: sig / np.sqrt(2) * rng.normal(0, 1, T)
    yA, yB = yA + nz(), yB + nz()
    if cfg["ana"] == "reg":       # per channel: the recorded FPGA output of the OTHER device is regressed out
        for ych, h in ((yA, hat[:, 1]), (yB, hat[:, 0])):
            if h.std() > 0:
                X = np.c_[h, np.ones(T)]
                ych -= X[:, 0] * np.linalg.lstsq(X, ych, rcond=None)[0][0]
    y = yA + yB
    W = np.fft.rfft(rng.normal(0, 1, T)); f = np.fft.rfftfreq(T)
    fl = np.fft.irfft(W * np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1 / T)), 0), n=T); fl /= fl.std()
    y = y + 0.4 * sig * fl + 0.3 * sA * np.cumsum(rng.normal(0, 1, T)) / np.sqrt(T)
    return y, s, dict(phi=phi.tolist(), tau=tau.tolist(), sig=float(sig))


def job(a):
    global NET
    name, i = a
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    cfg = dict(BASE, **CONDS[name])
    rng = np.random.default_rng(23000 + i)            # same devices / noise draw across conditions (paired)
    y, s, info = simulate(cfg, rng)
    try:
        st, f0, f1, *_ = lrt_cnn3_ms(y, 2, NET, ar1=True, rng=np.random.default_rng(i), n_start=3)
        return name, i, dict(stat=float(st), J=float(np.atleast_1d(f1["J"])[0]),
                             trans=[int(np.abs(np.diff(s[:, k])).sum()) for k in range(2)], **info)
    except Exception as ex:
        return name, i, dict(error=repr(ex)[:150])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = json.load(open(OUT)) if OUT.exists() else {}
    tasks = [(k, i) for k in CONDS for i in range(n * (2 if k == "J0_off" else 1)) if str(i) not in res.get(k, {})]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for k, i, r in p.imap_unordered(job, tasks):
            res.setdefault(k, {})[str(i)] = r
            json.dump(res, open(OUT, "w"), indent=1)
    null = np.array([r["stat"] for r in res["J0_off"].values() if "stat" in r])
    thr = float(np.quantile(null, 0.95))
    summ = {}
    for k, v in res.items():
        ok = [r for r in v.values() if "stat" in r]
        st = np.array([r["stat"] for r in ok]); Jh = np.array([r["J"] for r in ok])
        summ[k] = dict(n=len(ok), err=len(v) - len(ok), rate_thr=float(np.mean(st > thr)), rate_chi2=float(np.mean(st > 3.84)),
                       J_med=float(np.median(Jh)), J_iqr=[float(np.quantile(Jh, .25)), float(np.quantile(Jh, .75))],
                       stat_med=float(np.median(st)))
    out = dict(thr=thr, summary=summ, res=res)
    json.dump(out, open(OUT, "w"), indent=1)
    print("thr", round(thr, 2))
    for k, s_ in summ.items():
        print(f"{k:14s} n={s_['n']:3d} err={s_['err']} det(thr)={s_['rate_thr']:.2f} det(chi2)={s_['rate_chi2']:.2f} J={s_['J_med']:.2f} {np.round(s_['J_iqr'],2)} LRTmed={s_['stat_med']:.1f}")


if __name__ == "__main__":
    main()
