"""R2: practical detectors vs the oracle, and false positives under confounds.

A. Efficiency.  For each (J, SNR) at dt = 0.1 tau_c: fresh H1 (coupled) and H0*
   (closest independent) data at T = {0.5, 1, 2, 4, 8} x oracle T_min; every detector
   thresholded at its own 10% H1 quantile (90% sensitivity); FPR on H0*; T_min by
   log-interpolation; efficiency = T_min(oracle, same data) / T_min(detector).
B. Confounds (J = 1 kT, SNR = 4, dt = 0.1, T = 4 x oracle T_min).  Thresholds are
   calibrated on clean H0* data to a nominal 1% FPR, then applied to independent
   traps with a confound: shared drive (fast / slow, same / opposite sign),
   readout coupling, 1/f noise, drift.  Reported: actual FPR, and power on
   coupled data.

Outputs results/r2/r2_A.json, r2_B.json.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim import theory as th  # noqa: E402
from rtnsim import detectors as dd  # noqa: E402
from rtnsim.noise import add_flicker, add_drift  # noqa: E402
from r1_bound import base, tmin_from_curve  # noqa: E402

OUT = ROOT / "results" / "r2"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA, EPS, DT = 0.01, 0.1, 0.1
N1, N0 = 1000, 3000
DETS = ("glrt", "glrt_general", "corr", "ideal_lrt", "oracle")


def r1_oracle_tmin(J, snr, dt=DT):
    r = json.load(open(ROOT / "results" / "r1" / "r1.json"))
    x = next(z for z in r["part_c"] if z["J"] == J and z["snr"] == snr and z["dt"] == dt and z["amp2"] == 0.6)
    return x["tmin_hmm"]


def all_scores(y, cfg1, cfg0):
    s = dd.detector_scores(y)
    s["oracle"] = th.hmm_loglik(y, cfg1) - th.hmm_loglik(y, cfg0)
    return s


def part_a(conds, mults=(0.5, 1, 2, 4, 8), max_samples=40000, append=False):
    out = json.load(open(OUT / "r2_A.json")) if append and (OUT / "r2_A.json").exists() else []
    for J, snr in conds:
        cfg1 = base(J, sigma=1.0 / snr, dt=DT)
        _, cfg0 = th.coupling_information(cfg1)
        t_or = r1_oracle_tmin(J, snr)
        Ts = [int(round(m * t_or / DT)) for m in mults if m * t_or / DT <= max_samples]
        rows = []
        for i, T in enumerate(Ts):
            t0 = time.time()
            rng = np.random.default_rng(1000 * i + int(10 * J) + int(snr))
            y1 = th.sample_readout(cfg1, N1, T, rng)
            s1 = all_scores(y1, cfg1, cfg0)
            del y1
            y0 = th.sample_readout(cfg0, N0, T, rng)
            s0 = all_scores(y0, cfg1, cfg0)
            del y0
            row = dict(T=T * DT)
            for d in DETS:
                thr = np.quantile(s1[d], EPS)
                row[f"fpr_{d}"] = float(np.mean(s0[d] > thr))
                row[f"h0_median_{d}"] = float(np.median(s0[d]))
                row[f"h1_median_{d}"] = float(np.median(s1[d]))
            rows.append(row)
            print(f"A J={J} snr={snr} T={T * DT:.0f}: " +
                  " ".join(f"{d}={row['fpr_' + d]:.3f}" for d in DETS) + f"  ({time.time() - t0:.0f}s)", flush=True)
        T_arr = [r["T"] for r in rows]
        tmins = {d: tmin_from_curve(T_arr, [r[f"fpr_{d}"] for r in rows]) for d in DETS}
        eff = {d: (tmins["oracle"] / tmins[d] if np.isfinite(tmins[d]) else 0.0) for d in DETS}
        out.append(dict(J=J, snr=snr, dt=DT, r1_oracle_tmin=t_or, rows=rows, tmin=tmins, efficiency=eff))
        print(f"A J={J} snr={snr}: T_min " + " ".join(f"{d}={tmins[d]:.0f}" for d in DETS), flush=True)
        json.dump(out, open(OUT / "r2_A.json", "w"), indent=1, default=float)
    return out


def confound_sets(cfg0):
    sets = {}
    c = copy.deepcopy(cfg0)
    sets["clean"] = (c, None)
    for name, tau_d, g in (("drive_slow_same", 5.0, (1.0, 1.0)), ("drive_fast_same", 0.5, (1.0, 1.0)),
                           ("drive_slow_opposite", 5.0, (1.0, -1.0))):
        c = copy.deepcopy(cfg0)
        c.drive.enabled, c.drive.tau_on, c.drive.tau_off, c.drive.gamma = True, tau_d, tau_d, list(g)
        sets[name] = (c, None)
    c = copy.deepcopy(cfg0)
    c.coupling.B = [[0, 0.3], [0, 0]]
    sets["readout_coupling"] = (c, None)
    sets["flicker"] = (copy.deepcopy(cfg0), ("flicker", 0.5 * cfg0.readout.sigma))
    sets["drift"] = (copy.deepcopy(cfg0), ("drift", None))
    return sets


def part_b(J=1.0, snr=4.0, mult=4.0):
    cfg1 = base(J, sigma=1.0 / snr, dt=DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * r1_oracle_tmin(J, snr) / DT))
    rng = np.random.default_rng(77)
    res = dict(J=J, snr=snr, T=T * DT, n=N0)
    scores = {}
    for name, (cfg, extra) in confound_sets(cfg0).items():
        t0 = time.time()
        y = th.sample_readout(cfg, N0, T, rng)
        if extra is not None and extra[0] == "flicker":
            y = add_flicker(y, extra[1], fmin=0.3 / (T * DT), fmax=0.5 / DT, n_terms=8, dt=DT, rng=rng)
        if extra is not None and extra[0] == "drift":
            # random walk reaching ~ one step height (A1 = 1) rms over the record
            y = add_drift(y, sd_per_sqrt_time=1.0 / np.sqrt(T * DT), dt=DT, rng=rng)
        scores[name] = all_scores(y, cfg1, cfg0)
        print(f"B {name}: done ({time.time() - t0:.0f}s)", flush=True)
    y = th.sample_readout(cfg1, N1, T, rng)
    scores["coupled"] = all_scores(y, cfg1, cfg0)
    thr = {d: float(np.quantile(scores["clean"][d], 1 - ALPHA)) for d in DETS}
    res["threshold"] = thr
    res["fpr"] = {name: {d: float(np.mean(scores[name][d] > thr[d])) for d in DETS}
                  for name in scores if name != "coupled"}
    res["power"] = {d: float(np.mean(scores["coupled"][d] > thr[d])) for d in DETS}
    json.dump(res, open(OUT / "r2_B.json", "w"), indent=1, default=float)
    for name, f in res["fpr"].items():
        print(f"B FPR {name:22s} " + " ".join(f"{d}={f[d]:.3f}" for d in DETS), flush=True)
    print("B power (coupled)      " + " ".join(f"{d}={res['power'][d]:.3f}" for d in DETS), flush=True)
    return res


def generate_set(name, cfg0, cfg1, n, T, seed):
    """Deterministic data for one confound condition (shared with the CNN evaluation)."""
    rng = np.random.default_rng(seed)
    if name == "coupled":
        return th.sample_readout(cfg1, n, T, rng)
    cfg, extra = confound_sets(cfg0)[name]
    y = th.sample_readout(cfg, n, T, rng)
    if extra is not None and extra[0] == "flicker":
        y = add_flicker(y, extra[1], fmin=0.3 / (T * DT), fmax=0.5 / DT, n_terms=8, dt=DT, rng=rng)
    if extra is not None and extra[0] == "drift":
        y = add_drift(y, sd_per_sqrt_time=1.0 / np.sqrt(T * DT), dt=DT, rng=rng)
    return y


def detrend(y, window=201):
    """Remove slow baseline: subtract a running median much longer than the dwell times."""
    from scipy.ndimage import median_filter
    base = median_filter(y, size=(1, window), mode="nearest")
    return y - base + np.median(y, axis=1, keepdims=True)


CONFOUNDS = ("clean", "drive_slow_same", "drive_fast_same", "drive_slow_opposite", "readout_coupling",
             "flicker", "drift", "coupled")


def part_c(J=1.0, snr=4.0, mult=2.0):
    """Raw vs detrended pipelines at T = 2 x oracle T_min (power not saturated)."""
    cfg1 = base(J, sigma=1.0 / snr, dt=DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * r1_oracle_tmin(J, snr) / DT))
    res = dict(J=J, snr=snr, T=T * DT, n=N0)
    for pipe in ("raw", "detrended"):
        sc = {}
        for i, name in enumerate(CONFOUNDS):
            y = generate_set(name, cfg0, cfg1, N0, T, seed=500 + i)
            if pipe == "detrended":
                y = detrend(y)
            sc[name] = all_scores(y, cfg1, cfg0)
        thr = {d: float(np.quantile(sc["clean"][d], 1 - ALPHA)) for d in DETS}
        res[pipe] = dict(fpr={n: {d: float(np.mean(sc[n][d] > thr[d])) for d in DETS} for n in CONFOUNDS if n != "coupled"},
                         power={d: float(np.mean(sc["coupled"][d] > thr[d])) for d in DETS})
        for n, f in res[pipe]["fpr"].items():
            print(f"C {pipe:9s} FPR {n:20s} " + " ".join(f"{d}={f[d]:.3f}" for d in DETS), flush=True)
        print(f"C {pipe:9s} power " + " ".join(f"{d}={res[pipe]['power'][d]:.3f}" for d in DETS), flush=True)
        json.dump(res, open(OUT / "r2_C.json", "w"), indent=1, default=float)
    return res


def part_d(J=1.0, snr=4.0, mult=2.0, levels=(0.03, 0.1, 0.3, 1.0), n=1500):
    """Drift tolerance: random-walk drift whose end-of-record rms is `level` step heights.
    Raw pipeline vs model-based baseline correction (dd.baseline_correct)."""
    cfg1 = base(J, sigma=1.0 / snr, dt=DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * r1_oracle_tmin(J, snr) / DT))
    dets = ("glrt", "corr", "ideal_lrt")
    res = dict(J=J, snr=snr, T=T * DT, n=n, levels=list(levels), raw={}, corrected={})

    def data(cfg, lvl, seed):
        rng = np.random.default_rng(seed)
        y = th.sample_readout(cfg, n, T, rng)
        if lvl > 0:
            y = add_drift(y, sd_per_sqrt_time=lvl / np.sqrt(T * DT), dt=DT, rng=rng)
        return y

    for pipe in ("raw", "corrected"):
        prep = (lambda y: y) if pipe == "raw" else dd.baseline_correct
        clean = dd.detector_scores(prep(data(cfg0, 0.0, 900)))
        thr = {d: float(np.quantile(clean[d], 1 - ALPHA)) for d in dets}
        pw = dd.detector_scores(prep(data(cfg1, 0.0, 901)))
        res[pipe]["power_clean"] = {d: float(np.mean(pw[d] > thr[d])) for d in dets}
        for k, lvl in enumerate(levels):
            s0 = dd.detector_scores(prep(data(cfg0, lvl, 910 + k)))
            s1 = dd.detector_scores(prep(data(cfg1, lvl, 920 + k)))
            res[pipe][str(lvl)] = dict(fpr={d: float(np.mean(s0[d] > thr[d])) for d in dets},
                                       power={d: float(np.mean(s1[d] > thr[d])) for d in dets})
            print(f"D {pipe:9s} drift={lvl}: FPR " + " ".join(f"{d}={res[pipe][str(lvl)]['fpr'][d]:.3f}" for d in dets)
                  + " | power " + " ".join(f"{d}={res[pipe][str(lvl)]['power'][d]:.3f}" for d in dets), flush=True)
        json.dump(res, open(OUT / "r2_D.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "D" in which:
        part_d()
    if "C" in which:
        part_c()
    if "B" in which:
        part_b()
    if "A" in which:
        part_a([(2.0, 4.0), (2.0, 2.0), (1.0, 4.0), (1.0, 2.0), (0.5, 4.0)])
    if "E" in which:   # finish the weak-coupling condition with a shorter grid
        part_a([(0.5, 4.0)], mults=(0.5, 1, 2, 4), append=True)
