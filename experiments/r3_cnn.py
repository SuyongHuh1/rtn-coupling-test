"""R3b: confound-aware causal CNN.  Train, then evaluate with the R2 protocols on the
same data (same seeds) plus out-of-distribution confounds.

Outputs results/r3/ccnn_params.pkl, ccnn_train.json, r3_ccnn.json.
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
from rtnsim import learned_causal as LC  # noqa: E402
from rtnsim import theory as th  # noqa: E402
from rtnsim.noise import add_drift, add_flicker  # noqa: E402
import r2_detectors as R2  # noqa: E402
from r1_bound import base, tmin_from_curve  # noqa: E402

OUT = ROOT / "results" / "r3"
OUT.mkdir(parents=True, exist_ok=True)


def ood_set(name, cfg0, cfg1, n, T, seed):
    """Confounds stronger than anything in training."""
    rng = np.random.default_rng(seed)
    dt = R2.DT
    if name == "drive_very_slow_strong":
        c = copy.deepcopy(cfg0)
        c.drive.enabled, c.drive.tau_on, c.drive.tau_off, c.drive.gamma = True, 20.0, 20.0, [2.0, 2.0]
        return th.sample_readout(c, n, T, rng)
    y = th.sample_readout(cfg0, n, T, rng)
    if name == "drift_2x":
        return add_drift(y, sd_per_sqrt_time=2.0 / np.sqrt(T * dt), dt=dt, rng=rng)
    if name == "flicker_strong":
        return add_flicker(y, 1.0 * cfg0.readout.sigma, fmin=0.3 / (T * dt), fmax=0.5 / dt, n_terms=8, dt=dt, rng=rng)
    raise ValueError(name)


OOD = ("drive_very_slow_strong", "drift_2x", "flicker_strong")


def eval_confounds(score, J=1.0, snr=4.0, mult=2.0):
    cfg1 = base(J, sigma=1.0 / snr, dt=R2.DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * R2.r1_oracle_tmin(J, snr) / R2.DT))
    sc = {name: score(R2.generate_set(name, cfg0, cfg1, R2.N0, T, seed=500 + i)) for i, name in enumerate(R2.CONFOUNDS)}
    for i, name in enumerate(OOD):
        sc[name] = score(ood_set(name, cfg0, cfg1, R2.N0, T, seed=700 + i))
    thr = float(np.quantile(sc["clean"], 1 - R2.ALPHA))
    res = dict(T=T * R2.DT, fpr={n: float(np.mean(v > thr)) for n, v in sc.items() if n != "coupled"},
               power=float(np.mean(sc["coupled"] > thr)))
    print("confounds: " + " ".join(f"{n}={v:.3f}" for n, v in res["fpr"].items()) + f" power={res['power']:.3f}",
          flush=True)
    return res


def eval_efficiency(score):
    A = json.load(open(ROOT / "results" / "r2" / "r2_A.json"))
    out = []
    for cond in A:
        J, snr = cond["J"], cond["snr"]
        cfg1 = base(J, sigma=1.0 / snr, dt=R2.DT)
        _, cfg0 = th.coupling_information(cfg1)
        fprs = []
        for i, row in enumerate(cond["rows"]):
            T = int(round(row["T"] / R2.DT))
            rng = np.random.default_rng(1000 * i + int(10 * J) + int(snr))
            s1 = score(th.sample_readout(cfg1, R2.N1, T, rng))
            s0 = score(th.sample_readout(cfg0, R2.N0, T, rng))
            fprs.append(float(np.mean(s0 > np.quantile(s1, R2.EPS))))
        T_arr = [r["T"] for r in cond["rows"]]
        tm = tmin_from_curve(T_arr, fprs)
        out.append(dict(J=J, snr=snr, T=T_arr, fpr=fprs, tmin=tm,
                        efficiency=cond["tmin"]["oracle"] / tm if np.isfinite(tm) else 0.0))
        print(f"efficiency J={J} snr={snr}: T_min={tm:.0f} (oracle {cond['tmin']['oracle']:.0f})", flush=True)
    return out


if __name__ == "__main__":
    # usage: r3_cnn.py [level|diff]
    LC.INPUT = sys.argv[1] if len(sys.argv) > 1 else "level"
    tag = "" if LC.INPUT == "level" else "_diff"
    path = OUT / f"ccnn{tag}_params.pkl"
    if path.exists():
        p = LC.load(path)
    else:
        t0 = time.time()
        cache = OUT / ("ccnn_cache" if tag == "" else "ccnn_cache_diff")
        if tag and not (cache / "data.npz").exists() and (OUT / "ccnn_cache" / "data.npz").exists():
            cache.mkdir(parents=True, exist_ok=True)            # same traces, different input transform
            import shutil
            shutil.copy(OUT / "ccnn_cache" / "data.npz", cache / "data.npz")
        p, hist = LC.train(n_pairs=40000, T=1024, epochs=20, log=lambda m: print(m, flush=True), cache_dir=cache)
        LC.save(p, path)
        json.dump(dict(history=hist, seconds=time.time() - t0, params=LC.n_params(p),
                       macs_per_sample=LC.macs_per_sample(), receptive_field=LC.receptive_field()),
                  open(OUT / f"ccnn{tag}_train.json", "w"), indent=1)
    score = lambda y: LC.scores(p, y)
    res = dict(confounds=eval_confounds(score), efficiency=eval_efficiency(score))
    json.dump(res, open(OUT / f"r3_ccnn{tag}.json", "w"), indent=1, default=float)
