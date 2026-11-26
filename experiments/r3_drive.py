"""R3a: drive-aware GLRT vs the R2 GLRT.

1. Same confound sets as R2 part C (T = 2x oracle T_min, J = 1 kT, SNR 4): FPR at a
   threshold set to 1% on clean independent data, power on coupled data.
2. Timescale separation: independent traps + shared drive (gamma = 1, 1 kT) with drive
   dwell tau_d in {0.5, 2, 5, 20, 50} trap times; FPR of both tests vs tau_d.

Outputs results/r3/r3_drive.json.
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
from rtnsim import drive_hmm as DH  # noqa: E402
import r2_detectors as R2  # noqa: E402
from r1_bound import base  # noqa: E402

OUT = ROOT / "results" / "r3"
OUT.mkdir(parents=True, exist_ok=True)
P_MAX = 0.01
N = 1500


def scores(y):
    return dict(drive_aware=DH.drive_aware_scores(y, p_max=P_MAX), glrt=dd.detector_scores(y)["glrt"])


def main(J=1.0, snr=4.0, mult=2.0):
    cfg1 = base(J, sigma=1.0 / snr, dt=R2.DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * R2.r1_oracle_tmin(J, snr) / R2.DT))
    res = dict(J=J, snr=snr, T=T * R2.DT, n=N, p_max=P_MAX)
    sc = {}
    for i, name in enumerate(R2.CONFOUNDS):
        t0 = time.time()
        sc[name] = scores(R2.generate_set(name, cfg0, cfg1, N, T, seed=500 + i))
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    for k, tau_d in enumerate((0.5, 2.0, 5.0, 20.0, 50.0)):
        c = copy.deepcopy(cfg0)
        c.drive.enabled, c.drive.tau_on, c.drive.tau_off, c.drive.gamma = True, tau_d, tau_d, [1.0, 1.0]
        sc[f"drive_tau{tau_d:g}"] = scores(th.sample_readout(c, N, T, np.random.default_rng(800 + k)))
        print(f"drive tau {tau_d}: done", flush=True)
    thr = {d: float(np.quantile(sc["clean"][d], 1 - R2.ALPHA)) for d in ("drive_aware", "glrt")}
    res["fpr"] = {n: {d: float(np.mean(v[d] > thr[d])) for d in thr} for n, v in sc.items() if n != "coupled"}
    res["power"] = {d: float(np.mean(sc["coupled"][d] > thr[d])) for d in thr}
    json.dump(res, open(OUT / "r3_drive.json", "w"), indent=1)
    for n, f in res["fpr"].items():
        print(f"FPR {n:22s} drive_aware={f['drive_aware']:.3f} glrt={f['glrt']:.3f}", flush=True)
    print(f"power drive_aware={res['power']['drive_aware']:.3f} glrt={res['power']['glrt']:.3f}", flush=True)


if __name__ == "__main__":
    main()
