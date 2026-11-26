"""R2 learned detector: train the matched-pair CNN, then evaluate it on exactly the
data used for the classical detectors (same seeds) in parts A and C of r2_detectors.

Outputs results/r2/cnn_params.pkl, cnn_train.json, r2_cnn.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim import learned as L  # noqa: E402
from rtnsim import theory as th  # noqa: E402
import r2_detectors as R2  # noqa: E402
from r1_bound import base, tmin_from_curve  # noqa: E402

OUT = ROOT / "results" / "r2"


def train_model(tag, n_pairs, epochs):
    t0 = time.time()
    p, hist = L.train(n_pairs=n_pairs, T=1024, epochs=epochs, batch=64, log=lambda m: print(m, flush=True))
    L.save(p, OUT / f"cnn_params{tag}.pkl")
    json.dump(dict(history=hist, seconds=time.time() - t0, n_pairs=n_pairs, T=1024, epochs=epochs),
              open(OUT / f"cnn_train{tag}.json", "w"), indent=1)
    return p


def eval_efficiency(p):
    A = json.load(open(OUT / "r2_A.json"))
    out = []
    for cond in A:
        J, snr = cond["J"], cond["snr"]
        cfg1 = base(J, sigma=1.0 / snr, dt=R2.DT)
        _, cfg0 = th.coupling_information(cfg1)
        fprs = []
        for i, row in enumerate(cond["rows"]):
            T = int(round(row["T"] / R2.DT))
            rng = np.random.default_rng(1000 * i + int(10 * J) + int(snr))
            s1 = L.scores(p, th.sample_readout(cfg1, R2.N1, T, rng))
            s0 = L.scores(p, th.sample_readout(cfg0, R2.N0, T, rng))
            fprs.append(float(np.mean(s0 > np.quantile(s1, R2.EPS))))
        T_arr = [r["T"] for r in cond["rows"]]
        tm = tmin_from_curve(T_arr, fprs)
        out.append(dict(J=J, snr=snr, T=T_arr, fpr=fprs, tmin=tm,
                        efficiency=cond["tmin"]["oracle"] / tm if np.isfinite(tm) else 0.0))
        print(f"CNN J={J} snr={snr}: FPR {np.round(fprs, 3)} T_min={tm:.0f} (oracle {cond['tmin']['oracle']:.0f})",
              flush=True)
    return out


def eval_confounds(p, J=1.0, snr=4.0, mult=2.0):
    cfg1 = base(J, sigma=1.0 / snr, dt=R2.DT)
    _, cfg0 = th.coupling_information(cfg1)
    T = int(round(mult * R2.r1_oracle_tmin(J, snr) / R2.DT))
    res = {}
    for pipe in ("raw", "detrended"):
        sc = {}
        for i, name in enumerate(R2.CONFOUNDS):
            y = R2.generate_set(name, cfg0, cfg1, R2.N0, T, seed=500 + i)
            if pipe == "detrended":
                y = R2.detrend(y)
            sc[name] = L.scores(p, y)
        thr = float(np.quantile(sc["clean"], 1 - R2.ALPHA))
        res[pipe] = dict(fpr={n: float(np.mean(sc[n] > thr)) for n in R2.CONFOUNDS if n != "coupled"},
                         power=float(np.mean(sc["coupled"] > thr)))
        print(f"CNN {pipe}: " + " ".join(f"{n}={v:.3f}" for n, v in res[pipe]["fpr"].items()) +
              f" power={res[pipe]['power']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    # usage: r2_cnn.py [tag n_pairs epochs]; default = the first 6-epoch model
    tag, n_pairs, epochs = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ("", 30000, 6)
    path = OUT / f"cnn_params{tag}.pkl"
    p = L.load(path) if path.exists() else train_model(tag, n_pairs, epochs)
    res = dict(confounds=eval_confounds(p), efficiency=eval_efficiency(p), n_pairs=n_pairs, epochs=epochs)
    json.dump(res, open(OUT / f"r2_cnn{tag}.json", "w"), indent=1, default=float)
