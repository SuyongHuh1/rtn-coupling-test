"""R5f: does the idealisation-aware null keep its level, and does it keep power?

Synthetic records shaped like the public data (two traps, steps 1 and 0.6, ~300 switches per
trap) with coloured noise (white + 1/f-like) and a random-walk drift; conditions: independent,
coupled J = 1 and 2 kT.  For each record the pipeline null of r5_pipeline_null.py is built
from the record itself and the rejection rate at 5% is reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim import theory as th  # noqa: E402
from rtnsim.noise import add_drift, add_flicker  # noqa: E402
from rtnsim.robust import idealise  # noqa: E402
from r5_pipeline_null import null_stats  # noqa: E402
from r5_surrogate_validation import cfg  # noqa: E402

OUT = ROOT / "results" / "r5"


def main(n_rec=12, n_sim=30, T_tau=300):
    rng = np.random.default_rng(5)
    res = {}
    for name, J in (("independent", 0.0), ("coupled J=1", 1.0), ("coupled J=2", 2.0)):
        c = cfg(J=J)
        c.traps.amp, c.readout.sigma = [1.0, 0.6], 0.2
        n = int(T_tau / c.readout.dt)
        y = th.sample_readout(c, n_rec, n, rng)
        y = add_flicker(y, 0.1, fmin=0.3 / (T_tau), fmax=0.5 / c.readout.dt, n_terms=8, dt=c.readout.dt, rng=rng)
        y = add_drift(y, sd_per_sqrt_time=0.5 / np.sqrt(T_tau), dt=c.readout.dt, rng=rng)
        ps = []
        for i in range(n_rec):
            idl = idealise(y[i], baseline=True)
            obs, null = null_stats([(y[i], idl)], rng, n_sim, 101)
            ps.append((1 + np.sum(null >= obs)) / (n_sim + 1))
        res[name] = dict(reject_5pct=float(np.mean(np.array(ps) <= 0.05)), p=[float(p) for p in ps])
        print(name, res[name]["reject_5pct"], flush=True)
    json.dump(res, open(OUT / "r5_pipeline_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
