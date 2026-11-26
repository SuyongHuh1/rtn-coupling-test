"""R5a: validate the global/local shift-surrogate test on exact simulated paths.

Conditions: independent traps; kinetic coupling J = 1, 2 kT; slow shared drive (same and
opposite sign, gamma = 1.5 kT) with dwell 5, 20, 50, 200 trap times.  Two record lengths
matched to the public data (~150 and ~600 switches per trap).  Rejection rate at 5%.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim import RTNConfig  # noqa: E402
from rtnsim import theory as th  # noqa: E402
from rtnsim.surrogate import shift_test  # noqa: E402

OUT = ROOT / "results" / "r5"


def cfg(J=0.0, drive=None, sign=1, dt=0.1):
    c = RTNConfig()
    c.traps.tau_c, c.traps.tau_e = [1.0, 1.0], [1.0, 1.0]
    c.coupling.J, c.coupling.phi = ([[0, J], [J, 0]] if J else None), 0.5
    c.readout.dt = dt
    if drive:
        c.drive.enabled, c.drive.tau_on, c.drive.tau_off = True, drive, drive
        c.drive.gamma = [1.5, 1.5 * sign]
    return c


def run(c, T_tau, n_rep, rng):
    n = int(T_tau / c.readout.dt)
    x = th.sample_states(c, n_rep, n, rng)
    s1, s2 = x & 1, (x >> 1) & 1
    pg, pl = [], []
    for i in range(n_rep):
        _, a, b = shift_test(s1[i], s2[i], rng, n_sur=199)
        pg.append(a)
        pl.append(b)
    pg, pl = np.array(pg), np.array(pl)
    return dict(reject_global=float(np.mean(pg < 0.05)), reject_local=float(np.mean(pl < 0.05)))


def main(n_rep=200):
    rng = np.random.default_rng(1)
    conds = [("independent", cfg())] + [(f"coupled J={J}", cfg(J=J)) for J in (1.0, 2.0)]
    for d in (5, 20, 50, 200):
        conds.append((f"drive same dwell={d}", cfg(drive=d, sign=1)))
        conds.append((f"drive opposite dwell={d}", cfg(drive=d, sign=-1)))
    res = {}
    for T in (150, 600):
        for name, c in conds:
            r = run(c, T, n_rep, rng)
            res[f"T={T} {name}"] = r
            print(T, name, r, flush=True)
    json.dump(res, open(OUT / "r5_surrogate_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
