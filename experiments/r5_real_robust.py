"""R5c: drift-robust two-trap idealisation + shift-surrogate coupling test on public RTN.

For each record (same preprocessing as r5_real.py):
  * additive two-trap idealisation with a slow running-median baseline (rtnsim.robust);
  * resolvability: both traps >= 20 switches and the smaller step >= 1.5 noise sd;
  * conditional-rate statistic vs 999 global and 999 local (1-4 dwell) shift surrogates;
  * the same without baseline correction (drift sensitivity);
  * R1: smallest |J| identifiable at this record length from the idealised rates.
Output: results/r5/r5_robust.json, fig2_robust.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.realdata import load_b1500  # noqa: E402
from rtnsim.robust import idealise  # noqa: E402
from rtnsim.surrogate import shift_test, mean_dwell  # noqa: E402
from rtnsim import theory as th  # noqa: E402
from rtnsim import RTNConfig  # noqa: E402
import r5_real as R  # noqa: E402

OUT = ROOT / "results" / "r5"


def rates_from_path(s, dt):
    s = np.asarray(s)
    up = np.sum((s[:-1] == 0) & (s[1:] == 1))
    dn = np.sum((s[:-1] == 1) & (s[1:] == 0))
    t0, t1 = np.sum(s[:-1] == 0) * dt, np.sum(s[:-1] == 1) * dt
    return up / max(t0, dt), dn / max(t1, dt), int(up + dn)


def jmin(r1, r2, T_rec):
    for J in np.r_[np.linspace(0.1, 3, 30), np.linspace(3.5, 8, 10)]:
        c = RTNConfig()
        c.traps.tau_c, c.traps.tau_e = [1 / max(r1[0], 1e-9), 1 / max(r2[0], 1e-9)], [1 / max(r1[1], 1e-9), 1 / max(r2[1], 1e-9)]
        c.coupling.J, c.coupling.phi = [[0, J], [J, 0]], 0.5
        D, _ = th.coupling_information(c)
        if 1.4 * np.log(100) / D <= T_rec:
            return float(J)
    return float("inf")


def main():
    rng = np.random.default_rng(7)
    out, keep = [], {}
    for r in R.records():
        t, I, _ = load_b1500(r["path"])
        t, y, n_spk, dt = R.preprocess(t, I, r["trim_s"])
        row = dict(set=r["set"], file=r["file"], T=len(y), dt=dt, duration_s=float(t[-1] - t[0]))
        for tag, base in (("baseline", True), ("raw", False)):
            idl = idealise(y, baseline=base)
            r1, r2 = rates_from_path(idl["s1"], dt), rates_from_path(idl["s2"], dt)
            resolvable = bool(min(r1[2], r2[2]) >= 20 and abs(idl["a2"]) >= 1.5 * idl["sig"])
            stat, pg, pl = shift_test(idl["s1"], idl["s2"], rng, n_sur=999)
            row[tag] = dict(a1=idl["a1"], a2=idl["a2"], sig=idl["sig"], switches=[r1[2], r2[2]],
                            rates_per_s=[r1[:2], r2[:2]], resolvable=resolvable, stat=stat,
                            p_global=pg, p_local=pl,
                            jmin_kT=jmin(r1, r2, row["duration_s"]) if resolvable else None)
            if tag == "baseline":
                keep[r["file"]] = (y, idl, dt)
        print(r["file"][:44], {k: (v if not isinstance(v, dict) else
                                   {kk: v[kk] for kk in ("switches", "resolvable", "p_global", "p_local", "jmin_kT")})
                                for k, v in row.items() if k in ("baseline", "raw")}, flush=True)
        out.append(row)
    json.dump(out, open(OUT / "r5_robust.json", "w"), indent=1, default=float)
    figure(out, keep)


def figure(out, keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [r for r in out if r["baseline"]["resolvable"]]
    if not sel:
        return
    fig, axs = plt.subplots(len(sel), 1, figsize=(10, 1.7 * len(sel)), squeeze=False)
    for ax, r in zip(axs[:, 0], sel):
        y, idl, dt = keep[r["file"]]
        tt = np.arange(len(y)) * dt
        ax.plot(tt, y, lw=0.3, color="#9aa3b2")
        fit = idl["beta"] + idl["a1"] * idl["s1"] + idl["a2"] * idl["s2"]
        ax.plot(tt, fit - np.median(fit - y), lw=0.6, color="#2b6cd6")
        b = r["baseline"]
        ax.set_title(f"{r['set']} {r['file'][:38]}  switches {b['switches']}  p_global={b['p_global']:.3f}  "
                     f"p_local={b['p_local']:.3f}  J_min={b['jmin_kT']:.1f} kT", fontsize=8)
    axs[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_robust.png", dpi=120)


if __name__ == "__main__":
    main()
