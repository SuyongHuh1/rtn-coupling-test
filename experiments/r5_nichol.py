"""R5d: coupling tests on the public Si/SiGe charge-sensor data of Albrecht, Ye, Jacobson & Nichol
(arXiv:2601.08088; data Zenodo 10.5281/zenodo.18828196, CC BY 4.0).

Datasets: Gate_P (plunger RP1 sweep, 11 biases x 20 reps x 8192 samples at 60 Hz) - the paper
reports a conditional rate dependence (4-level model preferred) at RP1 = 0.567 V; MCtemp
(6 temperatures x 30 x 4096); LH_cond_Sgate (9 S-gate biases x 30 x 8192); LH_SDbias_*.

Per bias: each repetition is idealised separately (additive two-trap HMM + slow baseline,
rtnsim.robust) and the conditional-rate statistic is summed over repetitions; surrogates shift
trap 2 within each repetition (global: 10-90% of the repetition; local: 1-4 dwell times).
Also: the HMM GLRT (product vs kinetically coupled, per repetition, levels refitted per rep)
and the drive-aware GLRT, summed over repetitions with chi2(4 n_rep) references; and the R1
smallest identifiable |J| for the total duration.
Output: results/r5/r5_nichol.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.robust import idealise  # noqa: E402
from rtnsim.surrogate import shift_test_multi  # noqa: E402
from rtnsim import detectors as det  # noqa: E402
from rtnsim import drive_hmm as DH  # noqa: E402
from r5_real_robust import rates_from_path, jmin  # noqa: E402

D = Path(__import__("os").environ.get("RTN_NICHOL", "data/nichol"))
OUT = ROOT / "results" / "r5"
DT = 1 / 60


def analyse(block, rng, label, value, n_sur=199, with_hmm=True):
    """block: (n_rep, T) raw sensor signal."""
    z = (block - np.median(block)) / (np.median(np.abs(np.diff(block, axis=1))) / 0.6745 / np.sqrt(2))
    pairs, a1s, a2s, sigs = [], [], [], []
    for y in z:
        idl = idealise(y, baseline=True, min_win=61)
        pairs.append((idl["s1"], idl["s2"]))
        a1s.append(idl["a1"]); a2s.append(idl["a2"]); sigs.append(idl["sig"])
    s1 = np.concatenate([p[0] for p in pairs]); s2 = np.concatenate([p[1] for p in pairs])
    r1, r2 = rates_from_path(s1, DT), rates_from_path(s2, DT)
    stat, pg, pl = shift_test_multi(pairs, rng, n_sur=n_sur)
    row = dict(dataset=label, value=float(value), n_rep=len(z), T_rep=z.shape[1],
               a1=float(np.median(a1s)), a2=float(np.median(a2s)), sig=float(np.median(sigs)),
               switches=[r1[2], r2[2]], rates_per_s=[r1[:2], r2[:2]],
               resolvable=bool(min(r1[2], r2[2]) >= 20 and abs(np.median(a2s)) >= 1.5 * np.median(sigs)),
               shift_stat=stat, p_global=pg, p_local=pl,
               jmin_kT=jmin(r1, r2, z.size * DT))
    if with_hmm:
        prod = det.fit_product(z, n_iter=60)
        cpl = det.fit_coupled(z, prod, n_iter=40)
        g = float(np.sum(np.maximum(cpl["ll"] - prod["ll"], 0)))
        da = float(np.sum(DH.drive_aware_scores(z, p_max=0.01)))
        k = 4 * len(z)
        row.update(glrt_sum=g, p_glrt_chi2=float(stats.chi2.sf(g, k)), drive_aware_sum=da,
                   p_drive_chi2=float(stats.chi2.sf(da, k)))
    print(label, value, {k_: row[k_] for k_ in ("switches", "resolvable", "p_global", "p_local", "jmin_kT")},
          {k_: row.get(k_) for k_ in ("p_glrt_chi2", "p_drive_chi2")}, flush=True)
    return row


def main(which=("Gate_P", "MCtemp", "LH_cond_Sgate", "LH_SDbias")):
    rng = np.random.default_rng(11)
    res = []
    for name in which:
        if name == "LH_SDbias":
            for lvl in ("low", "medium", "high"):
                s = sio.loadmat(D / f"LH_SDbias_{lvl}.mat", squeeze_me=True, struct_as_record=False)["savedat"]
                res.append(analyse(s.data, rng, "SDbias", {"low": 0, "medium": 1, "high": 2}[lvl]))
            continue
        s = sio.loadmat(D / f"{name}.mat", squeeze_me=True, struct_as_record=False)["savedat"]
        vals = s.MCtemp if name == "MCtemp" else s.Vgate
        for i, v in enumerate(vals):
            res.append(analyse(s.data[i], rng, name, v))
        json.dump(res, open(OUT / "r5_nichol.json", "w"), indent=1, default=float)
    json.dump(res, open(OUT / "r5_nichol.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("Gate_P", "MCtemp", "LH_cond_Sgate", "LH_SDbias"))
