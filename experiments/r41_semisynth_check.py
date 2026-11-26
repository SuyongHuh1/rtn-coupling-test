"""R41: does the semi-synthetic construction (R38, continuous-time) impose the intended coupling?
Rebuilds the R38 records with the same seeds while recording B's state for every sample, then measures B's
capture and emission rates separately while A is empty and while A is occupied (transitions / time):
    J_eff = ln(c_B|A=0 / c_B|A=1) + ln(e_B|A=1 / e_B|A=0)        (= J for phi = 0.5, eq. 1)
and compares J_eff with the injected J, for coupled (J != 0) and independent (J = 0) records.
    RTN_NICHOL=<folder with Gate_P.mat> python experiments/r41_semisynth_check.py -> results/r8/r41_semisynth_check.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import r38_semisynth as R


def assemble_ct_states(sA, seg, L, rate, J, rng):
    """R.assemble_ct with B's state recorded per sample (same random draws, same output)."""
    T = len(sA); out, lab = [], []; t = 0
    s = int(rng.random() < rate[0] / (rate[0] + rate[1])); first = True
    while t < T:
        k = rate[s]
        k2 = k * (np.exp(-J / 2) if s == 0 else np.exp(J / 2)) if sA[t] == 1 else k
        w = np.exp(-(k2 - k) * (L[s] - L[s].min()))
        x = seg[s][rng.choice(len(w), p=w / w.sum())]
        if not first:
            x = x[1:] if len(x) > 1 else x
        sw = np.flatnonzero(sA[t:t + len(x)] != sA[t])
        if sw.size and J != 0:
            x = x[:sw[0]]
            out.append(x); lab.append(np.full(len(x), s)); t += len(x); first = False
            continue
        out.append(x); lab.append(np.full(len(x), s)); t += len(x); s = 1 - s; first = True
    return np.concatenate(out)[:T], np.concatenate(lab)[:T]


def j_eff(sA, sB):
    a, b0, b1 = sA[:-1], sB[:-1], sB[1:]
    r = {}
    for ai in (0, 1):
        m = a == ai
        c = np.sum(m & (b0 == 0) & (b1 == 1)) / max(1, np.sum(m & (b0 == 0)))
        e = np.sum(m & (b0 == 1) & (b1 == 0)) / max(1, np.sum(m & (b0 == 1)))
        r[ai] = (c, e, int(np.sum(m & (b0 != b1))))
    return float(np.log(r[0][0] / r[1][0]) + np.log(r[1][1] / r[0][1])), min(r[0][2], r[1][2])


def main():
    seg, L, rate = R.pools()
    A = R.a_records()
    rng = np.random.default_rng(382)
    n = 22
    out = {}
    for i in range(n):
        y, sA = A[i % len(A)]
        J = float(rng.choice(R.J_SET) * rng.choice([-1, 1]))
        r_seed = int(rng.integers(1 << 31))
        for lab, JJ in (("coup", J), ("indep", 0.0)):
            b, sB = assemble_ct_states(sA, seg, L, rate, JJ, np.random.default_rng(r_seed))
            b0 = R.assemble_ct(sA, seg, L, rate, JJ, np.random.default_rng(r_seed))
            assert np.array_equal(b, b0)                       # same record as R38
            je, nsw = j_eff(sA, sB)
            out[f"{lab}{i}"] = dict(J_true=JJ, J_eff=je, min_switches_per_A_state=nsw)
    pos = [v for v in out.values() if v["J_true"] > 0]; neg = [v for v in out.values() if v["J_true"] < 0]
    ind = [v for v in out.values() if v["J_true"] == 0]
    summ = {nm: dict(n=len(g), ratio_median=float(np.median([v["J_eff"] / v["J_true"] for v in g])) if nm != "indep" else None,
                     J_eff_median=float(np.median([v["J_eff"] for v in g])), J_eff_range=[float(min(v["J_eff"] for v in g)), float(max(v["J_eff"] for v in g))])
            for nm, g in (("J>0", pos), ("J<0", neg), ("indep", ind))}
    out["summary"] = summ
    json.dump(out, open(ROOT / "results/r8/r41_semisynth_check.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for k, v in out.items():
        if k != "summary" and k.startswith("coup"):
            print(k, v["J_true"], round(v["J_eff"], 2), v["min_switches_per_A_state"])


if __name__ == "__main__":
    main()
