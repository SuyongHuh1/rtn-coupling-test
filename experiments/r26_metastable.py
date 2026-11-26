"""R26: metastable-defect negative controls.  Two INDEPENDENT traps (J = 0) where trap B is not a simple
two-level fluctuator:
  onoff     B switches between an active configuration (normal RTN) and an inactive one (frozen empty);
            configuration dwell = (10-40) x B's switching time  ('anomalous RTN')
  threestate B has two filled configurations with the same current but different emission rates
            (fast / 10-30x slower), so its filled dwell is hyper-exponential
Full verdict pipeline as for the real records (r20): R9a 3 starts, checklist (1) resolution and (2)
transitions, pipeline-null calibration (39 nulls when p_asym < 0.2), (4) shared-drive alternative when
p_cal < 0.05.  A record is a false coupling verdict if it passes everything.
    python experiments/r26_metastable.py [n=40]
"""
from __future__ import annotations
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
OUT = ROOT / "results/r23/r26_metastable.json"
T = 4096


def simulate(kind, i):
    rng = np.random.default_rng(26000 + 1000 * (kind == "threestate") + i)
    tau = np.exp(rng.uniform(np.log(10), np.log(80), 2))
    p = rng.uniform(0.3, 0.7, 2)
    c, e = 2 / tau * p, 2 / tau * (1 - p)
    a = np.array([1.0, rng.uniform(0.3, 0.9)])
    sA = np.zeros(T, int); sB = np.zeros(T, int)
    sA[0] = rng.random() < p[0]
    m = 1; slow = rng.uniform(10, 30)
    md = tau[1] * rng.uniform(10, 40)
    u = rng.random((T, 4))
    for t in range(1, T):
        sA[t] = sA[t - 1] ^ (u[t, 0] < (c[0] if sA[t - 1] == 0 else e[0]))
        if kind == "onoff":
            m = m ^ (u[t, 2] < 1 / md)
            if m == 0:
                sB[t] = 0
            else:
                sB[t] = sB[t - 1] ^ (u[t, 1] < (c[1] if sB[t - 1] == 0 else e[1]))
        else:   # threestate: 0 empty, 1 filled-fast, 2 filled-slow (same current)
            st = sB[t - 1]
            if st == 0:
                sB[t] = (1 if u[t, 3] < 0.6 else 2) if u[t, 1] < c[1] else 0
            elif st == 1:
                sB[t] = 0 if u[t, 1] < e[1] else 1
            else:
                sB[t] = 0 if u[t, 1] < e[1] / slow else 2
    occB = (sB > 0).astype(int)
    sig = a[1] * rng.uniform(0.15, 0.35)
    W = np.fft.rfft(rng.normal(0, 1, T)); f = np.fft.rfftfreq(T)
    fl = np.fft.irfft(W * np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1 / T)), 0), n=T); fl /= fl.std()
    y = a[0] * sA + a[1] * occB + sig * rng.normal(0, 1, T) + 0.4 * sig * fl
    return y


def job(a):
    kind, i = a
    import r20_southampton as R
    r = R.job((f"{kind}{i}", simulate(kind, i), 2, 39, 9300 + i))
    r["kind"] = kind
    return kind, i, r


def final(r):
    if "error" in r:
        return "error"
    if r["dwell_min"] < 3:
        return "unresolvable"
    if min(r["trans"]) < 15:
        return "not_testable"
    p = r.get("p_cal", r["p_asym"])
    if p >= 0.05:
        return "no_evidence"
    if "stat_drive" in r and chi2.sf(r["stat_drive"], 1) >= 0.05:
        return "alternative"
    return "coupled"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    res = json.load(open(OUT)) if OUT.exists() else {}
    res = res.get("res", res)
    tasks = [(k, i) for k in ("onoff", "threestate") for i in range(n) if str(i) not in res.get(k, {})]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for k, i, r in p.imap_unordered(job, tasks):
            res.setdefault(k, {})[str(i)] = r
            json.dump(dict(res=res), open(OUT, "w"), indent=1)
    summ = {}
    for k, v in res.items():
        rows = list(v.values())
        fv = [final(r) for r in rows]
        ok = [r for r in rows if "error" not in r]
        summ[k] = dict(n=len(rows), asym=float(np.mean([r["p_asym"] < 0.05 for r in ok])),
                       cal=float(np.mean([r.get("p_cal", 1) < 0.05 for r in ok])),
                       verdicts={x: fv.count(x) for x in set(fv)})
    json.dump(dict(summary=summ, res=res), open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
