"""R27b: the gated variants of R27 did not help because the kurtosis gate missed many t(3) records.
Here: V4 = Hampel always; V5 = Hampel when the RAW record has isolated spikes (count of samples > 5 robust
sigma from a 7-point running median, both neighbours inside: Gaussian noise gives ~0 per 4,096 samples).
Reuses R27's V0 and its Hampel runs (same seeds); computes the Hampel run where R27 skipped it."""
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r27_robust as Q
NET = None


def spikes(y, k=5.0):
    from scipy.ndimage import median_filter
    m = median_filter(y, size=7, mode="nearest")
    r = y - m
    s = np.median(np.abs(np.diff(y))) / 0.6745 / np.sqrt(2)
    out = np.abs(r) > k * s
    iso = out & ~np.r_[False, out[:-1]] & ~np.r_[out[1:], False]
    return int(iso.sum())


def records(setname, i):
    if setname.startswith("heavy"):
        import r25_ood as R
        R.TDOF = int(setname[-1])
        ys, _ = R.make_pair("heavy", i)
        return list(zip((1, 0), ys, (i, i)))
    import r8_oracle as O
    global _REC
    if "_REC" not in globals():
        _REC = O.make(100, white=False)
    ys, ss, rows = _REC
    kk = [j for j in range(len(rows)) if rows[j]["K"] == 2]
    pos = [j for j in kk if rows[j]["coupled"] == 1][i]; neg = [j for j in kk if rows[j]["coupled"] == 0][i]
    return [(1, ys[pos], pos), (0, ys[neg], neg)]


def job(a):
    global NET
    setname, i, prev = a
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    from rtnsim.tnoise import hampel
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    out = []
    for (lab, y, seed), p in zip(records(setname, i), prev):
        r = dict(coupled=lab, V0=p["V0"], spikes=spikes(y))
        r["H"] = p["V1"] if p["kurt"] > Q.KGATE else float(lrt_cnn3_ms(hampel(y), 2, NET, ar1=True, rng=np.random.default_rng(seed), n_start=3)[0])
        out.append(r)
    return setname, i, out


if __name__ == "__main__":
    prev = json.load(open(ROOT / "results/r23/r27_robust.json"))["res"]
    tasks = [(s, int(i), v) for s, d in prev.items() for i, v in d.items()]
    res = {}
    with Pool(2) as p:
        for s, i, r in p.imap_unordered(job, tasks):
            res.setdefault(s, {})[str(i)] = r
    summ = {}
    for s, v in res.items():
        rows = [x for vv in v.values() for x in vv]
        for x in rows:
            x["V4"] = x["H"]; x["V5"] = x["H"] if x["spikes"] > 2 else x["V0"]
        d = dict(gated_V5=float(np.mean([x["spikes"] > 2 for x in rows])), spikes_med=float(np.median([x["spikes"] for x in rows])))
        for m in ("V0", "V4", "V5"):
            neg = np.array([x[m] for x in rows if x["coupled"] == 0]); pos = np.array([x[m] for x in rows if x["coupled"] == 1])
            thr = np.quantile(neg, 0.95)
            au = (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))
            d[m] = dict(power=round(float(np.mean(pos > thr)), 3), auc=round(float(au), 3), fpr_chi2=round(float(np.mean(neg > 3.84)), 3))
        summ[s] = d
    json.dump(dict(summary=summ, res=res), open(ROOT / "results/r23/r27b_spikegate.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
