"""R42 (review): calibrated FPR versus the ratio of integration time to dwell time (measurement guideline 1).

Independent two-trap records (J = 0) are simulated from exponential dwell times with the fast trap's mean dwell
d samples (both states, occupancy 0.5) and the slow trap's 3d; step heights 1.0 and 0.6, white noise
sigma = 0.25, T = 4096.  Each set of trap paths is sampled two ways:
    box    exact average occupancy over each sampling interval (boxcar integration time = sample interval)
    point  instantaneous occupancy at the end of each interval (no integration)
so the integration-to-dwell ratio is 1/d.  Every record goes through the paper's calibrated test (r8_calib.job,
pipeline nulls, B = 30); the nulls are point-sampled as in the paper.  Reported: calibrated FPR per d and
acquisition, before and after checklist steps 1-2 (shortest joint-state dwell >= 3 samples, >= 15 switches).

    RTN_MS=3 RTN_RESTART=3 RTN_JOINT=0 python experiments/r42_boxcar_dwell.py [workers] [n_box] [n_point] [B]
-> results/r8/r42_boxcar_dwell.json (resumable).
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
OUT = ROOT / "results/r8/r42_boxcar_dwell.json"
T, A, SIG = 4096, (1.0, 0.6), 0.25
D_SET = (2, 3, 4, 6, 8, 12, 20, 40)


def paths(rates, T, rng):
    """Event times and initial states of independent two-state traps with equal capture/emission rate."""
    out = []
    for r in rates:
        st = int(rng.random() < 0.5)
        n = int(3 * T * r + 50)
        while True:
            t = np.cumsum(rng.exponential(1.0 / r, n))
            if t[-1] > T:
                break
            n *= 2
        out.append((st, np.r_[0.0, t]))
    return out


def sample(pth, T, mode):
    edges = np.arange(T + 1.0)
    occ = np.empty((T, len(pth)))
    for k, (st, tt) in enumerate(pth):
        s_i = (st ^ (np.arange(tt.size - 1) & 1)).astype(float)       # state on [tt[i], tt[i+1])
        if mode == "box":
            I = np.r_[0.0, np.cumsum(s_i * np.diff(tt))]
            occ[:, k] = np.diff(np.interp(edges, tt, I))
        else:
            idx = np.searchsorted(tt, edges[1:], side="right") - 1
            occ[:, k] = s_i[np.clip(idx, 0, s_i.size - 1)]
    return occ


def make(d, i, mode):
    rng = np.random.default_rng(42_000 + 1000 * d + i)                # same seed -> same traps for box and point
    pth = paths((1.0 / d, 1.0 / (3 * d)), T, rng)
    occ = sample(pth, T, mode)
    noise = np.random.default_rng(43_000 + 1000 * d + i).normal(0, SIG, T)
    return occ @ np.array(A) + noise


def job(a):
    key, d, i, mode, B = a
    import r8_calib as C
    y = make(d, i, mode)
    try:
        _, r = C.job((50_000 + 1000 * d + i, y, 2, B))
    except Exception as e:                                             # noqa: BLE001
        return key, dict(d=d, mode=mode, err=str(e))
    return key, dict(d=d, mode=mode, p=r.get("p_r8"), stat=r.get("stat_r8"), n_null=r.get("n_null"),
                     dwell_min=r.get("dwell_min"), trans_min=r.get("trans_min"))


def summary(res):
    out = {}
    for mode in ("box", "point"):
        for d in D_SET:
            v = [x for x in res.values() if isinstance(x, dict) and x.get("mode") == mode and x.get("d") == d and x.get("p") is not None]
            if not v:
                continue
            ok = [x for x in v if (x.get("dwell_min") or 0) >= 3 and (x.get("trans_min") or 0) >= 15]
            out[f"{mode}_d{d}"] = dict(n=len(v), fpr=float(np.mean([x["p"] <= 0.05 for x in v])),
                                       n_pass12=len(ok), fpr_pass12=float(np.mean([x["p"] <= 0.05 for x in ok])) if ok else None)
    return out


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    npt = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    sys.argv = [sys.argv[0], "--r9"]
    res = json.load(open(OUT)) if OUT.exists() else {}
    tasks = [(f"{m}_d{d}_{i}", d, i, m, B) for d in D_SET for m, n in (("box", nb), ("point", npt)) for i in range(n)]
    tasks = [t for t in tasks if t[0] not in res]
    tasks.sort(key=lambda t: t[2])                                   # spread the d levels over time
    print(len(tasks), "to do", flush=True)
    with Pool(W) as p:
        for n, (k, r) in enumerate(p.imap_unordered(job, tasks)):
            res[k] = r
            if n % 5 == 0:
                res["summary"] = summary(res)
                json.dump(res, open(OUT, "w"), indent=1)
            print(k, r.get("p"), r.get("dwell_min"), flush=True)
    res["summary"] = summary(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
