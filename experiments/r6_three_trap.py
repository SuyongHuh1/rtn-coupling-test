"""R6: three-trap (K-trap) analysis.

validate   synthetic records (3 traps, steps 1 / 0.6 / 0.35, 1/f noise + drift): BIC choice of K,
           pipeline-null rejection for independent traps and for one coupled pair (J = 1, 2 kT).
south      Southampton records: K in {1, 2, 3} by BIC, dwell-time checks per trap, K-trap coupling
           statistic vs its pipeline null (100 simulations), multiple-testing correction.
nichol     Nichol plunger sweep (11 biases) and temperature sweep (6), first 6 repetitions each.

Outputs results/r6/r6_<mode>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim import RTNConfig  # noqa: E402
from rtnsim import theory as th  # noqa: E402
from rtnsim.noise import add_drift, add_flicker  # noqa: E402
from rtnsim.robustk import idealise_k, coupling_stat, pipeline_null  # noqa: E402

OUT = ROOT / "results" / "r6"
OUT.mkdir(parents=True, exist_ok=True)


def cfg3(J01=0.0, sigma=0.12):
    c = RTNConfig()
    c.traps.tau_c, c.traps.tau_e = [1.0, 0.7, 1.5], [1.0, 1.2, 0.8]
    c.traps.amp = [1.0, 0.6, 0.35]
    c.coupling.J = [[0, J01, 0], [J01, 0, 0], [0, 0, 0]] if J01 else None
    c.coupling.phi = 0.5
    c.readout.dt, c.readout.sigma = 0.1, sigma
    return c


def choose_k(y, rng, kmax=3, min_win=101):
    fits = {K: idealise_k(y, K, min_win=min_win, rng=rng) for K in range(1, kmax + 1)}
    K = min(fits, key=lambda k: fits[k]["bic"])
    return K, fits


def dwell_ks(s):
    from r5_adequacy import dwell_ks as dk
    return dk(s)


def validate(n_rec=12, n_sim=40, n=3000):
    rng = np.random.default_rng(31)
    res = {}
    for name, J in (("independent", 0.0), ("coupled pair J=1", 1.0), ("coupled pair J=2", 2.0)):
        y = th.sample_readout(cfg3(J), n_rec, n, rng)
        y = add_flicker(y, 0.06, fmin=0.3 / (n * 0.1), fmax=5.0, n_terms=8, dt=0.1, rng=rng)
        y = add_drift(y, sd_per_sqrt_time=0.3 / np.sqrt(n * 0.1), dt=0.1, rng=rng)
        ks, ps = [], []
        for i in range(n_rec):
            K, fits = choose_k(y[i], rng)
            ks.append(K)
            f = fits[3]
            obs, null = pipeline_null([(y[i], f)], 3, rng, n_sim=n_sim)
            ps.append((1 + np.sum(null >= obs)) / (n_sim + 1))
        res[name] = dict(K_chosen=ks, reject_5pct=float(np.mean(np.array(ps) <= 0.05)), p=[float(p) for p in ps])
        print(name, res[name]["K_chosen"], res[name]["reject_5pct"], flush=True)
    json.dump(res, open(OUT / "r6_validate.json", "w"), indent=1)


def south(n_sim=100):
    import r5_real as R
    from rtnsim.realdata import load_b1500
    rng = np.random.default_rng(32)
    keep = {r["file"] for r in json.load(open(ROOT / "results/r5/r5_pipeline_null_southampton.json"))}
    res = []
    for r in R.records():
        if r["file"] not in keep:
            continue
        t, I, _ = load_b1500(r["path"])
        t, y, _, dt = R.preprocess(t, I, r["trim_s"])
        K, fits = choose_k(y, rng)
        row = dict(set=r["set"], file=r["file"], K=K, bic={str(k): f["bic"] for k, f in fits.items()},
                   steps={str(k): f["a"].tolist() for k, f in fits.items()},
                   sig={str(k): f["sig"] for k, f in fits.items()})
        if K >= 2:
            f = fits[K]
            row["switches"] = [int(np.count_nonzero(np.diff(f["S"][:, k]))) for k in range(K)]
            row["dwell"] = {f"trap{k}": dwell_ks(f["S"][:, k]) for k in range(K)}
            obs, null = pipeline_null([(y, f)], K, rng, n_sim=n_sim)
            row.update(stat=obs, per_pair=coupling_stat(f["S"])[1], null_q95=float(np.quantile(null, .95)),
                       p_pipeline=float((1 + np.sum(null >= obs)) / (n_sim + 1)))
        print({k: row.get(k) for k in ("file", "K", "switches", "p_pipeline")}, flush=True)
        res.append(row)
        json.dump(res, open(OUT / "r6_south.json", "w"), indent=1, default=float)


def nichol(n_sim=30, reps=6):
    from r5_nichol import D
    rng = np.random.default_rng(33)
    res = []
    for name in ("Gate_P", "MCtemp"):
        s = sio.loadmat(D / f"{name}.mat", squeeze_me=True, struct_as_record=False)["savedat"]
        vals = s.MCtemp if name == "MCtemp" else s.Vgate
        for i, v in enumerate(vals):
            block = s.data[i][:reps]
            z = (block - np.median(block)) / (np.median(np.abs(np.diff(block, axis=1))) / 0.6745 / np.sqrt(2))
            Ks, segs = [], {2: [], 3: []}
            for y in z:
                K, fits = choose_k(y, rng, min_win=61)
                Ks.append(K)
                for k in (2, 3):
                    segs[k].append((y, fits[k]))
            Kmaj = int(np.bincount(Ks).argmax())
            row = dict(dataset=name, value=float(v), K_per_rep=Ks, K=Kmaj)
            for k in (2, 3):
                obs, null = pipeline_null(segs[k], k, rng, n_sim=n_sim, min_win=61)
                row[f"K{k}"] = dict(stat=obs, null_q95=float(np.quantile(null, .95)),
                                    p=float((1 + np.sum(null >= obs)) / (n_sim + 1)),
                                    steps=np.median([f["a"] for _, f in segs[k]], 0).tolist(),
                                    sig=float(np.median([f["sig"] for _, f in segs[k]])))
            print(name, v, Ks, {k: row[f"K{k}"]["p"] for k in (2, 3)}, flush=True)
            res.append(row)
            json.dump(res, open(OUT / "r6_nichol.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"validate": validate, "south": south, "nichol": nichol}[sys.argv[1]]()
