"""R8 on measured records (Southampton SSDM + EDTM): R8 LRT (AR(1) noise, Kalman baseline,
global decomposition search), path statistic on the same decomposition, CNN score (mean logit
over 4096-sample windows) and the CNN+R8 Fisher combination, each calibrated by the same
pipeline null (n_sim copies of the fitted J = 0 model with the record's own baseline and
block-bootstrapped residual).  Both K = 2 and K = 3 are analysed, with the AR(1) BIC of the
J = 0 fits, to see whether verdicts still flip with K.

    python experiments/r8_real.py [n_sim=30] [workers=2]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.cfhmm import lrt_ms, simulate_null  # noqa: E402
from rtnsim.robustk import bits  # noqa: E402
from rtnsim.realdata import load_b1500  # noqa: E402
from r8_calib import cnn_score as _cnn1, _rank_p  # noqa: E402

OUT = ROOT / "results" / "r8"


def cnn_score(y, win=4096):
    n = max(1, len(y) // win)
    return float(np.mean([_cnn1(y[k * win:(k + 1) * win]) for k in range(n)]))


def job(args):
    name, y, K, n_sim, seed = args
    rng = np.random.default_rng(seed)
    kw = dict(ar1=True, kalman=True)
    s_obs, f0, f1, beta, p_obs = lrt_ms(y, K, rng, **kw)
    c_obs = cnn_score(y)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    n8, npth, ncnn = [], [], []
    for _ in range(n_sim):
        ys = simulate_null(f0, beta, resid, len(y), rng)
        s, *_, p = lrt_ms(ys, K, rng, **kw)
        n8.append(s); npth.append(p); ncnn.append(cnn_score(ys))
    n8, npth, ncnn = map(np.array, (n8, npth, ncnn))

    def comb(s8, sc, excl=None):
        m = np.ones(n_sim, bool)
        if excl is not None:
            m[excl] = False
        return -2 * (np.log(_rank_p(s8, n8[m])) + np.log(_rank_p(sc, ncnn[m])))
    c_null = np.array([comb(n8[j], ncnn[j], j) for j in range(n_sim)])
    sw = [int(np.count_nonzero(np.diff(bits(K)[f0["path"]][:, k]))) for k in range(K)]
    return name, K, dict(bic_ar1=f0["bic"], steps=f0["coef"].tolist(), rho=f0["rho"], J=f1["J"].tolist(),
                         switches=sw, stat_r8=s_obs, stat_path=p_obs, cnn=c_obs,
                         p_r8=float(_rank_p(s_obs, n8)), p_path=float(_rank_p(p_obs, npth)),
                         p_cnn=float(_rank_p(c_obs, ncnn)), p_comb=float(_rank_p(comb(s_obs, c_obs), c_null)))


def main():
    import r5_real as R
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    keep = {r["file"]: r for r in json.load(open(ROOT / "results/r6/r6_south.json")) if r["K"] >= 2}
    tasks = []
    for j, r in enumerate(R.records()):
        if r["file"] not in keep:
            continue
        t, I, _ = load_b1500(r["path"])
        t, y, _, dt = R.preprocess(t, I, r["trim_s"])
        for K in (2, 3):
            tasks.append((r["file"], y, K, n_sim, 7000 + 10 * j + K))
    path = OUT / "r8_real.json"
    res = json.load(open(path)) if path.exists() else {}
    tasks = [t for t in tasks if f"{t[0]}|K{t[2]}" not in res]
    print(len(tasks), "tasks", flush=True)
    with Pool(workers) as pool:
        for name, K, out in pool.imap_unordered(job, tasks):
            res[f"{name}|K{K}"] = out
            json.dump(res, open(path, "w"), indent=1)
            print(name, K, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()
                            if k.startswith("p_") or k in ("bic_ar1", "switches")}, flush=True)


if __name__ == "__main__":
    main()
