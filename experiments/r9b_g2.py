"""G2: does joint fitting of a bias series fix the choice of the number of traps K?

For simulated series (K_true = 2 or 3 traps, N biases, 1/f + drift), compare
  per-record : each record alone (amplitudes and rates free), K by BIC
  joint      : all records together (shared amplitudes a_k + d_k v, log-linear rates), K by joint BIC
Baselines from the R9a step detector + Kalman smoother (independent of K).
    python experiments/r9b_g2.py [n_series_per_K=20] [N=6] [T=3000] [workers=2] [cond=full|white]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim.series import simulate_series, select_K, select_K_joint, choose_K_supported  # noqa: E402
from rtnsim.cfhmm import kalman_baseline  # noqa: E402
from rtnsim.r9 import load_stepnet, step_candidates  # noqa: E402
from rtnsim.stepnet import denoise  # noqa: E402

_net = None
MAXC = 9
AR1 = '--ar1' in sys.argv


def merge(c, tol=1.08):
    out = []
    for x in np.sort(c):
        if x > 0 and (not out or x > out[-1] * tol):
            out.append(x)
    return np.array(out)


def job(a):
    global _net
    sid, K, N, T, cond = a
    if _net is None:
        _net = load_stepnet(ROOT / "results/r9a_gpu/stepnet.pt")
    rng = np.random.default_rng(9000 + sid)
    ys, vs, truth, states = simulate_series(K, N, T, rng, flicker=(cond == "full"), drift=(cond == "full"))
    t0 = time.time()
    zs, cands_all, per, per_sup, H = [], [], [], [], []
    for y in ys:
        lev, edge, step = denoise(_net, y)
        beta, _ = kalman_baseline(y - lev)
        z = y - (beta - np.median(beta))
        c = step_candidates(edge, step, lev, 4)
        zs.append(z); cands_all.append(c)
        on = edge > 0.5
        idx = np.flatnonzero(on)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1) if idx.size else []
        H.append(np.array([abs(step[r].sum()) for r in runs if len(r)]))
    rec_fits = {K_: [] for K_ in (1, 2, 3, 4)}
    for n, z in enumerate(zs):
        c = merge(cands_all[n])
        if len(c) > MAXC:
            c = c[np.unique(np.linspace(0, len(c) - 1, MAXC).round().astype(int))]
        kb, fits = select_K([z], [0.0], c, slopes=False, ar1=AR1)
        per.append(int(kb))
        per_sup.append(int(choose_K_supported(fits, H[n], n_edges_min=5)))
        for K_ in rec_fits:
            if K_ in fits:
                rec_fits[K_].append(fits[K_])
    pooled = merge(np.concatenate(cands_all))
    if len(pooled) > MAXC:
        # keep the candidates supported by most records
        support = np.array([sum(np.any(np.abs(cc - p) < 0.08 * p) for cc in cands_all) for p in pooled])
        pooled = np.sort(pooled[np.argsort(-support, kind="stable")[:MAXC]])
    kj, fits = select_K_joint(zs, vs, rec_fits, pooled, ar1=AR1)
    kjs = int(choose_K_supported(fits, np.concatenate(H)))
    eff = [int(np.sum(np.abs(np.diff(S, axis=0)).sum(0) >= 10)) for S in states]
    return sid, dict(K=K, per_record=per, per_record_sup=per_sup, eff_K=eff, joint=int(kj), joint_sup=kjs, sec=time.time() - t0,
                     bic={str(k): f["bic"] for k, f in fits.items()},
                     a_hat=fits[kj]["a"].tolist() if kj in fits else None, a_true=truth["a"].tolist())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    cond = sys.argv[5] if len(sys.argv) > 5 else "full"
    tasks = [(1000 * K + i, K, N, T, cond) for K in (2, 3) for i in range(n)]
    out_path = ROOT / f"results/r8/r9b_g2_{cond}{'_ar1' if AR1 else ''}.json"
    res = {}
    if out_path.exists():
        res = {int(k): v for k, v in json.load(open(out_path)).items() if k != "summary"}
        tasks = [t for t in tasks if t[0] not in res]
    with Pool(w) as p:
        for sid, r in p.imap_unordered(job, tasks):
            res[sid] = r
            json.dump(res, open(out_path, "w"), indent=1)
            print(sid, r["K"], "per", r["per_record"], "per_sup", r["per_record_sup"], "eff", r["eff_K"], "joint", r["joint"], "joint_sup", r["joint_sup"], f"{r['sec']:.0f}s", flush=True)
    summ = {}
    for K in (2, 3):
        rr = [r for r in res.values() if r["K"] == K]
        pr = np.array([k for r in rr for k in r["per_record"]])
        eff = np.array([k for r in rr for k in r["eff_K"]])
        prs = np.array([k for r in rr for k in r["per_record_sup"]])
        summ[f"K={K}_supported"] = dict(
            joint_correct=float(np.mean([r["joint_sup"] == K for r in rr])),
            joint_counts={str(k): int(sum(r["joint_sup"] == k for r in rr)) for k in (1, 2, 3, 4)},
            per_record_correct_vs_device=float(np.mean(prs == K)),
            per_record_correct_vs_effective=float(np.mean(prs == eff)),
            per_record_over=float(np.mean(prs > eff)), per_record_under=float(np.mean(prs < eff)),
            per_record_max_correct=float(np.mean([max(r["per_record_sup"]) == K for r in rr])))
        summ[f"K={K}"] = dict(
            n_series=len(rr),
            joint_correct=float(np.mean([r["joint"] == K for r in rr])),
            joint_counts={str(k): int(sum(r["joint"] == k for r in rr)) for k in (1, 2, 3, 4)},
            per_record_correct_vs_device=float(np.mean(pr == K)),
            per_record_correct_vs_effective=float(np.mean(pr == eff)),
            per_record_over=float(np.mean(pr > eff)), per_record_under=float(np.mean(pr < eff)),
            per_record_majority_correct=float(np.mean([np.bincount(r["per_record"]).argmax() == K for r in rr])),
            per_record_max_correct=float(np.mean([max(r["per_record"]) == K for r in rr])))
    res["summary"] = summ
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
