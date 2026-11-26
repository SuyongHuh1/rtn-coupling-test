"""R8 synthetic comparison: HMM path statistic (R6) vs R8 soft LRT on identical matched pairs.

Data: R7 simulator (coupled record + its I-projection independent partner, same nuisance),
K_true in {2, 3}, T = 4096, no shared drive, point sampling (drift and 1/f on).  Each record is
analysed with K_fit = K_true and with a wrong K (K_true + 1 for K = 2, K_true - 1 for K = 3).
Discrimination is measured with the threshold set at the 95th percentile of the independent
partners (oracle calibration), so the comparison is about power, not about calibration.
Records are saved (results/r8/records.npz) so the R7 network can score the same data.

    python experiments/r8_compare.py [n_pairs_per_K=150] [workers=2]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim import sim2_torch as S  # noqa: E402
from rtnsim.robustk import idealise_k, coupling_stat  # noqa: E402
from rtnsim.cfhmm import lrt, lrt_ar1  # noqa: E402

OUT = ROOT / "results" / "r8"
OUT.mkdir(parents=True, exist_ok=True)
T = 4096


def make_records(n_pairs, seed=81):
    ys, rows = [], []
    for K in (2, 3):
        g = torch.Generator().manual_seed(seed + K)
        got = 0
        while got < n_pairs:
            y, lab = S.make_batch(50, T, g, "cpu", p_drive=0, p_box=0)
            if int(lab["K"][0]) != K:
                continue
            for r in range(y.shape[0]):
                ys.append(y[r].double().numpy())
                rows.append(dict(K=K, coupled=int(lab["coupled"][r]), pair=len(rows) // 2,
                                 Jmax=float(lab["Jvec"][r].abs().max()),
                                 tau_min=float(torch.exp(-lab["theta"][r, :K, 0]).min()) * 1.0))
            got += y.shape[0] // 2
        # trim to n_pairs
        extra = (got - n_pairs) * 2
        if extra:
            del ys[-extra:], rows[-extra:]
    return np.stack(ys), rows


def analyse(args):
    i, y, K = args
    rng = np.random.default_rng(1000 + i)
    out = {}
    for Kf in sorted({K, K + 1 if K == 2 else K - 1}):
        f = idealise_k(y, Kf, rng=rng)
        st, f0, f1, _ = lrt(y, Kf, rng, f=f)
        sa, g0, g1, _ = lrt_ar1(y, Kf, rng, f=f)
        out[str(Kf)] = dict(path=coupling_stat(f["S"])[0], lrt=st, J=f1["J"].tolist(), bic=f["bic"],
                            lrt_ar1=sa, J_ar1=g1["J"].tolist(), bic_ar1=g0["bic"], rho=g0["rho"])
    for Kf in (1, 2, 3):                                     # K selection, white vs AR(1) noise
        if str(Kf) not in out:
            f = idealise_k(y, Kf, rng=rng)
            g0 = lrt_ar1(y, Kf, rng, f=f)[1] if Kf > 1 else None
            out[str(Kf)] = dict(bic=f["bic"], bic_ar1=(g0["bic"] if g0 else _bic_ar1_k1(y, f)))
    return i, out


def _bic_ar1_k1(y, f):
    from rtnsim.cfhmm import fit_ar1, init_from_idealiser
    return fit_ar1(y - f["beta"], 1, False, init_from_idealiser(f))["bic"]


def summarise(rows, res):
    def stats(score, lab):
        s0, s1 = score[lab == 0], score[lab == 1]
        thr = np.quantile(s0, 0.95)
        return dict(auc=float((s1[:, None] > s0[None]).mean() + 0.5 * (s1[:, None] == s0[None]).mean()),
                    tpr_at_fpr5=float((s1 > thr).mean()), n_pairs=int(len(s1)))
    summ = {}
    for K in (2, 3):
        idx = [i for i, r in enumerate(rows) if r["K"] == K and i in res]
        lab = np.array([rows[i]["coupled"] for i in idx])
        for Kf in sorted({K, K + 1 if K == 2 else K - 1}):
            tag = f"Ktrue={K},Kfit={Kf}"
            summ[tag] = {m: stats(np.array([res[i][str(Kf)][m] for i in idx]), lab) for m in ("path", "lrt", "lrt_ar1")}
            # paired: coupled record beats its own independent partner
            pr = {}
            for m in ("path", "lrt", "lrt_ar1"):
                d = [res[i][str(Kf)][m] - res[i + 1][str(Kf)][m] for i in idx if rows[i]["coupled"] == 1 and i + 1 in res]
                pr[m] = float(np.mean(np.array(d) > 0))
            summ[tag]["paired_win"] = pr
            # power vs |J| (threshold from all independent records of this K)
            Jm = np.array([rows[i]["Jmax"] for i in idx])
            bins = [(0.7, 1.5), (1.5, 2.25), (2.25, 3.01)]
            pj = {}
            for m in ("path", "lrt", "lrt_ar1"):
                sc = np.array([res[i][str(Kf)][m] for i in idx])
                thr = np.quantile(sc[lab == 0], 0.95)
                pj[m] = {f"{a}-{b}": float((sc[(lab == 1) & (Jm >= a) & (Jm < b)] > thr).mean()) for a, b in bins}
            summ[tag]["power_by_J"] = pj
            # BIC choice of K
        ch = {}
        for b in ("bic", "bic_ar1"):
            ks = [min(("1", "2", "3"), key=lambda k: res[i][k][b]) for i in idx]
            ch[b] = {k: ks.count(k) for k in sorted(set(ks))}
        summ[f"Ktrue={K}"] = {"K_choice": ch}
    return summ


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rec = OUT / "records.npz"
    if rec.exists():
        z = np.load(rec, allow_pickle=True)
        ys, rows = z["y"], list(z["rows"])
    else:
        ys, rows = make_records(n)
        np.savez_compressed(rec, y=ys, rows=np.array(rows, dtype=object))
    part = OUT / "r8_compare_raw_v2.json"
    res = {int(k): v for k, v in json.load(open(part)).items()} if part.exists() else {}
    todo = [(i, ys[i], rows[i]["K"]) for i in range(len(rows)) if i not in res]
    print(f"{len(rows)} records, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for j, (i, out) in enumerate(pool.imap_unordered(analyse, todo, chunksize=2)):
            res[i] = out
            if j % 20 == 0:
                json.dump(res, open(part, "w"))
                print(f"{j + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(part, "w"))
    summ = summarise(rows, res)
    json.dump(summ, open(OUT / "r8_compare.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
