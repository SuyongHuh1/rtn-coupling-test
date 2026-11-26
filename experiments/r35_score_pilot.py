"""R35 pilot: co-flip (kappa) score test next to the unchanged J-only fit, in two observation models.
  inst   point-sampling emission (the pipeline's model); nulls: point-sampled J = 0 chain
  integ  integrated-sampling emission (rtnsim.integ: each sample is the boxcar average over the interval,
         moments conditional on both end states); nulls: boxcar-averaged J = 0 chain
Statistic: U_p = d loglik / d kappa_p at kappa = 0 (Fisher identity), T = sum_p max(U_p, 0); kappa is never
fitted, so the J-only estimate and interval are untouched.  Nulls are refitted from the fitted parameters
(not the full CNN pipeline) -- a pilot, not the calibrated test.
    python experiments/r35_score_pilot.py rtninja <nsrc> <rec,rec,...> [nsim]
    python experiments/r35_score_pilot.py sim <n_pairs> [nsim]
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
torch.set_num_threads(1)
os.environ["RTN_JOINT"] = "0"; os.environ.setdefault("RTN_RESTART", "1")


def _pairs(K):
    out = []
    for k in range(K):
        for l in range(k + 1, K):
            m = (1 << k) | (1 << l)
            src = [s for s in range(2 ** K) if not s & m]
            out.append((src, [s | m for s in src]))
    return out


def _add_kappa(Q0, kap, K):
    off = Q0 - torch.diag(torch.diagonal(Q0))
    for p, (src, dst) in enumerate(_pairs(K)):
        add = torch.zeros_like(off)
        add[src, dst] = 1.0; add[dst, src] = 1.0
        off = off + kap[p] * add
    return off - torch.diag(off.sum(1))


def _Q0(c, e, K):
    from rtnsim.sim2_torch import generator
    t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
    Q = generator(t(c)[None], t(e)[None], torch.zeros(1, K, K, dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64))[0]
    off = Q - torch.diag(torch.diagonal(Q))
    return off - torch.diag(off.sum(1))


def score_inst(z, K, f):
    from rtnsim.robustk import _fb, bits
    Q0 = _Q0(f["c"], f["e"], K)
    P = torch.linalg.matrix_exp(Q0).numpy(); P = P / P.sum(1, keepdims=True)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    N = _fb(z, P, np.full(2 ** K, 1.0 / 2 ** K), X @ np.asarray(f["coef"]), float(f["sig"]), False)[1]
    kap = torch.zeros(K * (K - 1) // 2, dtype=torch.float64, requires_grad=True)
    q = (torch.tensor(N) * torch.log(torch.linalg.matrix_exp(_add_kappa(Q0, kap, K)).clamp_min(1e-300))).sum()
    q.backward()
    return kap.grad.numpy()


def score_integ(z, K, f):
    from rtnsim.integ import interval_moments, _fb_pair
    Q0 = _Q0(f["c"], f["e"], K)
    coef = torch.tensor(np.asarray(f["coef"], float)); sig = float(f["sig"]); off = float(np.atleast_1d(f.get("off", 0.0))[0])

    def em(Q):
        P, W, C = interval_moments(Q, K)
        a = coef[1:]
        return P, coef[0] + W @ a, sig ** 2 + torch.einsum("ijkl,k,l->ij", C, a, a).clamp_min(0)
    with torch.no_grad():
        P, m, v = em(Q0)
        P = P.numpy(); P = P / P.sum(1, keepdims=True)
    n, sy, syy = (torch.tensor(x) for x in _fb_pair(z - off, P, np.full(2 ** K, 1.0 / 2 ** K), m.numpy(), v.numpy())[1:4])
    kap = torch.zeros(K * (K - 1) // 2, dtype=torch.float64, requires_grad=True)
    P_, m_, v_ = em(_add_kappa(Q0, kap, K))
    q = (n * torch.log(P_)).sum() - 0.5 * (n * torch.log(2 * np.pi * v_) + (syy - 2 * m_ * sy + m_ * m_ * n) / v_).sum()
    q.backward()
    return kap.grad.numpy()


def T_of(U):
    return float(np.sum(np.maximum(U, 0.0)))


def sim_chain(c, e, T, rng, box):
    """Independent traps; occupancy (T, K): point samples or exact boxcar averages over each interval."""
    K = len(c)
    occ = np.empty((T, K))
    edges = np.arange(T + 1.0)
    for k in range(K):
        st = int(rng.random() < c[k] / (c[k] + e[k]))
        n = int(3 * T * max(c[k], e[k]) + 50)
        while True:
            d = rng.exponential(1.0, 2 * n)
            d[0::2] /= (c[k] if st == 0 else e[k]); d[1::2] /= (e[k] if st == 0 else c[k])
            t = np.cumsum(d)
            if t[-1] > T:
                break
            n *= 2
        tt = np.r_[0.0, t]                                   # state on [tt[i], tt[i+1]) is st ^ (i & 1)
        s_i = (st ^ (np.arange(tt.size - 1) & 1)).astype(float)
        if box:
            I = np.r_[0.0, np.cumsum(s_i * np.diff(tt))]     # integral of occupancy up to tt
            F = np.interp(edges, tt, I)
            occ[:, k] = np.diff(F)
        else:
            occ[:, k] = s_i[np.searchsorted(tt, edges[1:], side="right") - 1]
    return occ


def analyse(y, K, seed, nsim):
    """J-only pipeline fit (unchanged) -> both score tests with their own nulls."""
    from rtnsim.cfhmm import lrt_ms, fit
    from rtnsim.integ import fit_integ_qn
    rng = np.random.default_rng(seed)
    t0 = time.time()
    statJ, f0, f1, beta, _ = lrt_ms(y, K, rng)
    z = y - beta
    out = dict(statJ=float(statJ), J=[float(x) for x in np.atleast_1d(f1["J"])], c=f0["c"].tolist(), e=f0["e"].tolist(),
               coef=np.asarray(f0["coef"]).tolist(), sig=float(f0["sig"]))
    ini = dict(coef=np.asarray(f0["coef"]), sig=float(f0["sig"]), c=f0["c"], e=f0["e"])
    # --- point-sampling model
    Ui = score_inst(z, K, f0); out["U_inst"] = Ui.tolist(); Ti = T_of(Ui)
    null = []
    for _ in range(nsim):
        zz = float(f0["coef"][0]) + sim_chain(f0["c"], f0["e"], len(y), rng, False) @ np.asarray(f0["coef"][1:]) + f0["sig"] * rng.standard_normal(len(y))
        try:
            g = fit(zz, K, False, ini); null.append(T_of(score_inst(zz, K, g)))
        except Exception:
            pass
    out["T_inst"], out["null_inst"] = Ti, null
    out["p_inst"] = float((1 + np.sum(np.array(null) >= Ti)) / (1 + len(null)))
    out["sec_inst"] = time.time() - t0; t0 = time.time()
    # --- integrated-sampling model
    fi = fit_integ_qn([z], K, False, ini)
    out["integ"] = dict(c=fi["c"].tolist(), e=fi["e"].tolist(), coef=fi["coef"].tolist(), sig=fi["sig"], ll=fi["ll"], ll_inst=float(f0["ll"]))
    Ug = score_integ(z, K, fi); out["U_integ"] = Ug.tolist(); Tg = T_of(Ug)
    inig = dict(coef=fi["coef"], sig=fi["sig"], c=fi["c"], e=fi["e"])
    null = []
    for _ in range(nsim):
        zz = float(fi["coef"][0]) + sim_chain(fi["c"], fi["e"], len(y), rng, True) @ fi["coef"][1:] + fi["sig"] * rng.standard_normal(len(y))
        try:
            g = fit_integ_qn([zz], K, False, inig, em_warm=0); null.append(T_of(score_integ(zz, K, g)))
        except Exception:
            pass
    out["T_integ"], out["null_integ"] = Tg, null
    out["p_integ"] = float((1 + np.sum(np.array(null) >= Tg)) / (1 + len(null)))
    out["sec_integ"] = time.time() - t0
    return out


def job(a):
    tag, y, K, seed, nsim = a
    try:
        return dict(tag=tag, **analyse(y, K, seed, nsim))
    except Exception as ex:
        import traceback
        return dict(tag=tag, error=repr(ex)[:200], tb=traceback.format_exc()[-600:])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rtninja":
        from r17_rtninja_real import DATA
        nsrc = int(sys.argv[2]); recs = sys.argv[3].split(","); nsim = int(sys.argv[4]) if len(sys.argv) > 4 else 19
        jobs = [(r, np.load(DATA / f"n{nsrc}" / f"noise05_{nsrc}source" / r / "MeasurementData.npy").astype(float), nsrc, 3500 + i, nsim)
                for i, r in enumerate(recs)]
        outp = ROOT / f"results/r35/rtninja_{nsrc}src.json"
    else:
        import runpy
        n = int(sys.argv[2]); nsim = int(sys.argv[3]) if len(sys.argv) > 3 else 19
        argv = sys.argv; sys.argv = ["r32", str(n), "1", "8192", "coop"]
        g = runpy.run_path(str(ROOT / "experiments/r32_box_power.py"), run_name="r32")
        sys.argv = argv
        jobs = [(f"{lab}{i}", y, 2, 3600 + i, nsim) for i in range(n) for lab, y, _ in g["make"](i)]
        outp = ROOT / "results/r35/sim_coop.json"
    res = json.load(open(outp)) if outp.exists() else []
    done = {r["tag"] for r in res}
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [j for j in jobs if j[0] not in done]):
            res.append(r); json.dump(res, open(outp, "w"))
            print(r["tag"][:12], "J", round(r.get("statJ", -1), 1), "inst", r.get("p_inst"), "integ", r.get("p_integ"),
                  round(r.get("sec_inst", 0)), round(r.get("sec_integ", 0)), r.get("error"), flush=True)
