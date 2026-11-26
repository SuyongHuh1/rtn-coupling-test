"""R34: checklist item (6) on Si/SiGe Gate_P (Nichol), the boxcar-integrated record.  Per repetition: the
pipeline statistic (lrt_cnn3_ms, ar1, 3 starts, mode from RTN_JOINT), then p from NSIM instant nulls and NSIM
boxcar nulls.  Also a pooled test: sum of the per-repetition statistics against sums of one null per repetition.
    python experiments/r34_gatep_box.py <vi> <K> <workers> [nsim]
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
VI, K, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NSIM = int(sys.argv[4]) if len(sys.argv) > 4 else 19
MODE = int(os.environ.get("RTN_JOINT", "0") or 0)
NET = None


def simulate_null_box(f0, beta, resid, T, rng, sub=20):
    """J = 0 null that reproduces the measurement chain: chain on a sub-sample grid (rates / sub per step),
    boxcar-averaged over `sub` steps; boxcar-induced transition outliers removed from the residual pool."""
    from rtnsim.robustk import _block_boot
    K = len(f0["c"])
    occ = np.zeros((T, K))
    for k in range(K):
        c, e = f0["c"][k] / sub, f0["e"][k] / sub
        kt = c + e
        q = 1 - np.exp(-kt)
        p01, p10 = c / kt * q, e / kt * q
        st = int(rng.random() < c / kt)
        u = rng.random(T * sub)
        x = np.empty(T * sub)
        for t in range(T * sub):
            st = st ^ int(u[t] < (p01 if st == 0 else p10))
            x[t] = st
        occ[:, k] = x.reshape(T, sub).mean(1)
    r = resid - np.median(resid)
    mad = np.median(np.abs(r)) / 0.6745
    r = np.where(np.abs(r) > 5 * mad, rng.normal(0, mad, r.size), r)
    return beta + f0["coef"][0] + occ @ f0["coef"][1:] + _block_boot(r, T, rng)


def job(a):
    global NET
    import rtnsim.r9 as R9
    from rtnsim.cfhmm import simulate_null
    from rtnsim.robustk import bits
    r, y = a
    if NET is None:
        NET = R9.load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    try:
        rng = np.random.default_rng(3400 + r)
        s, f0, f1, beta, _ = R9.lrt_cnn3_ms(y, K, NET, ar1=True, rng=rng, n_start=3)
        X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
        resid = y - beta - (X @ f0["coef"])[f0["path"]]
        out = dict(rep=r, stat=float(s), J=[float(x) for x in np.atleast_1d(f1.get("J", []))],
                   kappa=[float(x) for x in np.atleast_1d(f1.get("kappa", []))])
        for name, sim in (("inst", simulate_null), ("box", simulate_null_box)):
            null = []
            for _ in range(NSIM):
                try:
                    null.append(float(R9.lrt_cnn3_ms(sim(f0, beta, resid, len(y), rng), K, NET, ar1=True, rng=rng, n_start=3)[0]))
                except Exception:
                    null.append(np.nan)
            nn = np.array(null)
            ok = nn[np.isfinite(nn)]
            out["null_" + name] = null
            out["p_" + name] = float((1 + np.sum(ok >= s)) / (1 + ok.size))
        return out
    except Exception as ex:
        return dict(rep=r, error=repr(ex)[:200])


if __name__ == "__main__":
    s = sio.loadmat(N.DATA / "Gate_P.mat")["savedat"][0, 0]
    d = s["data"]; V = float(s["Vgate"].ravel()[VI])
    outp = ROOT / f"results/r34_gatep_v{VI}_K{K}_j{MODE}.json"
    res = json.load(open(outp))["rows"] if outp.exists() else []
    done = {r["rep"] for r in res}
    todo = [(r, d[VI, r].astype(float)) for r in range(d.shape[1]) if r not in done]
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, todo):
            res.append(r); json.dump(dict(rows=res), open(outp, "w"))
            print(r["rep"], r.get("stat"), r.get("p_inst"), r.get("p_box"), r.get("error"), flush=True)
    ok = [r for r in res if "error" not in r]
    summ = dict(V=V, K=K, mode=MODE, n=len(ok), n_err=len(res) - len(ok),
                rej_inst=int(sum(r["p_inst"] < 0.05 for r in ok)), rej_box=int(sum(r["p_box"] < 0.05 for r in ok)))
    if ok:
        obs = sum(r["stat"] for r in ok)
        for name in ("inst", "box"):
            M = np.array([r["null_" + name] for r in ok])          # reps x NSIM
            sums = np.nansum(M, 0)
            summ["sum_obs"], summ["p_sum_" + name] = obs, float((1 + np.sum(sums >= obs)) / (1 + sums.size))
    print(json.dumps(summ))
    json.dump(dict(summary=summ, rows=res), open(outp, "w"))
