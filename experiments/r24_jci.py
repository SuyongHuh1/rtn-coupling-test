"""R24: interval for J.  The Hessian/sandwich interval under-covers (white 0.88-0.90, 1/f 0.70; R9 era).
Replacement = pipeline bootstrap, the same machinery as the calibration: records from the FITTED COUPLED
model (J = J_hat) + the record's baseline + block-bootstrapped residual, the whole pipeline (CNN, search,
fit) re-run on each; basic bootstrap interval [2 J_hat - q97.5, 2 J_hat - q2.5] and percentile interval.
Coverage of the true J on R10 coupled records (K = 2; true phi random in 0.2-0.8, fit phi = 0.5).
    python experiments/r24_jci.py white|full [n=40] [B=39]
"""
from __future__ import annotations
import json, sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
COND = sys.argv[1] if len(sys.argv) > 1 else "full"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
B = int(sys.argv[3]) if len(sys.argv) > 3 else 39
OUT = ROOT / f"results/r23/r24_jci_{COND}.json"
NET = None


def net():
    global NET
    if NET is None:
        from rtnsim.r9 import load_stepnet
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    return NET


def job(a):
    i, y, J_true = a
    from rtnsim.r9 import lrt_cnn3_ms, lrt_cnn3
    from rtnsim.cfhmm import Kinetics, simulate_from_chain
    from rtnsim.robustk import bits
    rng = np.random.default_rng(i)
    try:
        st, f0, f1, beta, _ = lrt_cnn3_ms(y, 2, net(), ar1=True, rng=rng, n_start=3)
        J_hat = float(np.atleast_1d(f1["J"])[0])
        X = np.hstack([np.ones((4, 1)), bits(2)])
        path = f1.get("path", f0["path"])
        resid = y - beta - (X @ f1["coef"])[path]
        P = Kinetics(2, np.asarray(f1["c"]), np.asarray(f1["e"]), np.atleast_1d(f1["J"])).P_np()
        w, V = np.linalg.eig(P.T); pi0 = np.abs(np.real(V[:, np.argmin(np.abs(w - 1))]))
        fm = dict(P=P, pi0=pi0, Xd=X, coef=np.asarray(f1["coef"]))
        Jb = []
        for b in range(B):
            yb = simulate_from_chain(fm, beta, resid, len(y), rng)
            _, _, g1, *_ = lrt_cnn3(yb, 2, net(), ar1=True, rng=np.random.default_rng(rng.integers(1 << 31)))
            Jb.append(float(np.atleast_1d(g1["J"])[0]))
        return i, dict(J_true=J_true, J_hat=J_hat, stat=float(st), Jb=Jb)
    except Exception as ex:
        return i, dict(error=repr(ex)[:200], J_true=J_true)


def main():
    import r8_oracle as O
    ys, ss, rows = O.make(100, white=(COND == "white"))
    idx = [i for i in range(len(rows)) if rows[i]["K"] == 2 and rows[i]["coupled"] == 1][:N]
    import torch
    res = json.load(open(OUT)) if OUT.exists() else {}
    # true J: regenerate labels is not stored in rows -> Jmax carries |J| only; sign from fit is compared on |J|
    tasks = [(i, ys[i], rows[i]["Jsum"]) for i in idx if str(i) not in res]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for i, r in p.imap_unordered(job, tasks):
            res[str(i)] = r
            json.dump(res, open(OUT, "w"), indent=1)
    ok = [r for r in res.values() if "Jb" in r]
    cov_b, cov_p, wid = [], [], []
    for r in ok:
        Jt, Jh, Jb = r["J_true"], r["J_hat"], np.array(r["Jb"])
        lo, hi = np.quantile(Jb, [0.025, 0.975])
        cov_b.append(2 * Jh - hi <= Jt <= 2 * Jh - lo); cov_p.append(lo <= Jt <= hi); wid.append(hi - lo)
    summ = dict(n=len(ok), err=len(res) - len(ok), cover_basic=float(np.mean(cov_b)), cover_pct=float(np.mean(cov_p)),
                width_med=float(np.median(wid)),
                sign_ok=float(np.mean([np.sign(r["J_hat"]) == np.sign(r["J_true"]) for r in ok])),
                rel_bias_med=float(np.median([(r["J_hat"] - r["J_true"]) / r["J_true"] for r in ok])))
    json.dump(dict(summary=summ, res=res), open(OUT, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
