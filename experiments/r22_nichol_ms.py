"""R22: Nichol verdicts re-checked with the multi-start per-repetition fit (lrt_cnn3_ms, 3 starts, J = 0
likelihood picks).  Same pipeline as r11_nichol (+ r11_alt_qn for the drive models); outputs go to
results/nichol_ms so the single-start results stay untouched.
    python experiments/r22_nichol_ms.py <file> <vi> <K> [alt]
"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
from rtnsim.stepnet import denoise
N.OUT = ROOT / "results/nichol_ms"
N.OUT.mkdir(parents=True, exist_ok=True)
_net = None


def per_rep(a):
    global _net
    r, y, K = a
    if _net is None:
        _net = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    s, f0, f1, beta, _ = lrt_cnn3_ms(y, K, _net, ar1=True, rng=np.random.default_rng(r), n_start=3)
    lev, edge, step = denoise(_net, y)
    eh = np.abs(step[edge > 0.5]) if np.any(edge > 0.5) else np.array([])
    return r, dict(stat=float(s), coef=f0["coef"].tolist(), sig=f0["sig"], rho=f0["rho"],
                   c=f0["c"].tolist(), e=f0["e"].tolist(), J1=f1["J"].tolist(), beta=beta, ll=f0["ll"],
                   edges=eh.tolist())


def main():
    fname, vi, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    s = sio.loadmat(N.DATA / fname)["savedat"][0, 0]
    d = s["data"]; V = s["Vgate"].ravel()
    path = N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"
    if not path.exists():
        t0 = time.time()
        ys = [d[vi, r].astype(float) for r in range(d.shape[1])]
        with Pool(2) as p:
            reps = dict(p.imap_unordered(per_rep, [(r, ys[r], K) for r in range(len(ys))]))
        reps = [reps[r] for r in range(len(ys))]
        zs = [ys[r] - np.asarray(reps[r]["beta"]) for r in range(len(ys))]
        f0, f1 = N.joint(zs, K, reps)
        res = dict(V=float(V[vi]), K=K, lrt_joint=2 * max(f1["ll"] - f0["ll"], 0.0), J=f1["J"].tolist(),
                   bic0=f0["bic"], coef=f0["coef"].tolist(), sig=f0["sig"], rho=f0["rho"],
                   c=f0["c"].tolist(), e=f0["e"].tolist(), c1=f1["c"].tolist(), e1=f1["e"].tolist(),
                   rep_stats=[r["stat"] for r in reps], rep_J=[r["J1"] for r in reps], rep_ll=[r["ll"] for r in reps])
        json.dump(res, open(path, "w"))
        np.save(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy", np.array([r["beta"] for r in reps]))
        print(f"V={V[vi]} K={K} LRT_joint={res['lrt_joint']:.2f} J={np.round(res['J'], 3)} ({time.time()-t0:.0f}s)", flush=True)
    if len(sys.argv) > 4 and sys.argv[4] == "alt":
        import r11_alt_qn as A
        res = json.load(open(path))
        betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
        ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
        for r0 in range(0, d.shape[1], 10):
            op = N.OUT / f"{fname[:-4]}_v{vi}_K{K}_altqn_r{r0}.json"
            if op.exists():
                continue
            zs = [d[vi, r].astype(float) - betas[r] for r in range(r0, r0 + 10)]
            out, _ = A.analyse(zs, K, ini, np.random.default_rng(vi))
            out["V"] = res["V"]
            json.dump(out, open(op, "w"), indent=1)
            print(r0, {k: np.round(out[k], 2).tolist() for k in ("lrt_coupling", "lrt_drive", "lrt_coupling_given_drive", "J_given_drive")}, flush=True)


if __name__ == "__main__":
    main()
