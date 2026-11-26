"""Integrated-sampling coupling LRT on Nichol voltages, with an (oracle-start) parametric null that
includes the 60 Hz boxcar.   python experiments/r11_integ_real.py <vi> <K> <n_sim>
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from r11_nichol_null import sim_rep
from rtnsim.integ import fit_integ_qn


def lrt(zs, K, ini):
    f0 = fit_integ_qn(zs, K, False, ini)
    base = dict(coef=f0["coef"], c=f0["c"], e=f0["e"], sig=f0["sig"], off=f0["off"])
    f1 = None
    for j0 in (0.0, 0.3, -0.3):
        g = fit_integ_qn(zs, K, True, dict(base, J=np.full(K * (K - 1) // 2, j0)), em_warm=0)
        if f1 is None or g["ll"] > f1["ll"]:
            f1 = g
    f0b = fit_integ_qn(zs, K, False, dict(coef=f1["coef"], c=f1["c"], e=f1["e"], sig=f1["sig"], off=f1["off"]), em_warm=0)
    if f0b["ll"] > f0["ll"]:
        f0 = f0b
    return 2 * max(f1["ll"] - f0["ll"], 0.0), f0, f1


def null_job(a):
    j, f0, K, T, nrep, betas = a
    rng = np.random.default_rng(777 + j)
    res = dict(K=K, coef=f0["coef"].tolist(), c=f0["c"].tolist(), e=f0["e"].tolist(), rho=0.0, sig=f0["sig"])
    zs = [sim_rep(res, betas[r] - np.median(betas[r]), T, rng) for r in range(nrep)]
    # same analysis chain: per-rep baseline is not re-estimated here (oracle baseline removed)
    zs = [z - (b - np.median(b)) for z, b in zip(zs, betas)]
    s, *_ = lrt(zs, K, dict(coef=f0["coef"], c=f0["c"], e=f0["e"], sig=f0["sig"]))
    return j, s


def main():
    vi, K, n_sim = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    d = sio.loadmat(N.DATA / "Gate_P.mat")["savedat"][0, 0]["data"]
    res = json.load(open(N.OUT / f"Gate_P_v{vi}_K{K}.json"))
    betas = np.load(N.OUT / f"Gate_P_v{vi}_K{K}_beta.npy")
    zs = [d[vi, r].astype(float) - betas[r] for r in range(len(betas))]
    ini = dict(coef=np.array(res["coef"]), sig=res["sig"], c=np.array(res["c"]), e=np.array(res["e"]))
    s, f0, f1 = lrt(zs, K, ini)
    print(json.dumps(dict(V=res["V"], K=K, lrt_integ=s, J=f1["J"].tolist(), c=f0["c"].tolist(), e=f0["e"].tolist(),
                          coef=f0["coef"].tolist(), sig=f0["sig"], lrt_inst=res["lrt_joint"])), flush=True)
    null = []
    if n_sim:
        with Pool(2) as p:
            for j, x in p.imap_unordered(null_job, [(j, f0, K, betas.shape[1], len(betas), betas) for j in range(n_sim)]):
                null.append(x)
    out = dict(V=res["V"], K=K, lrt_integ=s, J=f1["J"].tolist(), c=f0["c"].tolist(), e=f0["e"].tolist(),
               c1=f1["c"].tolist(), e1=f1["e"].tolist(), coef=f0["coef"].tolist(), sig=f0["sig"], null=null,
               p=float((1 + sum(x >= s for x in null)) / (1 + len(null))) if null else None)
    json.dump(out, open(N.OUT / f"Gate_P_v{vi}_K{K}_integ.json", "w"), indent=1)
    print(json.dumps(dict(V=out["V"], K=K, lrt=s, p=out["p"], null_max=max(null) if null else None)), flush=True)


if __name__ == "__main__":
    main()
