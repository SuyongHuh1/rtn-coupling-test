"""Validate the integrated-sampling emission: independent boxcar-sampled records (16x oversampled),
instantaneous AR(1) model vs integrated model, coupling LRT.   python experiments/r11_integ_test.py n"""
import sys, time, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r11_nichol_null import sim_rep
from rtnsim.cfhmm import fit_ar1_multi
from rtnsim.integ import fit_integ
def job(j):
    rng = np.random.default_rng(j)
    K = 2 if j % 2 == 0 else 3
    res = dict(K=K, coef=[0.0, 1.0, 0.8, 0.4][:K + 1], c=[0.02, 0.15, 0.09][:K], e=[0.02, 1.5, 1.8][:K],
               rho=0.0, sig=0.25)
    zs = [sim_rep(res, np.zeros(4096), 4096, rng) for _ in range(5)]
    ini = dict(coef=np.array(res["coef"]), sig=0.3, rho=0.0, c=np.array(res["c"]), e=np.array(res["e"]))
    t = time.time()
    a0 = fit_ar1_multi(zs, K, False, ini, n_iter=40); a1 = fit_ar1_multi(zs, K, True, dict(ini, coef=a0["coef"], c=a0["c"], e=a0["e"], sig=a0["sig"], rho=a0["rho"]), n_iter=40)
    t1 = time.time()
    b0 = fit_integ(zs, K, False, ini); b1 = fit_integ(zs, K, True, dict(ini, coef=b0["coef"], c=b0["c"], e=b0["e"], sig=b0["sig"], off=b0["off"]))
    return dict(K=K, lrt_inst=2 * (a1["ll"] - a0["ll"]), lrt_integ=2 * (b1["ll"] - b0["ll"]), J_inst=a1["J"].tolist(),
                J_integ=b1["J"].tolist(), e_integ=b0["e"].tolist(), e_inst=a0["e"].tolist(), sec=(t1 - t, time.time() - t1))
if __name__ == "__main__":
    n = int(sys.argv[1])
    with Pool(2) as p:
        out = list(p.imap_unordered(job, range(n)))
    for o in out: print(json.dumps({k: (np.round(v, 2).tolist() if not isinstance(v, int) else v) for k, v in o.items()}))
