import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from r11_nichol_null import sim_rep
from rtnsim.cfhmm import fit_ar1_multi
from rtnsim.integ import fit_integ_qn as fit_integ
vi, K, nrep, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
res = json.load(open(ROOT / f"results/nichol/Gate_P_v{vi}_K{K}.json"))
rng = np.random.default_rng(seed)
zs = [sim_rep(res, np.zeros(8192), 8192, rng) for _ in range(nrep)]
ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
t = time.time()
a0 = fit_ar1_multi(zs, K, False, ini, n_iter=40)
a1 = fit_ar1_multi(zs, K, True, dict(ini, coef=a0["coef"], c=a0["c"], e=a0["e"], sig=a0["sig"], rho=a0["rho"]), n_iter=40)
print("inst", 2 * (a1["ll"] - a0["ll"]), np.round(a1["J"], 2), round(time.time() - t), flush=True)
ini2 = dict(coef=np.array(res["coef"]), sig=res["sig"], c=np.array(res["c"]), e=np.array(res["e"]))
b0 = fit_integ(zs, K, False, ini2)
b1 = fit_integ(zs, K, True, dict(ini2, coef=b0["coef"], c=b0["c"], e=b0["e"], sig=b0["sig"], off=b0["off"]))
b0b = fit_integ(zs, K, False, dict(ini2, coef=b1["coef"], c=b1["c"], e=b1["e"], sig=b1["sig"], off=b1["off"]))
l0 = max(b0["ll"], b0b["ll"])
print("integ", 2 * (b1["ll"] - l0), np.round(b1["J"], 2), "e true", np.round(res["e"], 3), "e integ", np.round(b0["e"], 3), round(time.time() - t), flush=True)
