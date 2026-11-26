"""Alternative explanations at one Nichol voltage: independent / coupled / independent + slow drive /
coupled + slow drive (drive rate <= 1/(10 x slowest trap)), joint over repetitions.
    python experiments/r11_alt.py <vi> <K> [file]
"""
import json, sys
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from rtnsim.cfhmm import fit_ar1_multi, fit_ar1_multi_drive

vi, K = int(sys.argv[1]), int(sys.argv[2])
fname = sys.argv[3] if len(sys.argv) > 3 else "Gate_P.mat"
d = sio.loadmat(N.DATA / fname)["savedat"][0, 0]["data"]
res = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"))
betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
zs = [d[vi, r].astype(float) - betas[r] for r in range(len(betas))]
ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
m0 = fit_ar1_multi(zs, K, False, ini, n_iter=40)
m1 = fit_ar1_multi(zs, K, True, dict(ini, J=np.array(res["J"]), c=np.array(res["c1"]), e=np.array(res["e1"])), n_iter=40)
slow = 1.0 / np.min(m0["c"] + m0["e"])
rng = np.random.default_rng(vi)
best2 = None
for m in (10, 30, 100):
    r = 1.0 / (m * slow)
    g = fit_ar1_multi_drive(zs, K, False, dict(ini, coef=m0["coef"], off=m0["off"], c=m0["c"], e=m0["e"],
                            gamma=rng.normal(0, 0.5, K), drive=(r, r), max_rate=1.0 / (10 * slow)), n_iter=40)
    if best2 is None or g["ll"] > best2["ll"]:
        best2 = g
m3 = fit_ar1_multi_drive(zs, K, True, dict(ini, coef=best2["coef"], off=best2["off"], c=best2["c"], e=best2["e"],
                         gamma=best2["gamma"], drive=best2["drive"], max_rate=1.0 / (10 * slow)), n_iter=40)
out = dict(V=res["V"], K=K, ll_indep=m0["ll"], ll_coupled=m1["ll"], ll_drive=best2["ll"], ll_drive_coupled=m3["ll"],
           lrt_coupling=2 * (m1["ll"] - m0["ll"]), lrt_drive=2 * (best2["ll"] - m0["ll"]),
           lrt_coupling_given_drive=2 * (m3["ll"] - best2["ll"]), J_given_drive=m3["J"].tolist(),
           gamma=best2["gamma"].tolist(), drive=best2["drive"].tolist())
json.dump(out, open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_alt.json", "w"), indent=1)
print(json.dumps(out))
