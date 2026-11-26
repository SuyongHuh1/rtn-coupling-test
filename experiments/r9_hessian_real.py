"""Hessian diagnostics on the measured records (BIC-chosen K from r8_real): minimum detectable J,
confounding of J, J estimate with sandwich SE."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r5_real as R
from rtnsim.realdata import load_b1500
from rtnsim.cfhmm import lrt_ms
from rtnsim.fisher import diagnostics
r8 = json.load(open(ROOT / "results/r8/r8_real.json"))
out = {}
for j, r in enumerate(R.records()):
    k2, k3 = r8.get(r["file"] + "|K2"), r8.get(r["file"] + "|K3")
    if not k2:
        continue
    K = 2 if k2["bic_ar1"] < k3["bic_ar1"] else 3
    t, I, _ = load_b1500(r["path"]); t, y, _, dt = R.preprocess(t, I, r["trim_s"])
    s, f0, f1, beta, p = lrt_ms(y, K, np.random.default_rng(7000 + 10 * j + K), ar1=True, kalman=True)
    d0 = diagnostics(y - beta, dict(f0, J=[0.0] * (K * (K - 1) // 2)))
    d1 = diagnostics(y - beta, f1)
    out[r["file"]] = dict(K=K, T=len(y), J_hat=f1["J"].tolist(), se_J=d1["se_J"], J_min=d0["J_min"], conf=d0["conf"],
                          p_r8=r8[r["file"] + f"|K{K}"]["p_r8"])
    print(r["file"][:40], K, len(y), "J", np.round(f1["J"], 2), "se", np.round(d1["se_J"], 2),
          "Jmin", np.round(d0["J_min"], 2), "conf", np.round(d0["conf"], 2), "p", out[r["file"]]["p_r8"], flush=True)
json.dump(out, open(ROOT / "results/r8/r9_hessian_real.json", "w"), indent=1)
