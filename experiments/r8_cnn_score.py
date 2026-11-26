"""Score the GPU-trained R7 network on the R8 oracle records (same seeds), so CNN, HMM path
statistic and R8 LRT are compared on identical data.  Writes results/r8/r8_cnn_<cond>.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from rtnsim.net_r7 import CouplingNet  # noqa: E402
from rtnsim.sim2_torch import preprocess  # noqa: E402
import r8_oracle as O  # noqa: E402

ck = torch.load(ROOT / "results/r7_gpu/model.pt", map_location="cpu")
net = CouplingNet(C=ck["config"]["C"]).eval()
net.load_state_dict(ck["state_dict"])
for cond in ("white", "full"):
    ys, ss, rows = O.make(100, white=(cond == "white"))
    with torch.no_grad():
        x = preprocess(torch.tensor(np.stack(ys), dtype=torch.float32))
        o = net(x)
    sc = o["coupled"].numpy()
    kp = o["K"].argmax(-1).numpy() + 2
    steps = o["theta"][:, :, 2].numpy()
    summ = {}
    for K in (2, 3):
        idx = np.array([i for i in range(len(rows)) if rows[i]["K"] == K])
        lab = np.array([rows[i]["coupled"] for i in idx])
        s = sc[idx]; s0, s1 = s[lab == 0], s[lab == 1]
        thr = np.quantile(s0, 0.95)
        summ[f"K={K}"] = dict(auc=float((s1[:, None] > s0[None]).mean()), tpr=float((s1 > thr).mean()),
                              K_acc=float((kp[idx] == K).mean()))
    json.dump(dict(summary=summ, score=sc.tolist(), K_pred=kp.tolist(), steps=steps.tolist()),
              open(ROOT / f"results/r8/r8_cnn_{cond}.json", "w"))
    print(cond, summ)
