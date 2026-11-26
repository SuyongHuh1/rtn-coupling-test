"""Conservative drive-null for the LH coupling candidates: simulate the fitted M2 (independent traps +
slow drive), then run the SAME analysis the real record went through (r11_alt_qn.analyse: start from the
per-repetition independent fit, M0 -> M1 -> M2 with three drive starts -> M3, nested refits), not the
oracle start.  Stat = 2 (ll3 - ll2).   python experiments/r14_null_cons.py <vi> <n_sim> [nrep]
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from r11_null_qn import simulate
from r11_alt_qn import analyse

FN, K = "LH_cond_Sgate.mat", 3


def job(a):
    j, d, res, betas = a
    rng = np.random.default_rng(9090 + j)
    ys = simulate(d, K, betas, betas.shape[1], rng)
    zs = [y - (b - np.median(b)) for y, b in zip(ys, betas)]
    ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
    out, _ = analyse(zs, K, ini, np.random.default_rng(j))
    return j, max(out["lrt_coupling_given_drive"], 0.0)


def main():
    vi, n = int(sys.argv[1]), int(sys.argv[2])
    nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    path = N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullcons.json"
    state = json.load(open(path)) if path.exists() else {"null": {}}
    d = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["m2"]
    res = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3.json"))
    betas = np.load(N.OUT / f"LH_cond_Sgate_v{vi}_K3_beta.npy")[:nrep]
    todo = [(j, d, res, betas) for j in range(n) if str(j) not in state["null"]]
    with Pool(int(__import__("os").environ.get("RTN_WORKERS", "2"))) as p:
        for j, s in p.imap_unordered(job, todo):
            state["null"][str(j)] = s
            json.dump(state, open(path, "w"), indent=1)
            print(j, round(s, 2), flush=True)
    obs = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_altqn.json"))["lrt_coupling_given_drive"]
    null = np.array(list(state["null"].values()))
    state["summary"] = dict(obs=obs, n=len(null), p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                            q95=float(np.quantile(null, 0.95)), max=float(null.max()))
    json.dump(state, open(path, "w"), indent=1)
    print(json.dumps(state["summary"]), flush=True)


if __name__ == "__main__":
    main()
