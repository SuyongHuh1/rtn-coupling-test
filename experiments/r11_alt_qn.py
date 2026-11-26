"""Alternative explanations at one Nichol voltage with the quasi-Newton multi-segment fit:
M0 independent, M1 coupled, M2 independent + slow drive (rate <= 1/(10 x slowest trap)),
M3 coupled + drive.  Several starts per model; nested refits guard against local optima.
    python experiments/r11_alt_qn.py <vi> <K> [file]
"""
import json, sys
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import r11_nichol as N
from rtnsim.multiqn import fit_multi_qn


def strip(f):
    return {k: v for k, v in f.items() if k not in ("fg", "x")}


def analyse(zs, K, ini, rng):
    nJ = K * (K - 1) // 2
    m0 = fit_multi_qn(zs, K, False, ini)
    base = dict(ini, coef=m0["coef"], c=m0["c"], e=m0["e"], sig=m0["sig"], rho=m0["rho"], off=m0["off"])
    m1 = max((fit_multi_qn(zs, K, True, dict(base, J=np.full(nJ, j))) for j in (0.0, 0.5)), key=lambda f: f["ll"])
    m0b = fit_multi_qn(zs, K, False, dict(base, coef=m1["coef"], c=m1["c"], e=m1["e"], sig=m1["sig"], rho=m1["rho"], off=m1["off"]))
    m0 = max((m0, m0b), key=lambda f: f["ll"])
    slow = 1.0 / np.min(m0["c"] + m0["e"])
    mr = 1.0 / (10 * slow)
    m2 = None
    for m in (10, 30, 100):
        for s in range(1):
            g = fit_multi_qn(zs, K, False, dict(base, coef=m0["coef"], c=m0["c"], e=m0["e"], sig=m0["sig"], rho=m0["rho"],
                                                 off=m0["off"], gamma=rng.normal(0, 1.0, K), drive=(1 / (m * slow),) * 2,
                                                 max_rate=mr), drive=True)
            if m2 is None or g["ll"] > m2["ll"]:
                m2 = g
    db = dict(base, coef=m2["coef"], c=m2["c"], e=m2["e"], sig=m2["sig"], rho=m2["rho"], off=m2["off"],
              gamma=m2["gamma"], drive=m2["drive"], max_rate=mr)
    m3 = max((fit_multi_qn(zs, K, True, dict(db, J=np.full(nJ, j)), drive=True) for j in (0.0, 0.5)),
             key=lambda f: f["ll"])
    m2b = fit_multi_qn(zs, K, False, dict(db, coef=m3["coef"], c=m3["c"], e=m3["e"], sig=m3["sig"], rho=m3["rho"],
                                          off=m3["off"], gamma=m3["gamma"], drive=m3["drive"]), drive=True)
    m2 = max((m2, m2b), key=lambda f: f["ll"])
    # nesting sanity: M0 within M1, M2; M2 within M3
    return dict(ll=[m0["ll"], m1["ll"], m2["ll"], m3["ll"]],
                lrt_coupling=2 * (m1["ll"] - m0["ll"]), lrt_drive=2 * (m2["ll"] - m0["ll"]),
                lrt_coupling_given_drive=2 * (m3["ll"] - m2["ll"]),
                J=m1["J"].tolist(), J_given_drive=m3["J"].tolist(), gamma=m2["gamma"].tolist(),
                drive=m2["drive"].tolist(), c=m0["c"].tolist(), e=m0["e"].tolist(), coef=m0["coef"].tolist(),
                sig=m0["sig"], rho=m0["rho"], msgs=[m["msg"][:30] for m in (m0, m1, m2, m3)]), (m0, m1, m2, m3)


def main():
    vi, K = int(sys.argv[1]), int(sys.argv[2])
    fname = sys.argv[3] if len(sys.argv) > 3 else "Gate_P.mat"
    nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    r0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    d = sio.loadmat(N.DATA / fname)["savedat"][0, 0]["data"]
    res = json.load(open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}.json"))
    betas = np.load(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_beta.npy")
    zs = [d[vi, r].astype(float) - betas[r] for r in range(r0, r0 + nrep)]
    ini = dict(coef=np.array(res["coef"]), sig=res["sig"], rho=res["rho"], c=np.array(res["c"]), e=np.array(res["e"]))
    out, _ = analyse(zs, K, ini, np.random.default_rng(vi))
    out["V"] = res["V"]
    json.dump(out, open(N.OUT / f"{fname[:-4]}_v{vi}_K{K}_altqn{'' if r0 == 0 else f'_r{r0}'}.json", "w"), indent=1)
    print(json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, (list, float)) and k != "msgs" else v) for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()
