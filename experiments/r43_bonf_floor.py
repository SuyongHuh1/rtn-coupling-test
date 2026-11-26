"""R43 (review): raise B for the two measured records whose calibrated p sits at the floor 1/(B+1), so the
floor lies below the Bonferroni level alpha/m of checklist step 3 (m = biases scanned on that device).

    lh    LH_cond_Sgate 0.151 V (v6, K = 3), same-analysis drive nulls of r14_null_cons; m = 9 -> B = 399
    soton Southampton EDTM RTS_vg-0.645vd10.3 (K = 3), pipeline nulls of r20_southampton; m = 7 -> B = 299
Existing nulls are kept; new nulls use new seeds (iid from the same fitted null model).
    RTN_MS=3 RTN_RESTART=3 RTN_JOINT=0 RTN_WORKERS=2 python experiments/r43_bonf_floor.py lh|soton [B] [time1]
-> results/nichol/LH_cond_Sgate_v6_K3_nullcons.json (extended) or results/r8/r43_soton_bonf.json
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
REC = "RTS_vg-0.645vd10.3_tsample1_25000_plc10.csv"
SOUT = ROOT / "results/r8/r43_soton_bonf.json"
_S = {}


def soton_setup():
    import r20_southampton as R20
    import r5_real as R
    from rtnsim.realdata import load_b1500
    src = json.load(open(ROOT / "results/r8/r20_southampton_v2.json"))
    row = [r for r in src["rows"] if r["rec"] == REC and r["K"] == 3][0]
    rec = [r for r in R.records() if r["file"] == REC][0]
    t, I, _ = load_b1500(rec["path"])
    y = R.preprocess(t, I, rec["trim_s"])[1]
    # reproduce the stored fit: seed 9100 + 10 i + K with i the rank by length among the analysed records
    names = [r["rec"] for r in src["summary"]]
    lens = {}
    for r in R.records():
        if r["file"] in names:
            tt, II, _ = load_b1500(r["path"]); lens[r["file"]] = len(R.preprocess(tt, II, r["trim_s"])[1])
    i = sorted(lens, key=lambda k: lens[k]).index(REC)
    rng = np.random.default_rng(9100 + 10 * i + 3)
    (s, f0, f1, beta, _), _ = R20.fit_best(y, 3, rng)
    return y, row, float(s), f0, beta


def soton_job(j):
    import r20_southampton as R20
    from rtnsim.cfhmm import simulate_null
    from rtnsim.robustk import bits
    y, row, s, f0, beta = _S["v"]
    rng = np.random.default_rng(77_000 + j)
    X = np.hstack([np.ones((8, 1)), bits(3)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    return j, float(R20.fit_best(simulate_null(f0, beta, resid, len(y), rng), 3, rng)[0][0])


def _init():
    _S["v"] = soton_setup()


def run_soton(B, time1):
    v = soton_setup(); _S["v"] = v
    y, row, s, f0, beta = v
    print("stored stat", row["stat"], "reproduced", s, flush=True)
    res = json.load(open(SOUT)) if SOUT.exists() else {"stat_stored": row["stat"], "stat": s, "null_old": row["null"], "null_new": {}}
    need = B - len(res["null_old"])
    todo = [j for j in range(need) if str(j) not in res["null_new"]][: (1 if time1 else None)]
    t0 = time.time()
    with Pool(int(os.environ.get("RTN_WORKERS", "2")), initializer=_init) as p:
        for j, x in p.imap_unordered(soton_job, todo):
            res["null_new"][str(j)] = x; json.dump(res, open(SOUT, "w"))
            print(j, round(x, 2), round(time.time() - t0), flush=True)
    null = np.r_[res["null_old"], list(res["null_new"].values())]
    obs = row["stat"]
    res["summary"] = dict(obs=obs, n=len(null), exceed=int((null >= obs).sum()), p=float((1 + (null >= obs).sum()) / (1 + len(null))))
    json.dump(res, open(SOUT, "w"), indent=1); print(res["summary"], flush=True)


def run_lh(B, time1):
    import r14_null_cons as C, r11_nichol as N
    vi = 6
    path = N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullcons.json"
    state = json.load(open(path))
    d = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["m2"]
    res = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3.json"))
    betas = np.load(N.OUT / f"LH_cond_Sgate_v{vi}_K3_beta.npy")[:10]
    todo = [(j, d, res, betas) for j in range(B) if str(j) not in state["null"]][: (1 if time1 else None)]
    t0 = time.time()
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for j, s in p.imap_unordered(C.job, todo):
            state["null"][str(j)] = s; json.dump(state, open(path, "w"), indent=1)
            print(j, round(s, 2), round(time.time() - t0), flush=True)
    obs = json.load(open(N.OUT / f"LH_cond_Sgate_v{vi}_K3_altqn.json"))["lrt_coupling_given_drive"]
    null = np.array(list(state["null"].values()))
    state["summary"] = dict(obs=obs, n=len(null), exceed=int((null >= obs).sum()), p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                            q95=float(np.quantile(null, 0.95)), max=float(null.max()))
    json.dump(state, open(path, "w"), indent=1); print(state["summary"], flush=True)


if __name__ == "__main__":
    which = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else (399 if which == "lh" else 299)
    t1 = len(sys.argv) > 3
    (run_lh if which == "lh" else run_soton)(B, t1)
