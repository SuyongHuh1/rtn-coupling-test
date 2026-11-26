"""R44 (review): power of the three-trap replication rule (checklist step 5, Table II row "3 traps, replicated").

The stored coupled set (records.npz) was simulated before the last simulator change and cannot be regenerated,
so a fresh coupled set is used: the 100 coupled three-trap partners of the exp_new independent records
(r8_oracle.make(100, seed=291), 1/f; same generator stream as R40, whose originals reproduce exactly).
Each coupled original is calibrated (B = 99, Besag-Clifford stop after 5 exceedances); each detected one gets a
replicate with the SAME true parameters, new state path and new noise (R40's patched make_batch), calibrated alike.
Replicated detection = both p <= 0.05 AND the same pair (largest |J|) with the same sign.

    RTN_MS=3 RTN_RESTART=3 RTN_JOINT=0 python experiments/r44_rep_power.py [workers]
-> results/r8/r44_rep_power.json (resumable).  Also gives a second estimate of the single-record power (0.56).
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
os.environ.setdefault("RTN_MS", "3")
OUT = ROOT / "results/r8/r44_rep_power.json"
B_FULL, STOP = 99, 5


def calib(y, seed):
    import r8_calib as C
    from rtnsim.cfhmm import simulate_null
    from rtnsim.robustk import bits
    rng = np.random.default_rng(seed)
    s_obs, f0, f1, beta, _ = C._r9_stat(y, 3, rng)
    X = np.hstack([np.ones((8, 1)), bits(3)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    null, ex = [], 0
    while len(null) < B_FULL and ex < STOP:
        try:
            s = C._r9_stat(simulate_null(f0, beta, resid, len(y), rng), 3, rng)[0]
        except ValueError:
            continue
        null.append(float(s)); ex += s >= s_obs
    p = (1 + ex) / (1 + B_FULL) if len(null) == B_FULL else None
    return dict(stat=float(s_obs), n_null=len(null), exceed=int(ex), p=p, sig=bool(p is not None and p <= 0.05),
                J=np.atleast_1d(f1["J"]).tolist(), **C._checklist12(f1, 3))


def top(J):
    J = np.atleast_1d(np.asarray(J, float)); i = int(np.argmax(np.abs(J))); return i, int(np.sign(J[i]))


def job(a):
    key, y, y_rep = a
    r = dict(orig=calib(y, 500 + key))
    if r["orig"]["sig"]:
        r["rep"] = calib(y_rep, 44_000 + key)
        r["same_pair_sign"] = top(r["rep"]["J"]) == top(r["orig"]["J"])
    return key, r


def ok12(x):
    return (x["dwell_min"] or 0) >= 3 and (x["trans_min"] or 0) >= 15


def main():
    sys.argv = [sys.argv[0], "--r9"] + sys.argv[1:]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    import r40_replication as R
    recs = R.make_pairs()
    keys = [k for k, v in recs.items() if v["coupled"] == 1]
    assert len(keys) == 100
    res = json.load(open(OUT)) if OUT.exists() else {}
    todo = [(k, recs[k]["y"], recs[k]["y_rep"]) for k in keys if str(k) not in res]
    print(len(todo), "to do", flush=True)
    with Pool(W) as p:
        for k, r in p.imap_unordered(job, todo):
            res[str(k)] = r; json.dump(res, open(OUT, "w"))
            print(k, "orig p", r["orig"]["p"], "rep p", r.get("rep", {}).get("p"), "same", r.get("same_pair_sign"), flush=True)
    v = [res[str(k)] for k in keys if str(k) in res]
    det = [x for x in v if x["orig"]["sig"]]
    rep = [x for x in det if x["rep"]["sig"] and x["same_pair_sign"]]
    res["summary"] = dict(done=len(v), power=len(det) / max(1, len(v)), power_pass12=sum(ok12(x["orig"]) for x in det) / max(1, len(v)),
                          rep_sig=sum(x["rep"]["sig"] for x in det), replicated=len(rep), power_rep=len(rep) / max(1, len(v)),
                          power_rep_pass12=sum(ok12(x["orig"]) and ok12(x["rep"]) for x in rep) / max(1, len(v)))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"]), flush=True)


if __name__ == "__main__":
    main()
