"""R40 (review): false-positive rate of the three-trap replication rule (checklist step 5).

For every three-trap null record of the exp_new set (r8_oracle.make(100, seed=291), the 1/f independent
records calibrated with B = 99 in r8_calib_exp_new_r9_ms3_B99.json) a replicate is generated with the SAME
true parameters (rates, occupancies, step heights, noise and drift amplitudes) but a new state path and new
noise: make_batch is re-run from the same generator state with the chain and the noise drawn from a second
generator, while the parameter stream is consumed exactly as in the original.  The replicate is analysed and
calibrated like the original (same pipeline, B = 99), with the Besag-Clifford sequential stop: nulls are drawn
until 5 of them reach the observed statistic (then p > 0.05 for B = 99 is certain) or all 99 are done, so the
decision at p <= 0.05 is identical to a full B = 99 run.

    RTN_MS=3 python experiments/r40_replication.py [workers] [which]    which = fp (default) | all
-> results/r8/r40_replication.json.  Replicated FPR = #(original and replicate both p <= 0.05) / #records.
"""
import inspect, json, os, sys, textwrap
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
os.environ.setdefault("RTN_MS", "3")
from rtnsim import sim2_torch as S

OUT = ROOT / "results/r8/r40_replication.json"
SRC = ROOT / "results/r8/r8_calib_exp_new_r9_ms3_B99.json"
T, SEED, N = 4096, 291, 100
B_FULL, STOP = 99, 5                     # p <= 0.05 at B = 99 needs at most 4 exceedances


def _patched_make_batch():
    """make_batch with the chain and the noise drawn from module-level REP generators; the parameter
    stream of `gen` is consumed exactly as in the original (the original chain draw is made and discarded)."""
    src = textwrap.dedent(inspect.getsource(S.make_batch))
    a = "    x = sample_chain(torch.cat([Q1, Q0]), Tf, dt, gen)"
    b = "    _ = sample_chain(torch.cat([Q1, Q0]), Tf, dt, gen)\n    x = sample_chain(torch.cat([Q1, Q0]), Tf, dt, REP['chain'])"
    c = "    gw = torch.Generator(device=dev).manual_seed(int(torch.randint(0, 2 ** 31, (1,), generator=gen, device=gen.device)))"
    d = "    _ = int(torch.randint(0, 2 ** 31, (1,), generator=gen, device=gen.device))\n    gw = REP['noise']"
    assert src.count(a) == 1 and src.count(c) == 1
    ns = dict(vars(S))
    exec(src.replace(a, b).replace(c, d).replace("def make_batch", "def make_batch_rep"), ns)
    return ns["make_batch_rep"], ns


def make_pairs(rep_seed=4040):
    """Three-trap records in r8_oracle.make order (index 200 + j) and their replicates."""
    mk, ns = _patched_make_batch()
    out, K = {}, 3
    g = torch.Generator().manual_seed(SEED + K)
    got, idx = 0, 200
    while got < N:
        st = g.get_state()
        y, lab = S.make_batch(50, T, g, "cpu", p_drive=0, p_box=0, return_states=True)
        if int(lab["K"][0]) != K:
            continue
        g2 = torch.Generator().manual_seed(0)
        g2.set_state(st)
        ns["REP"] = dict(chain=torch.Generator().manual_seed(rep_seed + got), noise=torch.Generator().manual_seed(rep_seed + 10_000 + got))
        y2, lab2 = mk(50, T, g2, "cpu", p_drive=0, p_box=0, return_states=True)
        assert torch.equal(lab["theta"], lab2["theta"]) and torch.equal(lab["Jvec"], lab2["Jvec"])
        take = min(50, N - got)
        for r in range(2 * take):
            out[idx] = dict(y=y[r].double().numpy(), y_rep=y2[r].double().numpy(), coupled=int(lab["coupled"][r]))
            idx += 1
        got += take
    return out


def job(a):
    key, y, K, seed = a
    import r8_calib as C
    from rtnsim.cfhmm import simulate_null
    from rtnsim.robustk import bits
    rng = np.random.default_rng(seed)
    s_obs, f0, f1, beta, _ = C._r9_stat(y, K, rng)
    X = np.hstack([np.ones((2 ** K, 1)), bits(K)])
    resid = y - beta - (X @ f0["coef"])[f0["path"]]
    null, ex = [], 0
    while len(null) < B_FULL and ex < STOP:
        try:
            s = C._r9_stat(simulate_null(f0, beta, resid, len(y), rng), K, rng)[0]
        except ValueError:
            continue
        null.append(float(s)); ex += s >= s_obs
    p = (1 + ex) / (1 + B_FULL) if len(null) == B_FULL else None      # None: stopped early, p > 0.05
    return key, dict(stat=float(s_obs), n_null=len(null), exceed=int(ex), p=p, fp=bool(p is not None and p <= 0.05),
                     null=null, **C._checklist12(f1, K))


def main():
    sys.argv = [sys.argv[0], "--r9"] + sys.argv[1:]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    which = sys.argv[3] if len(sys.argv) > 3 else "fp"
    orig = json.load(open(SRC))
    recs = make_pairs()
    # check: regenerated originals give the stored statistic for one record
    res = json.load(open(OUT)) if OUT.exists() else {}
    keys = [k for k, v in orig.items() if isinstance(v, dict) and v.get("K") == 3 and "p_r8" in v]
    assert all(recs[int(k)]["coupled"] == 0 for k in keys)
    sel = [k for k in keys if which == "all" or orig[k]["p_r8"] <= 0.05]
    todo = [(k, recs[int(k)]["y_rep"], 3, 40_000 + int(k)) for k in sel if k not in res]
    print(len(sel), "records,", len(todo), "to do", flush=True)
    with Pool(W) as p:
        for k, r in p.imap_unordered(job, todo):
            res[k] = dict(p_orig=orig[k]["p_r8"], **r)
            json.dump(res, open(OUT, "w"))
            print(k, "orig", orig[k]["p_r8"], "rep stat", round(r["stat"], 1), "n_null", r["n_null"], "p", r["p"], flush=True)
    fp1 = [k for k in keys if orig[k]["p_r8"] <= 0.05]
    both = [k for k in fp1 if res.get(k, {}).get("fp")]
    summ = dict(n_records=len(keys), fp_first=len(fp1), fp_both=len(both), done=sum(k in res for k in fp1),
                p_rep_given_fp=len(both) / max(1, sum(k in res for k in fp1)))
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
