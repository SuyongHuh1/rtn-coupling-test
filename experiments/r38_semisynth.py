"""R38: semi-synthetic positive control on real noise.

Two single-trap Si/SiGe records (Gate_P, bias index 9: slow 0.45 mV trap A; bias index 10: fast 0.35 mV
trap B, both with real 60 Hz boxcar integration) are idealised. Every dwell segment of B (raw samples,
real noise, real transition samples at its start) goes into a pool per state. A new B record is then
assembled segment by segment along the time axis of an A record: a segment for state s is drawn from the
pool of state s with weight exp(-(k' - k) L), L its length, where k is B's fitted rate out of s and k'
the target rate given A's state at the segment start. Exponential tilting of an exponential dwell
distribution with rate k by exp(-(k'-k) L) gives rate k', so B's rates depend on A's occupancy as in
eq. (1) with a known J (phi = 0.5):  c_B' = c_B exp(-J/2), e_B' = e_B exp(J/2) while A is occupied.
J = 0 uses the same procedure (independent control). The record is A + g (B' - median), g = 1.

    python experiments/r38_semisynth.py <n_pairs> <n_sim> [workers]   -> results/r8/r38_semisynth.json
Each pair i gives one coupled record (|J| from {1, 1.5, 2, 3}, random sign) and one J = 0 record that
share A and the segment pool.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
DATA = Path(os.environ.get("RTN_NICHOL", "data/nichol"))
OUT = ROOT / ("results/r8/r38_semisynth" + os.environ.get("RTN_TAG", "") + ".json")
J_SET = (1.0, 1.5, 2.0, 3.0)
CT = os.environ.get("RTN_CT", "1") == "1"         # continuous-time construction (default); 0 = segment-start


def _ideal(y, rng):
    from rtnsim.robustk import idealise_k
    f = idealise_k(y, 1, rng=rng)
    return f["S"][:, 0].astype(int), f


def pools():
    d = sio.loadmat(DATA / "Gate_P.mat")["savedat"][0, 0]["data"]
    rng = np.random.default_rng(380)
    seg = {0: [], 1: []}
    n_sw, t_occ = 0, 0
    for r in range(d.shape[1]):
        y = d[10, r].astype(float)
        s, _ = _ideal(y, rng)
        base = np.median(y[s == 0])
        cut = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
        for a, b in zip(cut[:-1], cut[1:]):
            if a == 0 or b == len(s):             # truncated first/last dwell: not a complete dwell
                continue
            seg[int(s[a])].append(y[a:b] - base)
        n_sw += len(cut) - 3
        t_occ += s.sum()
    L = {k: np.array([len(x) for x in v], float) for k, v in seg.items()}
    rate = {k: 1.0 / L[k].mean() for k in (0, 1)}        # c_B = rate[0], e_B = rate[1]
    return seg, L, rate


def a_records():
    d = sio.loadmat(DATA / "Gate_P.mat")["savedat"][0, 0]["data"]
    rng = np.random.default_rng(381)
    out = []
    for r in range(d.shape[1]):
        y = d[9, r].astype(float)
        s, _ = _ideal(y, rng)
        out.append((y, s))
    return out


def assemble(sA, seg, L, rate, J, rng):
    T = len(sA)
    out = np.empty(0)
    s = int(rng.random() < rate[0] / (rate[0] + rate[1]))
    while len(out) < T:
        k = rate[s]
        if sA[len(out)] == 1:
            k2 = k * (np.exp(-J / 2) if s == 0 else np.exp(J / 2))
        else:
            k2 = k
        w = np.exp(-(k2 - k) * (L[s] - L[s].min()))
        j = rng.choice(len(w), p=w / w.sum())
        out = np.r_[out, seg[s][j]]
        s = 1 - s
    return out[:T]


def assemble_ct(sA, seg, L, rate, J, rng):
    """Continuous-time version: B's rate follows A's *current* state. When A switches inside a B dwell,
    the dwell is cut there and continued with a fresh residual (exponential dwells are memoryless) drawn
    from the same state's pool under the new rate, without that segment's first (transition) sample."""
    T = len(sA)
    out = []
    t = 0
    s = int(rng.random() < rate[0] / (rate[0] + rate[1]))
    first = True
    while t < T:
        k = rate[s]
        k2 = k * (np.exp(-J / 2) if s == 0 else np.exp(J / 2)) if sA[t] == 1 else k
        w = np.exp(-(k2 - k) * (L[s] - L[s].min()))
        x = seg[s][rng.choice(len(w), p=w / w.sum())]
        if not first:
            x = x[1:] if len(x) > 1 else x
        sw = np.flatnonzero(sA[t:t + len(x)] != sA[t])
        if sw.size and J != 0:                         # A switches inside this dwell: cut, keep state
            x = x[:sw[0]]
            out.append(x); t += len(x); first = False
            continue
        out.append(x); t += len(x); s = 1 - s; first = True
    return np.concatenate(out)[:T]


def build(n_pairs):
    seg, L, rate = pools()
    A = a_records()
    rng = np.random.default_rng(382)
    recs = []
    for i in range(n_pairs):
        y, sA = A[i % len(A)]
        J = float(rng.choice(J_SET) * rng.choice([-1, 1]))
        r_seed = int(rng.integers(1 << 31))
        for lab, JJ in (("coup", J), ("indep", 0.0)):
            b = (assemble_ct if CT else assemble)(sA, seg, L, rate, JJ, np.random.default_rng(r_seed))
            recs.append((f"{lab}{i}", y + b, JJ))
    return recs, rate


def job(a):
    tag, y, J, n_sim, seed = a
    import r8_calib as C
    _, cal = C.job((seed, y, 2, n_sim))
    return tag, dict(J_true=J, p_cal=cal.get("p_r8"), stat=cal.get("stat_r8"), J=cal.get("J"),
                     dwell_min=cal.get("dwell_min"), trans_min=cal.get("trans_min"), n_null=cal.get("n_null"))


def main():
    sys.argv += ["--r9"]
    n, n_sim = int(sys.argv[1]), int(sys.argv[2])
    W = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2
    recs, rate = build(n)
    res = json.load(open(OUT)) if OUT.exists() else {}
    todo = [(t, y, J, n_sim, 3800 + i) for i, (t, y, J) in enumerate(recs) if t not in res]
    with Pool(W) as p:
        for t, r in p.imap_unordered(job, todo):
            res[t] = r
            json.dump(res, open(OUT, "w"), indent=1)
            print(t, round(r["J_true"], 2), r["p_cal"], round(r["stat"] or -1, 1), flush=True)
    rr = {k: v for k, v in res.items() if k != "summary"}
    pos = [v for k, v in rr.items() if k.startswith("coup")]
    neg = [v for k, v in rr.items() if k.startswith("indep")]
    res["summary"] = dict(rate_B=rate, n_pos=len(pos), n_neg=len(neg),
                          power=float(np.mean([v["p_cal"] <= 0.05 for v in pos])) if pos else None,
                          fpr=float(np.mean([v["p_cal"] <= 0.05 for v in neg])) if neg else None)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
