"""R25: power outside the CNN's training distribution.  The R9a step detector was trained on records from
the same generator as the R10 test records (sim2_torch.make_batch).  Here K = 2 matched pairs (coupled +
independent I-projection partner, same noise) are drawn with features the training generator never makes:
  lorentz   background = 30 weak telegraphs with log-uniform rates (non-Gaussian, Lorentzian 1/f) instead
            of spectrally shaped Gaussian 1/f
  heavy     Student-t (3 dof) white noise + Gaussian 1/f
  slow      long dwell (tau 200-500 samples, training 8-200), T = 16384
  skewed    one trap occupancy 0.05-0.12 (training 0.15-0.85)
Methods at a 5 % threshold set on each condition's independent partners: R9a (CNN-HMM, 3 starts),
R8 (HMM + global decomposition search, no CNN), oracle (coupling statistic on the true path).
    python experiments/r25_ood.py [n_pairs=30]
"""
from __future__ import annotations
import json, sys, os, math
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/r23/r25_ood.json"
CONDS = ("lorentz", "heavy", "slow", "skewed")
TDOF = 3
NET = None


def make_pair(cond, i):
    import torch
    from rtnsim import sim2_torch as S
    rng = np.random.default_rng(25000 + 1000 * CONDS.index(cond) + i)
    T = 16384 if cond == "slow" else 4096
    tau = np.exp(rng.uniform(np.log(200), np.log(500), 2)) if cond == "slow" else np.exp(rng.uniform(np.log(8), np.log(200), 2))
    p = rng.uniform(0.15, 0.85, 2)
    if cond == "skewed":
        p[1] = rng.uniform(0.05, 0.12)
    c, e = 2 / tau * p, 2 / tau * (1 - p)
    Jv = rng.uniform(0.7, 3.0) * rng.choice([-1, 1])
    phi = rng.uniform(0.2, 0.8)
    tt = lambda x: torch.tensor(np.asarray(x, np.float32))[None]
    J = torch.zeros(1, 2, 2); J[0, 0, 1] = J[0, 1, 0] = Jv
    ph = torch.tensor([phi], dtype=torch.float32)
    c0, e0 = S.i_projection(tt(c), tt(e), J, ph)
    Q1 = S.generator(tt(c), tt(e), J, ph); Q0 = S.generator(c0, e0, torch.zeros_like(J), ph)
    g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
    x = S.sample_chain(torch.cat([Q1, Q0]), T, 1.0, g).numpy()
    bits = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    st = [bits[x[0]], bits[x[1]]]
    a = np.array([1.0, rng.uniform(0.2, 0.9)])
    sig = a[1] * rng.uniform(0.15, 0.5)
    if cond == "heavy":
        w = rng.standard_t(TDOF, T) / math.sqrt(TDOF / (TDOF - 2))
    else:
        w = rng.normal(0, 1, T)
    noise = sig * w
    if cond == "lorentz":
        bg = np.zeros(T)
        for _ in range(30):
            r = 10 ** rng.uniform(-4, -0.5); s_ = int(rng.random() < 0.5); u = rng.random(T)
            tr = np.empty(T)
            for t in range(T):
                if u[t] < r:
                    s_ ^= 1
                tr[t] = s_
            bg += tr * rng.uniform(0.3, 1.0)
        bg = (bg - bg.mean()) / bg.std()
        noise = noise + 0.6 * sig * bg
    else:
        W = np.fft.rfft(rng.normal(0, 1, T)); f = np.fft.rfftfreq(T)
        fl = np.fft.irfft(W * np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1 / T)), 0), n=T); fl /= fl.std()
        noise = noise + 0.4 * sig * fl
    drift = 0.3 * np.cumsum(rng.normal(0, 1, T)) / np.sqrt(T) * (rng.random() < 0.5)
    ys = [s_ @ a + noise + drift for s_ in st]
    return ys, st


def job(a):
    global NET
    cond, i = a
    from rtnsim.r9 import load_stepnet, lrt_cnn3_ms
    from rtnsim.cfhmm import lrt_ms
    from rtnsim.robustk import coupling_stat
    if NET is None:
        NET = load_stepnet(str(ROOT / "results/r9a_gpu/stepnet.pt"))
    ys, st = make_pair(cond, i)
    out = []
    for lab, y, s in zip((1, 0), ys, st):
        rng = np.random.default_rng(i)
        r = dict(coupled=lab, oracle=float(coupling_stat(s)[0]), trans=[int(np.abs(np.diff(s[:, k])).sum()) for k in range(2)])
        try:
            r["r9a"] = float(lrt_cnn3_ms(y, 2, NET, ar1=True, rng=rng, n_start=3)[0])
        except Exception as ex:
            r["r9a_err"] = repr(ex)[:120]
        try:
            r["r8"] = float(lrt_ms(y, 2, np.random.default_rng(i), ar1=True)[0])
        except Exception as ex:
            r["r8_err"] = repr(ex)[:120]
        out.append(r)
    return cond, i, out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    res = json.load(open(OUT)) if OUT.exists() else {}
    res = res.get("res", res)
    tasks = [(c, i) for c in CONDS for i in range(n) if str(i) not in res.get(c, {})]
    with Pool(int(os.environ.get("RTN_WORKERS", "2"))) as p:
        for c, i, r in p.imap_unordered(job, tasks):
            res.setdefault(c, {})[str(i)] = r
            json.dump(dict(res=res), open(OUT, "w"), indent=1)
    summ = {}
    for c in CONDS:
        rows = [x for v in res.get(c, {}).values() for x in v]
        d = {}
        for m in ("r9a", "r8", "oracle"):
            ok = [x for x in rows if m in x]
            neg = [x[m] for x in ok if x["coupled"] == 0]; pos = [x[m] for x in ok if x["coupled"] == 1]
            if len(neg) < 5:
                continue
            thr = np.quantile(neg, 0.95)
            d[m] = dict(power=float(np.mean(np.array(pos) > thr)), auc=auc(pos, neg), fpr_chi2=float(np.mean(np.array(neg) > 3.84)) if m != "oracle" else None)
        summ[c] = d
    json.dump(dict(summary=summ, res=res), open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
