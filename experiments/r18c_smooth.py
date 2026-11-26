"""R18c: R18/R18b with continuous tails. Instantaneous dVth = sum a_k s_k + n, n ~ N(0, SIG^2), SIG = 0.1 eta
(measurement/other-variability noise), so tail probabilities and quantiles are well defined even for equal
amplitudes and deep tails.  Tail ratio = P_coupled(X > x*) / P_indep(X > x*), x* = independent-model quantile.
First-order := tail ratio outside 0.8-1.25 or quantile shift >= 5 %.
    python experiments/r18c_smooth.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import r18_sensitivity as R
SIG = 0.1
TAILS = (1e-4, 1e-6)

def tailfun(v, w):
    o = np.argsort(v); v, w = v[o], w[o]
    def P(x):
        lo = np.searchsorted(v, x - 8 * SIG)
        return float(w[lo:] @ (0.5 * erfc((x - v[lo:]) / (SIG * np.sqrt(2)))))
    return P, v[-1]

def evaluate(devs):
    vs, ps, qs = [], [], []
    for a, eps, J in devs:
        v, p, q, m = R.device_dists(a, eps, J); vs.append(v); ps.append(p); qs.append(q)
    v = np.concatenate(vs); p = np.concatenate(ps) / len(devs); q = np.concatenate(qs) / len(devs)
    Pc, vmax = tailfun(v, p); Pi, _ = tailfun(v, q)
    out = {}
    for t in TAILS:
        try:
            xi = brentq(lambda x: Pi(x) - t, 0, vmax + 10 * SIG)
            xc = brentq(lambda x: Pc(x) - t, 0, vmax + 10 * SIG)
            out[f"tail_ratio_{t:g}"] = Pc(xi) / t
            out[f"q_shift_{t:g}"] = xc / xi - 1
        except ValueError:
            out[f"tail_ratio_{t:g}"] = None; out[f"q_shift_{t:g}"] = None
    return out

def first(r, t):
    tr, qs = r[f"tail_ratio_{t:g}"], r[f"q_shift_{t:g}"]
    return None if tr is None else bool(tr < 0.8 or tr > 1.25 or abs(qs) >= 0.05)

rows = []
rng = np.random.default_rng(1818)
grid = [(amp, Wv, sign, f, Jm) for amp in ("exp", "equal") for Wv in (1.0, 4.0) for sign in (1, -1)
        for f in (0.1, 0.3, 1.0) for Jm in (0.25, 0.5, 1.0, 2.0)]
for amp, Wv, sign, f, Jm in grid:
    R.W, R.NDEV = Wv, 100000
    devs = R.population(4, f, Jm, sign, rng)
    if amp == "equal":
        devs = [(np.ones_like(a), e, J) for a, e, J in devs]
    r = evaluate(devs)
    rows.append(dict(amp=amp, W=Wv, sign="repulsive" if sign > 0 else "cooperative", f=f, Jm=Jm,
                     **{k: (None if v is None else round(v, 3)) for k, v in r.items()},
                     first_1e4=first(r, 1e-4), first_1e6=first(r, 1e-6)))
    print(rows[-1], flush=True)
json.dump(rows, open(ROOT / "results/r8/r18c_smooth.json", "w"), indent=1)
