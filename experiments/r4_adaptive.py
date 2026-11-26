"""R4a: information-optimal (adaptive) gate-bias choice for coupling identification.

Physics.  Gate bias V shifts each trap's energy linearly, eps_i(V) = eps_i0 - beta_i V
(kT), with capture/emission c_i = k_i exp(-eps_i/2), e_i = k_i exp(eps_i/2) (uncoupled
occupancy p_i = 1/(1+exp(eps_i))).  The coupling J and the splitting phi do not depend on V.

Because the expected log-likelihood ratio of a record made of segments at biases V_s is
sum_s D*(V_s) T_s (segments are independent given the bias), the record length needed
for a fixed (alpha, power) scales as 1 / <D*>_time.  Hence no bias schedule beats the
best fixed bias V*, and the question is how close a design that does NOT know the trap
parameters gets to it.

Designs (record budget measured in units of the reference trap time):
  nominal   one fixed bias V = 0 (what an experiment set up for something else records)
  sweep     equal time on a 9-point grid over [-Vmax, Vmax] (parameter-agnostic practice)
  oracle    the best fixed bias, true parameters known (upper bound for any bias design)
  adaptive  a short pilot sweep (state statistics only), fit ln c_i(V), ln e_i(V) per trap,
            pick V^ = argmax of the small-coupling closed form (its argmax does not depend
            on the unknown J), then record at V^.  The pilot data count toward the test.

Output: results/r4/r4_adaptive.json, fig1_adaptive.png
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim import RTNConfig  # noqa: E402
from rtnsim import theory as th  # noqa: E402

OUT = ROOT / "results" / "r4"
OUT.mkdir(parents=True, exist_ok=True)
VMAX, NGRID = 4.0, 9
GRID = np.linspace(-VMAX, VMAX, NGRID)
VFINE = np.linspace(-VMAX, VMAX, 161)
ALPHA = 0.01


def make_cfg(eps, k, J, phi=0.5, dt=0.05):
    c = RTNConfig()
    cap = k * np.exp(-eps / 2)
    emi = k * np.exp(eps / 2)
    c.traps.tau_c = (1.0 / cap).tolist()
    c.traps.tau_e = (1.0 / emi).tolist()
    c.traps.amp = [1.0, 0.6]
    c.coupling.J, c.coupling.phi = [[0, J], [J, 0]], phi
    c.readout.dt = dt
    return c


def D_at(pair, V, J):
    eps = pair["eps0"] - pair["beta"] * V
    D, _ = th.coupling_information(make_cfg(eps, pair["k"], J))
    return D


def small_D_shape(eps, k):
    """V-dependence of the small-coupling closed form (J^2, phi factors dropped)."""
    p = 1.0 / (1.0 + np.exp(eps))
    kk = 2 * k * np.cosh(eps / 2)
    return np.prod(p * (1 - p), axis=-1) * kk.sum(axis=-1)


def pilot_estimate(pair, J, t_per_point, rng, dt=0.05):
    """Noise-free point-sampled pilot at each grid bias -> per-trap (V, ln c, ln e) estimates."""
    n = int(round(t_per_point / dt))
    obs = [[], []]
    for V in GRID:
        cfg = make_cfg(pair["eps0"] - pair["beta"] * V, pair["k"], J, dt=dt)
        x = th.sample_states(cfg, 1, n, rng)[0]
        for i in range(2):
            s = (x >> i) & 1
            up = np.sum((s[:-1] == 0) & (s[1:] == 1))
            dn = np.sum((s[:-1] == 1) & (s[1:] == 0))
            t0, t1 = np.sum(s[:-1] == 0) * dt, np.sum(s[:-1] == 1) * dt
            obs[i].append((V, up, dn, t0, t1))
    return obs


def fit_trap(rows):
    """Weighted LS of ln c and ln e vs V on points with transitions; slopes constrained by the
    model (d ln c/dV = +beta/2, d ln e/dV = -beta/2) -> fit eps0, beta, k jointly."""
    V = np.array([r[0] for r in rows]); up = np.array([r[1] for r in rows], float)
    dn = np.array([r[2] for r in rows], float); t0 = np.array([r[3] for r in rows]); t1 = np.array([r[4] for r in rows])
    good = (up >= 3) & (dn >= 3)
    if good.sum() >= 2:
        lc, le = np.log(up[good] / t0[good]), np.log(dn[good] / t1[good])
        w = np.sqrt(np.minimum(up[good], dn[good]))
        # ln c = ln k - eps0/2 + beta V/2 ; ln e = ln k + eps0/2 - beta V/2
        A = np.vstack([np.r_[np.ones(good.sum()), np.ones(good.sum())],
                       np.r_[-0.5 * np.ones(good.sum()), 0.5 * np.ones(good.sum())],
                       np.r_[0.5 * V[good], -0.5 * V[good]]]).T
        b = np.r_[lc, le]
        W = np.r_[w, w]
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        return dict(k=np.exp(sol[0]), eps0=sol[1], beta=sol[2], ok=True)
    # trap seen switching at <= 1 grid point: centre the guess there (or at the edge it leans to)
    occ = np.array([r[4] / max(r[3] + r[4], 1e-9) for r in rows])
    j = int(np.argmin(np.abs(occ - 0.5)))
    return dict(k=1.0, eps0=float(V[j]), beta=1.0, ok=False)


def main(n_pairs=300, J=1.0, seed=0, pilot_frac_list=(2.0, 5.0, 10.0)):
    rng = np.random.default_rng(seed)
    L = np.log(1 / ALPHA)                       # Stein-type information target (same for all designs)
    res = dict(J=J, n_pairs=n_pairs, vmax=VMAX, grid=GRID.tolist(), designs={}, pairs=[])
    rows = []
    for n in range(n_pairs):
        pair = dict(eps0=rng.uniform(-4, 4, 2), beta=rng.uniform(0.5, 2.0, 2) * rng.choice([1, 1], 2),
                    k=np.exp(rng.uniform(np.log(0.3), np.log(3.0), 2)))
        Dfine = np.array([D_at(pair, V, J) for V in VFINE])
        Dgrid = np.array([D_at(pair, V, J) for V in GRID])
        D_nom, D_sweep, D_orc = D_at(pair, 0.0, J), Dgrid.mean(), Dfine.max()
        row = dict(D_nominal=D_nom, D_sweep=D_sweep, D_oracle=D_orc, V_oracle=float(VFINE[Dfine.argmax()]))
        for tp in pilot_frac_list:              # pilot time per grid point (trap time units)
            obs = pilot_estimate(pair, J, tp, rng)
            est = [fit_trap(obs[i]) for i in range(2)]
            eps_hat = np.stack([e["eps0"] - e["beta"] * VFINE for e in est], -1)
            k_hat = np.array([e["k"] for e in est])
            Vhat = float(VFINE[np.argmax(small_D_shape(eps_hat, k_hat))])
            D_hat = D_at(pair, Vhat, J)
            T_pilot = tp * NGRID
            info_pilot = D_sweep * T_pilot
            T_need = L / D_sweep if info_pilot >= L else T_pilot + (L - info_pilot) / D_hat
            row[f"adaptive_{tp:g}"] = dict(V=Vhat, D=D_hat, T=T_need, fit_ok=[e["ok"] for e in est])
        for name, D in (("nominal", D_nom), ("sweep", D_sweep), ("oracle", D_orc)):
            row[f"T_{name}"] = L / D if D > 0 else np.inf
        rows.append(row)
        if n % 50 == 0:
            print(n, flush=True)
    def q(a):
        a = np.asarray(a, float)
        return dict(median=float(np.median(a)), p25=float(np.quantile(a, .25)), p75=float(np.quantile(a, .75)),
                    p90=float(np.quantile(a, .9)))
    T_orc = np.array([r["T_oracle"] for r in rows])
    summ = {}
    for name in ("nominal", "sweep"):
        T = np.array([r[f"T_{name}"] for r in rows])
        summ[name] = dict(T_over_oracle=q(T / T_orc), frac_over_10x=float(np.mean(T / T_orc > 10)))
    for tp in pilot_frac_list:
        T = np.array([r[f"adaptive_{tp:g}"]["T"] for r in rows])
        summ[f"adaptive_pilot{tp:g}"] = dict(T_over_oracle=q(T / T_orc), frac_over_10x=float(np.mean(T / T_orc > 10)),
                                            gain_vs_sweep=q(np.array([r["T_sweep"] for r in rows]) / T),
                                            gain_vs_nominal=q(np.array([r["T_nominal"] for r in rows]) / T))
    res["summary"] = summ
    res["pairs"] = rows
    json.dump(res, open(OUT / f"r4_adaptive_J{J:g}.json", "w"), default=float, indent=1)
    print(json.dumps(summ, indent=1))
    figure(rows, pilot_frac_list)


def figure(rows, pilot_frac_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T_orc = np.array([r["T_oracle"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    series = [("fixed nominal bias (V = 0)", np.array([r["T_nominal"] for r in rows]) / T_orc, "#9aa3b2"),
              ("uniform 9-point sweep", np.array([r["T_sweep"] for r in rows]) / T_orc, "#5b6b84")]
    tp = max(pilot_frac_list)
    series.append((f"adaptive (pilot {tp:g} τ × 9 points)", np.array([r[f"adaptive_{tp:g}"]["T"] for r in rows]) / T_orc, "#2b6cd6"))
    for name, v, col in series:
        v = np.sort(np.clip(v, 1, 1e4))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=name, color=col, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("record length needed / best fixed bias (1 = optimal)")
    ax.set_ylabel("fraction of trap pairs (cumulative)")
    ax.legend(frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_adaptive.png", dpi=160)  # J=1 run overwrites last


if __name__ == "__main__":
    J = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(n_pairs=400, J=J)
