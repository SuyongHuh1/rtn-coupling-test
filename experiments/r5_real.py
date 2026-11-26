"""R5: the R1-R3 pipeline on public measured RTN (University of Southampton datasets).

Data (B1500A exports):
  SSDM  Ibukuro et al., SSDM 2018, DOI 10.5258/SOTON/D0507 (CC BY): silicon nanowire, drain
        current vs time at 11 front-gate biases, dt = 0.232 s, 1160 s per bias.
  EDTM  Li, Saito et al., 2017, DOI 10.5258/SOTON/399158: low-temperature silicon device,
        drain current, dt = 1 s, up to 25 000 s per record (14 long records).

Per record:
 1. preprocessing: trim the start-up transient (SSDM: first 150 s; EDTM: first 10 s),
    replace isolated one-sample spikes by the 3-point median, robust standardisation;
 2. number of current levels by BIC over Gaussian HMMs with 1-4 states (tied variance);
 3. records with >= 3 levels: two-trap product HMM vs kinetically coupled HMM (R2 GLRT),
    null calibrated by parametric bootstrap from the fitted independent model; drive-aware
    GLRT (R3, slow drive capped at 1/100 samples) with a chi2(4) reference;
 4. R1 detectability: the smallest |J| identifiable from this record length, from the
    fitted independent rates (noise-free path information, T_min = 1.4 x Stein time).

Output: results/r5/r5_real.json, fig1_records.png
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtnsim.realdata import load_b1500  # noqa: E402
from rtnsim import detectors as det  # noqa: E402
from rtnsim import drive_hmm as DH  # noqa: E402
from rtnsim import theory as th  # noqa: E402
from rtnsim import RTNConfig  # noqa: E402

OUT = ROOT / "results" / "r5"
OUT.mkdir(parents=True, exist_ok=True)
import os
EXT = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).exists() else Path(os.environ.get("RTN_EXT") or "data")
N_BOOT = 200


def records():
    out = []
    for f in sorted(glob.glob(str(EXT / "ssdm/DOI_document_SSDM/Raw_data/*timesampling*.csv"))):
        out.append(dict(set="SSDM", file=Path(f).name, path=f, trim_s=150.0))
    for f in sorted(glob.glob(str(EXT / "edtm/EDTM data/RTS_*25000*.csv"))):
        out.append(dict(set="EDTM", file=Path(f).name, path=f, trim_s=10.0))
    return out


def preprocess(t, I, trim_s):
    keep = t - t[0] >= trim_s
    t, y = t[keep], I[keep].astype(float)
    med = np.r_[y[0], np.median(np.stack([y[:-2], y[1:-1], y[2:]]), axis=0), y[-1]]
    d = np.abs(y - med)
    mad = np.median(np.abs(np.diff(y))) / 0.6745 + 1e-30
    spikes = (d > 6 * mad) & (np.abs(np.r_[0, np.diff(med)]) < 2 * mad)
    y = np.where(spikes, med, y)
    scale = np.median(np.abs(np.diff(y))) / 0.6745 / np.sqrt(2) + 1e-30
    return t, (y - np.median(y)) / scale, int(spikes.sum()), float(np.mean(np.diff(t)))


def n_levels(y):
    from hmmlearn.hmm import GaussianHMM
    best = None
    res = {}
    for k in range(1, 5):
        lls = []
        for seed in range(3):
            m = GaussianHMM(n_components=k, covariance_type="tied", n_iter=200, random_state=seed, tol=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(y[:, None])
            lls.append(m.score(y[:, None]))
        ll = max(lls)
        n_par = k * (k - 1) + k + 1 + (k - 1)
        bic = -2 * ll + n_par * np.log(len(y))
        res[k] = dict(ll=ll, bic=bic)
        if best is None or bic < res[best]["bic"]:
            best = k
    return best, res


def sample_product(fit, B, T, rng):
    P, mu, sig, pi0 = fit["P"][0], fit["mu"][0], float(fit["sig"][0]), fit["pi0"][0]
    C = np.cumsum(P, 1)
    C[:, -1] = 1
    w, v = np.linalg.eig(P.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    if not np.all(np.isfinite(pi)) or pi.sum() <= 0:
        pi = np.full(4, 0.25)
    x = rng.choice(4, B, p=pi / pi.sum())
    out = np.empty((B, T))
    u = rng.random((T, B))
    for k in range(T):
        x = (u[k][:, None] > C[x]).sum(1)
        out[:, k] = mu[x]
    return out + sig * rng.standard_normal((B, T))


def glrt(y):
    prod = det.fit_product(y, n_iter=60)
    cpl = det.fit_coupled(y, prod, n_iter=40)
    return prod, cpl["ll"] - prod["ll"]


def jmin_detectable(prod, dt, T_rec):
    """Smallest |J| with 1.4 x Stein time <= record length (noise-free path information)."""
    P1, P2 = prod["P1"][0], prod["P2"][0]
    rates = []
    for P in (P1, P2):
        c = -np.log(max(P[0, 0], 1e-12)) / dt
        e = -np.log(max(P[1, 1], 1e-12)) / dt
        rates.append((c, e))
    for J in np.r_[np.linspace(0.1, 3, 30), np.linspace(3.5, 8, 10)]:
        cfg = RTNConfig()
        cfg.traps.tau_c = [1 / rates[0][0], 1 / rates[1][0]]
        cfg.traps.tau_e = [1 / rates[0][1], 1 / rates[1][1]]
        cfg.coupling.J, cfg.coupling.phi = [[0, J], [J, 0]], 0.5
        D, _ = th.coupling_information(cfg)
        if 1.4 * np.log(100) / D <= T_rec:
            return float(J), rates
    return float("inf"), rates


def transitions(prod):
    """Expected number of switches of each trap from the fitted transition matrix."""
    return None


def main():
    rng = np.random.default_rng(0)
    res = []
    traces = {}
    for r in records():
        t, I, _ = load_b1500(r["path"])
        t, y, n_spk, dt = preprocess(t, I, r["trim_s"])
        k, bic = n_levels(y)
        row = dict(set=r["set"], file=r["file"], T=len(y), dt=dt, duration_s=float(t[-1] - t[0]),
                   spikes_replaced=n_spk, n_levels=k, bic={str(a): b for a, b in bic.items()})
        traces[r["file"]] = y
        if k >= 3:
            Y = y[None]
            prod, g = glrt(Y)
            if not (np.all(np.isfinite(prod["P"])) and np.all(np.isfinite(prod["mu"]))):
                row["fit_failed"] = True
                res.append(row)
                print(row["file"], "fit failed", flush=True)
                continue
            boot = sample_product(prod, N_BOOT, len(y), rng)
            _, gb = glrt(boot)
            p_boot = float((1 + np.sum(gb >= g[0])) / (N_BOOT + 1))
            da = DH.drive_aware_scores(Y, p_max=0.01)
            jmin, rates = jmin_detectable(prod, dt, row["duration_s"])
            sw = [float(prod["P1"][0][0, 1] * len(y) * 0.5 + prod["P1"][0][1, 0] * len(y) * 0.5),
                  float(prod["P2"][0][0, 1] * len(y) * 0.5 + prod["P2"][0][1, 0] * len(y) * 0.5)]
            row.update(glrt=float(g[0]), p_boot=p_boot, glrt_boot_q99=float(np.quantile(gb, 0.99)),
                       drive_aware=float(da[0]), p_drive_chi2=float(stats.chi2.sf(da[0], 4)),
                       jmin_kT=jmin, rates_per_s=rates, approx_switches=sw,
                       levels=prod["mu"][0].tolist(), sigma=float(prod["sig"][0]))
        res.append(row)
        print({k_: v for k_, v in row.items() if k_ not in ("bic",)}, flush=True)
    json.dump(res, open(OUT / "r5_real.json", "w"), indent=1, default=float)
    figure(res, traces)


def figure(res, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [r for r in res if r["n_levels"] >= 3 and not r.get("fit_failed")]
    fig, axs = plt.subplots(len(sel), 1, figsize=(10, 1.6 * max(len(sel), 1)), squeeze=False)
    for ax, r in zip(axs[:, 0], sel):
        y = traces[r["file"]]
        ax.plot(np.arange(len(y)) * r["dt"], y, lw=0.3, color="#33415c")
        ax.set_title(f"{r['set']} {r['file'][:40]}  levels={r['n_levels']}  GLRT p={r['p_boot']:.3f}  "
                     f"drive-aware p={r.get('p_drive_chi2', float('nan')):.3f}  J_min={r['jmin_kT']:.1f} kT", fontsize=8)
        ax.set_ylabel("z")
    axs[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_records.png", dpi=120)


if __name__ == "__main__":
    main()
