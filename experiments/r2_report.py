"""Figures and report for R2 (reads results/r2/r2_A/B/C/D.json and r2_cnn.json if present)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r2"
C = {"oracle": "#1E2A44", "glrt": "#2a78d6", "glrt_general": "#6250d6", "corr": "#eb6834",
     "ideal_lrt": "#1baf7a", "cnn": "#eda100"}
NAME = {"oracle": "oracle (known models)", "glrt": "GLRT, coupled-kinetics HMM", "glrt_general": "GLRT, unconstrained HMM",
        "corr": "|corr| of idealised states", "ideal_lrt": "idealise → chain LRT", "cnn": "CNN (matched-pair trained)"}
INK, MUTED, GRID = "#1E2A44", "#5A6478", "#E3E6EA"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def f0(x):
    return "> 8×" if x is None or not np.isfinite(x) else f"{x:.0f}"


def main():
    A, B, Cc, D = load("r2_A.json"), load("r2_B.json"), load("r2_C.json"), load("r2_D.json")
    N = load("r2_cnn_long.json") or load("r2_cnn.json")
    if N:
        NAME["cnn"] = f"CNN ({N.get('epochs', 6)} epochs, matched pairs)"
    cnn_eff = {(e["J"], e["snr"]): e for e in N["efficiency"]} if N else {}
    dets = ["oracle", "glrt", "glrt_general", "corr", "ideal_lrt"]
    L = ["# R2 report — practical detectors vs the oracle, and confounds\n",
         "Two traps, τ_c = τ_e = 1, A₁ = 1, A₂ = 0.6, Δt = 0.1 τ_c. FPR ≤ 1% at 90% sensitivity. "
         "Detectors do not know any parameter; the emission levels are fitted once on the pooled record "
         "(device property), kinetics per trace.\n"]

    if A and Cc and D and N:
        effg = [c["efficiency"]["glrt"] for c in A]
        effc = [c["efficiency"]["corr"] for c in A]
        effn = [e["efficiency"] for e in N["efficiency"]]
        rc, rr = Cc["raw"]["fpr"], N["confounds"]["raw"]["fpr"]
        L.append("## Findings\n")
        L.append(f"1. **Not knowing the parameters costs about 2× record length, and it is predictable.** The "
                 f"coupled-kinetics GLRT reaches {min(effg):.2f}–{max(effg):.2f} of oracle efficiency in every "
                 "condition; a Wilks (non-central χ²₄) prediction from D* and the R1 noise efficiency gives its "
                 "T_min within 2–16% (slightly optimistic).")
        L.append("2. **Physics structure helps a little.** The GLRT whose alternative only lets each trap’s rates "
                 "depend on the other trap (4 extra numbers) needs 3–13% shorter records than an unconstrained 4×4 "
                 "transition matrix (12 extra numbers), and it is far more robust to 1/f noise (1.3% vs 9% FPR).")
        L.append(f"3. **A simple occupancy correlation is near-oracle when states are resolvable, and fails when "
                 f"they are not.** |corr| of the idealised trap states reaches {max(effc):.2f} at SNR 4 but drops to "
                 f"{min(effc):.2f} at SNR 2, where idealisation errors dominate; idealise-then-chain-LRT is the "
                 "weakest and fails at SNR 2.")
        slow = ("drive_slow_same", "drive_slow_opposite")
        lo = min(rc[n][d] for n in slow for d in ("glrt", "corr", "ideal_lrt"))
        hi = max(rc[n][d] for n in slow for d in ("glrt", "corr", "ideal_lrt"))
        L.append(f"4. **The learned detector is less efficient but more robust.** A 24k-parameter CNN trained only "
                 f"on matched pairs reaches {min(effn):.2f}–{max(effn):.2f} efficiency (power {N['confounds']['raw']['power']:.2f} "
                 f"vs GLRT {Cc['raw']['power']['glrt']:.2f} at 2× oracle length), but keeps FPR ≤ 1% under both slow "
                 f"shared drives, readout coupling and 1/f noise, where model-based tests reach {lo:.1%}–{hi:.1%} "
                 "under slow drive. A shorter-trained version (6 epochs) was even drift-immune (0.0%) but much weaker.")
        L.append("5. **A slow shared drive (gate or temperature fluctuation) is the dangerous physical confound.** It "
                 "creates genuinely correlated switching without interaction: FPR rises 2–5× at 2× oracle length and "
                 "up to ~10× at 4×, because longer records make the spurious correlation more significant. A fast "
                 "drive is harmless. Even the oracle that knows both clean models is fooled (38–64%) — the data come "
                 "from neither hypothesis.")
        L.append(f"6. **Baseline drift destroys every HMM-based test.** A random-walk drift reaching one step "
                 f"height over the record gives 95–99% FPR; tolerance is ≈ 0.1 step height (FPR ≤ "
                 f"{max(D['raw']['0.1']['fpr'].values()):.1%}), and 0.3 already gives 12–44%. The CNN degrades much "
                 f"less ({rr['drift']:.1%}).")
        L.append("7. **Two common fixes did not work.** Running-median detrending pushes shared-drive FPR to ~95% "
                 "(the median follows correlated occupancy changes) and costs power; a residual-based baseline "
                 "correction did not improve drift tolerance. Drift and drive need to be in the model, or in the "
                 "training distribution.\n")
        L.append("Next (R3): train the detector with drive and drift in the null class (confound-aware learning), "
                 "add a drive-aware HMM alternative as the model-based counterpart, then port the network to "
                 "FPGA/SNN with measured latency and energy.\n")

    # ---------------- efficiency
    if A:
        L.append("## Efficiency (T_min in τ_c; efficiency = oracle T_min / detector T_min, same data)\n")
        hdr = dets + (["cnn"] if N else [])
        L.append("| J (kT) | SNR | " + " | ".join(NAME[d] for d in hdr) + " |")
        L.append("|---" * (len(hdr) + 2) + "|")
        for c in A:
            cells = []
            for d in hdr:
                if d == "cnn":
                    e = cnn_eff.get((c["J"], c["snr"]))
                    cells.append(f"{f0(e['tmin'])} ({e['efficiency']:.2f})" if e else "—")
                else:
                    cells.append(f"{f0(c['tmin'][d])} ({c['efficiency'][d]:.2f})")
            L.append(f"| {c['J']} | {c['snr']} | " + " | ".join(cells) + " |")
        fig, axes = plt.subplots(1, len(A), figsize=(3.4 * len(A), 3.6), constrained_layout=True, sharey=True)
        for ax, c in zip(np.atleast_1d(axes), A):
            T = [r["T"] for r in c["rows"]]
            for d in dets:
                ax.semilogy(T, np.maximum([r[f"fpr_{d}"] for r in c["rows"]], 2e-4), "o-", ms=4, color=C[d],
                            label=NAME[d])
            e = cnn_eff.get((c["J"], c["snr"]))
            if e:
                ax.semilogy(e["T"], np.maximum(e["fpr"], 2e-4), "o--", ms=4, color=C["cnn"], label=NAME["cnn"])
            ax.axhline(0.01, color=MUTED, lw=1)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("record length (τ_c)")
            ax.set_title(f"J = {c['J']} kT, SNR = {c['snr']}", loc="left", color=INK, fontsize=10)
        np.atleast_1d(axes)[0].set_ylabel("FPR at 90% sensitivity")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.savefig(RES / "fig1_efficiency.png")
        plt.close(fig)
        # Wilks prediction for the coupled-kinetics GLRT (k = 4 extra transition numbers)
        from scipy.stats import chi2, ncx2
        from scipy.optimize import brentq
        r1 = json.load(open(ROOT / "results" / "r1" / "r1.json"))
        ref = {x["J"]: x["tmin_hmm"] for x in r1["part_c"] if x["snr"] >= 1000}
        L.append("\nWilks prediction for the coupled-kinetics GLRT: T such that the 10% quantile of a non-central "
                 "χ²₄ with non-centrality 2·D_eff·T reaches the null 99% point χ²₄,0.99, where D_eff = D* × (R1 "
                 "noisy-oracle efficiency at this SNR and Δt):\n")
        L.append("| J | SNR | oracle T_min | GLRT measured | GLRT predicted | error |")
        L.append("|---|---|---|---|---|---|")
        for c in A:
            x = next(z for z in r1["part_c"] if z["J"] == c["J"] and z["snr"] == c["snr"] and z["dt"] == 0.1
                     and z["amp2"] == 0.6)
            Deff = x["D"] * ref[c["J"]] / x["tmin_hmm"]
            thr = chi2.ppf(0.99, 4)
            Tp = brentq(lambda T: ncx2.ppf(0.1, 4, 2 * Deff * T) - thr, 1e-3, 1e6)
            Tm = c["tmin"]["glrt"]
            L.append(f"| {c['J']} | {c['snr']} | {c['tmin']['oracle']:.0f} | {Tm:.0f} | {Tp:.0f} | {(Tp - Tm) / Tm:+.0%} |")
        L.append("\n![efficiency](fig1_efficiency.png)\n")

    # ---------------- confounds
    for tag, data, label in (("B", B, "T = 4× oracle T_min"), ("C", Cc["raw"] if Cc else None, "T = 2× oracle T_min")):
        if not data:
            continue
        L.append(f"## False positives under confounds ({label}; thresholds set to 1% on clean independent data)\n")
        names = [n for n in data["fpr"]]
        hdr = dets + (["cnn"] if (N and tag == "C") else [])
        L.append("| confound | " + " | ".join(NAME[d] for d in hdr) + " |")
        L.append("|---" * (len(hdr) + 1) + "|")
        for n in names:
            cells = [f"{data['fpr'][n][d]:.3f}" for d in dets]
            if N and tag == "C":
                cells.append(f"{N['confounds']['raw']['fpr'][n]:.3f}")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        cells = [f"{data['power'][d]:.3f}" for d in dets]
        if N and tag == "C":
            cells.append(f"{N['confounds']['raw']['power']:.3f}")
        L.append("| **power on coupled data** | " + " | ".join(cells) + " |\n")
    if Cc:
        L.append("Running-median detrending before detection (a common preprocessing step):\n")
        L.append("| confound | " + " | ".join(NAME[d] for d in dets) + " |")
        L.append("|---" * (len(dets) + 1) + "|")
        for n, f in Cc["detrended"]["fpr"].items():
            L.append(f"| {n} | " + " | ".join(f"{f[d]:.3f}" for d in dets) + " |")
        L.append("| **power** | " + " | ".join(f"{Cc['detrended']['power'][d]:.3f}" for d in dets) + " |\n")
        # figure: confound FPR bars (raw, T = 2x)
        names = [n for n in Cc["raw"]["fpr"] if n != "clean"]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        show = ["glrt", "corr", "ideal_lrt"] + (["cnn"] if N else [])
        w = 0.8 / len(show)
        for k, d in enumerate(show):
            vals = [N["confounds"]["raw"]["fpr"][n] if d == "cnn" else Cc["raw"]["fpr"][n][d] for n in names]
            ax.bar(np.arange(len(names)) + (k - (len(show) - 1) / 2) * w, np.maximum(vals, 1e-3), w * 0.92,
                   color=C[d], label=NAME[d])
        ax.axhline(0.01, color=MUTED, lw=1)
        ax.text(-0.45, 0.0115, "nominal 1%", color=MUTED, fontsize=8, ha="left")
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylabel("actual FPR (threshold = 1% on clean null)")
        ax.set_title("Confounds that break independence tests (T = 2× oracle T_min)", loc="left", color=INK,
                     fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(RES / "fig2_confounds.png")
        plt.close(fig)
        L.append("![confounds](fig2_confounds.png)\n")

    # ---------------- drift tolerance
    if D:
        L.append("## Drift tolerance (random-walk baseline; level = end-of-record rms in step heights)\n")
        dd_ = ["glrt", "corr", "ideal_lrt"]
        L.append("| pipeline | drift | " + " | ".join(f"FPR {NAME[d]}" for d in dd_) + " | " +
                 " | ".join(f"power {NAME[d]}" for d in dd_) + " |")
        L.append("|---" * (2 + 2 * len(dd_)) + "|")
        for pipe in ("raw", "corrected"):
            for lvl in D["levels"]:
                r = D[pipe][str(lvl)]
                L.append(f"| {pipe} | {lvl} | " + " | ".join(f"{r['fpr'][d]:.3f}" for d in dd_) + " | " +
                         " | ".join(f"{r['power'][d]:.3f}" for d in dd_) + " |")
        L.append("\nPower on clean coupled data: raw " +
                 ", ".join(f"{d} {D['raw']['power_clean'][d]:.2f}" for d in dd_) + "; corrected " +
                 ", ".join(f"{d} {D['corrected']['power_clean'][d]:.2f}" for d in dd_) + ".\n")
    (RES / "R2_report.md").write_text("\n".join(L) + "\n")
    print((RES / "R2_report.md").read_text())


if __name__ == "__main__":
    main()
