"""Figures and markdown report for R1 (reads results/r1/r1.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1E2A44", "#5A6478", "#E3E6EA"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def path_tmin(r, J):
    """Noise-free reference T_min, estimated through the same HMM pipeline (SNR 1000, dt 0.005)."""
    b = next(x for x in r["part_c"] if x["J"] == J and x["snr"] >= 1000)
    return b["tmin_hmm"]


def fig_information(r):
    a = r["part_a"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True)
    ax = axes[0]
    for k, phi in enumerate((0.0, 0.5, 1.0)):
        rows = [x for x in a["J_phi"] if x["phi"] == phi]
        J = [x["J"] for x in rows]
        ax.loglog(J, [x["D"] for x in rows], "o-", color=C[k], ms=5, label=f"φ = {phi}: exact")
        ax.loglog(J, [x["D_small"] for x in rows], "--", color=C[k], lw=1.2, label=f"φ = {phi}: J² closed form")
    ax.set_xlabel("interaction J (kT)")
    ax.set_ylabel("D*  (per τ, symmetric traps, p = ½, k = 2)")
    ax.set_title("Coupling information grows as J²", loc="left", color=INK, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    occ = a["occupancy"]
    ax.plot([x["p1"] for x in occ], [x["D"] for x in occ], "o-", color=C[0], ms=5, label="exact")
    ax.plot([x["p1"] for x in occ], [x["D_small"] for x in occ], "--", color=C[0], lw=1.2, label="closed form")
    ax.set_xlabel("occupancy p₁ of trap 1 (switching rate fixed)")
    ax.set_ylabel("D*  (J = 1 kT)")
    ax.set_title("Most information at half occupancy", loc="left", color=INK, fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    sp = a["speed"]
    ax.loglog([x["k2_over_k1"] for x in sp], [x["D"] for x in sp], "o-", color=C[0], ms=5, label="exact")
    ax.loglog([x["k2_over_k1"] for x in sp], [x["D_small"] for x in sp], "--", color=C[0], lw=1.2,
              label="closed form")
    ax.set_xlabel("speed ratio k₂ / k₁")
    ax.set_ylabel("D*  (J = 1 kT, per τ₁)")
    ax.set_title("The faster trap reports the coupling", loc="left", color=INK, fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(RES / "fig1_information.png")
    plt.close(fig)


def fig_path(r):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    for k, b in enumerate(r["part_b"]):
        T = np.array(b["T"]) / b["stein"]
        axes[0].semilogy(T, np.maximum(b["fpr"], 3e-4), "o-", color=C[k], ms=4, label=f"J = {b['J']} kT")
        axes[1].plot(T, np.array(b["llr1_mean_over_T"]) / b["D"], "o-", color=C[k], ms=4, label=f"J = {b['J']} kT")
    axes[0].axhline(r["alpha"], color=MUTED, lw=1)
    axes[0].axvline(1.0, color=MUTED, lw=1, ls=":")
    axes[0].set_xlabel("observation time / Stein time  ln(1/α)/D*")
    axes[0].set_ylabel("false-positive rate at 90% sensitivity")
    axes[0].set_title("Noise-free path oracle", loc="left", color=INK, fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color=MUTED, lw=1)
    axes[1].set_xlabel("observation time / Stein time")
    axes[1].set_ylabel("E[LLR] / (D* T)")
    axes[1].set_ylim(0.6, 1.2)
    axes[1].set_title("Evidence accrues at the rate D*", loc="left", color=INK, fontsize=10)
    axes[1].legend(fontsize=8)
    fig.savefig(RES / "fig2_path_oracle.png")
    plt.close(fig)


def fig_efficiency(r):
    rows = [x for x in r["part_c"] if x["amp2"] == 0.6 and x["snr"] < 1000]
    Js = sorted({x["J"] for x in rows})
    snrs = sorted({x["snr"] for x in rows})
    dts = sorted({x["dt"] for x in rows})
    fig, axes = plt.subplots(1, len(Js) + 1, figsize=(4.0 * (len(Js) + 1), 3.8), constrained_layout=True)
    for ax, J in zip(axes, Js):
        tp = path_tmin(r, J)
        M = np.zeros((len(snrs), len(dts)))
        for x in rows:
            if x["J"] == J:
                M[snrs.index(x["snr"]), dts.index(x["dt"])] = tp / x["tmin_hmm"] if np.isfinite(x["tmin_hmm"]) else 0
        im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(snrs)):
            for j in range(len(dts)):
                ax.text(j, i, f"{M[i, j]:.2f}" if M[i, j] > 0 else "> 6×", ha="center", va="center",
                        color="#F7F6F2" if M[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(len(dts)), [f"{d:g}" for d in dts])
        ax.set_yticks(range(len(snrs)), [f"{s:g}" for s in snrs])
        ax.set_xlabel("sampling interval Δt / τ_c")
        ax.set_ylabel("SNR = step height / σ")
        ax.set_title(f"Oracle efficiency, J = {J} kT", loc="left", color=INK, fontsize=10)
        ax.grid(False)
    ax = axes[-1]
    sl = [x for x in r["part_c"] if x["J"] == 1.0 and x["dt"] == 0.1 and x["snr"] in (1.0, 2.0, 4.0)]
    tp = path_tmin(r, 1.0)
    for k, snr in enumerate((1.0, 2.0, 4.0)):
        pts = sorted([x for x in sl if x["snr"] == snr], key=lambda x: x["amp2"])
        uniq = {}
        for x in pts:
            uniq[x["amp2"]] = x
        a2 = sorted(uniq)
        ax.plot(a2, [tp / uniq[v]["tmin_hmm"] if np.isfinite(uniq[v]["tmin_hmm"]) else 0 for v in a2],
                "o-", color=C[k], ms=5, label=f"SNR = {snr:g}")
    ax.set_xlabel("step-height ratio A₂ / A₁")
    ax.set_ylabel("efficiency  T_ref / T_HMM")
    ax.set_ylim(0, 1.05)
    ax.set_title("The smaller step sets the cost", loc="left", color=INK, fontsize=10)
    ax.legend(fontsize=8)
    fig.colorbar(im, ax=list(axes[:-1]), shrink=0.8, label="T_ref / T_HMM  (“> 6×”: not reached within 6× the Stein time)")
    fig.savefig(RES / "fig3_noise_sampling.png")
    plt.close(fig)


if __name__ == "__main__":
    r = json.load(open(RES / "r1.json"))
    fig_information(r)
    fig_path(r)
    fig_efficiency(r)
    print("figures written")
