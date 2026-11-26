"""Paper figures 1, 2, 4, 5 (vector PDF + PNG) from the stored result files.
    python experiments/figs_paper.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as Beta

ROOT = Path(__file__).resolve().parents[1]
R8, NI, OUT = ROOT / "results/r8", ROOT / "results/nichol", ROOT / "figures/out"
OUT.mkdir(parents=True, exist_ok=True)

# reference categorical palette (fixed order), text in neutral ink
C = dict(ours="#2a78d6", zh="#6250d6", pku="#eb6834", alb="#1baf7a", son="#eda100", chi2="#e87ba4", upper="#52514e")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300})
W1, W2 = 3.5, 7.16      # IEEE column / page width (in)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def wilson(k, n):
    lo = Beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = Beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def fig1():
    w = json.load(open(R8 / "r10_baselines_white.json"))["summary"]
    f = json.load(open(R8 / "r10_baselines_full.json"))["summary"]
    rn = json.load(open(R8 / "r12_realneg.json"))["summary"]
    _c = {k: v for k, v in json.load(open(R8 / "r8_calib_exp_r9.json")).items() if k != "summary"}
    cal = {f"K={K}": {"p_r8": float(np.mean([r["p_r8"] <= 0.05 for r in _c.values() if r["K"] == K]))} for K in (2, 3)}
    zw = json.load(open(R8 / "r10b_zhang_white.json"))["summary"]
    zf = json.load(open(R8 / "r10b_zhang_full.json"))["summary"]
    Z = [zw["K=2"], zw["K=3"], zf["K=2"], zf["K=3"]]
    AW = json.load(open(R8 / "r19_albrecht_white.json"))["summary"]; AF = json.load(open(R8 / "r19_albrecht_full.json"))["summary"]
    SW = json.load(open(R8 / "r19c_son_white.json"))["summary"]; SF = json.load(open(R8 / "r19c_son_full.json"))["summary"]
    AB = json.load(open(R8 / "r19b_albrecht_realneg.json"))["summary"]["alb_full"]
    SB = json.load(open(R8 / "r19c_son_realneg.json"))["summary"]["son_faithful_fpr"]
    conds = ["White\n2 traps", "White\n3 traps", "1/f + drift\n2 traps", "1/f + drift\n3 traps", "Real-noise\ncontrol"]
    S = [w["K=2"], w["K=3"], f["K=2"], f["K=3"]]
    rows = [("Zhang'17 missing state", "zh", [z["zhang_bic"]["fpr"] for z in Z] + [np.nan]),
            ("Guo'18-style Δτ/τ, |d| > 0.5", "pku", [s["pku_0p5"]["fpr"] for s in S] + [rn["pku_0p5"]]),
            ("Albrecht'26 (full procedure)", "alb", [AW["K=2"]["alb_full"]["fpr"], AW["K=3"]["alb_full"]["fpr"], AF["K=2"]["alb_full"]["fpr"], AF["K=3"]["alb_full"]["fpr"], AB]),
            ("Son'17 (trap2 | trap1, one test)", "son", [SW["K=2"]["fpr"], SW["K=3"]["fpr"], SF["K=2"]["fpr"], SF["K=3"]["fpr"], SB]),
            ("Ours, asymptotic χ²", "chi2", [s["ours_chi2"]["fpr"] for s in S] + [rn["ours_chi2"]]),
            ("Ours, pipeline-calibrated", "ours", [np.nan, np.nan, cal["K=2"]["p_r8"], cal["K=3"]["p_r8"], rn["ours_calibrated"]])]
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    x = np.arange(len(conds)); bw = 0.13
    for i, (lab, key, v) in enumerate(rows):
        xs = x + (i - 2.5) * bw
        v = np.array(v, float)
        ax.bar(xs, np.nan_to_num(v), bw - 0.02, color=C[key], label=lab, edgecolor="white", linewidth=0.6)
        for xi, vi in zip(xs, v):
            if np.isnan(vi):
                ax.text(xi, 0.075, "n/a", ha="center", va="bottom", fontsize=5.5, color=INK2, rotation=90)
            elif key == "ours":
                ax.text(xi, max(vi, 0.05) + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(4.42, 0.03, "5%", fontsize=6, color=INK, ha="left", va="top"); ax.set_xlim(-0.55, 4.6)
    ax.set_xticks(x, conds); ax.set_ylim(0, 1.05); ax.set_ylabel("False-positive rate\n(independent traps)")
    grid_y(ax)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6, handlelength=1.0, columnspacing=0.9)
    save(fig, "fig1_fpr")


def fig2():
    d = json.load(open(R8 / "r10_length_full_v2.json"))
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.7), sharey=True)
    for ax, K in zip(axs, (2, 3)):
        Ts = (4096, 16384, 65536)
        sw = [d[f"T={T},K={K}"]["median_switches"] for T in Ts]
        for key, lab in (("pku", "Guo'18-style Δτ/τ"), ("alb", "Albrecht'26 (full procedure)"), ("son", "Son'17 (one test)")):
            jk = {"alb": "albrecht_f", "son": "son_f"}.get(key, key)
            ax.plot(sw, [d[f"T={T},K={K}"][jk]["fpr"] for T in Ts], color=C[key], lw=2, marker="o", ms=4,
                    mec="white", mew=0.8, label=lab)
        ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xscale("log"); ax.set_xticks(sw, [f"{s:.0f}" for s in sw]); ax.minorticks_off()
        ax.set_title(f"{K} traps", fontsize=7.5, color=INK); ax.set_xlabel("Switches per trap (median)")
        ax.set_ylim(0, 1.0); grid_y(ax)
    axs[0].set_ylabel("False-positive rate")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.2))
    save(fig, "fig2_length")


def fig4():
    conds = ["White\n2 traps", "White\n3 traps", "1/f + drift\n2 traps", "1/f + drift\n3 traps"]
    vals = [("HMM + global decomposition search (R8)", "pku", [0.86, 0.80, 0.59, 0.36]),
            ("CNN–HMM, this work (R9a)", "ours", [0.87, 0.84, 0.81, 0.68]),
            ("Upper bound (true state path)", "upper", [0.88, 0.83, 0.88, 0.83])]
    cal = json.load(open(R8 / "r8_calib_exp_r9.json"))
    rows = {k: v for k, v in cal.items() if k != "summary"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 1.9), gridspec_kw=dict(width_ratios=[3, 1.1], wspace=0.45))
    x = np.arange(4); bw = 0.26
    for i, (lab, key, v) in enumerate(vals):
        xs = x + (i - 1) * bw
        a.bar(xs, v, bw - 0.03, color=C[key], label=lab, edgecolor="white", linewidth=0.6,
              alpha=0.55 if key == "upper" else 1)
        if key == "ours":
            for xi, vi in zip(xs, v):
                a.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    a.set_xticks(x, conds); a.set_ylim(0, 1.0); a.set_ylabel("Power at 5% false-positive rate"); grid_y(a)
    a.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=6.5, handlelength=1.2)
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontsize=7.5, color=INK)
    for j, K in enumerate((2, 3)):
        rr = [r for r in rows.values() if r["K"] == K]
        k = sum(r["p_r8"] <= 0.05 for r in rr); n = len(rr)
        lo, hi = wilson(k, n)
        b.errorbar(j, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=C["ours"], ms=5, mec="white", mew=0.8,
                   capsize=3, lw=1.2)
        b.text(j + 0.12, k / n, f"{k}/{n}", va="center", fontsize=6.5, color=INK)
    b.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.set_xticks([0, 1], ["2 traps", "3 traps"]); b.set_xlim(-0.5, 1.7); b.set_ylim(0, 0.25)
    b.set_ylabel("Calibrated FPR, 1/f (95% CI)"); grid_y(b)
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontsize=7.5, color=INK)
    save(fig, "fig4_power")


def fig5():
    pw = json.load(open(NI / "Gate_P_v8_K2_power.json"))["summary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw=dict(wspace=0.3))
    Js = sorted(float(k) for k in pw["power"])
    a.plot(Js, [pw["power"][str(j) if str(j) in pw["power"] else f"{j}"] for j in Js], color=C["ours"], lw=2,
           marker="o", ms=4, mec="white", mew=0.8)
    J0, sd = pw["J_obs"][0], 0.025
    a.axvspan(J0 - 2 * sd, J0 + 2 * sd, color=C["ours"], alpha=0.12, lw=0)
    a.text(J0 + 2 * sd + 0.01, 0.15, f"observed J = {J0:.3f}\n(band: ±2 s.d.)", fontsize=6.5, color=INK)
    a.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    a.set_xlabel("Injected coupling J (kT)"); a.set_ylabel("Power (calibrated 5%)"); a.set_ylim(0, 1.05)
    a.set_title("(a) Gate_P 0.567 V: no evidence, |J| ≥ 0.1 kT excluded", loc="left", fontsize=7, color=INK)
    grid_y(a)
    obs = {6: [69.3, 105.7, 191.0], 8: [168.2, 32.4, 65.8]}
    for j, (vi, V) in enumerate(((6, "0.151 V"), (8, "0.152 V"))):
        nc = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullcons.json"))["null"].values()))
        no = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["null"].values()))
        rng = np.random.default_rng(vi)
        for off, arr, lab, col in ((-0.18, no, "Null, oracle start", INK2), (0.06, nc, "Null, same analysis as data", C["pku"])):
            b.scatter(j + off + rng.uniform(-0.05, 0.05, arr.size), np.maximum(arr, 0.3), s=9, color=col,
                      edgecolor="white", linewidth=0.4, label=lab if j == 0 else None, zorder=2)
        b.scatter([j + 0.3] * 3, obs[vi], marker="D", s=22, color=C["ours"], edgecolor="white", linewidth=0.6,
                  label="Observed, 3 repetition blocks" if j == 0 else None, zorder=3)
    b.set_yscale("log"); b.set_ylim(0.2, 2e4); b.set_xticks([0, 1], ["0.151 V\ncandidate", "0.152 V\nnot passed"])
    b.set_xlim(-0.5, 1.6); b.set_ylabel("LRT, coupling given hidden state")
    b.set_title("(b) LH_cond_Sgate: calibration must rerun the real analysis", loc="left", fontsize=7, color=INK)
    b.yaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(loc="upper left", fontsize=6, handletextpad=0.2, borderaxespad=0.1, ncol=2)
    save(fig, "fig5_nichol")


if __name__ == "__main__":
    fig1(); fig2(); fig4(); fig5()
    print(sorted(p.name for p in OUT.iterdir()))
