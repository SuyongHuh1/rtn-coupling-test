"""Figures of the I2MTC 2027 submission, regenerated from the result files.

Print-ready: every figure is drawn at the width it is printed (text width 7.14 in, column width 3.49 in) and
saved without bbox cropping, so the font sizes below are the printed sizes (all text 8 pt, IEEE minimum);
fonts are embedded as TrueType (Type 42), not Type 3, for IEEE PDF eXpress.
    python figures/make_figures.py [power thr null fpr len fig5]   -> figures/out/*.pdf  (RTN_FIG_OUT overrides)
"""
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import figs_paper as F
import matplotlib.pyplot as plt

FS = 8
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS, "axes.labelsize": FS,
                     "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                     "figure.constrained_layout.use": True})
TW, CW = 516 / 72.27, 252 / 72.27                     # IEEEtran conference \textwidth, \columnwidth (in)
F.OUT = Path(os.environ.get("RTN_FIG_OUT", Path(__file__).resolve().parent / "out"))
F.OUT.mkdir(parents=True, exist_ok=True)
C, INK, INK2 = F.C, F.INK, F.INK2
R8, NI = F.R8, F.NI
KT = r"$k_\mathrm{B}T$"

# calibrated counts, J only + pooled restarts: 1/f B = 99 (two independently drawn sets), white B = 30
CAL = {2: (11, 200), 3: (23, 200)}
CALW = {2: (0, 50), 3: (2, 50)}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(F.OUT / f"{name}.{ext}", dpi=300)          # no bbox cropping: printed size = figsize
    plt.close(fig)


def fig_fpr():
    w = json.load(open(R8 / "r10_baselines_white.json"))["summary"]
    f = json.load(open(R8 / "r10_baselines_full.json"))["summary"]
    rn = json.load(open(R8 / "r12_realneg.json"))["summary"]
    zw = json.load(open(R8 / "r10b_zhang_white.json"))["summary"]
    zf = json.load(open(R8 / "r10b_zhang_full.json"))["summary"]
    Z = [zw["K=2"], zw["K=3"], zf["K=2"], zf["K=3"]]
    AW = json.load(open(R8 / "r19_albrecht_white.json"))["summary"]; AF = json.load(open(R8 / "r19_albrecht_full.json"))["summary"]
    SW = json.load(open(R8 / "r19c_son_white.json"))["summary"]; SF = json.load(open(R8 / "r19c_son_full.json"))["summary"]
    AB = json.load(open(R8 / "r19b_albrecht_realneg.json"))["summary"]["alb_full"]
    SB = json.load(open(R8 / "r19c_son_realneg.json"))["summary"]["son_faithful_fpr"]
    S = [w["K=2"], w["K=3"], f["K=2"], f["K=3"]]
    conds = ["White\n2 traps", "White\n3 traps", "1/f + drift\n2 traps", "1/f + drift\n3 traps", "Real-noise\ncontrol"]
    rows = [("Missing states", "zh", [z["zhang_bic"]["fpr"] for z in Z] + [np.nan]),
            ("Δτ/τ, |d| > 0.5", "pku", [s["pku_0p5"]["fpr"] for s in S] + [rn["pku_0p5"]]),
            ("Evidence ratio", "alb", [AW["K=2"]["alb_full"]["fpr"], AW["K=3"]["alb_full"]["fpr"],
                                            AF["K=2"]["alb_full"]["fpr"], AF["K=3"]["alb_full"]["fpr"], AB]),
            ("Cond. dwell", "son", [SW["K=2"]["fpr"], SW["K=3"]["fpr"], SF["K=2"]["fpr"], SF["K=3"]["fpr"], SB]),
            ("Ours, χ²", "chi2", [s["ours_chi2"]["fpr"] for s in S] + [rn["ours_chi2"]]),
            ("Ours, calibrated", "ours", [CALW[2][0] / CALW[2][1], CALW[3][0] / CALW[3][1], CAL[2][0] / CAL[2][1], CAL[3][0] / CAL[3][1], 0.0])]
    fig, ax = plt.subplots(figsize=(TW, 1.9))
    x = np.arange(len(conds)); bw = 0.13
    for i, (lab, key, v) in enumerate(rows):
        xs = x + (i - 2.5) * bw
        v = np.array(v, float)
        ax.bar(xs, np.nan_to_num(v), bw - 0.02, color=C[key], label=lab, edgecolor="white", linewidth=0.6)
        for xi, vi in zip(xs, v):
            if np.isnan(vi):
                ax.text(xi, 0.075, "n/a", ha="center", va="bottom", fontsize=FS, color=INK2, rotation=90)
            elif key == "ours":
                ax.text(xi, max(vi, 0.05) + 0.02, f"{vi:.2f}", ha="center", va="bottom", fontsize=FS, color=INK, rotation=90)
    ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(-0.5, 0.065, "5%", fontsize=FS, color=INK, ha="left", va="bottom"); ax.set_xlim(-0.55, 4.62)
    ax.set_xticks(x, conds); ax.set_ylim(0, 1.05); ax.set_ylabel("False-positive rate\n(independent traps)")
    F.grid_y(ax)
    fig.legend(*ax.get_legend_handles_labels(), ncol=6, loc="outside upper center", fontsize=FS, handlelength=1.0,
               columnspacing=0.8)
    save(fig, "fig_fpr")


def fig_power():
    conds = ["White\n2 traps", "White\n3 traps", "1/f + drift\n2 traps", "1/f + drift\n3 traps"]
    vals = [("HMM, global decomposition search", "pku", [0.86, 0.80, 0.59, 0.36]),
            ("CNN–HMM (this work)", "ours", [0.87, 0.85, 0.81, 0.69]),
            ("Oracle (true state path)", "upper", [0.88, 0.83, 0.88, 0.83])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.05), gridspec_kw=dict(width_ratios=[3, 1.1]))
    x = np.arange(4); bw = 0.26
    for i, (lab, key, v) in enumerate(vals):
        xs = x + (i - 1) * bw
        a.bar(xs, v, bw - 0.03, color=C[key], label=lab, edgecolor="white", linewidth=0.6,
              alpha=0.55 if key == "upper" else 1)
        if key == "ours":
            for xi, vi in zip(xs, v):
                a.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=FS, color=INK)
    a.set_xticks(x, conds); a.set_ylim(0, 1.0); a.set_ylabel("Power at 5% FPR"); F.grid_y(a)
    fig.legend(*a.get_legend_handles_labels(), ncol=3, loc="outside upper center", fontsize=FS, handlelength=1.2)
    a.set_title("(a)", loc="left", fontsize=FS, color=INK)
    for j, K in enumerate((2, 3)):
        k, n = CAL[K]
        lo, hi = wilson(k, n)
        b.errorbar(j, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=C["ours"], ms=5, mec="white", mew=0.8,
                   capsize=3, lw=1.2)
        b.text(j, hi + 0.008, f"{k}/{n}", va="bottom", ha="center", fontsize=FS, color=INK)
    b.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    b.set_xticks([0, 1], ["2 traps", "3 traps"]); b.set_xlim(-0.5, 1.7); b.set_ylim(0, 0.25)
    b.set_ylabel("Calibrated FPR, 1/f\n(95% Wilson CI)"); F.grid_y(b)
    b.set_title("(b)", loc="left", fontsize=FS, color=INK)
    save(fig, "fig_power")


def fig_len():
    d = json.load(open(R8 / "r10_length_full_v2.json"))
    fig, axs = plt.subplots(1, 2, figsize=(CW, 1.9), sharey=True)
    for ax, K in zip(axs, (2, 3)):
        Ts = (4096, 16384, 65536)
        sw = [d[f"T={T},K={K}"]["median_switches"] for T in Ts]
        for key, jk, lab in (("pku", "pku", "Δτ/τ rule"), ("alb", "albrecht_f", "Evidence ratio"), ("son", "son_f", "Conditional dwell")):
            ax.plot(sw, [d[f"T={T},K={K}"][jk]["fpr"] for T in Ts], color=C[key], lw=1.6, marker="o", ms=3.5,
                    mec="white", mew=0.7, label=lab)
        ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xscale("log"); ax.set_xticks(sw, [f"{s:.0f}" for s in sw]); ax.minorticks_off()
        ax.set_title(f"{K} traps, 1/f", fontsize=FS, color=INK); ax.set_xlabel("Switches per trap")
        ax.set_ylim(0, 1.0); F.grid_y(ax)
    axs[0].set_ylabel("False-positive rate")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=FS,
               handlelength=1.2, columnspacing=0.8)
    save(fig, "fig_len")


def fig_thr():
    """FPR on the independent records versus the decision threshold of the published statistics."""
    import r8_oracle as O
    curves = {}
    for cond in ("white", "full"):
        _, _, rows = O.make(100, white=(cond == "white"))
        A = json.load(open(R8 / f"r19_albrecht_{cond}.json"))["res"]
        S = json.load(open(R8 / f"r19c_son_{cond}.json"))["p"]
        G = json.load(open(R8 / f"r10_baselines_{cond}.json"))["res"]
        for K in (2, 3):
            ind = [i for i, r in enumerate(rows) if r["K"] == K and r["coupled"] == 0]
            def fpr(get, ths, larger=True):
                v = np.array([get(i) for i in ind if get(i) is not None], float)
                return [float(np.mean(v > t) if larger else np.mean(v < t)) for t in ths]
            curves[(cond, K)] = dict(
                d=fpr(lambda i: G.get(str(i), {}).get("pku_max"), TH_D),
                e=fpr(lambda i: A.get(str(i), {}).get("e"), TH_E),
                p=fpr(lambda i: S.get(str(i)), TH_P, larger=False))
    fig, axs = plt.subplots(1, 3, figsize=(CW, 2.2), sharey=True)
    spec = [("d", TH_D, 0.5, "|d| (Δτ/τ)", [0.25, 1, 4], [".25", "1", "4"]),
            ("e", TH_E, 2.0, "e (evid. ratio)", [1, 5, 20], ["1", "5", "20"]),
            ("p", TH_P, 0.05, "p (cond. rate)", [0.05, 1e-3, 1e-6], [".05", "1e-3", "1e-6"])]
    sty = {("white", 2): ("-", "o"), ("white", 3): ("-", "s"), ("full", 2): ("--", "o"), ("full", 3): ("--", "s")}
    lab = {("white", 2): "white, 2", ("white", 3): "white, 3", ("full", 2): "1/f, 2", ("full", 3): "1/f, 3"}
    for ax, (key, ths, ours, xl, tk, tl) in zip(axs, spec):
        for ck, cv in curves.items():
            ls, mk = sty[ck]
            ax.plot(ths, cv[key], ls=ls, marker=mk, ms=2.5, lw=1.0, color=C["pku"] if ck[1] == 2 else C["alb"], label=lab[ck])
            j = int(np.argmin(np.abs(np.log(np.array(ths)) - np.log(ours))))
            ax.plot(ths[j], cv[key][j], "o", mfc="white", mec=INK, ms=4, mew=0.8, zorder=5)
        ax.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(tk, tl, fontsize=FS)
        ax.set_xlabel(xl + "\nstricter \u2192", fontsize=FS); ax.set_ylim(0, 1.02); F.grid_y(ax)
        if key == "p":
            ax.invert_xaxis()
    axs[0].set_ylabel("FPR")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=FS, handlelength=1.6,
               columnspacing=0.7, handletextpad=0.4)
    save(fig, "fig_thr")


def fig5():
    """Si/SiGe data: (a) power at the fitted parameters of the 0.567 V record, (b) LH_cond_Sgate nulls."""
    pw = json.load(open(NI / "Gate_P_v8_K2_power.json"))["summary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 1.75))
    Js = sorted(float(k) for k in pw["power"])
    a.plot(Js, [pw["power"][str(j) if str(j) in pw["power"] else f"{j}"] for j in Js], color=C["ours"], lw=2,
           marker="o", ms=4, mec="white", mew=0.8)
    J0, sd = pw["J_obs"][0], 0.025
    a.axvspan(J0 - 2 * sd, J0 + 2 * sd, color=C["ours"], alpha=0.12, lw=0)
    a.text(J0 + 2 * sd + 0.01, 0.12, f"observed $J$ = {J0:.3f}\n(band: ±2 s.e.)", fontsize=FS, color=INK)
    a.axhline(0.05, color=INK, lw=0.7, ls=(0, (3, 2)))
    a.set_xlabel(f"Injected coupling $J$ ({KT})"); a.set_ylabel("Power (calibrated 5%)"); a.set_ylim(0, 1.05)
    a.set_title(f"(a) Gate_P 0.567 V: $|J|\\geq0.1$ {KT} excluded (TOST)", loc="left", fontsize=FS, color=INK)
    F.grid_y(a)
    obs = {6: [69.3, 105.7, 191.0], 8: [168.2, 32.4, 65.8]}
    for j, (vi, V) in enumerate(((6, "0.151 V"), (8, "0.152 V"))):
        nc = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullcons.json"))["null"].values()))
        no = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["null"].values()))
        rng = np.random.default_rng(vi)
        for off, arr, lab, col in ((-0.18, no, "Null, oracle start", INK2), (0.06, nc, "Null, same analysis", C["pku"])):
            b.scatter(j + off + rng.uniform(-0.05, 0.05, arr.size), np.maximum(arr, 0.3), s=9, color=col,
                      edgecolor="white", linewidth=0.4, label=lab if j == 0 else None, zorder=2)
        b.scatter([j + 0.3] * 3, obs[vi], marker="D", s=22, color=C["ours"], edgecolor="white", linewidth=0.6,
                  label="Observed, 3 blocks" if j == 0 else None, zorder=3)
    b.set_yscale("log"); b.set_ylim(0.2, 2e5); b.set_xticks([0, 1], ["0.151 V\ncandidate", "0.152 V\nnot passed"])
    b.set_xlim(-0.5, 1.6); b.set_ylabel("LRT, coupling\ngiven hidden state")
    b.set_title("(b) LH: calibration must re-run the analysis", loc="left", fontsize=FS, color=INK)
    b.yaxis.grid(True, color=F.GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(loc="upper left", fontsize=FS, handletextpad=0.2, borderaxespad=0.2, ncol=2, columnspacing=0.8)
    save(fig, "fig5_nichol")


def fig_null():
    """LH_cond_Sgate: oracle-start versus same-analysis nulls (v14 Fig. 4, column width)."""
    fig, b = plt.subplots(figsize=(CW, 2.3))
    obs = {6: [69.3, 105.7, 191.0], 8: [168.2, 32.4, 65.8]}
    for j, (vi, V) in enumerate(((6, "0.151 V"), (8, "0.152 V"))):
        nc = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullcons.json"))["null"].values()))
        no = np.array(list(json.load(open(NI / f"LH_cond_Sgate_v{vi}_K3_nullqn.json"))["null"].values()))
        rng = np.random.default_rng(vi)
        for off, arr, lab, col in ((-0.2, no, "Oracle-start null", INK2), (0.05, nc, "Same-analysis null", C["pku"])):
            b.scatter(j + off + rng.uniform(-0.06, 0.06, arr.size), np.maximum(arr, 0.3), s=7, color=col,
                      edgecolor="white", linewidth=0.3, label=lab if j == 0 else None, zorder=2)
        b.scatter([j + 0.3] * 3, obs[vi], marker="D", s=20, color=C["ours"], edgecolor="white", linewidth=0.6,
                  label="Observed (3 blocks)" if j == 0 else None, zorder=3)
    b.set_yscale("log"); b.set_ylim(0.2, 5e3); b.set_xticks([0, 1], ["LH 0.151 V", "LH 0.152 V"])
    b.set_xlim(-0.5, 1.5); b.set_ylabel("Coupling LRT inside\nthe drive model")
    b.yaxis.grid(True, color=F.GRID, lw=0.5); b.set_axisbelow(True)
    b.legend(loc="upper left", fontsize=FS, handletextpad=0.2, borderaxespad=0.3, ncol=1, frameon=False)
    save(fig, "fig_null")


TH_D = [0.25, 0.5, 1.0, 2.0, 4.0]
TH_E = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
TH_P = [0.05, 0.01, 1e-3, 1e-4, 1e-6]


if __name__ == "__main__":
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "experiments"))
    which = _s.argv[1:] or ["power", "thr", "null"]
    for w in which:
        dict(fpr=fig_fpr, power=fig_power, len=fig_len, fig5=fig5, thr=fig_thr, null=fig_null)[w]()
    print(sorted(p.name for p in F.OUT.iterdir()))
