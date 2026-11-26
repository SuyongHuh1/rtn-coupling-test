"""Figures and report for R3 (results/r3/*.json; R2 results for comparison)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r3"
R2D = ROOT / "results" / "r2"
INK, MUTED, GRID = "#1E2A44", "#5A6478", "#E3E6EA"
COL = {"glrt": "#2a78d6", "drive_aware": "#6250d6", "corr": "#eb6834", "cnn_r2": "#eda100", "ccnn": "#1baf7a",
       "ccnn_int8": "#008300"}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def main():
    drv, cc, hw = load(RES / "r3_drive.json"), load(RES / "r3_ccnn.json"), load(RES / "r3_hw_diff.json")
    ccd, hwl = load(RES / "r3_ccnn_diff.json"), load(RES / "r3_hw.json")
    r2c, r2cnn, r2a = load(R2D / "r2_C.json"), load(R2D / "r2_cnn_long.json"), load(R2D / "r2_A.json")
    tr = load(RES / "ccnn_train.json")
    trd = load(RES / "ccnn_diff_train.json")
    L = ["# R3 report — confound-robust detection and the hardware path\n",
         "Setting as in R2 (two traps, J = 1 kT, SNR 4, Δt = 0.1 τ_c, record = 2× oracle T_min for the confound "
         "table). Thresholds are set to 1% FPR on clean independent data.\n"]

    if drv and cc and ccd and hw:
        dtaus = [k for k in drv["fpr"] if k.startswith("drive_tau")]
        dkeys = dtaus + ["drive_slow_same", "drive_fast_same", "drive_slow_opposite"]
        effd = [e["efficiency"] for e in ccd["efficiency"]]
        cm, em = hw["cost"]["cnn"], hw["cost"]["em_glrt"]
        L.append("## Findings\n")
        L.append(f"1. **Modelling a slow drive removes its false positives, at a price in power.** The drive-aware "
                 f"GLRT keeps FPR ≤ {max(drv['fpr'][k]['drive_aware'] for k in dkeys):.1%} "
                 f"for every drive tested (dwell 0.5–50 trap times, same and opposite sign), where the R2 GLRT reaches "
                 f"{max(drv['fpr'][k]['glrt'] for k in dkeys):.1%} on the same data; power at 2× oracle length falls from "
                 f"{drv['power']['glrt']:.2f} to {drv['power']['drive_aware']:.2f}.")
        L.append("2. **Coupling is identifiable from a drive only with timescale separation.** If the hidden drive "
                 "may switch as fast as the traps, the drive-only model mimics coupling (the drive tracks one trap, "
                 "the other follows the drive) and the test has no power at all; capping the drive at ≥ 10 trap "
                 "times restores it. Physical gate and temperature fluctuations are slow, and fast drives do not bias "
                 "the plain test anyway (R2), so the cap costs nothing physically.")
        L.append(f"3. **Neither HMM test survives baseline drift** (drive-aware {drv['fpr']['drift']['drive_aware']:.0%}).")
        L.append(f"4. **Confound-aware training fixes the drives but not drift, unless the input is offset-free.** "
                 f"With the calibrated level as input the causal CNN still gives {cc['confounds']['fpr']['drift']:.1%} "
                 f"under drift ({cc['confounds']['fpr']['drift_2x']:.1%} at twice the drift). Feeding the first "
                 f"difference instead makes drift structurally invisible: {ccd['confounds']['fpr']['drift']:.1%} and "
                 f"{ccd['confounds']['fpr']['drift_2x']:.1%} (out of distribution), with every in-range confound ≤ 1% and "
                 f"the strongest out-of-range drive at {ccd['confounds']['fpr']['drive_very_slow_strong']:.1%}.")
        L.append(f"5. **Robustness costs efficiency.** The Δ-input CNN reaches {max(effd):.2f} of oracle efficiency at "
                 f"J = 2 kT and {min(e for e in effd if e > 0):.2f} at J = 1 kT, and does not reach 1% FPR within 4× "
                 f"the oracle length at J = 0.5 kT; power at 2× oracle length is {ccd['confounds']['power']:.2f} vs "
                 f"0.95 for the R2 GLRT on clean data.")
        L.append(f"6. **The streaming hardware path works.** int8 quantisation changes power by "
                 f"{hw['confounds_int8']['power'] - hw['confounds_float']['power']:+.3f} and no FPR by more than 0.002; the "
                 f"emitted one-sample-per-call C++ is bit-exact against the integer model; {cm['macs_per_sample']:,} "
                 f"MAC/sample → {cm['energy_int8_J'] * 1e9:.1f} nJ/sample (int8, Horowitz 45 nm, compute only), "
                 f"{cm['power_int8_W_at_fs'] * 1e3:.2f} mW at 100 kS/s with ≈ {cm['dsp_needed_at_200MHz']:.0f} MAC units at "
                 f"200 MHz, vs ~{em['energy_J'] * 1e9:.0f} nJ/sample for a 140-pass fp32 HMM-EM that also needs the "
                 "whole record in memory.\n")
        L.append("Picture for the paper: the GLRT is the most powerful clean-data test and its cost is predicted "
                 "by theory (R2); a slow shared drive and drift are what break it; a drive-aware model fixes the "
                 "drive only under timescale separation; an offset-free, confound-trained streaming network is "
                 "robust to all tested confounds and fits a sub-mW FPGA budget, but needs 3–8× longer records than the oracle (more at 0.5 kT). "
                 "Closing that gap (larger receptive field, likelihood-derived features, a CNN-screen + GLRT-confirm "
                 "hybrid) and synthesising on the real board are the remaining steps.\n")

    names = ["clean", "drive_slow_same", "drive_fast_same", "drive_slow_opposite", "readout_coupling", "flicker",
             "drift"]
    cols, heads = [], []
    if r2c:
        cols += [("R2 GLRT", lambda n: r2c["raw"]["fpr"][n]["glrt"], r2c["raw"]["power"]["glrt"]),
                 ("R2 |corr|", lambda n: r2c["raw"]["fpr"][n]["corr"], r2c["raw"]["power"]["corr"])]
    if drv:
        cols.append(("drive-aware GLRT", lambda n: drv["fpr"][n]["drive_aware"], drv["power"]["drive_aware"]))
    if r2cnn:
        cols.append(("R2 CNN (clean training)", lambda n: r2cnn["confounds"]["raw"]["fpr"][n],
                     r2cnn["confounds"]["raw"]["power"]))
    if cc:
        cols.append(("causal CNN, level input", lambda n: cc["confounds"]["fpr"][n], cc["confounds"]["power"]))
    if ccd:
        cols.append(("causal CNN, Δ input", lambda n: ccd["confounds"]["fpr"][n], ccd["confounds"]["power"]))
    if hw:
        cols.append(("Δ input, int8", lambda n: hw["confounds_int8"]["fpr"][n], hw["confounds_int8"]["power"]))
    if cols:
        L.append("## False positives under confounds\n")
        L.append("| confound | " + " | ".join(c[0] for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for n in names:
            L.append(f"| {n} | " + " | ".join(f"{c[1](n):.3f}" for c in cols) + " |")
        L.append("| **power on coupled data** | " + " | ".join(f"{c[2]:.3f}" for c in cols) + " |\n")
        if cc and ccd:
            L.append("Out-of-distribution confounds (stronger than anything in training):\n")
            L.append("| confound | level input | Δ input | Δ input, int8 |")
            L.append("|---|---|---|---|")
            for n in ("drive_very_slow_strong", "drift_2x", "flicker_strong"):
                i8 = f"{hw['confounds_int8']['fpr'][n]:.3f}" if hw else "—"
                L.append(f"| {n} | {cc['confounds']['fpr'][n]:.3f} | {ccd['confounds']['fpr'][n]:.3f} | {i8} |")
            L.append("")

    if drv:
        L.append("## Drive timescale and identifiability\n")
        taus = [k for k in drv["fpr"] if k.startswith("drive_tau")]
        L.append("| drive dwell (trap times) | R2 GLRT FPR | drive-aware GLRT FPR |")
        L.append("|---|---|---|")
        for k in taus:
            L.append(f"| {k.replace('drive_tau', '')} | {drv['fpr'][k]['glrt']:.3f} | {drv['fpr'][k]['drive_aware']:.3f} |")
        L.append(f"\nThe drive-aware null caps the drive at p_max = {drv['p_max']} switches per sample "
                 f"(dwell ≥ {1 / drv['p_max'] / 10:.0f} trap times). Without a cap the drive-only model can mimic "
                 "coupling (the drive tracks one trap and the other trap follows the drive): in a pilot run the "
                 "uncapped test had no power at all.\n")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        x = [float(k.replace("drive_tau", "")) for k in taus]
        ax.semilogx(x, [drv["fpr"][k]["glrt"] for k in taus], "o-", color=COL["glrt"], label="R2 GLRT")
        ax.semilogx(x, [drv["fpr"][k]["drive_aware"] for k in taus], "o-", color=COL["drive_aware"],
                    label="drive-aware GLRT")
        ax.axhline(0.01, color=MUTED, lw=1)
        ax.axvline(1 / drv["p_max"] / 10, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("shared-drive dwell time (trap times)")
        ax.set_ylabel("FPR (threshold 1% on clean)")
        ax.set_title("A slow drive mimics coupling unless modelled", loc="left", color=INK, fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig(RES / "fig1_drive_timescale.png")
        plt.close(fig)
        L.append("![drive](fig1_drive_timescale.png)\n")

    if cc and r2a:
        L.append("## Efficiency on clean data (T_min in τ_c, efficiency vs oracle)\n")
        L.append("| J | SNR | oracle | R2 GLRT | R2 CNN | causal CNN, level | causal CNN, Δ |")
        L.append("|---|---|---|---|---|---|---|")
        r2e = {(e["J"], e["snr"]): e for e in r2cnn["efficiency"]} if r2cnn else {}
        def cell(e):
            return f"{e['tmin']:.0f} ({e['efficiency']:.2f})" if np.isfinite(e["tmin"]) else "> 8× (0)"
        for i, c in enumerate(r2a):
            k = (c["J"], c["snr"])
            r2s = cell(r2e[k]) if k in r2e else "—"
            dcell = cell(ccd["efficiency"][i]) if ccd else "—"
            L.append(f"| {c['J']} | {c['snr']} | {c['tmin']['oracle']:.0f} | {c['tmin']['glrt']:.0f} "
                     f"({c['efficiency']['glrt']:.2f}) | {r2s} | {cell(cc['efficiency'][i])} | {dcell} |")
        L.append("")

    if hw:
        cm, em = hw["cost"]["cnn"], hw["cost"]["em_glrt"]
        L.append("## Hardware path\n")
        L.append(f"- Network: {hw['cost']['params']:,} parameters, {cm['macs_per_sample']:,} MACs per sample, "
                 f"receptive field {hw['cost']['receptive_field']} samples, causal; int8 weights "
                 f"{hw['cost']['weight_bytes_int8'] / 1024:.1f} KiB, line buffers "
                 f"{hw['cost']['line_buffer_bytes'] / 1024:.1f} KiB.")
        L.append(f"- Emitted streaming C++ (one sample per call) is **bit-exact** against the integer reference in "
                 f"C simulation: Δ-input model {hw['csim_bit_exact']}" +
                 (f", level-input model {hwl['csim_bit_exact']}." if hwl else "."))
        L.append(f"- int8 vs float (Δ input): power {hw['confounds_int8']['power']:.3f} vs "
                 f"{hw['confounds_float']['power']:.3f}; largest confound FPR change "
                 f"{max(abs(hw['confounds_int8']['fpr'][n] - hw['confounds_float']['fpr'][n]) for n in hw['confounds_float']['fpr']):.3f}.")
        L.append(f"- Analytical compute energy (Horowitz 45 nm, no memory): {cm['energy_int8_J'] * 1e9:.1f} nJ/sample "
                 f"int8 vs {cm['energy_fp32_J'] * 1e9:.0f} nJ fp32; at 100 kS/s that is "
                 f"{cm['power_int8_W_at_fs'] * 1e3:.2f} mW and ≈ {cm['dsp_needed_at_200MHz']:.0f} MAC units at 200 MHz.")
        L.append(f"- The HMM-EM GLRT needs ~{em['passes']} passes over a stored record "
                 f"(~{em['flop_equiv_per_sample']:,} flop-equivalents and ~{em['energy_J'] * 1e9:.0f} nJ per sample "
                 "in fp32, before memory traffic), so it cannot run as a streaming monitor.\n")
    fm_path = RES / "fpga_measured.json"
    if fm_path.exists():
        fm = json.load(open(fm_path))
        st, f0 = fm["signal_tap"], fm["clock_hz"]
        n = fm["selftest"]["traces"] * fm["selftest"]["samples_per_trace"]
        fmax = min(fm["fmax_mhz"].values())
        r = fm["resources"]
        overhead = (st["run_cycles"] - n * st["sample_period"]) / fm["selftest"]["traces"]
        L.append("## FPGA measurement (Verilog RTL on hardware)\n")
        L.append(f"- {fm['board']}, {fm['tool']}, {f0 / 1e6:.0f} MHz board clock.")
        L.append(f"- On-board self-test, {fm['selftest']['traces']} traces × {fm['selftest']['samples_per_trace']} "
                 f"samples: **{fm['selftest']['result']}** against the numpy integer reference.")
        L.append(f"- Signal Tap: **{st['sample_period']} cycles per sample** (constant); {st['run_cycles']:,} cycles for "
                 f"the whole test = {st['run_cycles'] / f0 * 1e3:.1f} ms → {n / (st['run_cycles'] / f0) / 1e3:.1f} kS/s "
                 f"including ≈{overhead:.0f} cycles of per-trace clear/compare; streaming rate "
                 f"{f0 / st['sample_period'] / 1e3:.1f} kS/s.")
        L.append(f"- Timing closure: worst-corner Fmax {fmax:.2f} MHz → bound {fmax * 1e6 / st['sample_period'] / 1e3:.1f} kS/s.")
        L.append(f"- Resources: ALM {r['alm'][0]:,}/{r['alm'][1]:,} ({100 * r['alm'][0] / r['alm'][1]:.1f}%), "
                 f"M10K {r['m10k'][0]}/{r['m10k'][1]} ({100 * r['m10k'][0] / r['m10k'][1]:.0f}%), "
                 f"DSP {r['dsp'][0]}/{r['dsp'][1]} ({100 * r['dsp'][0] / r['dsp'][1]:.0f}%) ({r['note']}).")
        if "power_mW" in fm:
            pw, fs = fm["power_mW"], f0 / st["sample_period"]
            mac = hw["cost"]["cnn"]["macs_per_sample"] if hw else 21728
            v, a = pw["rtl_vcd"], pw["vectorless"]
            e_core = v["ccnn_core_total"] * 1e-3 / fs
            L.append(f"- Power ({pw['conditions']}): CNN core (`ccnn_core`, logic + routing + M10K + DSP) "
                     f"**{v['ccnn_core_total']:.1f} mW** with simulation activity ({a['ccnn_core_total']:.1f} mW vectorless); "
                     f"whole design dynamic {v['top_hier_total']:.1f} mW incl. test-vector ROMs and clock; device static "
                     f"{v['core_static']:.0f} mW; total {v['total']:.0f} mW.")
            L.append(f"- **Energy per sample: {e_core * 1e9:.0f} nJ** for the CNN core ({e_core / mac * 1e12:.0f} pJ per MAC), "
                     f"{v['top_hier_total'] * 1e-3 / fs * 1e9:.0f} nJ for the whole dynamic design, "
                     f"{v['total'] * 1e-3 / fs * 1e6:.2f} µJ including device static power. The 45 nm arithmetic-only "
                     f"estimate above ({hw['cost']['cnn']['energy_int8_J'] * 1e9:.1f} nJ) is ≈{e_core / hw['cost']['cnn']['energy_int8_J']:.0f}× lower: "
                     "the FPGA figure includes weight/line-buffer memory reads, routing and clocking.")
            L.append(f"- Activity source: {v['vcd']}; Quartus still rates confidence '{v['confidence']}', so treat the "
                     "power numbers as estimates.\n")
        else:
            L.append("")
    for name, t in (("level input", tr), ("Δ input", trd)):
        if t:
            L.append(f"Training ({name}): {len(t['history'])} epochs, final validation accuracy "
                     f"{t['history'][-1]['val_acc']:.3f} on confound-augmented matched pairs, {t['seconds'] / 60:.0f} min "
                     "on 2 CPU cores.\n")
    (RES / "R3_report.md").write_text("\n".join(L) + "\n")
    print((RES / "R3_report.md").read_text())


if __name__ == "__main__":
    main()
