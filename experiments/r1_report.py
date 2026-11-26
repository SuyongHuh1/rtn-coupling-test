"""Write results/r1/R1_report.md from r1.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r1"


def f(x, d=0):
    return "> 6×Stein" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main():
    r = json.load(open(RES / "r1.json"))
    a, b, c = r["part_a"], r["part_b"], r["part_c"]
    tp = {x["J"]: x["tmin_hmm"] for x in c if x["snr"] >= 1000}   # noise-free reference, same pipeline
    c = [x for x in c if x["snr"] < 1000]
    L = ["# R1 report — when can trap–trap coupling be identified from RTN?\n",
         f"Criterion: false-positive rate ≤ {r['alpha']:.0%} at {1 - r['eps']:.0%} sensitivity. Two traps, "
         "τ_c = τ_e = 1 (time unit), so each trap switches at k = 2 per unit time and p = ½. "
         "Null = the closest independent-trap model (I-projection), which matches each trap’s occupancy "
         "and switching flux exactly.\n"]

    ref = tp
    path = {x["J"]: x for x in b}
    L.append("## Findings\n")
    L.append("1. **Closed-form coupling information.** For small J, D* ≈ (J²/2)·[φ² + (1−φ)²]·p₁(1−p₁)·p₂(1−p₂)·(k₁+k₂) "
             "(sympy-derived, exact formula converges to it as O(J)). Information grows as J², peaks at half "
             "occupancy, is set by the sum of switching rates (the faster trap reports the slower one’s state) and "
             "depends on the Brønsted splitting by up to 2× (minimum at φ = ½). At J = 1 kT the formula is 12% high.")
    L.append("2. **The null that matters is the best independent fit.** The closest independent-trap model matches "
             "each trap’s occupancy and switching flux exactly, so one-trap statistics cannot reveal coupling; "
             "only joint statistics can. This is the model an independent-trap analysis returns.")
    L.append("3. **D* is achieved and the time scale is universal.** E[LLR]/T converges to D* within 1% on exact "
             "simulator paths; FPR curves for J = 0.5, 1, 2 kT collapse when time is scaled by the Stein time "
             "ln(1/α)/D*, and the finite-sample T_min is 1.3–1.5× the Stein time.")
    L.append(f"4. **What it costs in transitions.** With p = ½ each trap makes about one transition per time unit, so "
             f"noise-free identification needs ≈ {ref[0.5]:.0f} transitions per trap at J = 0.5 kT, ≈ {ref[1.0]:.0f} at "
             f"1 kT and ≈ {ref[2.0]:.0f} at 2 kT (≈ 1000 at 0.3 kT). Typical RTN records hold 10³–10⁵ transitions, so "
             "sub-kT couplings are identifiable in principle.")
    L.append("5. **Bandwidth costs more than noise.** With SNR ≥ 2 and Δt ≤ 0.02 τ_c the noisy oracle matches the "
             "noise-free reference (efficiency ≈ 1 within ~5% Monte Carlo error). Sampling at Δt = τ_c keeps only "
             "~0.3 even at SNR 4. Rule of thumb for efficiency ≥ 0.7: Δt ≲ 0.1 τ_c at SNR ≥ 4, or Δt ≲ 0.02 τ_c "
             "at SNR ≥ 1 (SNR of the larger step; the smaller step is 0.6×).")
    L.append("6. **The smaller step sets the cost; level degeneracy does not.** Efficiency falls as A₂/A₁ shrinks; "
             "equal step heights (states (1,0) and (0,1) on the same level) cost nothing for identical traps "
             "(only identical kinetics tested).\n")
    L.append("Limits: two traps; point sampling with white noise; the oracle knows both models (D* is the right "
             "exponent for a test that fits the independent model, but finite-sample fitting costs extra); "
             "common drive (gate/temperature fluctuation), 1/f noise, drift and readout coupling are generated "
             "but not yet in the oracle — these are R2’s confound experiments.\n")
    L.append("## Closed form\n")
    L.append("For small J, the coupling information per unit time is\n")
    L.append("    D* ≈ (J²/2) · [φ² + (1 − φ)²] · p₁(1 − p₁) · p₂(1 − p₂) · (k₁ + k₂)\n")
    L.append("(sympy-derived; φ = Brønsted splitting, p = occupancy, k = c + e switching rate). "
             "Minimum observation time (Stein): T_lower = ln(1/α) / D*.\n")
    L.append("| J (kT) | φ | D* exact | D* closed form | ratio |")
    L.append("|---|---|---|---|---|")
    for x in a["J_phi"]:
        L.append(f"| {x['J']} | {x['phi']} | {x['D']:.4g} | {x['D_small']:.4g} | {x['D'] / x['D_small']:.2f} |")

    L.append("\n## Noise-free path oracle (exact simulator paths, 1000 H1 / 3000 H0 traces)\n")
    L.append("| J (kT) | D* | Stein time | path-oracle T_min | T_min / Stein | E[LLR]/(D*·T) at the longest T |")
    L.append("|---|---|---|---|---|---|")
    for x in b:
        L.append(f"| {x['J']} | {x['D']:.4f} | {x['stein']:.0f} | {x['tmin_path']:.0f} | "
                 f"{x['tmin_path'] / x['stein']:.2f} | {x['llr1_mean_over_T'][-1] / x['D']:.3f} |")
    L.append("\n![information](fig1_information.png)\n\n![path](fig2_path_oracle.png)\n")

    L.append("## Noisy, sampled readout (HMM oracle, parameters known)\n")
    L.append("Efficiency = T_min(noise-free reference) / T_min(noisy sampled); the reference runs through the "
             "identical estimation pipeline with SNR = 1000 and Δt/τ_c = 0.005 (sampled-chain information "
             "≥ 99.9% of D*). Step heights A₁ = 1, A₂ = 0.6. 2000 H1 / 6000 H0 traces per condition.\n")
    for J in sorted({x["J"] for x in c if x["amp2"] == 0.6}):
        rows = [x for x in c if x["J"] == J and x["amp2"] == 0.6]
        dts = sorted({x["dt"] for x in rows})
        snrs = sorted({x["snr"] for x in rows})
        L.append(f"\nJ = {J} kT (noise-free reference T_min = {tp[J]:.0f}): T_min of the HMM oracle (efficiency)\n")
        L.append("| SNR \\ Δt/τ_c | " + " | ".join(f"{d:g}" for d in dts) + " |")
        L.append("|---" * (len(dts) + 1) + "|")
        for s in snrs:
            cells = []
            for d in dts:
                x = next(z for z in rows if z["snr"] == s and z["dt"] == d)
                eff = tp[J] / x["tmin_hmm"] if np.isfinite(x["tmin_hmm"]) else 0.0
                cells.append(f"{f(x['tmin_hmm'])} ({eff:.2f})")
            L.append(f"| {s:g} | " + " | ".join(cells) + " |")
    L.append("\nStep-height ratio slice (J = 1 kT, Δt/τ_c = 0.1):\n")
    sl = [x for x in c if x["J"] == 1.0 and x["dt"] == 0.1 and x["snr"] in (1.0, 2.0, 4.0)]
    a2s = sorted({x["amp2"] for x in sl})
    L.append("| SNR \\ A₂/A₁ | " + " | ".join(f"{v:g}" for v in a2s) + " |")
    L.append("|---" * (len(a2s) + 1) + "|")
    for s in (1.0, 2.0, 4.0):
        cells = []
        for v in a2s:
            x = [z for z in sl if z["snr"] == s and z["amp2"] == v][0]
            cells.append(f"{f(x['tmin_hmm'])} ({tp[1.0] / x['tmin_hmm'] if np.isfinite(x['tmin_hmm']) else 0:.2f})")
        L.append(f"| {s:g} | " + " | ".join(cells) + " |")
    L.append("\n![noise and sampling](fig3_noise_sampling.png)\n")
    (RES / "R1_report.md").write_text("\n".join(L) + "\n")
    print((RES / "R1_report.md").read_text())


if __name__ == "__main__":
    main()
