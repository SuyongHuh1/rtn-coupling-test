# Supplementary results to the I2MTC 2027 submission

Results that support the paper but are not needed for its main claim. Every number comes from the
result files named in brackets (`results/r8/` unless stated). Measured imec RTNinja data are not
redistributed (CC BY-NC-ND); only derived per-record statistics are in this repository.

## S1. Published procedures
- **Rule FPR by condition**. The decision rules call 9–100 % of independent simulated
  records and 7.5–95 % of real-noise controls coupled. Only the conditional-rate test is near nominal on
  real noise (0.075). [r10_baselines_*, r19_albrecht_*, r19c_son_*, r12_realneg_ms3, r19b_albrecht_realneg]
- **Conditional rate over all trap pairs**: AUC 0.75–0.88, power at 5 % FPR
  0.24–0.46, AUC on the true state path 0.95.
- **Our LRT with the asymptotic χ² threshold**: FPR 0.19–0.41 in simulation,
  0.075 on real-noise controls.
- **Fed our decomposition**, the published statistics reach AUC 0.77–0.94 (ours 0.89–0.95). The evidence
  ratio was designed for ≈2.5×10⁵-sample traces with drift removal.
- **Threshold scan:** for three traps no |d| threshold up to 4 brings Δτ/τ below an FPR of 0.77. The
  evidence ratio and the conditional-rate test reach 5 % only at e ≥ 10–20 and p < 10⁻²–10⁻³, where they
  keep at most 0.04 and 0.34 power.
- **Record length** (4096 → 65 536 samples, 1/f): the rule FPRs rise. The evidence ratio goes from
  0.20–0.30 to 0.40–0.58 and the conditional rate from 0.05–0.18 to 0.28–0.35, because decomposition bias
  does not average out while its variance shrinks. [r10_length_full_v2]
- **RTNinja decompositions:** the published rules call 34–100 % of the 200 + 200 imec independent-source
  records coupled, and 28–100 % of those that pass checklist steps 1–2. [r17c_reimpl_vs_rtninja, r19d_son_rtninja]

## S2. Proposed test, simulation
- **Power at matched 5 % FPR, white noise:** CNN–HMM 0.87 and 0.85 against the oracle's 0.88 and 0.83
  [0.74–0.90]. The two are statistically indistinguishable. [r8_oracle_white_v3, r9a_g1_white_v3_ms3]
- **Power by effect size (largest |J|):** 0.36–0.64 for 0.7–1.2 kT (14–19 records per condition),
  0.79–0.92 for 1.2–1.8 kT and 0.73–0.94 above.
- **White noise, calibrated (B = 30):** power 0.90 and 0.60 for two and three traps. The FPR is 0/50 and
  2/50. [r8_calib_white_r9_ms3, r8_calib_white_coupled_r9_ms3]
- **Checklist yield:** steps 1–2 keep 62–80 % of the detected coupled records.
- **Three-trap false positives:**
  - 8/100 and 15/100 in the two sets (Fisher p = 0.18 between the sets).
  - No fit property singles them out: step-to-noise, level spacing, occupancy, dwell and switches all
    give Mann–Whitney p > 0.05. [r39a_fp_features]
  - A 2.5 % level does not remove them (5/100, 13/100).
  - Replication: 3 of 15 recur in a replicate with the same parameters. [r40_replication]
- **Replicated power (R44):** on a fresh set of 100 coupled three-trap records (the coupled partners of the
  exp_new set, 1/f), single-record power is 0.64 [0.54, 0.73], consistent with 0.56 [0.46, 0.65] on the
  stored set. 51 replicates are significant and 41 with the same pair and sign: replicated power 0.41
  [0.32, 0.51], 0.25 with steps 1–2 on both records. The stored coupled set (records.npz) predates the last
  simulator change and cannot be regenerated, hence the fresh set. [r44_rep_power]
- **J interval:**
  - The bootstrap interval covers the true J in 32/35 records (1/f) and 31/34 records (white), using
    40-record subsets: 0.91 [0.82, 0.96].
  - The Hessian interval covers 0.70.
  - Ĵ is biased low by 6–8 % (median). [r9_hessian_*]
- **CNN:**
  - Architecture: 11 residual dilated blocks, 64 channels, one kernel-3 convolution per block, dilations
    1–2¹⁰, kernel-5 input convolution.
  - Training: 25 000 steps, AdamW, 64 records per step.
  - Loss: Huber on the signal, weighted cross-entropy on step positions, and 0.5 × Huber on step heights.
  - Step F1: 0.885.
  - Training data: freshly simulated records with two or three traps, coupling, common drive, boxcar,
    drift and 1/f noise. [results/r9a_gpu]
- **Robustness (FPR ≤ 0.07 in nine variants):**
  - Bootstrap blocks of 50 or 1000 samples: 1–3/50.
  - Erlang-2 dwells: 1/30.
  - Common drive without a model: 2/30; with the drive model of step 4: 1/30.
  - Amplitude coupling: 0.05 and 0.025.
  - Student-t₃ spikes: 2/30.
  - Metastable on/off and three-state single defects: 0/40 and 1/40 after step 4.
  - Out of the training distribution (Lorentzian background, longer dwells), power is 0.70–0.80.
  - Rare occupancy (5–12 %) and t₃ spikes reduce power to 0.37 and 0.27.
  - [r8_calib_erlang_r9, r8_calib_drive*_r9, r9_ampcoup_*, r18*, r31_misspec_*]
- **One-way coupling and co-flips:**
  - One-way coupling is detected with AUC 0.70–0.84.
  - Co-flips give AUC 0.53, whereas the evidence ratio detects them.
  - A co-flip score statistic exceeded its nominal rate on RTNinja (9.0 %). [r31_misspec_*]

- **phi misspecification and sign (R45):** on the R44 coupled set (three traps, 1/f; the model fixes phi = 0.5
  while the simulator draws phi in [0.2, 0.8]), power is 19/31, 23/37 and 22/32 for |phi − 0.5| < 0.1,
  0.1–0.2 and ≥ 0.2, and 31/44 repulsive versus 33/56 attractive (Fisher p = 0.30). [r45_phi_sign]

## S3. Real-noise positive control
- **Attractive couplings** (J = −1 to −2): 3 of 13 detected [0.08, 0.50], and three more at p ≤ 0.1.
- **Weaker detection of attraction:**
  - Repulsive 8/9 against attractive 3/13 gives Fisher p = 0.008.
  - The construction does impose the coupling: J_eff/J is 0.80 for repulsive and 0.99 for attractive
    pairs, and |J_eff| ≤ 0.38 for independent ones. [r41_semisynth_check]
  - Simulation detects both signs equally, with or without boxcar. [r39b_box_sign]
  - Dwell resolution is symmetric: 7–8 samples for either sign.
  - An untested candidate cause is the sensor's level-dependent noise (1.4× with the fast trap occupied).

## S4. Public data
- **Gate_P 0.567 V:**
  - Parametric-bootstrap s.e. 0.026 (19 records at the fitted parameters).
  - Power curve at the fitted parameters. [results/nichol/Gate_P_v8_K2_power.json]
- **Gate_P others:**
  - 0.5655 and 0.566 V are undecidable (emission dwell 0.55 samples plus boxcar).
  - The other biases from 0.565 to 0.568 V are not testable: the weak trap has step-to-noise 1.0–1.4
    without CNN step support.
- **LH 0.148–0.1505 V:** alternative. The drive LRT is much larger than the coupling LRT; the residual is
  not calibrated.
- **LH 0.152 V:**
  - The drive fit sits at its bound (|γ| = 8 kT).
  - Same-analysis nulls reach 589, and the coupling p is 0.14 (20 nulls).
- **LH 0.151 V (details):**
  - The coupling LRT inside the drive model is 69, above all 99 same-analysis nulls (maximum 14.9).
  - Blocks 10–29 replicate pair and sign (LRT 106, 191) but were not fully independent of the selection.
  - J₁₂ > 0 encodes the near-absence of joint occupancy (stationary probability < 10⁻⁴). The likelihood
    bounds J only from below (3.1–5.0 kT).
  - In this limit two strongly repelling traps are observationally equivalent to one defect with two
    charged configurations.
  - Re-calibrated for the multiplicity rule (9 biases, level 0.0056; R43,
    `results/nichol/LH_cond_Sgate_v6_K3_nullcons.json`): 0 of 399 same-analysis nulls reach 69.3 (maximum
    14.9), p = 0.0025. Verdict: candidate (replication not held out): the recurring blocks were part of the
    analysis, so step 5 is strictly not met.
- **Southampton:**
  - Of the four calls removed by the checklist, two are undecidable, one is not testable and one is an
    alternative (common drive). [r20_southampton_v2, r20b_drive_boundary]
  - RTS_vg −0.645 V re-calibrated for the multiplicity rule (7 biases on one device, level 0.0071, R43,
    `r43_soton_bonf.json`): p = 0.025 at B = 39 (floor), 0.0067 at B = 299, 0.0083 at B = 599 (5 of 600
    reach the observed LRT 38.0). The paper uses B = 599: no evidence (step 3),
    borderline (Monte Carlo s.e. of p ≈ 0.004 at the level 0.0071).

## S5. Measurement guideline 1 (integration versus dwell)
- **Setup:** independent two-trap records sampled with boxcar integration over the whole sample. [r42_boxcar_dwell]
- **Calibrated FPR:**
  - 4/240 = 1.7 % [0.6, 4.2] for fastest dwells of 4–40 samples.
  - 4/40 at 2 samples and 3/40 at 3 samples; step 1 removes these.
  - Point sampling: 2/160.
- **Why the paper keeps the instantaneous observation model:**
  - An integrated-observation emission (in the code) is costlier.
  - It still cannot identify traps faster than the sample interval, while slower traps need only
    calibration.
  - The price is the records that step 1 excludes.
