# Calibrated identification of trap–trap coupling in multi-trap RTN

Code, trained model and result files for the paper
**"Calibrated Identification of Trap–Trap Coupling in Multi-Trap Random Telegraph Noise"** (submitted to IEEE I2MTC 2027).
The manuscript itself is not part of this repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24556035.svg)](https://doi.org/10.5281/zenodo.24556035)

A coupling claim in multi-trap random telegraph noise (RTN) is treated as a measurement with a stated
false-positive rate and power. The pipeline is divided:

1. a step-detecting CNN interprets the observation (baseline, decomposition candidates, trap-number support);
2. a coupled factorial HMM with coupling energy `J` performs a likelihood-ratio test;
3. the test is calibrated by re-running the **whole** pipeline, CNN included, on simulated null records;
4. a five-step checklist (resolvability, trap number, calibrated test, alternative models, replication) gives the verdict;

Calibrated operating characteristics (1/f noise, B = 99, decision at p <= 0.05): two traps FPR 11/200 (5.5 %),
power 0.78; three traps FPR 23/200 (11.5 %, decomposition errors that the nulls do not reproduce), power 0.56.
The paper reports the three-trap FPR as measured and requires replication (step 5) for three-trap calls; calls
replicated on a second record give FPR 3/100 (3.0 %) at power 0.41. On a measurement chain (NI USB-6421 AO -> divider
-> AI loopback, 130 records, pre-registered in `experiments/r46_loopback_design.md`) the two-trap test gives FPR 2/40
and detects 27/30, 29/30 and 29/30 records at J = +1, +2 and -1 k_BT.

The code and results used for the I2MTC 2027 submission are release `i2mtc-v15`, archived on Zenodo:
doi:10.5281/zenodo.24556035 (all versions), doi:10.5281/zenodo.24556036 (`i2mtc-v15`).

Not used in the paper: a score statistic for simultaneous two-trap transitions ("co-flips", `rtnsim/score.py`,
`experiments/r36_joint.py`, `results/r36/`). It detects co-flips in simulation (0.94) but exceeded its nominal
rate on the RTNinja records (7.9 %).

All numbers in the paper come from one pipeline version: `J` only (`RTN_JOINT=0`, default) with pooled restarts
(`RTN_RESTART=3`, default): three CNN-derived starts under three seeds, both models refitted from all solutions,
`Lambda = 2 (max l1 - max l0)`, so neither start selection nor the seed changes the statistic.

## Layout

| path | contents |
|---|---|
| `rtnsim/` | library: simulators (`sim2_torch.py`, `simulate.py`), coupled / drive-augmented factorial HMMs (`cfhmm.py`, `drive_hmm.py`, `multiqn.py`), CNN step detector and final pipeline (`stepnet.py`, `r9.py`), co-flip score test (`score.py`), integrated-sampling emission (`integ.py`), Student-t emission and Hampel filter (`tnoise.py`), real-data loaders (`realdata.py`) |
| `experiments/` | one script per experiment (`rNN_*.py`); see the map below |
| `training/` | CNN training script (GPU) |
| `models/stepnet.pt` | trained step detector used in the paper |
| `results/r8/`, `results/r23/`, `results/nichol/`, `results/nichol_ms/`, `results/r36/`, `results/r46/` | result files (JSON) behind every number and figure in the paper |
| `figures/make_figures.py` | regenerates the paper figures from the result files (output: `figures/out/`) |
| `docs/supplementary_results.md` | supplementary results with the result file behind each number |
| `rtl/` | Verilog for a streaming detector on DE10-Nano (Cyclone V), not used in the paper |
| `tests/` | unit tests |

## Paper ↔ scripts

| paper | script(s) | result file(s) |
|---|---|---|
| Sec. II, Table I, Fig. 1 — published procedures | `r10_baselines.py`, `r10b_zhang.py`, `r19_albrecht_faithful.py`, `r19c_son_faithful.py`, `r12_realneg.py`, `r19b_albrecht_realneg.py` | `r10_baselines_*.json`, `r10b_zhang_*.json`, `r19*_*.json`, `r19_power_faithful.json`, `r12_realneg.json` |
| Table I, AUC with true decomposition (fairness check) | `r30_oracle_baselines.py` | `r30_oracle_baselines_*.json` |
| Sec. II/VI — coupling outside the model, component ablation, long records | `r31_fairness.py misspec|ablation|long` | `r31_*.json` |
| Fig. 2 — record length | `r10_length.py` | `r10_length_full_v2.json` |
| Sec. II — RTNinja decompositions | `r15_rtninja.py`, `r17_rtninja_real.py`, `r17c_reimpl_vs_rtninja.py` | `r15_*.json`, `r17*.json` |
| Sec. III — pipeline | `rtnsim/r9.py` (`lrt_cnn3_ms`), `rtnsim/cfhmm.py` | |
| Fig. 4(a) — power | `r9a_g1.py full|white --v3` (set `RTN_MS=3`), `r8_oracle.py` | `r9a_g1_*_v3_ms3.json`, `r8_oracle*_v3.json` |
| Fig. 4(b) — calibrated FPR | `r8_calib.py exp|white 50 30 W --r9` (set `RTN_MS=3`), `p_r8` | `r8_calib_*_r9_ms3.json` |
| Sec. IV — J interval, confounders, OOD, heavy tails | `r24_jci.py`, `r8_calib.py drive|erlang [--drive-model]`, `r26_metastable.py`, `r25_ood.py`, `r25b_heavy.py`, `r27*_*.py`, `r28_heavy_cal.py`, `r21_ms_check.py` | `results/r23/*.json`, `r8_calib_{drive,drive_dm,erlang}_r9.json` |
| Sec. V, Table II, Fig. 5 — Si/SiGe data | `r11_nichol.py`, `r11_alt_qn.py`, `r13_power.py`, `r13_power_drive.py`, `r14_null_cons.py`, `r22_nichol_ms.py` | `results/nichol/`, `results/nichol_ms/` |
| Sec. V — Southampton | `r20_southampton.py` | `r20_southampton_v2.json` |
| Table III — checklist funnel | `r17e_checklist.py 2|3`, `r17f_cal3.py 2|3` | `r17e_checklist_*.json`, `r17f_cal2.json`, `r17f_cal3.json` |
| Sec. IV — calibrated power and checklist yield | `r8_calib.py coupled 100 99 W --r9` (`RTN_TAG=_B99`), `white_coupled 50 30 W --r9` | `r8_calib_coupled_r9_ms3_B99.json`, `r8_calib_white_coupled_r9_ms3.json` (fields `p_r8`, `dwell_min`, `trans_min`) |
| Sec. IV, Fig. 4(b) — calibrated FPR, 200 records per K | `r8_calib.py exp 100 99 W --r9` and `exp_new 100 99 W --r9` (`RTN_TAG=_B99`) | `r8_calib_exp_r9_ms3_B99.json`, `r8_calib_exp_new_r9_ms3_B99.json` |
| Sec. IV — residual-bootstrap block length | `RTN_BLOCK=50|1000 RTN_TAG=_L50|_L1000 r8_calib.py exp 50 30 W --r9` | `r8_calib_exp_r9_ms3_L50.json`, `_L1000.json` |
| Sec. IV — positive control on real noise | `r38_semisynth.py 24 30 W` (`RTN_CT=1`, `RTN_TAG=_ct`) | `r38_semisynth_ct.json` (`_segstart`: earlier construction, superseded) |
| Sec. IV — replication (step 5): FPR and replicated power | `r40_replication.py`, `r44_rep_power.py` | `r40_replication.json`, `r44_rep_power.json` |
| Sec. IV — effective coupling of the semi-synthetic records | `r41_semisynth_check.py` | `r41_semisynth_check.json` |
| Sec. IV / guideline 1 — boxcar integration and dwell length | `r42_boxcar_dwell.py` | `r42_boxcar_dwell.json` |
| Sec. IV — power by phi and coupling sign | `r45_phi_sign.py` | `r45_phi_sign.json` |
| Sec. V — Bonferroni over scanned biases, B raised to the floor | `r43_bonf_floor.py` | `r43_soton_bonf.json`, `results/nichol/LH_cond_Sgate_v6_K3_nullcons.json` |
| Sec. IV-B, Table I — measurement-chain positive control | `r46_loopback_make.py` (AO waveforms), `r46_loopback_analyze.py` | `results/r46/` (step-0/1 chain checks, truth, results; the recorded AI files are not included) |
| Fig. 2 — threshold sensitivity of published rules | `figures/make_figures.py thr` | uses `r10_baselines_*`, `r19_albrecht_*`, `r19c_son_*` |
| not in the paper — co-flip score test | `r36_joint.py sim|rtninja|realneg|south|conf ...`, `rtnsim/score.py` | `results/r36/*.json` (B = 39) |
| Sec. V — acquisition model vs. statistic (pilot), boxcar nulls | `r35_score_pilot.py`, `r32_box_power.py`, `r34_gatep_box.py`, `r17g_calbox.py` | not in the paper's tables |
| Sec. VI — reliability sensitivity | `r18_sensitivity.py`, `r18c_smooth.py` | `r18*.json` |
| Figures | `figures/make_figures.py [power thr null fpr len fig5]` (uses `experiments/figs_paper.py`) | `figures/out/` |

## Install and run

```bash
pip install -r requirements.txt
python -m pytest -q tests
# example: calibrated FPR, white noise, 50 + 50 independent records, 30 pipeline nulls each, 4 workers
RTN_MS=3 python experiments/r8_calib.py white 50 30 4 --r9
```
A calibrated verdict costs about 31 full pipeline runs; a calibrated two-trap verdict takes about 30 s on a 12-core laptop.
The co-flip indicator runs on the same nulls, e.g. `python experiments/r36_joint.py coflip white 2 40 39 4 point`.
Set `NUMBA_NUM_THREADS=1` when running many workers.

## Data (not redistributed)

Download the public datasets yourself and point the scripts to them:

| dataset | source | licence | env. variable / default |
|---|---|---|---|
| Si/SiGe charge sensor (Albrecht et al., arXiv:2601.08088) | Zenodo, doi:10.5281/zenodo.18828196 | CC BY 4.0 | `RTN_NICHOL` / `data/nichol` |
| Southampton silicon devices | doi:10.5258/SOTON/D0507, doi:10.5258/SOTON/399158 | see repository | `RTN_EXT` / `data` |
| imec RTNinja independent-source records | Zenodo, doi:10.5281/zenodo.16750729 | CC BY-NC-ND 4.0 (analysis only) | `RTN_RTNINJA` / `data/rtninja` |

## Licence

Code: MIT (see `LICENSE`). Third-party datasets keep their own licences and are not included.

## Citation

See `CITATION.cff`. Until the paper is published, cite the software archive:
S. Huh, *rtn-coupling-test: code and results for calibrated identification of trap–trap coupling in multi-trap RTN*,
Zenodo, doi:10.5281/zenodo.24556035.
