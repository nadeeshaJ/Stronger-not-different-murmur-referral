# Registered protocol (DRAFT — not frozen)

**Study:** When do AI second opinions help in paediatric heart murmur screening? Matched-workload evaluation on CirCor DigiScope v1.0.3.
**Author:** D. Nadeesha Jayasuriya, MSc AI, Dublin Business School. **Supervisor:** Dr. Shahram.
**Status:** final pre-freeze version, 2026-11-10. All design decisions taken; frozen by `scripts/make_freeze.py`.
**Code:** every number traces to `src/`, `scripts/` and `configs/`; the freeze records their SHA-256 hashes (`protocol/FREEZE.json`).
**AI assistance:** code, tests, simulations and this draft were produced with an AI assistant (Claude, Anthropic); design decisions and sign-off are the author's.
**Independent implementation:** the entire implementation (preprocessing, models, training, calibration, referral, statistics) was designed and written from scratch for this study, using only the revised proposal, this protocol, the original CirCor v1.0.3 data and the published literature. No code, architectures, configurations, checkpoints or predictions from the earlier (archival) project were inspected, copied or reused. The study does not attempt to reproduce archival results. Archival figures appear only where the proposal itself reports them as motivation (Section 9 power anchors).

---

## 1. Data and cohort
- CirCor DigiScope v1.0.3: all 10,434 files verified against `SHA256SUMS.txt`.
- **Eligibility.** 942 encounters, minus 68 with Murmur = Unknown, minus 72 with missing age (68 of them pregnant participants) = **802 encounters, 744 individuals, 2,770 recordings, 174 Present / 628 Absent**.
- Age groups: Neonate 5 (1 Present), Infant 101 (25), Child 627 (132), Adolescent 69 (16).
- **Units.** Decisions are per encounter. Grouping and inference are per individual: Patient IDs linked via Additional ID by transitive union-find. 58 individuals have 2 encounters.

## 2. Partitions
- 5 partitions (seeds 2026110901–05) × 5 outer × 5 inner folds, grouped by individual.
- Stratification is at the individual level on (n_encounters, n_present) and age group, using StratifiedKFold over individuals mapped back to encounters. The inner seed is derived with SeedSequence(partition seed, outer fold).
- Achieved balance: test-fold n 159–161; prevalence 0.206–0.230; present infants 5 in every test fold.
- `folds.csv` sha256 `091d7697c3c92c42c5cfbb6c7d830095e74eaa084b3a3b4193d962eb20e2aebc`.
- **Deviation from the draft proposal:** sklearn StratifiedGroupKFold(shuffle) was rejected because its test-fold prevalence ranged up to 0.31.

## 3. Preprocessing (identical for all architectures)
- int16 → float; native 4 kHz (no resampling); zero-phase Butterworth(4) band-pass 20–1000 Hz.
- Evaluation segments: 4 s windows (several cardiac cycles even at low paediatric heart rates), 2 s hop, tail-aligned last window, zero-padding for recordings shorter than 4 s.
- Training: random 4 s crops, 4 per recording per epoch.

## 4. Models and training
- Members: waveform CNN, spectrogram CNN, CRNN × 3 seeds, trained per inner split (1,125 runs). The architectures were designed for this study (`src/murmur/models.py`):
  - waveform CNN: strided stem plus 4 double-conv blocks, about 0.3 M parameters;
  - spectrogram CNN: 4 double-conv 2-D blocks;
  - CRNN: 3 conv blocks pooling frequency, then a BiGRU(64) with mean+max pooling.
- Input normalisation is fitted on each model's own training set and frozen as buffers:
  - waveform: 1 / training RMS;
  - spectrogram: per-bin mean/std of the log-power STFT (n_fft 256, hop 64, bins ≤ 1000 Hz).
- **Training labels:** location-level (a recording is positive iff its location is listed in Murmur locations). This uses CirCor's location annotation and is standard practice in CinC 2022 entries.
- **Encounter score:** segment probabilities → recording mean → encounter max. An encounter is murmur-present if any location carries a murmur.
- Optimisation:
  - AdamW, lr 1e-3, weight decay 1e-4, batch 128, BCE with pos_weight = negatives/positives, AMP;
  - fixed epoch count with no early stopping: **14 epochs**, fixed from the Phase 4 pilot on development data only (section 8).
- Determinism: seeded per (partition, outer, inner, arch, seed); deterministic cuDNN; any remaining GPU nondeterminism is measured.
- **Saved per run:** per-epoch encounter probabilities for the inner OOF fold and the outer test, the final checkpoint, and a manifest.
- **Sealing:** outer-test scoring is refused unless `protocol/FREEZE.json` exists and its code hash equals the running code.
- **Inner-inner models are not trained** (deviation from the proposal's secondary design). Secondary models only rank the primary's negatives and the H2 standalone uses label-free top-M, so no ensemble threshold is required. Ensemble rankings use the members of the same inner split, which is the same mechanism on development and test data.

## 5. Calibration and thresholds
- Per (partition, outer, arch, seed): an unpenalised Platt fit on pooled inner-OOF scores (Newton IRLS).
- Threshold = the largest value giving ≥ 90% sensitivity on calibrated pooled OOF.
- Every outer-test encounter is scored by each of the 5 inner models separately. A replicate is (partition, seed, inner model): 75 replicates per encounter.
- **Transfer check (label-free):** mean test predicted-negative share vs dev share per (partition, seed, primary), pooled over outer folds and inner models. A gap above 0.05 is reported as a transfer failure.

## 6. Study C — confirmatory
- **Primaries:** weak = waveform CNN; strong = CRNN.
- **Budget (decided 2026-11-09):** K = ceil(b × number of the primary's predicted negatives) on each scored test fold. b = 10% (confirmatory); 5% and 20% (secondary); curve over 1–30%.
- **Secondary candidates:**
  - single other architectures (same seed);
  - same-architecture 3-seed means;
  - heterogeneous mean and heterogeneous disagreement (SD of the architecture means).
  The candidate is chosen per (partition, outer, seed, primary, budget) by inner-OOF capture; ties go to the earlier, simpler candidate.
- **H1:** detections with secondary-ranked referral minus detections with own-probability referral, same K.
- **H2 (identical total flagged workload):** two-stage workflow vs the strongest standalone model (by dev pooled-OOF AUROC) flagging its top-M encounters, where M = the workflow's realised total flags on the same fold and replicate (label-free).
- **Estimand:** over all 174 murmur-present encounters, the replicate-averaged detection difference per encounter, summed within individual. The denominator is fixed.
- **Test:** exact sign-flip randomisation over individuals, computed by lattice convolution (resolution 75).
- **Intervals:** individual-clustered percentile bootstrap (10,000 draws, drawn once and reused).
- **Multiplicity:** Holm across H1-weak, H1-strong, H2-weak and H2-strong; two-sided α = 0.05.
- **Also reported:**
  - spread across partitions (training/partition variability);
  - beta-binomial posterior on mean discordant counts (weakly informative prior Beta(2,2));
  - imperfect reviewer (80%, 90%);
  - oracle ceiling, random floor, eligible FNs, referral precision, total flags, sensitivity and specificity.
- **No equivalence testing (decided 2026-11-09).** No clinically justified equivalence margin is available, so the TOST analysis in the proposal is removed. Every comparison is reported as a paired effect estimate with a 95% individual-clustered CI and the registered superiority test. A non-significant result is reported as "no detectable difference at this sample size", never as equivalence or non-inferiority.

## 7. Exploratory analyses
- **H3 (strength-matched architecture diversity).** Heterogeneous pair = the two non-primary architectures (same seed). Same-architecture pair = two seeds of the stronger of those two.
  - Strength = pooled inner-OOF AUROC per member and epoch.
  - The weaker pair stays at the final epoch; the stronger pair's members move to earlier checkpoints so that |Δ mean strength| ≤ 0.01, preferring the latest checkpoints.
  - Route (natural / checkpoint / unmatched) and member strengths are reported with every result. Intervals only.
  - Adjusted analysis: all 84 pairs and triples of non-primary members; OLS of capture on mean strength, dev error correlation, a mixed-architecture indicator and a triple indicator; bootstrap over (partition, seed).
- **Study A:**
  - AUROC (CI from individual resampling, averaging per-replicate single-model AUROCs), PR-AUC, sensitivity/specificity at the dev threshold, calibration slope/intercept, Brier, reliability bins;
  - FN overlap: Jaccard, nestedness, Q-statistic, error correlation.
- **Study B:**
  - Conditions: identity; white and pink noise at 20/10/0 dB; bursts 2 and 5 per 10 s at 0 dB; clipping at |x| percentiles 99.9/99.5/99 (0.1/0.5/1% of samples clipped; the label-free check showed percentile 95/90 clipping removes most S1/S2 energy, median effective SNR 3.5/2.4 dB); dropout 5/10/20% in 100 ms chunks.
  - Corruption is applied to raw audio with seeds matched per (recording, condition). Frozen models, calibrators, thresholds and referral choices are reused.
  - The identity condition must reproduce clean probabilities within 1e-3 before anything else is analysed.
  - Effective SNR is reported at the filtered waveform and at each architecture's input. Changes in sensitivity, specificity, FN and FP counts, probability shifts and FN-set overlap are reported.
  - Referral under corruption separates the eligible-FN population from the capture rate.
- **Age (RQ5):** per-stratum sensitivity/specificity and Study C detection with clustered intervals; neonates descriptive only.
  - Age × effective-SNR models only for strata with ≥ 10 present encounters (Infant, Adolescent vs Child).
  - Implemented as GEE logistic with exchangeable correlation within individual (population-averaged effects; deviation from the proposal's mixed-effects wording, because no frequentist GLMM is available in the analysis stack). Exploratory.

## 8. Pilot (Phase 4) — development data only
Partition 0, outer fold 0, inner models only; the outer test is never scored. The pilot fixes the epoch count (smallest epoch within 0.005 AUROC of each architecture's best) and measures runtime. No comparison with archival performance is made.

**Pilot result (2026-11-10, 45 runs, `runs_pilot/pilot_report.txt`, `pilot_curves.csv`):**
- Pooled inner-OOF AUROC, mean over seeds, peak: CRNN 0.931 (epoch 17), spectrogram CNN 0.921 (epoch 11), waveform CNN 0.920 (epoch 14). Curves plateau after about epoch 6–11 and are noisy (seed SD 0.01–0.025).
- Rule → CRNN 6, spectrogram 11, waveform 14 → **14 epochs**.
- The development-data strength gap between the registered "weak" (waveform CNN) and "strong" (CRNN) primaries is small, about 0.01–0.02 AUROC. The primaries keep their registered roles. Results are reported with their measured strengths, and "weak/strong" is read as relative, not as a large capability gap.
- Runtime: about 8 minutes per 30-epoch run with 2 concurrent workers (RTX 4060 Laptop, 8 GB). The full run at 14 epochs is estimated at about 35–40 hours. Results are recorded in `runs_pilot/pilot_report.txt`.

## 9. Power and feasibility (simulation, `protocol/mdd.csv`, `protocol/tost_feasibility.csv`)
Model: 174 present encounters, 75 replicates. D = expected discordant murmurs per replicate. Consistency c = the share of replicates in which a discordance-prone murmur is discordant.

| D | c | MDD α=0.05 | MDD α=0.0125 (Holm) |
|---|---|---|---|
| 5 | 1.0 | not reachable | not reachable |
| 5 | 0.25 | 3.5 murmurs (2.0 pp) | 4.0 (2.3 pp) |
| 10 | 1.0 | 9.0 (5.2 pp) | 10.0 (5.7 pp) |
| 10 | 0.5 | 6.5 (3.7 pp) | 7.5 (4.3 pp) |
| 20 | 0.5 | 9.0 (5.2 pp) | 10.5 (6.0 pp) |
| 30 | 0.5 | 11.0 (6.3 pp) | 12.5 (7.2 pp) |

- **The archival H2-strong pattern** (5 discordant, 4 vs 1) has power ≈ 0.09 at α = 0.05 and ≈ 0.02 after Holm if discordance is consistent across replicates. It rises to 0.73 only if discordance is mostly training noise (c = 0.25). H2-strong is therefore expected to be an **estimation** result.
- **Equivalence was not pursued.** Simulation under no true difference showed that a 1 pp margin is met with probability ≤ 0.035 (`protocol/tost_feasibility.csv`), and no clinically justified wider margin exists. H2-strong is therefore reported as an estimate with its CI.

## 10. Reporting
TRIPOD+AI. All tables come from `scripts/run_analysis.py` (the only place outer-test labels are joined), with SHA-256 manifests. Per-encounter predictions and fold assignments are released; raw audio stays on PhysioNet.

## 11. Literature search and novelty framing
- Search 1 (preliminary web search) and Search 2 (author's focused novelty check, verified citations; `protocol/literature/`) are in `protocol/lit-search.md`. Native database searches were run on 2026-11-09 (PubMed 12, arXiv 21, IEEE Xplore 44 for the registered strings; `protocol/search-counts.md`). The title screen is AI-assisted and preliminary; author verification is pending.
- **Prior work this study builds on, not invents:**
  - deferral to clinicians with workload trade-offs: CoDoC, Dvijotham et al. 2023;
  - uncertainty-based learning to defer: Liu et al. 2022;
  - disagreement prediction: Sanchez et al. 2023;
  - ensemble-disagreement rejection: DEGRE, Hong et al. 2026;
  - calibration-dependent referral (2023);
  - CirCor murmur detection with uncertainty: Zhang et al. 2024;
  - PhysioNet 2022 cost-based screening (Reyna et al.);
  - noise robustness and aggregation in murmur detection: Ladislaus et al. 2026.
- **Registered framing (not a definitive novelty claim):** "We evaluate whether a secondary murmur classifier offers incremental false-negative recovery over strong standalone and threshold-adjusted baselines at matched total flagged workload, and assess whether any heterogeneous-ensemble benefit persists once individual model strength is controlled. We examine robustness and age-dependent failure patterns descriptively."
- **Boundaries:**
  - AI-assisted code audit is not independent human review.
  - New partitions are not external patients.
  - CHD labels cannot validate murmur detection.
  - Referral benefits assume reviewer detection unless imperfect review is modelled.

## 12. Decision log
| ID | Decision | Resolution |
|---|---|---|
| D1 | Architectures | Independently designed (section 4); no previous-project material used |
| D2 | Segments and aggregation | 4 s / 2 s hop; recording mean, encounter max |
| D3 | Training label level | Location-level |
| D4 | Epoch count | 14, by the registered pilot rule (2026-11-10) |
| D5 | Budget base | Primary predicted negatives; 10% confirmatory, 5%/20% secondary; H2 at identical total flagged workload (2026-11-09, author) |
| D6 | Equivalence margin | No TOST; estimates, 95% CIs, superiority tests only (2026-11-09, author) |
| D7 | Tolerances | Transfer gap 0.05 (pooled); identity 1e-3 |
| D8 | Corruption levels | Section 7 |
| D9 | Age × corruption model | GEE (exploratory) |
| D10 | Bayesian prior | Beta(2,2) |
| D11 | Literature and novelty review | Focused novelty check plus native searches 2026-11-09 (PubMed 12 / arXiv 21 / IEEE 44); author verification of the screen pending |
| D12 | OSF registration | After freeze (author) |
