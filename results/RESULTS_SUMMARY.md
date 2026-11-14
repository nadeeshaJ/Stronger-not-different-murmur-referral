# Results summary: AI second opinions for paediatric murmur screening (CirCor, registered analysis)

Run completed 2026-11-12. Frozen protocol plus two documented operational deviations (`protocol/deviations.md`; scheduling and folder handling only, no effect on results).
- **Data and design:** 802 encounters, 744 individuals, 174 murmur-present. 5 partitions × 5 outer × 5 inner folds, 3 architectures × 3 seeds, so 1,125 models; 75 replicates per encounter.
- **Integrity checks:**
  - all 1,125 runs present;
  - Study B identity condition reproduced clean predictions exactly (max |Δ| = 0.0);
  - no transfer failures (dev vs test predicted-negative share gap ≤ 0.018; tolerance 0.05);
  - the operating threshold transferred (test sensitivity 0.90 vs 0.90 target).
- **Scope:** internal validation only (new CirCor partitions, not new patients). Referral benefits assume the reviewer detects referred murmurs unless stated (80% and 90% reviewer sensitivity are also reported).

## Study A: baseline models (outer test, mean over 75 replicates)
| Model | AUROC (95% CI) | PR-AUC | Sensitivity | Specificity | Calibration slope | Missed murmurs |
|---|---|---|---|---|---|---|
| Waveform CNN ("weak") | 0.908 (0.883–0.933) | 0.84 | 0.903 | 0.637 | 1.07 | 16.9 |
| Spectrogram CNN | 0.916 (0.892–0.937) | 0.85 | 0.901 | 0.671 | 1.07 | 17.3 |
| CRNN ("strong") | 0.935 (0.914–0.955) | 0.88 | 0.899 | 0.781 | 1.04 | 17.5 |

- At the same ~90% sensitivity, the CRNN flags far fewer murmur-absent children: specificity 0.78 vs 0.64. Its AUROC advantage is modest (+0.027), so "weak" vs "strong" is relative.
- **Errors overlap partially.** False-negative Jaccard is 0.33–0.39 between architecture pairs. 61–68% of the smaller miss set is shared, and error correlation is about 0.3.

## Study C: confirmatory (budget = 10% of the primary's predicted negatives; 174 murmur-present encounters; individual-clustered 95% CI; exact sign-flip test; Holm across 4)
| Hypothesis | Comparison | Extra murmurs detected (95% CI) | Holm p |
|---|---|---|---|
| H1-weak | Secondary-ranked referral vs waveform CNN's own next-most-likely | **+4.1 (+2.6 to +5.8)** | < 0.001 |
| H2-weak | Two-stage workflow vs strongest standalone model, identical total flags | +0.4 (−0.8 to +1.6) | 0.55 |
| H1-strong | Secondary-ranked referral vs CRNN's own next-most-likely | **+1.5 (+0.7 to +2.4)** | 0.001 |
| H2-strong | Two-stage workflow vs strongest standalone model, identical total flags | **+1.6 (+0.8 to +2.5)** | < 0.001 |

- **Context, mean per replicate over all 802 encounters:**
  - weak primary: about 17 murmurs missed; K = 44 referrals; the secondary captured 7.4 of the misses vs 3.2 by own threshold, 2.0 at random, and an oracle ceiling of 16.9;
  - strong primary: K = 53; the secondary captured 6.3 vs 4.8 by own threshold, 1.9 at random.
- Total flags were 429 (weak workflow) and 347 (strong workflow). The H2 standalone was the CRNN in most folds.
- **Robustness of the result:**
  - Partition-to-partition spread was small (SD 0.3–0.6 pp).
  - With an imperfect reviewer, H1 holds at both 80% and 90% sensitivity.
  - H1-strong and H2-strong are no longer significant at a 20% budget (secondary, unadjusted).
- **Reading:**
  - A second model beats lowering the primary's own threshold, for both primaries.
  - Against the strongest standalone model at identical workload, the weak-primary workflow shows **no detectable difference** (not equivalence).
  - The strong-primary workflow gains about 1.6 of 174 murmurs (~0.9 pp sensitivity).

## H3 (exploratory): does architectural diversity help beyond member strength?
- **Strength-matched comparison.** Heterogeneous pair vs same-architecture pair (matched within 0.01 dev AUROC; 100% matched, 41–51% without checkpoint adjustment):
  - weak primary: +0.05 murmurs (−0.3 to +0.5);
  - strong primary: −0.01 (−0.4 to +0.4).
- **Adjusted analysis over all 84 member pairs and triples:**
  - capture rises with mean member strength (weak primary: about +0.6 murmurs per +0.01 AUROC; CI excludes 0);
  - lower development error correlation helps for the weak primary;
  - the mixed-architecture indicator is about 0 (weak −0.18, CI −0.39 to +0.05; strong +0.09, CI −0.09 to +0.27).
- **Reading:** once strength is controlled, there is no evidence that mixing architectures adds value. The secondary's benefit tracks its strength (and decorrelated errors).

## Age (RQ5; neonates n = 1 present, descriptive only)
- **Sensitivity** in children, infants and adolescents: waveform CNN 0.92 / 0.90 / 0.81; spectrogram CNN 0.91 / 0.92 / 0.82; CRNN 0.91 / 0.91 / 0.85. The adolescent intervals are wide (16 present).
- The second-opinion workflows raise sensitivity in every stratum with data.
- **Age model (GEE, exploratory).** Adolescents have lower detection odds than children for all three architectures (log-odds −0.70 to −0.83; CIs exclude 0). Infants do not differ. There is no evidence of an age × noise interaction.

## Study B: controlled corruption (frozen models, calibrators, thresholds and referral choices)
- **Effective SNR at the model input differs from nominal** and by representation. Band-limited white/pink noise reaches the waveform input about 3 dB cleaner than nominal but the log-spectrogram input 5–15 dB worse. Dropout drives the spectrogram input below 0 dB.
- **Corruption mostly moves the operating point rather than simply degrading the model:**
  - Strong noise (0 dB) and heavy dropout push models to flag almost everyone: specificity 0.09–0.36, while sensitivity is stable or even rises (e.g. dropout 20%: CRNN sensitivity 0.98, specificity 0.15).
  - Moderate noise (10 dB) lowers both: sensitivity 0.78–0.83, specificity 0.63–0.73.
  - Clipping (≤1% of samples) and bursts change little.
  - **Rising sensitivity under severe corruption is alarm flooding, not robustness.**
- **The set of missed murmurs changes substantially** (FN-set Jaccard vs clean 0.12–0.43 under noise and dropout).
- **Referral under corruption:**
  - The primary's eligible-miss pool roughly doubles under 10 dB noise (weak 17 → 37–39), while referral capture rate falls (0.43 → 0.32).
  - The H1 benefit persists under every noise and clipping condition, and is smaller under dropout.
  - H2-weak becomes positive under moderate noise (e.g. white 10 dB +2.5, CI +0.5 to +4.6) and negative under 20% dropout (−1.1, CI −2.1 to −0.3).
  - These are exploratory, unadjusted results.
- Overlaps with Ladislaus et al. 2026 (noise robustness and aggregation); report Study B descriptively.

## Caveats for the write-up
- Internal validation on CirCor only; no external patients.
- The weak/strong AUROC gap is small (0.908 vs 0.935).
- Referral is simulated.
- Effects are a few murmurs out of 174. CIs, not p-values, carry the clinical interpretation.
- Non-significant results are not equivalence.
- Study B and age analyses are exploratory.
- The age × SNR model uses condition-level median SNR in a linear term, which the non-monotonic operating-point shifts violate. Interpret it as a coarse summary only.
