# Protocol addendum E1: external transportability check (ZCHSound, PASCAL B)

**Registered:** 2026-11-12, before either dataset was downloaded or inspected. The SHA-256 of this file is recorded in `protocol/addendum_external.sha256`. Any change after download is a new, dated deviation.
**Author:** D. Nadeesha Jayasuriya. **Drafted with:** Claude (Anthropic), AI-assisted.

## 1. Purpose and boundary
- Test whether the frozen CirCor models, calibrators, thresholds and secondary selections **transport** to independent recordings.
- **This is not external validation of murmur detection.** ZCHSound labels are congenital heart disease (CHD) diagnoses, not murmur labels. PASCAL B has murmur labels, but it is small, has no patient IDs or ages, and comes from a related clinical setting and device (DigiScope).
- No retraining, recalibration or threshold tuning on external data. No external label is used in any choice.
- Results are reported as transportability of a related endpoint (ZCHSound) or as a small near-external murmur check (PASCAL B). They are never pooled with CirCor results.

## 2. Data
| Dataset | Use | Target | Unit |
|---|---|---|---|
| ZCHSound clean subset (about 941 participants, 8 kHz) | primary transportability set | CHD (any of ASD, VSD, PDA, PFO) = 1; normal = 0 | participant |
| ZCHSound noisy subset (about 318 newborns) | descriptive only | same | participant |
| PASCAL Classifying Heart Sounds 2011, dataset B (labelled files, clean and noisy) | near-external murmur check | murmur = 1; normal or extrasystole = 0 | recording |
- **Licence and terms** are checked and recorded before use. If a dataset cannot be obtained or its terms forbid this use, it is dropped and the reason is reported.
- **PASCAL B sensitivity analysis:** exclude extrasystole recordings.
- **ZCHSound sensitivity analysis:** exclude PFO (often clinically insignificant and frequently murmur-free).
- Recordings shorter than 4 s are zero-padded exactly as in CirCor. Unreadable files are excluded and counted.

## 3. Preprocessing (identical to CirCor except resampling)
- ZCHSound: resample 8 kHz → 4 kHz with a polyphase filter (`scipy.signal.resample_poly`, up 1, down 2). PASCAL B: native 4 kHz; any file at another rate is resampled to 4 kHz the same way.
- Then the frozen pipeline: zero-phase Butterworth(4) 20–1000 Hz; 4 s windows, 2 s hop, tail-aligned; segment probability → recording mean → unit (participant) max.
- If ZCHSound provides several recordings per participant, the participant score is the max over recordings (the CirCor rule). Location metadata, if any, is not used.

## 4. Models and scoring
- All 1,125 frozen models. An **external replicate** is (partition, outer fold, seed, inner model): 375 per architecture.
- Each model's score is calibrated with its own frozen (partition, outer fold, architecture, seed) Platt calibrator and thresholded at that frozen 90%-development-sensitivity threshold.
- Study C rules are applied unchanged, per external replicate, at b = 10% of the primary's predicted negatives: the frozen secondary selection for (partition, outer fold, seed, primary, b = 0.10) and the frozen strongest-standalone choice. Random tie-breaks use the same seeds as CirCor.

## 5. Outcomes (all descriptive: estimates and 95% CIs, no hypothesis tests)
1. **Discrimination:** AUROC per architecture (mean over replicates), CI by participant (ZCHSound) or recording (PASCAL B) bootstrap, 2,000 draws.
2. **Operating-point transport:** sensitivity, specificity and predicted-positive share at the frozen thresholds; label-free predicted-negative share compared with the CirCor development share.
3. **Calibration:** calibration slope and intercept; reliability bins.
4. **Referral (b = 10%):** for weak and strong primaries, extra positives detected by (a) second-model referral vs own threshold (H1 contrast), (b) the workflow vs the strongest single standalone model at identical total flags (H2 contrast), (c) the workflow vs a 3-seed ensemble of the strongest architecture at identical total flags. Estimand: mean detection difference over all positive units, averaged over replicates; CI by unit bootstrap (10,000 draws, seed 20261109).
5. **Subgroups (descriptive):** ZCHSound by diagnosis (ASD, VSD, PDA, PFO) and, if available, by age group; ZCHSound noisy subset reported separately.

## 6. Interpretation rules
- A CI that includes zero is "no detectable difference", never equivalence.
- A change in sensitivity/specificity at the frozen threshold is reported as operating-point shift; it is not "re-tuned".
- ZCHSound results are statements about a CHD endpoint in a different population (China, different device, 8 kHz, many neonates in the noisy subset). They cannot confirm or refute CirCor murmur results.
- PASCAL B results are a small near-external check; possible overlap in clinical setting, device and team with CirCor is acknowledged.

## 7. Software
- Code: `external/run_external.py` (written after this addendum, reusing frozen `src/murmur` modules unchanged). Outputs: `results/external/` with SHA-256 manifest. Nothing in `src/`, `scripts/`, `configs/` or `tests/` changes, so the frozen code hash is unaffected.
