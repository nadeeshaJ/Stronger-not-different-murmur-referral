# Deviations from the frozen protocol

## Deviation 1 — operational job scheduling (2026-11-10T08:06:27Z)
- **What changed:** `scripts/run_train.py` only (launcher and worker loop). Workers now share one job queue with atomic claims, instead of a fixed half of the job list each. Each worker process also opts out of Windows power throttling and runs at above-normal priority.
- **Why:** after about 100 of 1,125 full runs, worker 0 trained CRNN models about 4x slower than worker 1 (about 1,500 s vs about 360 s per run; identical code, GPU and data; consistent with Windows scheduling the process onto efficiency cores). With static halves, the slow worker would have stretched the run from about 40 h to about 100 h.
- **Effect on results:** none. A run's data split, seed, model, training and outputs are fully determined by its run spec (partition, outer fold, inner fold, architecture, seed). Nothing in `src/`, `configs/` or `tests/` changed. Runs finished before the change are kept, unmodified.
- **Code hash:** 02a49cf118e9d42d -> 0d7393274327c1f8 (only `scripts/run_train.py` differs; SHA-256 d4bcc5e3da4ea570 -> 5d0b566ca27aefd7).
- **Tests:** full suite re-run and passed before re-freezing.

## Deviation 2: robust scratch folders and per-launch job claims (2026-11-10T10:24:57Z)
- **What changed (file handling only):**
  - `src/murmur/train.py`: each attempt writes to a uniquely named scratch folder; leftovers are never deleted or reused; a finished run is never overwritten.
  - `scripts/run_train.py`: each launch keeps its own job-claim folder.
  - `src/murmur/analysis.py`, `analysis_h3.py`, `simulate.py`, `scripts/run_corrupt.py`, `scripts/pilot_report.py`: run discovery ignores scratch (`.tmp`) and moved-aside (`.stale`) folders.
- **Why:** on the training laptop, Windows refused to delete empty scratch folders and old claim folders left by interrupted processes, most likely because the folder is shared with the AI assistant's session, which holds handles. One worker crashed on such a folder, and two runs were blocked by stale claims.
- **Effect on results:** none. Training, data, seeds, models and analysis computations are unchanged; only folder naming and discovery changed. Runs already finished are untouched.
- **Code hash:** 0d7393274327c1f8 -> d625d7ea23b8d126. Full test suite re-run and passed.

## Deviation 3: registered budget curve computed after the frozen analysis (2026-11-12)
- **What happened:** protocol section 6 registers a descriptive budget curve over 1–30% (`configs/analysis.json` → `budget_curve`), but the frozen `scripts/run_analysis.py` only wrote b = 5%, 10% and 20%. The curve was missing from `results/`.
- **What was done:** `post_hoc/budget_curve.py` calls the frozen Study C functions (`murmur.analysis.study_c`, `compare`, `descriptive_table`) unchanged, with the registered `budget_curve` budgets. It writes to `results/post_hoc/` only. Nothing in `src/`, `scripts/`, `configs/` or `tests/` changed, so the frozen code hash (d625d7ea23b8d126) and the `results/` manifest are untouched.
- **Check:** re-running at b = 10% in the same way reproduces the confirmatory estimates exactly (H1-weak +4.107, H2-weak +0.400, H1-strong +1.493, H2-strong +1.613 murmurs).
- **Status:** descriptive. Estimates with individual-clustered 95% CIs only; no new hypothesis tests; nothing in the confirmatory family changes.
- **Also post hoc, presentation only:** `post_hoc/make_figures.py` draws figures from the result CSVs; it reads no labels or predictions.
- **Correction to protocol text:** section 4 gives the waveform CNN as "about 0.3 M parameters". Counted from `src/murmur/models.py`, it has 587,793 (spectrogram CNN 293,841; CRNN 509,089). The code is unchanged; only the descriptive figure in the protocol was wrong.

## Deviation 4: exploratory ensemble-standalone comparator for H2 (2026-11-12, post hoc)
- **Why:** reading the frozen selections showed that the development-selected secondary was often a 3-seed mean, and for the strong primary most often the CRNN's own 3-seed mean (`hom_mean:crnn`, 33% of selections at b = 10%). The registered H2 standalone is a single model. So H2 partly compares an ensemble-ranked workflow with a single model.
- **What was done:** `post_hoc/ensemble_standalone.py` reuses the frozen Study C functions and the frozen secondary/standalone selections (no reselection, no labels used in any choice). At the same realised total flags M, it adds two label-free comparators: (a) a 3-seed ensemble of the strongest architecture flagging its top-M; (b) the selected secondary used alone, top-M. Outputs go to `results/post_hoc/h2_ensemble_*` only.
- **Check:** the single-model comparator reproduces the confirmatory H2 estimates exactly (+0.400 weak, +1.613 strong).
- **Status:** exploratory and post hoc. Estimates with individual-clustered 95% CIs only; no tests. The confirmatory H2 result stands as registered; this analysis qualifies its interpretation.

## Deviations 5–7: post-hoc audits after external review (2026-11-12, evening)
An external AI review of the draft raised three methodological questions. Each was checked with frozen code and saved predictions; no model was retrained and nothing in `src/`, `scripts/`, `configs/` or `tests/` changed. All outputs are in `results/post_hoc/`.

**5. Epoch selection (`post_hoc/epoch_audit.py`).**
- **Concern:** the epoch count (14) was chosen on partition 0 / outer fold 0 development data, and those individuals reappear in other partitions' outer tests.
- **Audit:** epochs were re-selected inside every (partition, outer fold, architecture, seed) from that fold's pooled inner out-of-fold AUROC only. Two rules were used: the registered pilot rule applied per fold, and argmax. Median chosen epochs were 8–10. The confirmatory analysis was then re-run.
- **Result:** conclusions unchanged. H1-weak +4.45 / +4.04 (frozen +4.11); H2-weak +0.55 / +0.40 (frozen +0.40, not significant); H1-strong +1.45 / +1.48 (frozen +1.49); H2-strong +1.67 / +1.59 (frozen +1.61). All Holm decisions are identical. Per-replicate AUROC was 0.004–0.013 higher with fold-wise selection, so the fixed epoch did not flatter the results.

**6. Validity of the inference (`post_hoc/null_simulation.py`).**
- **Concern:** whether the sign-flip test and bootstrap hold their nominal error rates under the fixed-budget referral design with replicate averaging.
- **Simulation:** 2,000 datasets per scenario with two exchangeable rankers (null true). The simulation used real labels, individuals, outer folds, the 75-replicate structure and fixed K. Model-specific error components persisted across replicates.
- **Result:** type I error 0.041–0.042 at α = 0.05 and 0.006–0.008 at α = 0.01; bootstrap non-coverage 0.053–0.054. Error rates were approximately nominal under the simulated scenarios.
- **Bayesian vs sign-flip:** the beta-binomial posterior is computed from mean per-replicate discordant counts. It closely equals the share of single trained replicates in which the workflow beat the comparator: 0.89 vs 0.91 (H1-weak), 0.69 vs 0.71 (H1-strong), 0.71 vs 0.72 (H2-strong). It therefore describes how reliably one trained system wins, not whether the expected gain is positive.

**7. H3 natural matches only (`post_hoc/h3_natural.py`).**
- **Concern:** residual strength differences and checkpoint adjustment may confound H3.
- **Restriction:** cells matched with final checkpoints only, 31 (weak) and 38 (strong) of 75; mean strength gap 0.002 and 0.001.
- **Result:** heterogeneous minus same-architecture pair +0.04 (−0.35 to +0.52) for the weak primary and +0.21 (−0.26 to +0.70) for the strong primary. This is consistent with the main H3 result: no evidence of a diversity benefit, but a small benefit of up to about 0.5–0.7 murmurs of 174 cannot be excluded.

## Deviation 8: exploratory pilot with a large pretrained audio model (2026-11-13, post hoc)
- **Why:** to check whether "stronger, not different" holds when the second opinion comes from a large model pretrained on general audio, which is very different from the three CNNs.
- **Plan:** `protocol/deviation8_fm_pilot.md`, hashed before any pilot run (`protocol/deviation8_fm_pilot.sha256`). A dry run on synthetic predictions found one pandas bug in the analysis script. It was fixed and re-hashed before the real run.
- **What was done:**
  - The frozen Audio Spectrogram Transformer (AST, `MIT/ast-finetuned-audioset-10-10-0.4593`) produced embeddings for the frozen 4-s segments (`post_hoc/fm_extract.py`).
  - Small MLP heads were trained on the same 375 training sets as the CNNs (`post_hoc/fm_heads.py`, outputs in `runs_fm/`).
  - Frozen calibration, threshold, referral and comparison functions were applied at b = 10% (`post_hoc/fm_pilot_analysis.py`).
  - Nothing in `src/`, `scripts/`, `configs/`, `tests/`, `runs/` or the frozen `results/` files changed.
- **Check:** the frozen-secondary arms reproduce the confirmatory H1 gains exactly (+4.107 weak, +1.493 strong).
- **Result:**
  - AST was the weakest model standalone: test AUROC 0.900 against 0.921 (waveform CNN), 0.936 (spectrogram CNN) and 0.942 (CRNN).
  - Its development false-negative correlation with the CNNs was somewhat lower than between the CNNs (0.40–0.47 against 0.43–0.50).
  - As a second opinion it did worse than the frozen development-selected secondary:
    - weak primary −2.2 murmurs (−4.0 to −0.2);
    - strong primary −2.5 (−4.2 to −0.7).
  - It did not clearly beat threshold lowering: weak +1.9 (−0.2 to +4.8); strong −1.0 (−2.6 to +0.6).
  - With AST as the primary, a CRNN second opinion added +6.2 (+3.2 to +9.8) over threshold lowering. It still fell short of the CRNN 3-seed ensemble alone at the same total flags: −2.2 (−4.2 to −0.1).
- **Status:** exploratory and post hoc. Estimates with individual-clustered 95% CIs only; no tests. The AST was not fine-tuned, so the result concerns this frozen-embedding adaptation, not transformers in general.

## Deviation 9: fine-tuned pretrained transformer, reduced design (2026-11-14, post hoc)
- **Why:** under deviation 8 the frozen AST was weaker than the CNNs, so it could not test whether a much stronger model changes the second-opinion picture.
- **Plan:** `protocol/deviation9_ast_finetune.md`, hashed before the run (`protocol/deviation9_ast_finetune.sha256`) after a dry run with a tiny random model and synthetic predictions.
- **What was done:**
  - AST was fine-tuned end to end on the frozen 4-s segments (400 frames, centre-cropped position embeddings; 5 epochs, fixed settings).
  - Reduced design: partition 0, seed 0 only, so 25 models and 5 replicates per encounter. Run on the RTX 4060 laptop in 5.4 hours (`post_hoc/ast_finetune.py`, outputs in `runs_ft/` and `results/post_hoc/ft_pilot_*`).
  - Frozen calibration, threshold, referral and comparison functions were applied at b = 10%. Nothing frozen changed.
- **Check:** the frozen arms reproduce the confirmatory partition-0, seed-0 values exactly (own threshold 160.6 / 162.4; frozen secondary 165.8 / 164.0 detected).
- **Result:**
  - The fine-tuned AST was the strongest model: test AUROC 0.959 against 0.947 (CRNN), 0.935 (spectrogram CNN) and 0.925 (waveform CNN). It was higher than the CRNN in 23 of 25 fold-level comparisons.
  - With it as primary, a CRNN second opinion added nothing over threshold lowering: −0.2 murmurs (−2.8 to +2.1).
  - As a second opinion for the CNN primaries it beat threshold lowering: +5.8 (+3.0 to +8.9) weak; +2.8 (+0.4 to +5.4) strong.
  - It was not clearly better than the CRNN 3-seed ensemble alone at the same total flags: +0.4 (−1.8 to +2.8) weak; +0.6 (−1.4 to +2.9) strong.
- **Status:** exploratory and post hoc, 5 replicates only; estimates with individual-clustered 95% CIs, no tests.
