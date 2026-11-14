# Stronger, not different: AI second opinions in paediatric heart murmur screening

**When do AI second opinions help in paediatric heart murmur screening? A matched-workload evaluation on CirCor DigiScope v1.0.3.**

Code, frozen protocol, fold assignments and result tables for the MSc dissertation and the accompanying manuscript.

D. Nadeesha Jayasuriya, MSc in Artificial Intelligence, Dublin Business School (supervisor: Dr. Shahram), 2026.

This repository has the frozen protocol, all code, fold assignments and result tables. Raw audio is not included; download it from PhysioNet. The per-run predictions of all models (`runs*/`) are too large for GitHub and are deposited on Zenodo (see *Archived copies* below).

## Main result (internal validation, b = 10%)

| Contrast (extra murmurs detected, of 174; 95% CI) | Weak primary (waveform CNN) | Strong primary (CRNN) |
|---|---|---|
| H1: second model vs the primary's own lowered threshold | +4.1 (+2.6 to +5.8) | +1.5 (+0.7 to +2.4) |
| H2: two-stage workflow vs strongest single model, same total flags | +0.4 (−0.8 to +1.6) | +1.6 (+0.8 to +2.5) |
| Post hoc: vs 3-seed ensemble of the strongest architecture, same total flags | −0.7 (−2.0 to +0.6) | −0.04 (−0.70 to +0.65) |

In short, a second model beats threshold lowering, but not a stronger model or a seed ensemble at the same workload.

## Layout

| Path | Contents |
|---|---|
| `protocol/protocol.md`, `protocol/FREEZE.json` | frozen protocol and SHA-256 freeze (code hash d625d7ea23b8d126); earlier freezes kept as `FREEZE_superseded_*.json` |
| `protocol/mdd.csv`, `protocol/tost_feasibility.csv` | pre-freeze power and feasibility simulations |
| `protocol/deviations.md` | deviations 1–9 (operational changes, budget curve, ensemble comparator, post-hoc audits, pretrained-transformer pilots) |
| `protocol/deviation8_fm_pilot.md`, `protocol/deviation9_ast_finetune.md` | plans and pre-run hashes for the two pretrained audio transformer pilots |
| `protocol/addendum_external.md` | pre-registered external transportability addendum (E1) |
| `configs/` | cohort, training and analysis parameters (frozen) |
| `src/murmur/` | cohort, partitions, preprocessing, models, training, calibration, referral, statistics, corruption |
| `scripts/` | build, train, corrupt, analyse, freeze, simulations |
| `tests/` | 65 unit tests |
| `post_hoc/` | budget curve (deviation 3), ensemble comparator (deviation 4), audits (5–7), transformer pilots (8–9), figures |
| `external/` | external transportability check on ZCHSound (addendum E1); labels are rebuilt from the downloaded dataset by `prepare_zchsound.py` |
| `data/derived/` | cohort tables, `folds.csv` (sha256 091d7697…), derived from CirCor (ODC-By 1.0) |
| `runs/<run>/preds.npz` | per-epoch out-of-fold and outer-test probabilities for all 1,125 models (**Zenodo**, not in this repository) |
| `results/` | frozen result tables + `manifest.json`; `results/post_hoc/`; `results/figures/` |

## Reproduce (Windows, NVIDIA GPU)

1. Put `circor-heart-sound-1.0.3.zip` from PhysioNet next to this folder.
2. `setup_windows.bat` — creates the virtual environment, installs CUDA PyTorch, extracts and verifies CirCor, builds the cohort and folds, runs the tests and checks the GPU.
3. `run_full.bat` — freezes, trains 1,125 models (about 35–40 GPU-hours on an RTX 4060 Laptop with 2 workers), runs Study B corruption scoring and the frozen analysis.
4. Post hoc: `python post_hoc/budget_curve.py`, `python post_hoc/ensemble_standalone.py`, `python post_hoc/make_figures.py` (needs `matplotlib`). The transformer pilots (`run_fm_pilot.bat`, `run_ast_finetune.bat`) also need `transformers`.

To re-run only the analyses without retraining, download the predictions archive from Zenodo and extract it into this folder so that `runs/` sits next to `src/`.

The frozen analysis (`scripts/run_analysis.py`) is the only code that joins outer-test labels. It refuses to run unless the code hash matches `protocol/FREEZE.json`.

## Data

- CirCor DigiScope v1.0.3: Oliveira J. et al., IEEE JBHI 26(6):2524–2535, 2022; PhysioNet, doi:10.13026/tshs-mw03, Open Data Commons Attribution Licence v1.0.
- ZCHSound (external check): download from its authors' release and run `external/prepare_zchsound.py`; no ZCHSound files are redistributed here.

## Licence

- Code: MIT (`LICENSE`).
- Protocol, text, result tables and figures: CC BY 4.0 (`LICENSE-docs.md`).
- `data/derived/`: derived from CirCor, under the dataset's ODC-By 1.0 licence.

## How to cite

Use GitHub's *Cite this repository* button (from `CITATION.cff`) or the Zenodo DOI once a release is archived.

## Archived copies

- Code release: Zenodo DOI to be added after the first GitHub release.
- Predictions of all models (`runs/`, `runs_corrupt/`, `runs_fm/`, `runs_ft/`): Zenodo DOI to be added.
