# Deviation 8: exploratory pilot with a large pretrained audio model (planned 2026-11-13)

Written and hashed before any pilot model was run. Post hoc and exploratory: the confirmatory results, the frozen code (hash d625d7ea23b8d126) and `results/` are not changed. The pilot code lives in `post_hoc/`, outputs in `runs_fm/` and `results/post_hoc/fm_pilot_*`.

## Why
Study C found that second-opinion gains were explained by model strength, not by architectural diversity. The three architectures were small CNNs trained only on CirCor. This pilot asks whether that conclusion holds for a model that is very different from all three: a large transformer pretrained on general audio.

## Model (fixed in advance)
- Audio Spectrogram Transformer (AST), Hugging Face checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`. It has about 87 M parameters and was pretrained on ImageNet and AudioSet. CirCor was not used in its pretraining, as far as is known. The AudioSet ontology includes heart-sound classes.
- **Frozen.** There is no fine-tuning. The embedding is the model's pooled output (768 values).
- Input is the frozen pipeline's own data and segmentation: the 4 kHz waveform cache (20–1000 Hz band-pass), cut into 4-s segments with a 2-s hop by the same rule as `murmur.train.DeviceData.fixed_segments`. Each segment is resampled to 16 kHz (polyphase, ×4) and passed through the AST feature extractor with its default settings.
- Embeddings are extracted once for all recordings. No labels are used at this step.

## Classifier head (fixed in advance, no tuning)
- Training sets are the same 375 as the frozen CNN members: each (partition, outer fold, held-out inner fold j) × seed s ∈ {0, 1, 2}.
- Inputs are standardised with training-set statistics.
- Architecture: MLP 768 → 128 (ReLU, dropout 0.2) → 1.
- Training: AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch 256, 20 epochs, no early stopping. Loss is binary cross-entropy with pos_weight = negative/positive training segments.
- Segment labels use the frozen location-level rule.
- Aggregation is the frozen rule: segment → recording by mean probability, recording → encounter by maximum.
- The head writes OOF and outer-test probabilities in the frozen `preds.npz` format under the architecture name `ast`.

## Analysis (`post_hoc/fm_pilot_analysis.py`)
Everything below uses the frozen calibration, thresholds, referral and comparison functions, at b = 10%. Results are estimates with individual-clustered 95% bootstrap CIs. There are no hypothesis tests.

**A. Standalone discrimination.** Test AUROC per (partition, outer, seed, inner model), and pooled development OOF AUROC, for AST and the three CNNs.

**B. Error correlation.** On development OOF data, the correlation of false-negative indicators among murmur-present encounters at the 90%-sensitivity thresholds, for AST against each CNN.

**C. AST as the second opinion for the registered primaries (weak: waveform CNN; strong: CRNN).** The secondary score is the AST 3-seed mean, fixed in advance with no development selection. It is compared with:
1. the primary's own threshold lowering (H1-type);
2. the frozen development-selected secondary;
3. the CRNN 3-seed ensemble flagging its top M at the same total flags (H2-type).

**D. AST as the primary, with the CRNN 3-seed mean as its secondary.** It is compared with:
1. AST's own threshold lowering;
2. the CRNN 3-seed ensemble top M;
3. the AST 3-seed ensemble top M.

The AST ensemble top M is also compared with the CRNN ensemble top M.

## Interpretation rule (stated in advance)
- If AST is weaker than the CRNN standalone, the pilot cannot answer whether a much stronger model changes the conclusion. It then answers a narrower question: does a maximally different second opinion add value at matched workload?
- If AST is stronger, part D shows whether the second-opinion benefit shrinks further when the primary is stronger.

Fine-tuning the transformer is not part of this pilot. If it is done later, it will be documented separately.
