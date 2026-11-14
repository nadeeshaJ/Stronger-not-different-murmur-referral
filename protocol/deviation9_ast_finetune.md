# Deviation 9: fine-tuned pretrained transformer, reduced design (planned 2026-11-13, before any run)

Post hoc and exploratory. This follows deviation 8, where the frozen AST was weaker than the CNNs. It does not change the frozen code (hash d625d7ea23b8d126) or any confirmatory result. Code: `post_hoc/ast_finetune.py`. Outputs: `runs_ft/`, `results/post_hoc/ft_pilot_*`.

## Question
Does a fine-tuned AST become stronger than the CRNN? And when the first model is that strong, does a second opinion still add anything at matched workload?

## Reduced design (compute limit: one laptop GPU)
- Partition 0 only, seed 0 only, all 5 outer × 5 inner folds: 25 fine-tuned models, named `astft`. This gives 5 replicates per encounter, against 75 in the main study, so the CIs will be wide.
- The comparators use the same partition and replicates: the frozen CNNs (seed 0, plus the CRNN 3-seed mean where named) and the frozen development-selected secondary.

## Model and training (fixed in advance, no tuning)
- **Model:** AST, `MIT/ast-finetuned-audioset-10-10-0.4593`, all layers fine-tuned.
- **Input:** the frozen 4-s segments (frozen cache and segmentation rule), resampled to 16 kHz.
  - Features are the AST feature extractor's log-mel filterbanks at 400 frames.
  - The time axis of the pretrained position embeddings is centre-cropped from 101 to 39 patches, the standard AST practice for shorter input.
- **Head:** LayerNorm + linear layer on the pooled output.
- **Training:**
  - Loss: BCE with pos_weight = negative/positive training recordings.
  - Segment labels use the frozen location-level rule.
  - Each epoch samples 4 fixed segments per training recording, with replacement.
  - Optimiser: AdamW, learning rate 2×10⁻⁵ for the backbone and 10⁻³ for the head, weight decay 10⁻⁴. Batch 16, mixed precision.
  - 5 epochs, no early stopping.
- **Scoring:** every fixed segment of the OOF and outer-test encounters is scored. Segments are combined into recordings by mean and recordings into encounters by maximum (the frozen rules).

## Analysis
The frozen calibration, thresholds and referral functions are applied at b = 10%, with the same arms as deviation 8 and AST replaced by `astft`:
- standalone AUROC;
- `astft` as the second opinion for the weak and strong primaries;
- `astft` as the primary, with the CRNN 3-seed mean as its second opinion;
- `astft` alone against the CRNN 3-seed ensemble alone, at the same total flags.

Results are estimates with individual-clustered 95% CIs, without tests.

## Interpretation rule (stated in advance)
- If `astft` is not stronger than the CRNN (mean test AUROC difference ≤ 0), the question "does a much stronger model change the picture" stays unanswered at this scale.
- If it is stronger, the key result is whether a second opinion on top of `astft` beats the `astft` ensemble alone at the same total flags. Here that ensemble has a single seed, so it is the standalone model.
