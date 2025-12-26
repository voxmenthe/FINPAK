# Assessment: v8 backbone improvements

Date: 2025-12-26
Files referenced: src/finpak/transformer_predictions/main_v8.py, src/finpak/transformer_predictions/train_v8.py, src/finpak/transformer_predictions/timeseries_decoder_v8.py

## Context
The v8 stack now supports:
- Partial checkpoint load with IO expansion for new features/targets.
- Optional IO warmup (backbone freeze) to stabilize adaptation.
- Checkpoint metadata tracking for previous/current IO dimensions.

This enables incremental growth without full retraining, but creates new opportunities to make the process more robust and faster to adapt.

## Opportunities / Improvements
1) **IO expansion initialization choices**
   - Current behavior zero-initializes new rows/cols. Consider configurable init (e.g., small normal, kaiming) and optional scaling of copied weights to reduce shift.
   - Benefit: faster adaptation of newly added targets/features.

2) **Selective optimizer reinit**
   - When expanding IO layers, optimizers keep stale state for old params and missing state for new ones.
   - Add an option to reinitialize optimizer state for only expanded tensors (or rebuild optimizers) to avoid odd LR dynamics.

3) **Backbone unfreeze schedule**
   - Current warmup is a hard freeze/unfreeze. Consider a staged unfreeze (e.g., unfreeze top blocks first) or gradual LR scaling.
   - Benefit: smoother adaptation and less instability.

4) **Feature registry for stable ordering**
   - Persist a feature/target name registry in checkpoints to detect reordering vs true expansion.
   - Benefit: protects against accidental mismatch (e.g., changing return_periods order).

5) **Expanded layer logging & metrics**
   - Track separate losses for “old” vs “new” targets during warmup to ensure new heads are learning without degrading old outputs.

6) **Automated compatibility checks**
   - Validate d_model, n_heads, n_layers before loading; fail fast if architecture mismatch would make reuse meaningless.

## Plan (recommended next iteration)
1) Add configurable initialization and optimizer state reset for expanded tensors.
2) Add feature/target registry serialization and a compatibility check on load.
3) Add optional staged-unfreeze schedule and per-target loss reporting during warmup.
4) Add a load-time summary report (architecture + IO diffs) to improve auditability.
