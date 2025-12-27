# Assessment: Expanding features/targets while resuming from checkpoint

Date: 2025-12-26
Files referenced: src/finpak/transformer_predictions/main_v7.py, src/finpak/transformer_predictions/configs.py, src/finpak/transformer_predictions/timeseries_decoder_v6.py

## Assessment (current behavior)
- The model’s **input projection** is `HybridInputEmbedding.continuous_projection` with shape `[d_continuous_proj, d_continuous]`. Changing the number of continuous features changes `d_continuous`, which makes this weight incompatible with the checkpoint.
- The **output heads** depend on `n_continuous_outputs` (len(target_periods)) and (optionally) `n_categorical_outputs`. Changing target count changes `continuous_projection` (and `categorical_projections`) shapes, making those weights incompatible.
- Everything else (transformer blocks, layer norms, attention, etc.) is independent of feature/target counts and **can be reused** if you can load the checkpoint with those incompatible layers skipped or partially initialized.

## Answer to your question
Yes. You can increase features and/or targets and still get meaningful resumed training by **reusing the backbone weights and selectively reinitializing only the layers whose shapes changed** (input projection and/or output heads). This is not a full restart: you preserve the bulk of the model’s learned temporal representations and only “grow” the I/O surfaces.

## Practical options (most useful first)
1) **Partial state_dict load (skip incompatible layers)**
   - Load the checkpoint with `strict=False` or manually filter keys to drop those with shape mismatches.
   - This reuses the backbone while reinitializing only the input projection and output heads.
   - Best when you add features/targets and are ok with the new I/O layers learning from scratch.

2) **Weight expansion with copy for old dimensions**
   - For input projection: create a new weight with the larger `d_continuous` and copy old columns into the left block; initialize new columns (for new features) to zeros or small std.
   - For continuous output head: create a new weight with the larger `n_continuous_outputs` and copy old rows; initialize new rows.
   - This keeps the previous predictions intact and learns new ones faster.

3) **Staged unfreeze**
   - Load backbone + reinit I/O; freeze transformer blocks for N epochs while I/O adapts; then unfreeze.
   - Helps prevent catastrophic drift in the reused representation when adding new signals or targets.

4) **Feature gating / masking**
   - Keep `d_continuous` fixed by defining a superset feature vector from the beginning, and “turn on” new features later by unmasking them. This avoids shape changes altogether.
   - This requires planning the max feature set in advance.

## What this means for your configs (vMP007a_stability → followon)
- In `configs.py`, `vMP007a_stability` has `target_periods: [1, 2]` and the followon uses `[1, 3]` and longer `sequence_length`.
- The sequence length change is fine for the backbone (no parameter mismatch).
- The target period change changes `n_continuous_outputs` and **will break a strict load** unless you skip or expand the output head.
- If you also add features (e.g., more return/sma/momentum/volatility), you will also break strict load of the input projection.

## Plan (recommended implementation path)
1) **Add a safe partial-load path**
   - In `main_v7.py` (or a helper), load the checkpoint state dict and drop keys whose shapes don’t match the current model.
   - Log which keys are skipped so you can confirm only I/O layers were reinitialized.

2) **Optional: add “expand weights” utilities**
   - Implement a small function to expand `continuous_projection.weight`, `continuous_projection.bias`, and `input_embedding.continuous_projection.weight` with old weights copied into matching sub-blocks.
   - Initialize new rows/columns with zeros or small normal.

3) **Train with short I/O warmup**
   - Freeze transformer blocks for 1–3 epochs (or a fixed number of steps) while new I/O weights adapt.
   - Unfreeze and continue normal training.

4) **Record compatibility metadata**
   - Save the old `d_continuous` and `n_continuous_outputs` in checkpoint metadata if not already present, so expansion logic can be automatic.

## Minimal change suggestion (if you want the quickest win)
- Use partial load (skip mismatched) + a short warmup/freeze. This preserves most learned capacity and avoids retraining from scratch.

## Open questions (to confirm best path)
- Do you want automatic weight expansion (copy old weights) or are you ok reinitializing the changed layers?
- Are you adding only continuous features, or also categorical features/bins?
- Do you want an explicit feature superset to avoid shape changes going forward?
