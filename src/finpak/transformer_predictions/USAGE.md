Usage

V7 (strict checkpoint compatibility)
- V7 expects identical input/output shapes when resuming from a checkpoint. You must keep the same feature/target counts and config dimensions.
- Use V7 if you are not changing the number/order of features or targets.

Examples (V7)
- Checkpoint path example:
  - "checkpoints/vMP007a_stability_e58_valloss_0.0004361_tc0_vc0.pt"
- Resume with identical dims:
  - python src/finpak/transformer_predictions/main_v7.py --config "vMP007a_stability_followon" --checkpoint "checkpoints/vMP007a_stability_e58_valloss_0.0004361_tc0_vc0.pt"
- Train fresh:
  - python src/finpak/transformer_predictions/main_v7.py --config "vMP007b_stability"

Additional V7 examples (same dims required)
- python src/finpak/transformer_predictions/main_v7.py --config "vMP007a_stability_followon_2" --checkpoint "checkpoints/vMP007b_stability_e248_valloss_0.0000600_tc10_vc0.pt"
- vMP007a_stability_followon_2_e620_valloss_0.0000486_tc12_vc0.pt
- python src/finpak/transformer_predictions/main_v7.py --config "vMP007a_stability" --checkpoint "checkpoints/vMP007a_stability_followon_2_e620_valloss_0.0000486_tc12_vc0.pt"

V8 (supports IO expansion + registry checks)
- V8 can expand input features and/or targets when resuming from a checkpoint.
- It will reuse the backbone and expand only IO layers when `--expand-checkpoint-io` is provided.
- Feature/target registry checks detect reorder/shrink; use `--enforce-registry` to fail fast.

Examples (V8)
- Train fresh:
  - python src/finpak/transformer_predictions/main_v8.py --config "vMP007b_stability"
- Resume with expanded IO + warmup:
  - python src/finpak/transformer_predictions/main_v8.py --config vMP007a_stability_followon --checkpoint checkpoints/…pt --expand-checkpoint-io --io-warmup-epochs 3
- Resume with custom init and registry enforcement:
  - python src/finpak/transformer_predictions/main_v8.py --config vMP007a_stability_followon --checkpoint checkpoints/…pt --expand-checkpoint-io --checkpoint-io-init normal --enforce-registry
