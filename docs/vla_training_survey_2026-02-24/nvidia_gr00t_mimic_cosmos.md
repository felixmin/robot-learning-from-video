# NVIDIA GR00T-Mimic / Cosmos Policy-Mimic Context

## Primary sources
- GR00T-Mimic docs: https://docs.isaacsim.omniverse.nvidia.com/latest/gr00t_mimic/tutorials/gr00t_mimic.html
- NVIDIA blog (GR00T-Mimic): https://developer.nvidia.com/blog/introducing-nvidia-isaac-gr00t-mimic-a-blueprint-for-generating-synthetic-manipulation-motion-data/
- Cosmos-Transfer1 benchmark docs: https://docs.nvidia.com/cosmos/latest/transfer1/benchmark/index.html

## What this is
- GR00T-Mimic is primarily a synthetic data generation blueprint, not a standalone VLA training paper with a full optimizer run-card.
- In practice, it is used to augment datasets for downstream robot policy/VLA training.

## Training configuration availability
- Public docs are rich on pipeline mechanics and data generation flow.
- Public docs do not provide a complete VLA training config card with exact LR, batch, total steps, and GPU count in the same way as a full model training paper.

## Full vs LoRA
- Not directly applicable for GR00T-Mimic itself as a data-generation blueprint.

## Gaps
- If you want exact training knobs tied to a specific policy shown in your cosmos policy-mimic video, we need the exact experiment/paper/repo artifact name for that run.
