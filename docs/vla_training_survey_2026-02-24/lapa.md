# LAPA

## Primary sources
- Paper: https://arxiv.org/abs/2410.11758
- Paper (HTML mirror): https://ar5iv.org/html/2410.11758
- Code: https://github.com/ByteDance-Seed/LAPA
- Repo README (raw): https://raw.githubusercontent.com/ByteDance-Seed/LAPA/main/README.md

## Code and implementation
- LAPA uses a two-stage pipeline:
- Stage 1: latent action quantization (LAQ) pretraining.
- Stage 2: VLA training over latent action tokens.
- The repo documents separate scripts/checkpoints for both stages.

## Training configuration (reported)
- Hardware (paper table): `64 x H100`.
- Stage 1 batch size: `512`.
- Stage 2 batch size: `2048`.
- Stage 1 learning rate: `2e-4`.
- Stage 2 learning rate: `5e-5`.
- Warmup steps: `5,000` for both stages.
- Validation interval: Stage 1 `20,000`, Stage 2 `5,000`.
- Visualization interval: `5,000`.

## Full vs LoRA
- Paper framing is full training/fine-tuning for the main two-stage pipeline.
- No LoRA-centric training recipe is the main reported method.

## Gaps
- Paper and public repo do not expose every single run-level knob (for example exact total steps for every final variant) in one centralized config table.
