# pi0 (PI Zero)

## Primary sources
- Paper: https://arxiv.org/abs/2410.24164
- Paper (HTML mirror): https://ar5iv.org/html/2410.24164
- Open-source code (OpenPI): https://github.com/Physical-Intelligence/openpi
- OpenPI train script example (raw): https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/scripts/train_pi0_fast_droid.sh
- OpenPI training config (raw): https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py

## Code and implementation
- pi0 is described as a flow-based VLA policy with a VLM + action expert structure.
- Public OpenPI code provides practical training scripts/configs for open-source pi0 variants and finetuning workflows.

## Training configuration (reported)
- Paper-reported high-level data scale: over `10,000` hours of robot data (plus internet-scale pretraining inputs).
- OpenPI example finetune (`pi0_fast_droid` script): `batch_size=32`, `learning_rate=5e-5`.
- OpenPI default optimizer config includes `weight_decay=0.0` in base train config.

## Full vs LoRA
- Paper does not present LoRA as the central method for pi0 pretraining.
- OpenPI public scripts mostly show full finetuning-style config flows (not LoRA-first recipes).

## Gaps
- Paper does not publish a complete run-card with exact global batch, total steps, and hardware counts for the full pi0 pretraining recipe.
