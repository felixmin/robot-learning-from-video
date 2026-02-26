# OpenVLA

## Primary sources
- Paper: https://arxiv.org/abs/2406.09246
- Paper (HTML mirror): https://ar5iv.org/html/2406.09246
- Code: https://github.com/openvla/openvla
- Repo README (raw): https://raw.githubusercontent.com/openvla/openvla/main/README.md
- VLA config registry: https://raw.githubusercontent.com/openvla/openvla/main/prismatic/conf/vla.py

## Code and implementation
- OpenVLA is implemented as a vision-language model finetuned for action-token generation.
- The official repo exposes both full finetuning and LoRA finetuning paths.
- Action prediction is framed as next-token generation over discretized action bins.

## Training configuration (reported)
- Pretrain/foundation source: finetunes Prismatic-7B on robot data mixture.
- Data mixture in paper: OXE + BridgeData V2 (MAGICAL SOUP++).
- Full training in paper: 1 epoch, about 45,000 gradient steps.
- Full training in paper: learning rate `2e-5`, global batch size `256`.
- Full finetune code path (repo): typically run with `8 x A100 80GB` in provided command template.
- LoRA finetune code path (repo): single-GPU path shown with `lora_rank=32`, `lora_dropout=0.0`, `learning_rate=5e-4`, `batch_size=16`, 4-bit loading.

## Full vs LoRA
- Full training: supported and used in the paper recipe.
- LoRA: supported in repo for cheaper adaptation; repo command defaults indicate LoRA on all linear layers.

## Gaps
- Paper does not give a full public hardware-time breakdown for every experiment.
- Some code defaults (for specific benchmark scripts) differ from paper-level summary values.
