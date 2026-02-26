# NVIDIA GR00T (N1 / N1.5)

## Primary sources
- GR00T N1 paper: https://arxiv.org/abs/2503.14734
- Paper (HTML mirror): https://ar5iv.org/html/2503.14734
- NVIDIA Isaac GR00T code: https://github.com/NVIDIA/Isaac-GR00T
- LoRA finetune config (raw): https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/examples/configs/finetune/gr00t_n1_5_lora.yaml
- Full finetune config (raw): https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/examples/configs/finetune/gr00t_n1_5_full_ft.yaml

## Code and implementation
- Isaac-GR00T exposes both LoRA and full-finetune config paths.
- Finetuning configs are Hydra YAMLs with explicit optimizer, batching, and adaptation settings.

## Training configuration (reported in public configs)
- Node/GPU config: `num_nodes=1`, `num_gpus=8`.
- Epochs: `max_epochs=50`.
- Batch config: `global_batch_size=256`, `micro_batch_size=4`.
- Optimizer: `lr=5e-6`, `weight_decay=1e-2`, `warmup_steps=2000`.
- Precision: bf16 enabled in LoRA config.
- LoRA config: `enable_lora=true`, `lora_rank=32`, `lora_alpha=64`, `lora_dropout=0.05`.
- Encoder freezing in LoRA config: vision and LLM encoders frozen.
- Full finetune config: inherits LoRA config but sets `freeze_vision_encoder=false` and `enable_lora=false`.

## Full vs LoRA
- Both are first-class in official configs.
- LoRA is the explicitly provided lightweight adaptation path.

## Gaps
- Technical report and public materials do not provide one complete, standardized paper run-card for every GR00T training stage.
