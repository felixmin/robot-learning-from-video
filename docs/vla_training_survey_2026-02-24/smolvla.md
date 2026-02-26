# SmolVLA ("Small VLA")

## Primary sources
- Paper: https://arxiv.org/abs/2506.01844
- Paper (HTML mirror): https://ar5iv.org/html/2506.01844
- LeRobot repo: https://github.com/huggingface/lerobot
- SmolVLA model card: https://huggingface.co/lerobot/smolvla_base

## Code and implementation
- SmolVLA combines a compact VLM backbone with an action expert for robot control.
- The implementation is integrated in LeRobot and designed for smaller compute budgets than very large VLAs.

## Training configuration (reported)
- Pretraining length: `300,000` iterations.
- Pretraining batch size: `256`.
- Fine-tuning batch size (paper simulation setup): `64`.
- Learning rate schedule: base `1e-3`, warmup `1,000` steps, cosine decay.
- Weight decay: `0` (as reported in training setup).
- Hardware: paper reports the full pretraining setup runs on `4 GPUs` and can also run on `1 GPU` with a small throughput decrease.

## Full vs LoRA
- Reported main training is full pretraining/fine-tuning style.
- LoRA is not the central reported recipe in the paper.

## Gaps
- Public sources do not always provide one complete consolidated table for every downstream fine-tuning run across all tasks.
