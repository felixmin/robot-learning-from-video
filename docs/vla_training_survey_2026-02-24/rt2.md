# RT-2

## Primary sources
- Paper: https://arxiv.org/abs/2307.15818
- Paper (HTML mirror): https://ar5iv.org/html/2307.15818

## Code and implementation
- RT-2 casts robot control as VLM token generation with actions represented as text tokens.
- Core method is co-fine-tuning a pretrained vision-language model on robot trajectories plus web-scale vision-language data.

## Training configuration (reported)
- Training method is clearly disclosed (co-fine-tuning with mixed robot + web data).
- Exact low-level hyperparameters (single canonical LR, global batch size, total steps, and hardware count) are not comprehensively disclosed in one public config card.

## Full vs LoRA
- Reported method is full co-fine-tuning.
- LoRA is not the headline method in the original RT-2 recipe.

## Gaps
- The paper focuses on methodology and capability gains; full reproducibility-grade run settings are only partially available.
