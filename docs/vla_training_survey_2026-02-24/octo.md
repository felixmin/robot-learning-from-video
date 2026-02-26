# Octo

## Primary sources
- Paper: https://arxiv.org/abs/2405.12213
- Paper (HTML mirror): https://ar5iv.org/html/2405.12213
- Code: https://github.com/octo-models/octo
- Training docs: https://github.com/octo-models/octo/blob/main/docs/training.md

## Code and implementation
- Octo is an open-source generalist robot policy stack with both pretraining and finetuning pipelines.
- Public training docs expose concrete config knobs for pretraining/finetuning.

## Training configuration (reported)
- Paper pretraining length: around `700,000` update steps.
- Code docs default pretraining batch size: `2048`.
- Code docs LR schedule example: warmup `2000` steps to peak `3e-4`, cosine decay to `3e-5` by `500,000` steps.
- Paper discusses downstream finetuning from pretrained Octo models as the default adaptation path.

## Full vs LoRA
- Core recipe uses full pretraining and full finetuning.
- LoRA is not the main published adaptation route in core Octo docs.

## Gaps
- Hardware counts are not always centralized in one single table across all experiments.
