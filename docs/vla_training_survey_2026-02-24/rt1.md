# RT-1

## Primary sources
- Paper: https://arxiv.org/abs/2212.06817
- Paper (HTML mirror): https://ar5iv.org/html/2212.06817

## Code and implementation
- RT-1 is a transformer policy over tokenized image + language + action streams.
- It was trained as a large-scale multitask robot manipulation policy.

## Training configuration (reported)
- Data scale: about `130,000` episodes over more than `700` tasks from `13` robots.
- Model size: about `100M` parameters.
- Batch size: `1024`.

## Full vs LoRA
- Full model training is the reported approach.
- LoRA is not part of the original RT-1 training recipe.

## Gaps
- Public paper does not provide a complete low-level optimizer run-card (exact LR schedule + total steps + hardware count) for all experiments.
