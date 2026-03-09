from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.constrained_decode import ActionTokenIds
from stage2.policy_inputs import ChatConfig
from stage2.legacy.policy_module_legacy import (
    PolicyTokenLightningModule,
    PolicyOptimizerConfig,
)


class DummyCodeProvider:
    codebook_size = 8
    code_seq_len = 4

    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        b = video.shape[0]
        return torch.tensor([[3, 1, 7, 0]] * b, dtype=torch.long)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=2)

    def apply_chat_template(
        self, messages, tokenize: bool, add_generation_prompt: bool
    ):
        parts: List[str] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("<image>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join([p for p in parts if p])

    def __call__(self, *, text, images, return_tensors: str, padding: bool):
        assert return_tensors == "pt"
        lengths = [len(t.split()) for t in text]
        max_len = max(lengths)
        input_ids = torch.zeros((len(text), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyPolicyModelWithGenerate(torch.nn.Module):
    def __init__(self, token_ids: ActionTokenIds):
        super().__init__()
        self.token_ids = token_ids

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        # Just return a finite loss.
        return SimpleNamespace(loss=input_ids.float().mean())

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        do_sample,
        prefix_allowed_tokens_fn,
        **kwargs,
    ):
        b, t = input_ids.shape
        # Emit the correct sequence for every sample.
        sep = (
            self.token_ids.between_token_ids[0]
            if self.token_ids.between_token_ids
            else None
        )
        suffix = torch.tensor(
            [
                self.token_ids.action_start_id,
                *([sep] if sep is not None else []),
                self.token_ids.action_code_ids[3],
                *([sep] if sep is not None else []),
                self.token_ids.action_code_ids[1],
                *([sep] if sep is not None else []),
                self.token_ids.action_code_ids[7],
                *([sep] if sep is not None else []),
                self.token_ids.action_code_ids[0],
                *([sep] if sep is not None else []),
                self.token_ids.action_end_id,
            ],
            dtype=torch.long,
        )
        suffix = suffix.unsqueeze(0).repeat(b, 1)
        return torch.cat([input_ids, suffix], dim=1)


def test_validation_step_logs_token_accuracy_without_error():
    token_ids = ActionTokenIds(
        action_start_id=10,
        action_end_id=11,
        action_code_ids=list(range(20, 28)),
        between_token_ids=[99],
        eos_token_id=2,
        code_seq_len=4,
    )

    module = PolicyTokenLightningModule(
        policy_model=DummyPolicyModelWithGenerate(token_ids),
        processor=FakeProcessor(),
        code_provider=DummyCodeProvider(),  # type: ignore[arg-type]
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
        chat=ChatConfig(system_prompt="You are a robot."),
        optimizer=PolicyOptimizerConfig(lr=1e-4, weight_decay=0.0),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
        action_token_ids=token_ids,
    )

    batch = {
        "frames": torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8),
        "language": ["pick up block", "push button"],
    }
    loss = module.validation_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert hasattr(module, "_last_val_sample")
    assert module._last_val_sample is not None
    assert module._last_val_sample["gt_codes"][0] == [3, 1, 7, 0]
    assert module._last_val_sample["pred_codes"][0] == [3, 1, 7, 0]

    pred = module._predict_codes(frames=batch["frames"], instructions=batch["language"])
    metrics = module._compute_generation_metrics(
        gt_codes=torch.tensor([[3, 1, 7, 0], [3, 1, 7, 0]], dtype=torch.long),
        pred_codes=pred,
    )
    assert float(metrics["token_accuracy"].cpu().item()) == 1.0
    assert float(metrics["sequence_accuracy"].cpu().item()) == 1.0


def test_predict_codes_slices_generated_suffix_after_padded_prompt():
    token_ids = ActionTokenIds(
        action_start_id=10,
        action_end_id=11,
        action_code_ids=list(range(20, 28)),
        between_token_ids=[99],
        eos_token_id=2,
        code_seq_len=4,
    )

    module = PolicyTokenLightningModule(
        policy_model=DummyPolicyModelWithGenerate(token_ids),
        processor=FakeProcessor(),
        code_provider=DummyCodeProvider(),  # type: ignore[arg-type]
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
        chat=ChatConfig(system_prompt="You are a robot."),
        optimizer=PolicyOptimizerConfig(lr=1e-4, weight_decay=0.0),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
        action_token_ids=token_ids,
    )

    frames = torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8)
    # Different prompt lengths => padding in the prompt batch.
    instructions = ["a b c d", "x"]

    _pred_codes, debug = module._predict_codes_with_debug(
        frames=frames, instructions=instructions
    )
    assert len(debug) == 2
    for rec in debug:
        suffix = rec.get("generated_suffix_ids")
        assert (
            isinstance(suffix, list) and suffix
        ), "missing generated_suffix_ids in debug"
        assert suffix[0] == token_ids.action_start_id
        assert rec.get("has_action_start") is True
        assert rec.get("has_action_end") is True
        assert rec.get("prompt_padded_len") is not None
        assert rec.get("prompt_true_len") is not None
