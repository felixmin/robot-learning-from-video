from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.policy_inputs import ChatConfig
from stage2.legacy.policy_module_legacy import (
    PolicyTokenLightningModule,
    PolicyOptimizerConfig,
)


class DummyCodeProvider:
    codebook_size = 8
    code_seq_len = 4

    def __init__(self):
        self.last_video_shape = None

    def codes_from_video(self, video: torch.Tensor) -> torch.Tensor:
        self.last_video_shape = tuple(video.shape)
        b = video.shape[0]
        return torch.tensor([[3, 1, 7, 0]] * b, dtype=torch.long)


class FakeProcessor:
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
        assert padding is True
        lengths = [len(t.split()) for t in text]
        max_len = max(lengths)
        input_ids = torch.zeros((len(text), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyPolicyModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        assert labels.shape == input_ids.shape
        # Padding should be masked.
        assert torch.all(labels[attention_mask == 0] == -100)
        # There should be at least one supervised token.
        assert torch.any(labels[attention_mask == 1] != -100)
        loss = input_ids.float().mean()
        return SimpleNamespace(loss=loss)


def test_policy_module_training_step_cpu_smoke():
    provider = DummyCodeProvider()
    module = PolicyTokenLightningModule(
        policy_model=DummyPolicyModel(),
        processor=FakeProcessor(),
        code_provider=provider,  # type: ignore[arg-type]
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
        chat=ChatConfig(system_prompt="You are a robot."),
        optimizer=PolicyOptimizerConfig(lr=1e-4, weight_decay=0.0),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )

    # Simulate a temporal-frame dict batch: frames [B, 2, H, W, 3] uint8, language as list[str]
    batch = {
        "frames": torch.randint(0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8),
        "language": ["pick up block", "push button"],
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert provider.last_video_shape == (2, 3, 2, 16, 16)
