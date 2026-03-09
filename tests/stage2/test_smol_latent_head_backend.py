from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.backends.interfaces import BackendMode, Stage2Batch
from stage2.legacy.backends.smol_latent_head_backend import (
    SmolFlowActionBackend,
    SmolFlowActionBackendConfig,
    SmolLatentHeadBackend,
    SmolLatentHeadBackendConfig,
)
from stage2.policy_inputs import ChatConfig


class FakeProcessor:
    def apply_chat_template(
        self, messages, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is False
        parts: List[str] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    parts.append("<image>")
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join([p for p in parts if p])

    def __call__(self, *, text, images, return_tensors: str, padding: bool):
        assert return_tensors == "pt"
        assert padding is True
        assert isinstance(text, list)
        assert len(text) == len(images)
        assert all(isinstance(x, list) and len(x) == 1 for x in images)
        b = len(text)
        # Pretend tokenized length depends on word count.
        lengths = [max(1, len(str(t).split())) for t in text]
        max_len = max(lengths)
        input_ids = torch.zeros((b, max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, seq_len in enumerate(lengths):
            input_ids[i, :seq_len] = torch.arange(1, seq_len + 1, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        pixel_values = torch.zeros((b, 3, 8, 8), dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


class DummyVLM(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )
        self.proj = torch.nn.Linear(1, hidden_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        assert output_hidden_states is True
        assert return_dict is True
        batch_size, seq_len = input_ids.shape
        x = torch.ones(
            (batch_size, seq_len, 1),
            dtype=torch.float32,
            device=input_ids.device,
        )
        last = self.proj(x)
        # mimic HF: tuple of hidden states, last at [-1]
        return SimpleNamespace(hidden_states=(last,))


class DummyVLMWithBf16Hidden(DummyVLM):
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        out = super().forward(
            input_ids,
            attention_mask,
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last = out.hidden_states[-1].to(torch.bfloat16)
        return SimpleNamespace(hidden_states=(last,))


def test_smol_latent_head_backend_loss_and_latents():
    vlm = DummyVLM(hidden_size=16)
    backend = SmolLatentHeadBackend(
        config=SmolLatentHeadBackendConfig(
            model_name="dummy",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            chat=ChatConfig(system_prompt="sys"),
            action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
            use_gpu_preprocessing=False,
        ),
        vlm=vlm,
        processor=FakeProcessor(),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )
    backend.setup(device=torch.device("cpu"))

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 8, 8, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["pick", "place"],
        target_codes=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
    )
    out = backend.loss_from_batch(batch, mode=BackendMode.CODES)
    assert torch.is_tensor(out.loss)

    latent = backend.latent_from_batch(batch, mode=BackendMode.CODES)
    assert latent.logits is not None
    assert latent.tokens is not None
    assert latent.logits.shape == (2, 4, 8)
    assert latent.tokens.shape == (2, 4)


def test_smol_latent_head_backend_handles_dtype_mismatch():
    vlm = DummyVLMWithBf16Hidden(hidden_size=16)
    backend = SmolLatentHeadBackend(
        config=SmolLatentHeadBackendConfig(
            model_name="dummy",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            chat=ChatConfig(system_prompt="sys"),
            action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
            use_gpu_preprocessing=False,
        ),
        vlm=vlm,
        processor=FakeProcessor(),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )
    backend.setup(device=torch.device("cpu"))

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 8, 8, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["pick", "place"],
        target_codes=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
    )
    latent = backend.latent_from_batch(batch, mode=BackendMode.CODES)
    assert latent.tokens is not None


def test_smol_flow_action_backend_multitask():
    vlm = DummyVLM(hidden_size=16)
    backend = SmolFlowActionBackend(
        config=SmolFlowActionBackendConfig(
            model_name="dummy",
            latent_vector_dim=8,
            action_dim=3,
            torch_dtype=torch.float32,
            trust_remote_code=False,
            chat=ChatConfig(system_prompt="sys"),
            action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
            use_gpu_preprocessing=False,
            flow_hidden_dim=32,
            flow_steps=4,
            latent_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
        vlm=vlm,
        processor=FakeProcessor(),
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )
    backend.setup(device=torch.device("cpu"))

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 8, 8, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["pick", "place"],
        target_latent_vectors=torch.randn(2, 4, 2),
        target_actions=torch.randn(2, 3),
    )
    out = backend.loss_from_batch(batch, mode=BackendMode.MULTITASK)
    assert torch.is_tensor(out.loss)
    latent = backend.latent_from_batch(batch, mode=BackendMode.MULTITASK)
    assert latent.vector is not None
    assert latent.actions is not None
    assert latent.vector.shape == (2, 8)
    assert latent.actions.shape == (2, 3)
