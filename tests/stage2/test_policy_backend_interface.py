from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.backends.interfaces import BackendMode, Stage2Batch
from stage2.legacy.backends.qwen3vl_chat_backend import (
    Qwen3VLChatActionTokenBackend,
    Qwen3VLChatBackendConfig,
)
from stage2.policy_inputs import ChatConfig


class FakeTokenizer:
    """
    Minimal tokenizer stub for backend tests.

    Requirements:
    - add_special_tokens({"additional_special_tokens": [...]})
    - convert_tokens_to_ids(token_str)
    - encode(text, add_special_tokens=...)
    - eos_token_id
    - __len__

    Note: encode() injects a dedicated `space_token_id` between space-separated tokens
    so `_infer_between_action_token_ids` can discover `between_token_ids`.
    """

    def __init__(self):
        self.eos_token_id = 2
        self.unk_token_id = 1
        self._next_id = 10
        self._tok_to_id: dict[str, int] = {}
        self.space_token_id = 99

    def __len__(self) -> int:
        return max([self._next_id, self.space_token_id + 1, self.eos_token_id + 1])

    def add_special_tokens(self, special_tokens_dict: dict[str, list[str]]) -> None:
        toks = list(special_tokens_dict.get("additional_special_tokens", []))
        for t in toks:
            if t in self._tok_to_id:
                continue
            self._tok_to_id[t] = self._next_id
            self._next_id += 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._tok_to_id.get(token, self.unk_token_id))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        parts = [p for p in text.split(" ") if p]
        out: list[int] = []
        for i, p in enumerate(parts):
            if i > 0:
                out.append(int(self.space_token_id))
            out.append(int(self.convert_tokens_to_ids(p)))
        return out

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        # Only used for debug; keep it simple.
        inv = {v: k for k, v in self._tok_to_id.items()}
        out = []
        for tid in token_ids:
            if tid == self.space_token_id:
                out.append("<space>")
            else:
                out.append(inv.get(int(tid), f"<id:{tid}>"))
        return " ".join(out)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

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
        lengths = [len(str(t).split()) for t in text]
        max_len = max(lengths) if lengths else 0
        input_ids = torch.zeros((len(text), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(lengths):
            if length <= 0:
                continue
            input_ids[i, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyPolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_called_with: int | None = None
        self._suffix_ids: list[int] | None = None

    def resize_token_embeddings(self, n: int):
        self.resize_called_with = int(n)
        return None

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        # Basic sanity checks to ensure prompt masking happened.
        assert labels.shape == input_ids.shape
        assert torch.all(labels[attention_mask == 0] == -100)
        assert torch.any(labels[attention_mask == 1] != -100)
        loss = input_ids.float().mean()
        return SimpleNamespace(loss=loss)

    def set_suffix_ids(self, suffix_ids: list[int]) -> None:
        self._suffix_ids = list(map(int, suffix_ids))

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        do_sample,
        prefix_allowed_tokens_fn,
        **kwargs,
    ):
        assert (
            self._suffix_ids is not None
        ), "test must set suffix ids after backend.setup()"
        b = int(input_ids.shape[0])
        suffix = (
            torch.tensor(self._suffix_ids, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        )
        return torch.cat([input_ids, suffix], dim=1)


def _make_backend() -> (
    tuple[Qwen3VLChatActionTokenBackend, DummyPolicyModel, FakeProcessor]
):
    model = DummyPolicyModel()
    processor = FakeProcessor()
    cfg = Qwen3VLChatBackendConfig(
        model_name="dummy",
        torch_dtype=torch.float32,
        attn_implementation=None,
        chat=ChatConfig(system_prompt="You are a robot."),
        action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
    )
    backend = Qwen3VLChatActionTokenBackend(
        config=cfg,
        policy_model=model,
        processor=processor,
        frames_to_images=lambda frames: [object() for _ in range(frames.shape[0])],
    )
    return backend, model, processor


def test_backend_setup_adds_tokens_and_infers_between_ids():
    backend, model, processor = _make_backend()
    backend.setup(device=torch.device("cpu"))
    assert model.resize_called_with is not None

    token_ids = backend._action_token_ids  # noqa: SLF001 (test-only)
    assert token_ids is not None
    assert token_ids.code_seq_len == 4
    assert len(token_ids.action_code_ids) == 8
    assert (
        token_ids.between_token_ids
    ), "expected non-empty between_token_ids in test tokenizer"

    # Build a suffix matching the expected parsed code sequence [3,1,7,0]
    tok = processor.tokenizer
    space = int(tok.space_token_id)
    suffix = [
        int(token_ids.action_start_id),
        space,
        int(token_ids.action_code_ids[3]),
        space,
        int(token_ids.action_code_ids[1]),
        space,
        int(token_ids.action_code_ids[7]),
        space,
        int(token_ids.action_code_ids[0]),
        space,
        int(token_ids.action_end_id),
    ]
    model.set_suffix_ids(suffix)

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["pick up block", "push button"],
    )
    out = backend.latent_from_batch(batch, mode=BackendMode.CODES)
    assert out.tokens is not None
    assert out.tokens.tolist() == [[3, 1, 7, 0], [3, 1, 7, 0]]


def test_backend_loss_from_batch_requires_target_codes():
    backend, model, processor = _make_backend()
    backend.setup(device=torch.device("cpu"))

    token_ids = backend._action_token_ids  # noqa: SLF001
    assert token_ids is not None
    tok = processor.tokenizer
    space = int(tok.space_token_id)
    model.set_suffix_ids(
        [
            int(token_ids.action_start_id),
            space,
            int(token_ids.action_code_ids[3]),
            space,
            int(token_ids.action_code_ids[1]),
            space,
            int(token_ids.action_code_ids[7]),
            space,
            int(token_ids.action_code_ids[0]),
            space,
            int(token_ids.action_end_id),
        ]
    )

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["a", "b"],
        target_codes=torch.tensor([[3, 1, 7, 0], [3, 1, 7, 0]], dtype=torch.long),
    )
    out = backend.loss_from_batch(batch, mode=BackendMode.CODES)
    assert torch.is_tensor(out.loss)


def test_backend_loss_raises_without_target_codes():
    backend, _model, _processor = _make_backend()
    backend.setup(device=torch.device("cpu"))

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool)
        },
        task_text=["a", "b"],
        target_codes=None,
    )
    try:
        backend.loss_from_batch(batch, mode=BackendMode.CODES)
    except ValueError as e:
        assert "target_codes" in str(e)
    else:
        raise AssertionError("Expected ValueError when target_codes is missing")
