from __future__ import annotations

from types import SimpleNamespace

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.backends.interfaces import BackendMode, Stage2Batch
from stage2.backends.smolvla_shared.config import SmolVLASharedBackendConfig
from stage2.backends.smolvla_shared_backend import SmolVLASharedBackend
from stage2.policy_inputs import ChatConfig


class FakeTokenizer:
    def __call__(
        self,
        texts,
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ):
        assert return_tensors == "pt"
        assert truncation is True
        bsize = len(texts)
        lengths = [max(1, min(max_length, len(str(t).split()))) for t in texts]
        max_len = max(lengths)
        input_ids = torch.zeros((bsize, max_len), dtype=torch.long)
        attention_mask = torch.zeros((bsize, max_len), dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()


class FakeSmolModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, expert_hidden_size: int = 12):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=hidden_size,
                head_dim=4,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        )
        self.expert_hidden_size = expert_hidden_size
        self.processor = FakeProcessor()
        self.vlm = torch.nn.Linear(1, 1, bias=False)
        self.prefix_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.suffix_proj = torch.nn.Linear(expert_hidden_size, expert_hidden_size)
        self.prefix_to_suffix = torch.nn.Linear(hidden_size, expert_hidden_size)
        self.lang_embed = torch.nn.Embedding(2048, hidden_size)
        self.image_proj = torch.nn.Linear(3, hidden_size)

        self.train_expert_only = True
        for p in self.vlm.parameters():
            p.requires_grad = False
        self.vlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_expert_only:
            self.vlm.eval()
        return self

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, C, H, W] -> two visual tokens
        bsize = int(image.shape[0])
        pooled = (
            image.mean(dim=(2, 3)).transpose(1, 2)
            if image.ndim == 5
            else image.mean(dim=(2, 3))
        )
        pooled = pooled.to(torch.float32)
        token = self.image_proj(pooled)
        return token[:, None, :].expand(bsize, 2, -1)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lang_embed(tokens)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: dict[str, torch.Tensor] | None,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[torch.Tensor | None], dict[str, torch.Tensor] | None]:
        del attention_mask, position_ids

        prefix, suffix = inputs_embeds
        cache = {} if past_key_values is None else dict(past_key_values)

        prefix_out = None
        if prefix is not None:
            prefix_out = self.prefix_proj(prefix)
            if fill_kv_cache and use_cache:
                cache["prefix_ctx"] = prefix_out.mean(dim=1, keepdim=True)

        ctx = cache.get("prefix_ctx")
        suffix_out = None
        if suffix is not None:
            suffix_out = self.suffix_proj(suffix)
            if ctx is not None:
                suffix_out = suffix_out + self.prefix_to_suffix(ctx).to(
                    dtype=suffix_out.dtype
                )

        return [prefix_out, suffix_out], cache


def _make_backend() -> SmolVLASharedBackend:
    return SmolVLASharedBackend(
        config=SmolVLASharedBackendConfig(
            model_name="dummy",
            latent_vector_dim=8,
            action_dim=3,
            action_chunk_size=50,
            torch_dtype=torch.float32,
            trust_remote_code=False,
            chat=ChatConfig(system_prompt="sys"),
            action_tokens=ActionTokenConfig(codebook_size=8, code_seq_len=4),
            use_gpu_preprocessing=True,
            image_size=(64, 64),
            flow_hidden_dim=32,
            flow_steps=4,
            latent_loss_weight=1.0,
            action_loss_weight=1.0,
            max_state_dim=4,
            freeze_vlm=True,
        ),
        smol_model=FakeSmolModel(),
    )


def _make_batch() -> Stage2Batch:
    return Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (2, 2, 16, 16, 3), dtype=torch.uint8
            ),
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((2, 2), dtype=torch.bool),
        },
        task_text=["pick", "place"],
        target_latent_vectors=torch.randn(2, 4, 2),
        target_actions=torch.randn(2, 50, 3),
        action_is_pad=torch.zeros((2, 50), dtype=torch.bool),
        state=torch.randn(2, 2),
    )


def test_smolvla_shared_backend_latent_flow() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    batch = _make_batch()
    out = backend.loss_from_batch(batch, mode=BackendMode.LATENT_FLOW)
    assert torch.is_tensor(out.loss)

    latent = backend.latent_from_batch(batch, mode=BackendMode.LATENT_FLOW)
    assert latent.vector is not None
    assert latent.vector.shape == (2, 8)


def test_smolvla_shared_backend_actions_and_multitask() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    batch = _make_batch()

    actions_out = backend.loss_from_batch(batch, mode=BackendMode.ACTIONS)
    assert torch.is_tensor(actions_out.loss)

    both_out = backend.loss_from_batch(batch, mode=BackendMode.MULTITASK)
    assert torch.is_tensor(both_out.loss)

    latent = backend.latent_from_batch(batch, mode=BackendMode.MULTITASK)
    assert latent.vector is not None
    assert latent.actions is not None
    assert latent.vector.shape == (2, 8)
    assert latent.actions.shape == (2, 3)


def test_smolvla_shared_backend_freezes_vlm_by_default() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    vlm_params = list(backend.core.vlm.parameters())
    assert len(vlm_params) > 0
    assert all(not p.requires_grad for p in vlm_params)
    assert backend.core.vlm.training is False

    backend.train()
    assert backend.core.vlm.training is False


def test_action_flow_loss_mean_uses_global_masked_mean() -> None:
    backend = _make_backend()
    backend.setup(device=torch.device("cpu"))

    b, t, a = 2, 50, 3
    target_actions = torch.zeros((b, t, a), dtype=torch.float32)
    target_actions[0] = 1.0
    target_actions[1] = 2.0
    mask = torch.zeros((b, t), dtype=torch.bool)
    mask[0, 1:] = True  # sample 0: only 1 valid step; sample 1: all 50 valid steps

    batch = Stage2Batch(
        image_streams={
            "observation.images.rgb": torch.randint(
                0, 256, (b, 2, 16, 16, 3), dtype=torch.uint8
            )
        },
        image_padding_masks={
            "observation.images.rgb": torch.ones((b, 2), dtype=torch.bool)
        },
        task_text=["pick", "place"],
        state=torch.randn(b, 2),
        action_is_pad=mask,
    )

    noise = torch.randn_like(target_actions)
    time = torch.full((b,), 0.3, dtype=torch.float32)
    per_sample = backend.core.action_flow_loss(
        batch=batch,
        target_actions=target_actions,
        action_is_pad=mask,
        noise=noise,
        time=time,
        reduction="none",
    )
    mean_loss = backend.core.action_flow_loss(
        batch=batch,
        target_actions=target_actions,
        action_is_pad=mask,
        noise=noise,
        time=time,
        reduction="mean",
    )

    valid_counts = (~mask).sum(dim=1).to(dtype=per_sample.dtype) * a
    expected = (per_sample * valid_counts).sum() / valid_counts.sum()
    assert torch.allclose(mean_loss, expected, atol=1e-6)
