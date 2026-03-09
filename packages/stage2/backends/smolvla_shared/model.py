from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from stage2.backends.interfaces import Stage2Batch
from stage2.backends.smolvla_shared.config import SmolVLASharedCoreConfig
from stage2.backends.smolvla_shared.flow import (
    create_sinusoidal_pos_embedding,
    make_noisy_target,
    sample_beta_time,
)
from stage2.backends.smolvla_shared.input_transform import (
    ImageTransformConfig,
    LanguageTransformConfig,
    prepare_image_inputs,
    prepare_language_inputs,
    prepare_state_tensor,
    resolve_action_pad_mask,
    to_action_chunk,
)
from stage2.backends.smolvla_shared.preprocess import pad_vector
from stage2.backends.smolvla_shared.smolvlm_with_expert import SmolVLMWithExpertModel


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_tensor(
    tensor: torch.Tensor, max_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


class SmolVLASharedCore(torch.nn.Module):
    """SmolVLA-style expert backbone with latent-flow output and optional real-action output."""

    def __init__(
        self,
        *,
        config: SmolVLASharedCoreConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        smol_model: SmolVLMWithExpertModel | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)
        if self.code_seq_len <= 0:
            raise ValueError(
                f"action_tokens.code_seq_len must be > 0, got {self.code_seq_len}"
            )
        self.action_chunk_size = int(self.cfg.action_chunk_size)
        if self.action_chunk_size <= 0:
            raise ValueError(
                f"action_chunk_size must be > 0, got {self.action_chunk_size}"
            )
        if int(self.cfg.latent_vector_dim) % self.code_seq_len != 0:
            raise ValueError(
                "latent_vector_dim must be divisible by action_tokens.code_seq_len, "
                f"got latent_vector_dim={self.cfg.latent_vector_dim} code_seq_len={self.code_seq_len}"
            )
        self.latent_step_dim = int(self.cfg.latent_vector_dim) // self.code_seq_len

        self.smol_model: SmolVLMWithExpertModel | None = smol_model
        self.vlm: torch.nn.Module | None = None
        self.processor: Any | None = None
        self._injected_vlm = vlm
        self._injected_processor = processor

        self.state_proj: torch.nn.Linear | None = None
        self.action_in_proj: torch.nn.Linear | None = (
            None  # latent-domain input projection
        )
        self.real_action_in_proj: torch.nn.Linear | None = None
        self.action_time_mlp_in: torch.nn.Linear | None = None
        self.action_time_mlp_out: torch.nn.Linear | None = None
        self.latent_out_proj: torch.nn.Linear | None = None
        self.action_out_proj: torch.nn.Linear | None = None

        self.global_image_start_token: torch.Tensor | None = None
        self.image_end_token: torch.Tensor | None = None

    def setup(self, *, device: torch.device) -> None:
        if self.smol_model is None:
            self.smol_model = SmolVLMWithExpertModel(
                model_id=self.cfg.model_name,
                load_vlm_weights=bool(self.cfg.load_vlm_weights),
                train_expert_only=bool(self.cfg.freeze_vlm),
                freeze_vision_encoder=bool(self.cfg.freeze_vision_encoder),
                attention_mode=str(self.cfg.attention_mode),
                num_expert_layers=int(self.cfg.num_expert_layers),
                num_vlm_layers=int(self.cfg.num_vlm_layers),
                self_attn_every_n_layers=int(self.cfg.self_attn_every_n_layers),
                expert_width_multiplier=float(self.cfg.expert_width_multiplier),
                torch_dtype=self.cfg.torch_dtype,
                trust_remote_code=bool(self.cfg.trust_remote_code),
                vlm=self._injected_vlm,
                processor=self._injected_processor,
            )

        self.smol_model.to(device=device, dtype=self.cfg.torch_dtype)
        self.vlm = self.smol_model.vlm
        self.processor = self.smol_model.processor

        text_hidden = int(self.smol_model.config.text_config.hidden_size)
        expert_hidden = int(self.smol_model.expert_hidden_size)
        model_dtype = next(self.smol_model.parameters()).dtype

        if self.state_proj is None:
            self.state_proj = torch.nn.Linear(
                int(self.cfg.max_state_dim), text_hidden
            ).to(
                device=device,
                dtype=model_dtype,
            )
            self.action_in_proj = torch.nn.Linear(
                self.latent_step_dim, expert_hidden
            ).to(
                device=device,
                dtype=model_dtype,
            )
            self.action_time_mlp_in = torch.nn.Linear(
                expert_hidden * 2, expert_hidden
            ).to(
                device=device,
                dtype=model_dtype,
            )
            self.action_time_mlp_out = torch.nn.Linear(expert_hidden, expert_hidden).to(
                device=device,
                dtype=model_dtype,
            )
            self.latent_out_proj = torch.nn.Linear(
                expert_hidden, self.latent_step_dim
            ).to(
                device=device,
                dtype=model_dtype,
            )
            self.real_action_in_proj = torch.nn.Linear(
                int(self.cfg.action_dim), expert_hidden
            ).to(
                device=device,
                dtype=model_dtype,
            )
            self.action_out_proj = torch.nn.Linear(
                expert_hidden, int(self.cfg.action_dim)
            ).to(
                device=device,
                dtype=model_dtype,
            )

        if self.cfg.add_image_special_tokens:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(
                    "processor.tokenizer is required when add_image_special_tokens=true"
                )
            fake_image_token = getattr(tokenizer, "fake_image_token_id", None)
            global_image_token = getattr(tokenizer, "global_image_token_id", None)
            if fake_image_token is None or global_image_token is None:
                raise RuntimeError(
                    "processor tokenizer must expose fake_image_token_id and global_image_token_id "
                    "when add_image_special_tokens=true"
                )
            self.global_image_start_token = torch.tensor(
                [int(fake_image_token), int(global_image_token)],
                dtype=torch.long,
                device=device,
            )
            self.image_end_token = torch.tensor(
                [int(fake_image_token)], dtype=torch.long, device=device
            )

    def train(self, mode: bool = True) -> SmolVLASharedCore:
        super().train(mode)
        if self.smol_model is not None:
            self.smol_model.train(mode)
        return self

    def _require_ready(self) -> tuple[torch.device, SmolVLMWithExpertModel, Any]:
        if self.smol_model is None or self.processor is None:
            raise RuntimeError("Core not initialized. Call setup(device=...) first.")
        device = next(self.smol_model.parameters()).device
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("processor.tokenizer is required")
        return device, self.smol_model, tokenizer

    def _require_heads(
        self,
    ) -> tuple[
        torch.nn.Linear,
        torch.nn.Linear,
        torch.nn.Linear,
        torch.nn.Linear,
        torch.nn.Linear,
    ]:
        if (
            self.state_proj is None
            or self.action_in_proj is None
            or self.action_time_mlp_in is None
            or self.action_time_mlp_out is None
            or self.latent_out_proj is None
        ):
            raise RuntimeError(
                "Core projection heads are not initialized. Call setup(device=...) first."
            )
        return (
            self.state_proj,
            self.action_in_proj,
            self.action_time_mlp_in,
            self.action_time_mlp_out,
            self.latent_out_proj,
        )

    def _require_action_heads(
        self,
    ) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        if (
            self.real_action_in_proj is None
            or self.action_time_mlp_in is None
            or self.action_time_mlp_out is None
            or self.action_out_proj is None
        ):
            raise RuntimeError(
                "Action projection heads are not initialized. Set action_dim and call setup(device=...)."
            )
        return (
            self.real_action_in_proj,
            self.action_time_mlp_in,
            self.action_time_mlp_out,
            self.action_out_proj,
        )

    def _prepare_images(
        self,
        *,
        batch: Stage2Batch,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return prepare_image_inputs(
            batch=batch,
            device=device,
            config=ImageTransformConfig(
                image_size=tuple(self.cfg.image_size),
                normalize_to_neg_one_to_one=True,
                camera_keys=self.cfg.camera_keys,
                empty_cameras=int(self.cfg.empty_cameras),
            ),
        )

    def _prepare_language(
        self,
        *,
        batch: Stage2Batch,
        tokenizer: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_language_inputs(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
            config=LanguageTransformConfig(
                system_prompt=self.cfg.chat.system_prompt,
                tokenizer_max_length=int(self.cfg.tokenizer_max_length),
                pad_language_to=str(self.cfg.pad_language_to),
            ),
        )

    def _prepare_state(
        self,
        *,
        batch: Stage2Batch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return prepare_state_tensor(
            batch=batch,
            device=device,
            dtype=dtype,
            max_state_dim=int(self.cfg.max_state_dim),
        )

    def _to_latent_sequence(
        self, x: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 2:
            if int(x.shape[1]) != int(self.cfg.latent_vector_dim):
                raise ValueError(
                    "Expected latent tensor [B, latent_vector_dim], "
                    f"got shape {tuple(x.shape)} and latent_vector_dim={self.cfg.latent_vector_dim}"
                )
            return x.reshape(x.shape[0], self.code_seq_len, self.latent_step_dim)
        if x.ndim == 3:
            if (
                int(x.shape[1]) != self.code_seq_len
                or int(x.shape[2]) != self.latent_step_dim
            ):
                raise ValueError(
                    f"Expected latent tensor [B,{self.code_seq_len},{self.latent_step_dim}], got {tuple(x.shape)}"
                )
            return x
        raise ValueError(f"Expected latent tensor rank 2 or 3, got rank={x.ndim}")

    def _flatten_latent_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq.reshape(x_seq.shape[0], -1)

    def _to_action_sequence(
        self,
        x: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = x.to(device=device, dtype=dtype)
        x = to_action_chunk(actions=x, chunk_size=int(self.action_chunk_size))
        if x.ndim != 3:
            raise ValueError(
                f"Expected action tensor [B,T,A], got shape {tuple(x.shape)}"
            )
        if int(x.shape[-1]) > int(self.cfg.action_dim):
            raise ValueError(
                f"Action dim exceeds configured action_dim={int(self.cfg.action_dim)}: got {int(x.shape[-1])}"
            )
        return pad_vector(x, int(self.cfg.action_dim))

    def embed_prefix(
        self,
        *,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_proj, *_ = self._require_heads()
        _, smol_model, _ = self._require_ready()

        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.cfg.add_image_special_tokens:
                if self.global_image_start_token is None:
                    raise RuntimeError("global_image_start_token is not initialized")
                image_start_token = (
                    smol_model.embed_language_tokens(self.global_image_start_token)
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones(
                    (img.shape[0], image_start_token.shape[1]),
                    dtype=torch.bool,
                    device=img.device,
                )
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                att_masks += [0] * int(image_start_mask.shape[1])

            img_emb = smol_model.embed_image(img)
            img_emb = img_emb * torch.tensor(
                img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.cfg.add_image_special_tokens:
                if self.image_end_token is None:
                    raise RuntimeError("image_end_token is not initialized")
                image_end_token = (
                    smol_model.embed_language_tokens(self.image_end_token)
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones(
                    (img.shape[0], image_end_token.shape[1]),
                    dtype=torch.bool,
                    device=img.device,
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * int(image_end_mask.shape[1])

        lang_emb = smol_model.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * int(lang_emb.shape[1])

        state_emb = state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        states_seq_len = int(state_emb.shape[1])
        state_mask = torch.ones(
            (state_emb.shape[0], states_seq_len),
            dtype=torch.bool,
            device=state_emb.device,
        )
        embs.append(state_emb)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        out_embs = torch.cat(embs, dim=1)
        out_pad_masks = torch.cat(pad_masks, dim=1)
        out_att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=out_pad_masks.device
        )[None, :]

        prefix_length = int(self.cfg.prefix_length)
        if prefix_length > 0 and out_pad_masks.shape[1] < prefix_length:
            out_embs = pad_tensor(out_embs, prefix_length, pad_value=0)
            out_pad_masks = pad_tensor(out_pad_masks, prefix_length, pad_value=0)
            out_att_masks = pad_tensor(out_att_masks, prefix_length, pad_value=0)

        out_att_masks = out_att_masks.expand(out_pad_masks.shape[0], -1)
        return out_embs, out_pad_masks, out_att_masks

    def embed_suffix(
        self,
        *,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, action_in_proj, action_time_mlp_in, action_time_mlp_out, _ = (
            self._require_heads()
        )

        action_emb = action_in_proj(noisy_latents)
        bsize = action_emb.shape[0]
        device = action_emb.device
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            int(action_emb.shape[-1]),
            float(self.cfg.min_period),
            float(self.cfg.max_period),
            device=device,
        ).to(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = action_time_mlp_out(action_time_emb)

        action_mask = torch.ones(
            (bsize, action_time_emb.shape[1]), dtype=torch.bool, device=device
        )
        att_masks = torch.ones(
            (bsize, action_time_emb.shape[1]), dtype=torch.bool, device=device
        )
        return action_time_emb, action_mask, att_masks

    def embed_action_suffix(
        self,
        *,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_in_proj, action_time_mlp_in, action_time_mlp_out, _ = (
            self._require_action_heads()
        )

        action_emb = action_in_proj(noisy_actions)
        bsize = action_emb.shape[0]
        device = action_emb.device
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            int(action_emb.shape[-1]),
            float(self.cfg.min_period),
            float(self.cfg.max_period),
            device=device,
        ).to(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = action_time_mlp_out(action_time_emb)

        action_mask = torch.ones(
            (bsize, action_time_emb.shape[1]), dtype=torch.bool, device=device
        )
        att_masks = torch.ones(
            (bsize, action_time_emb.shape[1]), dtype=torch.bool, device=device
        )
        return action_time_emb, action_mask, att_masks

    def _predict_from_suffix(
        self,
        *,
        suffix_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, _, _, _, latent_out_proj = self._require_heads()
        seq_len = int(suffix_out.shape[1])

        latent_in = suffix_out
        if latent_in.dtype != latent_out_proj.weight.dtype:
            latent_in = latent_in.to(latent_out_proj.weight.dtype)
        latent_seq = latent_out_proj(latent_in)

        action_seq: torch.Tensor | None = None
        if self.action_out_proj is not None:
            action_in = suffix_out
            if action_in.dtype != self.action_out_proj.weight.dtype:
                action_in = action_in.to(self.action_out_proj.weight.dtype)
            action_seq = self.action_out_proj(action_in)

        return latent_seq[:, -seq_len:], (
            action_seq[:, -seq_len:] if action_seq is not None else None
        )

    def _predict_actions_from_suffix(
        self,
        *,
        suffix_out: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, action_out_proj = self._require_action_heads()
        seq_len = int(suffix_out.shape[1])
        action_in = suffix_out
        if action_in.dtype != action_out_proj.weight.dtype:
            action_in = action_in.to(dtype=action_out_proj.weight.dtype)
        action_seq = action_out_proj(action_in)
        return action_seq[:, -seq_len:]

    def _forward_expert(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, smol_model, _ = self._require_ready()

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = smol_model.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -suffix_embs.shape[1] :].to(dtype=torch.float32)
        return self._predict_from_suffix(suffix_out=suffix_out)

    def _predict_latent_and_actions(
        self,
        *,
        batch: Stage2Batch,
        x_t_seq: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device, _, tokenizer = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        images, img_masks = self._prepare_images(batch=batch, device=device)
        lang_tokens, lang_masks = self._prepare_language(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
        )
        state = self._prepare_state(batch=batch, device=device, dtype=model_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            noisy_latents=x_t_seq,
            timestep=time,
        )
        return self._forward_expert(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_embs=suffix_embs,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
        )

    def _predict_action_velocity(
        self,
        *,
        batch: Stage2Batch,
        x_t_seq: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        device, _, tokenizer = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        images, img_masks = self._prepare_images(batch=batch, device=device)
        lang_tokens, lang_masks = self._prepare_language(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
        )
        state = self._prepare_state(batch=batch, device=device, dtype=model_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_suffix(
            noisy_actions=x_t_seq,
            timestep=time,
        )
        _, smol_model, _ = self._require_ready()
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = smol_model.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -suffix_embs.shape[1] :].to(dtype=torch.float32)
        return self._predict_actions_from_suffix(suffix_out=suffix_out)

    def latent_flow_loss(
        self,
        *,
        batch: Stage2Batch,
        target_vectors: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device, _, _ = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        target_seq = self._to_latent_sequence(
            target_vectors, device=device, dtype=model_dtype
        )
        if noise is None:
            noise_seq = torch.randn_like(target_seq)
        else:
            noise_seq = self._to_latent_sequence(
                noise, device=device, dtype=model_dtype
            )

        if time is None:
            time = sample_beta_time(
                batch_size=target_seq.shape[0],
                device=target_seq.device,
                dtype=target_seq.dtype,
                alpha=float(self.cfg.time_beta_alpha),
                beta=float(self.cfg.time_beta_beta),
            )
        else:
            time = time.to(device=target_seq.device, dtype=target_seq.dtype)

        x_t, u_t = make_noisy_target(target=target_seq, noise=noise_seq, time=time)
        v_t, _ = self._predict_latent_and_actions(batch=batch, x_t_seq=x_t, time=time)
        return F.mse_loss(v_t, u_t)

    def action_flow_loss(
        self,
        *,
        batch: Stage2Batch,
        target_actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        device, _, _ = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        target_seq = self._to_action_sequence(
            target_actions, device=device, dtype=model_dtype
        )
        if noise is None:
            noise_seq = torch.randn_like(target_seq)
        else:
            noise_seq = self._to_action_sequence(
                noise, device=device, dtype=model_dtype
            )

        if time is None:
            time = sample_beta_time(
                batch_size=target_seq.shape[0],
                device=target_seq.device,
                dtype=target_seq.dtype,
                alpha=float(self.cfg.time_beta_alpha),
                beta=float(self.cfg.time_beta_beta),
            )
        else:
            time = time.to(device=target_seq.device, dtype=target_seq.dtype)

        x_t, u_t = make_noisy_target(target=target_seq, noise=noise_seq, time=time)
        v_t = self._predict_action_velocity(batch=batch, x_t_seq=x_t, time=time)
        losses = F.mse_loss(v_t, u_t, reduction="none")

        pad_mask = resolve_action_pad_mask(
            action_is_pad=action_is_pad,
            batch_size=int(target_seq.shape[0]),
            chunk_size=int(self.action_chunk_size),
            device=target_seq.device,
        )
        valid_mask = (~pad_mask).unsqueeze(-1).expand_as(losses)

        valid_f = valid_mask.to(dtype=losses.dtype)
        per_sample_denom = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample = (losses * valid_f).sum(dim=(1, 2)) / per_sample_denom

        if reduction == "none":
            return per_sample
        if reduction != "mean":
            raise ValueError(
                f"Unsupported reduction={reduction!r}; expected 'mean' or 'none'."
            )
        denom = valid_f.sum().clamp(min=1.0)
        return (losses * valid_f).sum() / denom

    def _denoise_step(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict[int, dict[str, torch.Tensor]],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        _, smol_model, _ = self._require_ready()

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            noisy_latents=x_t,
            timestep=timestep,
        )

        suffix_len = int(suffix_pad_masks.shape[1])
        batch_size = int(prefix_pad_masks.shape[0])
        prefix_len = int(prefix_pad_masks.shape[1])

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = smol_model.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.code_seq_len :].to(dtype=torch.float32)
        v_t, _ = self._predict_from_suffix(suffix_out=suffix_out)
        return v_t

    def _denoise_action_step(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict[int, dict[str, torch.Tensor]],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        _, smol_model, _ = self._require_ready()

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_suffix(
            noisy_actions=x_t,
            timestep=timestep,
        )

        suffix_len = int(suffix_pad_masks.shape[1])
        batch_size = int(prefix_pad_masks.shape[0])
        prefix_len = int(prefix_pad_masks.shape[1])

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = smol_model.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.action_chunk_size :].to(
            dtype=torch.float32
        )
        return self._predict_actions_from_suffix(suffix_out=suffix_out)

    @torch.no_grad()
    def sample_latent_vectors(
        self,
        *,
        batch: Stage2Batch,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device, smol_model, tokenizer = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        images, img_masks = self._prepare_images(batch=batch, device=device)
        lang_tokens, lang_masks = self._prepare_language(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
        )
        state = self._prepare_state(batch=batch, device=device, dtype=model_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

        if noise is None:
            x_t = torch.randn(
                (prefix_embs.shape[0], self.code_seq_len, self.latent_step_dim),
                device=device,
                dtype=model_dtype,
            )
        else:
            x_t = self._to_latent_sequence(noise, device=device, dtype=model_dtype)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = smol_model.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        num_steps = int(self.cfg.flow_steps)
        if num_steps <= 0:
            raise ValueError(f"flow_steps must be > 0, got {num_steps}")

        dt = -1.0 / float(num_steps)
        for step in range(num_steps):
            t_val = 1.0 + step * dt
            time = torch.full((x_t.shape[0],), t_val, dtype=model_dtype, device=device)
            v_t = self._denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
            )
            x_t = x_t + dt * v_t

        return self._flatten_latent_sequence(x_t)

    @torch.no_grad()
    def sample_action_chunk(
        self,
        *,
        batch: Stage2Batch,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, _, _ = self._require_action_heads()
        device, smol_model, tokenizer = self._require_ready()
        model_dtype = next(self.smol_model.parameters()).dtype

        images, img_masks = self._prepare_images(batch=batch, device=device)
        lang_tokens, lang_masks = self._prepare_language(
            batch=batch,
            tokenizer=tokenizer,
            device=device,
        )
        state = self._prepare_state(batch=batch, device=device, dtype=model_dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

        if noise is None:
            x_t = torch.randn(
                (
                    int(prefix_embs.shape[0]),
                    int(self.action_chunk_size),
                    int(self.cfg.action_dim),
                ),
                device=device,
                dtype=model_dtype,
            )
        else:
            x_t = self._to_action_sequence(noise, device=device, dtype=model_dtype)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = smol_model.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        num_steps = int(self.cfg.flow_steps)
        if num_steps <= 0:
            raise ValueError(f"flow_steps must be > 0, got {num_steps}")
        dt = -1.0 / float(num_steps)
        for step in range(num_steps):
            t_val = 1.0 + step * dt
            time = torch.full((x_t.shape[0],), t_val, dtype=model_dtype, device=device)
            v_t = self._denoise_action_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time,
            )
            x_t = x_t + dt * v_t

        return x_t

    def predict_actions(self, *, batch: Stage2Batch) -> torch.Tensor:
        chunk = self.sample_action_chunk(batch=batch)
        return chunk[:, 0, :]
