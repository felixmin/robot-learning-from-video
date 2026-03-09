from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10_000
) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(
        torch.float32
    )
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def get_intermediate_size(
    hidden_dim: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256
) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        load_vlm_weights: bool,
        train_expert_only: bool,
        freeze_vision_encoder: bool,
        attention_mode: str,
        num_expert_layers: int,
        num_vlm_layers: int,
        self_attn_every_n_layers: int,
        expert_width_multiplier: float,
        torch_dtype: torch.dtype,
        trust_remote_code: bool,
        vlm: nn.Module | None = None,
        processor: Any | None = None,
    ) -> None:
        super().__init__()

        if vlm is None:
            if load_vlm_weights:
                self.vlm = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=True,
                )
            else:
                cfg = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
                try:
                    from transformers import SmolVLMForConditionalGeneration
                except ImportError as exc:
                    raise RuntimeError(
                        "SmolVLMForConditionalGeneration is required when load_vlm_weights=false"
                    ) from exc
                self.vlm = SmolVLMForConditionalGeneration(config=cfg)
            config = self.vlm.config
        else:
            self.vlm = vlm
            config = self.vlm.config

        if processor is None:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.processor = processor

        if num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = (
                self.get_vlm_model().text_model.layers[:num_vlm_layers]
            )

        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config

        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = int(lm_expert_config.hidden_size)
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(
            int(hidden_size * expert_width_multiplier)
        )
        lm_expert_config.num_hidden_layers = self.num_vlm_layers

        if num_expert_layers > 0:
            if len(self.get_vlm_model().text_model.layers) % num_expert_layers != 0:
                raise ValueError(
                    "Number of VLM layers must be divisible by num_expert_layers, "
                    f"got {len(self.get_vlm_model().text_model.layers)} and {num_expert_layers}"
                )
            lm_expert_config.num_hidden_layers = num_expert_layers

        self.lm_expert = AutoModel.from_config(lm_expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = int(self_attn_every_n_layers)
        self.attention_mode = str(attention_mode)

        if "cross" in self.attention_mode:
            for layer_idx in range(len(self.lm_expert.layers)):
                if (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                ):
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads
                    * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads
                    * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )

        self.lm_expert.embed_tokens = None
        self.num_attention_heads = int(self.config.text_config.num_attention_heads)
        self.num_key_value_heads = int(self.config.text_config.num_key_value_heads)

        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.train_expert_only = bool(train_expert_only)
        self.expert_hidden_size = int(lm_expert_config.hidden_size)
        self.set_requires_grad()

    def get_vlm_model(self) -> nn.Module:
        if not hasattr(self.vlm, "model"):
            raise AttributeError(
                "Expected VLM to expose `.model` like SmolVLMForConditionalGeneration"
            )
        return self.vlm.model

    def set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False

        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")
            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False

        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True) -> SmolVLMWithExpertModel:
        super().train(mode)
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
        if self.train_expert_only:
            self.vlm.eval()
        return self

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            )
            .last_hidden_state
        )
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers: list[list[nn.Module]],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool,
        fill_kv_cache: bool,
        past_key_values: dict[int, dict[str, torch.Tensor]] | None,
    ) -> tuple[list[torch.Tensor], dict[int, dict[str, torch.Tensor]] | None]:
        query_states = []
        key_states = []
        value_states = []

        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue

            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        query_states = apply_rope(query_states, _position_ids)
        key_states = apply_rope(key_states, _position_ids)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states], dim=1
                )

        att_output = self.get_attention_interface()(
            _attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers: list[list[nn.Module]],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool,
        fill_kv_cache: bool,
        past_key_values: dict[int, dict[str, torch.Tensor]] | None,
    ) -> tuple[list[torch.Tensor | None], dict[int, dict[str, torch.Tensor]] | None]:
        att_outputs: list[torch.Tensor | None] = []

        if len(inputs_embeds) == 2 and past_key_values is None:
            seq_len = inputs_embeds[0].shape[1]
            position_id = position_ids[:, :seq_len]
            expert_position_id = position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]
            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = self.get_attention_interface()(
                prefix_attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (
                *expert_input_shape,
                -1,
                expert_layer.self_attn.head_dim,
            )

            expert_hidden_states = expert_hidden_states.to(
                dtype=expert_layer.self_attn.q_proj.weight.dtype
            )
            expert_query_state = expert_layer.self_attn.q_proj(
                expert_hidden_states
            ).view(expert_hidden_shape)

            _key_states = key_states.to(
                dtype=expert_layer.self_attn.k_proj.weight.dtype
            ).view(*key_states.shape[:2], -1)
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            _value_states = value_states.to(
                dtype=expert_layer.self_attn.v_proj.weight.dtype
            ).view(*value_states.shape[:2], -1)
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id
                - torch.min(
                    expert_position_id,
                    dim=1,
                    keepdim=True,
                ).values
            )
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1]
            ]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)
            att_output = self.get_attention_interface()(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def get_model_layers(self, models: list[nn.Module]) -> list[list[nn.Module | None]]:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: dict[int, dict[str, torch.Tensor]] | None,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[torch.Tensor | None], dict[int, dict[str, torch.Tensor]] | None]:
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)

        batch_size: int | None = None
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = int(hidden_states.shape[0])
                break
        if batch_size is None:
            raise ValueError("inputs_embeds must contain at least one non-None tensor")

        num_layers = self.num_vlm_layers
        head_dim = int(self.vlm.config.text_config.head_dim)
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                )
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache,
                    fill_kv_cache,
                    past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache,
                    fill_kv_cache,
                    past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue

                end = start + hidden_states.shape[1]
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)

                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb += after_first_residual
                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0

            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        return self.eager_attention_forward

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            num_key_value_heads * num_key_value_groups,
            head_dim,
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(
            attention_mask[:, None, :, :], att_weights, big_neg
        )
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )
        return att_output
