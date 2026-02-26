from __future__ import annotations

from dataclasses import dataclass, field

import torch

from foundation.action_tokens import ActionTokenConfig
from foundation.vla_inputs import ChatConfig


@dataclass
class SmolVLASharedCoreConfig:
    model_name: str
    latent_vector_dim: int
    action_dim: int | None = None
    freeze_vlm: bool = True
    freeze_vision_encoder: bool = True
    load_vlm_weights: bool = True
    attention_mode: str = "cross_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    add_image_special_tokens: bool = False
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    max_state_dim: int = 32
    prefix_length: int = -1
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(
        default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4)
    )
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0


@dataclass
class SmolVLASharedBackendConfig:
    model_name: str
    latent_vector_dim: int
    action_dim: int | None = None
    freeze_vlm: bool = True
    freeze_vision_encoder: bool = True
    load_vlm_weights: bool = True
    attention_mode: str = "cross_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    add_image_special_tokens: bool = False
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    max_state_dim: int = 32
    prefix_length: int = -1
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(
        default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4)
    )
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    latent_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    def to_core_config(self) -> SmolVLASharedCoreConfig:
        return SmolVLASharedCoreConfig(
            model_name=self.model_name,
            latent_vector_dim=self.latent_vector_dim,
            action_dim=self.action_dim,
            freeze_vlm=self.freeze_vlm,
            freeze_vision_encoder=self.freeze_vision_encoder,
            load_vlm_weights=self.load_vlm_weights,
            attention_mode=self.attention_mode,
            num_expert_layers=self.num_expert_layers,
            num_vlm_layers=self.num_vlm_layers,
            self_attn_every_n_layers=self.self_attn_every_n_layers,
            expert_width_multiplier=self.expert_width_multiplier,
            add_image_special_tokens=self.add_image_special_tokens,
            tokenizer_max_length=self.tokenizer_max_length,
            pad_language_to=self.pad_language_to,
            max_state_dim=self.max_state_dim,
            prefix_length=self.prefix_length,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            chat=self.chat,
            action_tokens=self.action_tokens,
            use_gpu_preprocessing=self.use_gpu_preprocessing,
            image_size=self.image_size,
            flow_hidden_dim=self.flow_hidden_dim,
            flow_steps=self.flow_steps,
            min_period=self.min_period,
            max_period=self.max_period,
            time_beta_alpha=self.time_beta_alpha,
            time_beta_beta=self.time_beta_beta,
        )
