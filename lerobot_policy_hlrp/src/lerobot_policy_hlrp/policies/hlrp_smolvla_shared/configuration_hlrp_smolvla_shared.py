from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("hlrp_smolvla_shared")
@dataclass
class HLRPSmolVLASharedConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    horizon: int = 1
    n_action_steps: int = 1

    max_action_dim: int = 32

    model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    torch_dtype: str = "bf16"
    trust_remote_code: bool = True
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)

    latent_vector_dim: int = 128
    codebook_size: int = 2048
    code_seq_len: int = 1
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    stage2_artifact: Path | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        start = 1 - self.n_obs_steps
        return list(range(start, start + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("Policy requires an output ACTION feature named 'action'.")

        if len(self.image_features) == 0:
            raise ValueError("Policy requires at least one VISUAL input feature.")
