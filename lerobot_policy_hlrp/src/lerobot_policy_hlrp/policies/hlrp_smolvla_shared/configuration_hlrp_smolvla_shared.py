from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from omegaconf import MISSING


@PreTrainedConfig.register_subclass("hlrp_smolvla_shared")
@dataclass
class HLRPSmolVLASharedConfig(PreTrainedConfig):
    init_mode: str = MISSING
    stage3_training_mode: str = MISSING
    action_loss_weight: float = MISSING
    latent_loss_weight: float = MISSING
    alternating_latent_steps_per_action_step: int | None = None

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_action_dim: int = 32
    max_state_dim: int = 32

    model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    train_expert_only: bool = True
    freeze_vision_encoder: bool = True
    attention_mode: str = "cross_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    add_image_special_tokens: bool = False
    prefix_length: int = -1
    torch_dtype: str = "bf16"
    trust_remote_code: bool = True
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (512, 512)
    camera_keys: tuple[str, ...] | None = None
    empty_cameras: int = 0

    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"
    system_prompt: str | None = None

    latent_vector_dim: int = 128
    codebook_size: int = 2048
    code_seq_len: int = 1
    flow_hidden_dim: int = 1024
    flow_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    stage2_artifact: Path | None = MISSING
    lam_checkpoint_path: Path | None = None
    lam_future_frames: int | None = None
    lam_camera_keys: tuple[str, ...] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.init_mode not in {"artifact", "scratch"}:
            raise ValueError(f"init_mode must be one of {{'artifact','scratch'}}, got {self.init_mode!r}")
        if self.stage3_training_mode not in {"action", "latent", "multitask", "alternating"}:
            raise ValueError(
                "stage3_training_mode must be one of {'action','latent','multitask','alternating'}, "
                f"got {self.stage3_training_mode!r}"
            )
        if self.init_mode == "artifact" and self.stage2_artifact is None:
            raise ValueError("init_mode='artifact' requires non-null stage2_artifact")
        if self.init_mode == "scratch" and self.stage2_artifact is not None:
            raise ValueError("init_mode='scratch' requires stage2_artifact=null")
        if self.action_loss_weight <= 0.0:
            raise ValueError(f"action_loss_weight must be > 0, got {self.action_loss_weight}")
        if self.latent_loss_weight <= 0.0:
            raise ValueError(f"latent_loss_weight must be > 0, got {self.latent_loss_weight}")
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps must be <= chunk_size, got n_action_steps={self.n_action_steps} chunk_size={self.chunk_size}"
            )
        uses_latent_targets = self.stage3_training_mode in {"latent", "multitask", "alternating"}
        if uses_latent_targets:
            if self.lam_checkpoint_path is None:
                raise ValueError(
                    "lam_checkpoint_path is required when stage3_training_mode uses latent targets"
                )
            if self.lam_future_frames is None or self.lam_future_frames <= 0:
                raise ValueError(
                    "lam_future_frames must be > 0 when stage3_training_mode uses latent targets"
                )
            if self.lam_camera_keys is None or len(self.lam_camera_keys) == 0:
                raise ValueError(
                    "lam_camera_keys must be set to one or more camera keys when "
                    "stage3_training_mode uses latent targets"
                )

        if self.stage3_training_mode == "alternating":
            if (
                self.alternating_latent_steps_per_action_step is None
                or self.alternating_latent_steps_per_action_step < 1
            ):
                raise ValueError(
                    "alternating_latent_steps_per_action_step must be >= 1 when "
                    "stage3_training_mode='alternating'"
                )

    @property
    def observation_delta_indices(self) -> list[int]:
        if self.stage3_training_mode in {"latent", "multitask", "alternating"}:
            if self.lam_future_frames is None:
                raise ValueError("lam_future_frames must be set for latent target modes")
            return [0, int(self.lam_future_frames)]
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("Policy requires an output ACTION feature named 'action'.")

        if len(self.image_features) == 0:
            raise ValueError("Policy requires at least one VISUAL input feature.")
