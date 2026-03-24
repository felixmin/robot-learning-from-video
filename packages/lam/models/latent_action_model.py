"""
Latent Action Model (LAM).

A VQ-backed model that encodes frame-to-frame transitions into discrete latent
action codes. Supports modular decoder objectives for reconstruction,
representation supervision, and motion supervision.

Decoder types:
- DINO decoder: Predicts next-frame DINO tokens
- Pixel decoder: Predicts next-frame pixels with gradients flowing to the encoder
- Flow decoder: Predicts optical flow via RAFT knowledge distillation
- Aux decoder: Predicts pixels for visualization only with detached gradients
"""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from torch import nn

from lam.models.attention import ContinuousPositionBias, Transformer
from lam.models.decoder_losses import (
    _decode_spatial_tokens,
    compute_aux_outputs,
    compute_dino_loss,
    compute_flow_loss_term,
    compute_pixel_loss,
)
from lam.models.dino import DINOEncoder, DINOFeatureExtractor, DINOWrapper
from lam.models.nsvq import NSVQ

logger = logging.getLogger(__name__)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    if len(ret) != 2:
        raise ValueError(f"Expected pair, got {ret}")
    return ret


@dataclass
class DinoConfig:
    """Configuration for DINO supervision."""

    loss_weight: float = 1.0
    warmup_steps: int = 0

    def __post_init__(self):
        if self.loss_weight <= 0:
            raise ValueError(
                f"dino.loss_weight must be positive, got {self.loss_weight}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"dino.warmup_steps must be non-negative, got {self.warmup_steps}"
            )

    def get_weight(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.loss_weight
        warmup_factor = min(1.0, step / self.warmup_steps)
        return self.loss_weight * warmup_factor


@dataclass
class EncodedFrames:
    """Pre-VQ encoder outputs for one frame pair.

    Representation summary:
    - `enc_*_tokens`: frame token grids [B, 1, h, w, d], one token per image patch cell
    - `first_tokens` / `last_tokens`: the same frame information packed to [B, h*w, d]
      because the VQ module consumes a flat token sequence per frame
    """

    batch_size: int
    device: torch.device
    first_frame: torch.Tensor
    rest_frames: torch.Tensor
    enc_first_frame_tokens: torch.Tensor
    enc_rest_frames_tokens: torch.Tensor
    first_tokens: torch.Tensor
    last_tokens: torch.Tensor


@dataclass
class EncodedBatch(EncodedFrames):
    """Encoded frame pair plus quantized action tokens [B, code_seq_len, quant_dim]."""

    tokens: torch.Tensor
    indices: torch.Tensor
    perplexity: torch.Tensor


@dataclass
class CodebookStats:
    current_threshold: float
    window_total: torch.Tensor
    codebook_replaced: float
    replaced_count: float
    used_count: float
    min_count: float


class LatentActionModel(nn.Module):
    """
    Latent Action Model.

    Encodes frame pairs into discrete latent action codes using VQ-VAE style
    quantization. Uses transformer-based encoder/decoder with modular objectives.

    Architecture:
    - Encoder: Processes frame pairs through spatial/temporal transformers
    - VQ (NSVQ): Quantizes latent representations to discrete codes
    - Training Decoders (at least one required):
      - DINO decoder: Predicts next frame's DINO tokens
      - Pixel decoder: Predicts next frame's pixels (gradients to encoder)
      - Flow decoder: Predicts optical flow (gradients to encoder)
    - Interpretability Decoder (optional):
      - Aux decoder: Predicts pixels for visualization (gradients detached)

    Returns:
        (loss, metrics_dict) where metrics_dict contains diagnostic values
    """

    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head,
        heads,
        channels,
        attn_dropout,
        ff_dropout,
        code_seq_len,
        vq_discarding_threshold: float,
        vq_discarding_threshold_schedule,
        latent_ablation,
        use_dinov3_encoder,
        dinov3_model_name,
        dinov3_pool_to_grid,  # Pool DINO features to this grid size (e.g., 8 for 8x8)
        metrics_num_unique_codes_every_n_steps: int,
        # DINO supervision config (required when use_dino_decoder=True)
        dino_config,
        # Training decoder flags (at least one must be True, or flow_config must be set)
        use_dino_decoder,
        use_pixel_decoder,
        # Interpretability decoder flag (optional, for visualization only)
        use_aux_decoder,
        # Flow supervision config (optional - set to enable flow loss)
        flow_config,
        # Codebook replacement schedule (optional)
        codebook_replace_schedule,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self._store_core_configuration(
            latent_ablation=latent_ablation,
            metrics_num_unique_codes_every_n_steps=metrics_num_unique_codes_every_n_steps,
            vq_discarding_threshold=vq_discarding_threshold,
            vq_discarding_threshold_schedule=vq_discarding_threshold_schedule,
            code_seq_len=code_seq_len,
            image_size=image_size,
            patch_size=patch_size,
            codebook_replace_schedule=codebook_replace_schedule,
        )

        self._configure_decoder_flags(
            dino_config=dino_config,
            use_dino_decoder=use_dino_decoder,
            use_pixel_decoder=use_pixel_decoder,
            use_aux_decoder=use_aux_decoder,
            flow_config=flow_config,
        )

        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size
        self._initialize_encoder_stack(
            dim=dim,
            channels=channels,
            image_height=image_height,
            image_width=image_width,
            patch_height=patch_height,
            patch_width=patch_width,
            use_dinov3_encoder=use_dinov3_encoder,
            dinov3_model_name=dinov3_model_name,
            dinov3_pool_to_grid=dinov3_pool_to_grid,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self._initialize_vq(
            dim=dim,
            quant_dim=quant_dim,
            codebook_size=codebook_size,
            vq_discarding_threshold=vq_discarding_threshold,
        )

        eff_patch_h, eff_patch_w = self._effective_patch_shape(
            image_height=image_height,
            image_width=image_width,
            grid_size=self._effective_grid_size,
        )
        self._initialize_decoder_modules(
            spatial_depth=spatial_depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            channels=channels,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            eff_patch_h=eff_patch_h,
            eff_patch_w=eff_patch_w,
        )
        self._initialize_flow_components(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self._action_shape = self._derive_action_shape(code_seq_len)

    def _store_core_configuration(
        self,
        *,
        latent_ablation,
        metrics_num_unique_codes_every_n_steps: int,
        vq_discarding_threshold: float,
        vq_discarding_threshold_schedule,
        code_seq_len: int,
        image_size,
        patch_size,
        codebook_replace_schedule,
    ) -> None:
        if latent_ablation not in ("none", "permute_batch"):
            raise ValueError(
                f"latent_ablation must be one of ['none', 'permute_batch'], got {latent_ablation!r}"
            )
        self.latent_ablation = latent_ablation
        self.metrics_num_unique_codes_every_n_steps = int(
            metrics_num_unique_codes_every_n_steps
        )
        if self.metrics_num_unique_codes_every_n_steps <= 0:
            raise ValueError(
                "metrics_num_unique_codes_every_n_steps must be a positive integer"
            )
        if float(vq_discarding_threshold) < 0.0:
            raise ValueError("vq_discarding_threshold must be >= 0")

        self.codebook_replace_schedule = list(codebook_replace_schedule or [])
        self.vq_discarding_threshold = float(vq_discarding_threshold)
        self.vq_discarding_threshold_schedule = (
            self._validate_vq_discarding_threshold_schedule(
                vq_discarding_threshold_schedule
            )
        )
        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)

    def _initialize_encoder_stack(
        self,
        *,
        dim: int,
        channels: int,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        use_dinov3_encoder: bool,
        dinov3_model_name: str,
        dinov3_pool_to_grid,
    ) -> None:
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        self._initialize_encoder_projections(
            dim=dim,
            channels=channels,
            image_height=image_height,
            image_width=image_width,
            patch_height=patch_height,
            patch_width=patch_width,
            use_dinov3_encoder=use_dinov3_encoder,
            dinov3_model_name=dinov3_model_name,
            dinov3_pool_to_grid=dinov3_pool_to_grid,
        )
        self.enc_spatial_transformer = self._build_transformer(
            depth=spatial_depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self.enc_temporal_transformer = self._build_transformer(
            depth=temporal_depth,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def _initialize_vq(
        self,
        *,
        dim: int,
        quant_dim: int,
        codebook_size: int,
        vq_discarding_threshold: float,
    ) -> None:
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            discarding_threshold=vq_discarding_threshold,
            code_seq_len=self.code_seq_len,
            patch_size=self.patch_size,
            image_size=self.image_size,
            grid_size=self._effective_grid_size,
        )

    def _configure_decoder_flags(
        self,
        *,
        dino_config,
        use_dino_decoder,
        use_pixel_decoder,
        use_aux_decoder,
        flow_config,
    ) -> None:
        self.use_dino_decoder = bool(use_dino_decoder)
        if self.use_dino_decoder and dino_config is None:
            raise ValueError("dino_config is required when use_dino_decoder=true")
        if (not self.use_dino_decoder) and dino_config is not None:
            raise ValueError("dino_config must be null when use_dino_decoder=false")
        self.dino_config = dino_config
        self.use_pixel_decoder = bool(use_pixel_decoder)
        self.use_aux_decoder = bool(use_aux_decoder)
        self.flow_config = flow_config

        training_decoders = self._get_enabled_training_decoders()
        if not training_decoders:
            raise ValueError(
                "At least one training decoder must be enabled. "
                "Set dino.enabled=true, use_pixel_decoder=true, or provide flow_config."
            )

        logger.info("Enabled training decoders: %s", training_decoders)
        if self.dino_config is not None:
            logger.info(
                "DINO supervision enabled (weight=%s, warmup_steps=%s)",
                self.dino_config.loss_weight,
                self.dino_config.warmup_steps,
            )
        if self.use_aux_decoder:
            logger.info("Aux decoder enabled for interpretability")

    @staticmethod
    def _build_transformer(
        *,
        depth,
        dim,
        dim_head,
        heads,
        attn_dropout,
        ff_dropout,
        has_cross_attn: bool = False,
    ) -> Transformer:
        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        if has_cross_attn:
            transformer_kwargs["has_cross_attn"] = True
            transformer_kwargs["dim_context"] = dim
        return Transformer(depth=depth, **transformer_kwargs)

    @staticmethod
    def _effective_patch_shape(
        *,
        image_height: int,
        image_width: int,
        grid_size: tuple[int, int],
    ) -> tuple[int, int]:
        eff_h, eff_w = grid_size
        return image_height // eff_h, image_width // eff_w

    def _build_pixel_projection(
        self,
        *,
        dim: int,
        channels: int,
        image_height: int,
        image_width: int,
        grid_size: tuple[int, int],
    ) -> nn.Sequential:
        pixel_p1, pixel_p2 = self._effective_patch_shape(
            image_height=image_height,
            image_width=image_width,
            grid_size=grid_size,
        )
        return nn.Sequential(
            Rearrange(
                "b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=pixel_p1, p2=pixel_p2
            ),
            nn.LayerNorm(channels * pixel_p1 * pixel_p2),
            nn.Linear(channels * pixel_p1 * pixel_p2, dim),
            nn.LayerNorm(dim),
        )

    @staticmethod
    def _build_pixels_head(
        *,
        dim: int,
        channels: int,
        patch_height: int,
        patch_width: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, channels * patch_height * patch_width),
            Rearrange(
                "b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
            ),
        )

    def _initialize_encoder_projections(
        self,
        *,
        dim: int,
        channels: int,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        use_dinov3_encoder: bool,
        dinov3_model_name: str,
        dinov3_pool_to_grid,
    ) -> None:
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.dino_feature_extractor = None
        self.dino_encoder = None
        self.encoder_projection = None

        if use_dinov3_encoder:
            logger.info("Using DINOv3 Encoder: %s", dinov3_model_name)
            self.dino_feature_extractor = DINOFeatureExtractor(
                model_name=dinov3_model_name,
                target_size=self.image_size[0],
            )
            self.dino_encoder = DINOEncoder(
                self.dino_feature_extractor,
                dim,
                pool_to_grid=dinov3_pool_to_grid,
            )
            self.encoder_projection = DINOWrapper(self.dino_encoder)
            output_grid = self.dino_encoder.output_grid_size
            self._effective_grid_size = (output_grid, output_grid)
            logger.info("  - Effective grid size: %s", self._effective_grid_size)
        else:
            self._effective_grid_size = (
                image_height // patch_height,
                image_width // patch_width,
            )

        # Projects pixels [B, C, 1, H, W] to grid tokens [B, 1, h, w, dim].
        self.pixel_projection = self._build_pixel_projection(
            dim=dim,
            channels=channels,
            image_height=image_height,
            image_width=image_width,
            grid_size=self._effective_grid_size,
        )
        self.decoder_context_projection = self.pixel_projection

        if self.encoder_projection is None:
            self.encoder_projection = self.pixel_projection

    def _initialize_decoder_modules(
        self,
        *,
        spatial_depth: int,
        dim: int,
        dim_head: int,
        heads: int,
        channels: int,
        attn_dropout: float,
        ff_dropout: float,
        eff_patch_h: int,
        eff_patch_w: int,
    ) -> None:
        self.dino_decoder = None
        if self.use_dino_decoder:
            self.dino_decoder = self._build_transformer(
                depth=spatial_depth,
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                has_cross_attn=True,
            )

        self.pixel_decoder = None
        self.pixel_to_pixels = None
        if self.use_pixel_decoder:
            self.pixel_decoder = self._build_transformer(
                depth=spatial_depth,
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                has_cross_attn=True,
            )
            self.pixel_to_pixels = self._build_pixels_head(
                dim=dim,
                channels=channels,
                patch_height=eff_patch_h,
                patch_width=eff_patch_w,
            )

        self.aux_decoder = None
        self.aux_to_pixels = None
        if self.use_aux_decoder:
            self.aux_decoder = self._build_transformer(
                depth=spatial_depth,
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                has_cross_attn=True,
            )
            self.aux_to_pixels = self._build_pixels_head(
                dim=dim,
                channels=channels,
                patch_height=eff_patch_h,
                patch_width=eff_patch_w,
            )

    def _initialize_flow_components(
        self,
        *,
        dim: int,
        dim_head: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
    ) -> None:
        self.flow_decoder = None
        self.flow_teacher = None
        if self.flow_config is None:
            return

        from lam.models.flow import FlowDecoder, RAFTTeacher

        logger.info(
            "Initializing flow supervision (model=%s, depth=%s, weight=%s)",
            self.flow_config.model,
            self.flow_config.decoder_depth,
            self.flow_config.loss_weight,
        )
        self.flow_decoder = FlowDecoder(
            dim=dim,
            depth=self.flow_config.decoder_depth,
            heads=heads,
            dim_head=dim_head,
            image_size=self.image_size,
            effective_grid_size=self._effective_grid_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self.flow_teacher = RAFTTeacher(
            self.flow_config.model,
            chunk_size=self.flow_config.teacher_chunk_size,
            num_flow_updates=self.flow_config.teacher_num_flow_updates,
        )

    @staticmethod
    def _derive_action_shape(code_seq_len: int) -> tuple[int, int]:
        if math.sqrt(code_seq_len) % 1 == 0:
            action_size = int(math.sqrt(code_seq_len))
            return action_size, action_size
        if code_seq_len == 2:
            return 2, 1
        raise ValueError(
            f"code_seq_len must be a square number or 2, got {code_seq_len}"
        )

    def _get_enabled_training_decoders(self) -> List[str]:
        """
        Get list of enabled training decoders.

        Training decoders contribute gradients to the encoder/VQ.
        At least one must be enabled for meaningful training.

        Returns:
            List of enabled decoder names (e.g., ["dino", "flow"])
        """
        decoders = []
        if self.use_dino_decoder:
            decoders.append("dino")
        if self.use_pixel_decoder:
            decoders.append("pixel")
        if self.flow_config is not None:
            decoders.append("flow")
        return decoders

    def _should_replace_codebook(self, step: int) -> bool:
        """
        Check if unused codebook entries should be replaced at this step.

        Uses codebook_replace_schedule for diminishing frequency replacement:
        - More frequent early in training to ensure good codebook utilization
        - Less frequent later to reduce overhead
        """
        if not self.codebook_replace_schedule:
            return False
        for interval, until_step in self.codebook_replace_schedule:
            if step < until_step and step % interval == 0:
                return True
        return False

    def _validate_vq_discarding_threshold_schedule(
        self,
        schedule: Optional[list],
    ) -> Optional[list[tuple[float, int]]]:
        if schedule is None:
            return None
        if len(schedule) == 0:
            raise ValueError("vq_discarding_threshold_schedule must not be empty")

        parsed: list[tuple[float, int]] = []
        previous_until_step = -1
        for i, entry in enumerate(schedule):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"vq_discarding_threshold_schedule[{i}] must be [threshold, until_step], got {entry!r}"
                )
            threshold, until_step = entry
            threshold = float(threshold)
            until_step = int(until_step)
            if threshold < 0.0:
                raise ValueError(
                    f"vq_discarding_threshold_schedule[{i}] threshold must be >= 0, got {threshold}"
                )
            if until_step <= 0:
                raise ValueError(
                    f"vq_discarding_threshold_schedule[{i}] until_step must be > 0, got {until_step}"
                )
            if until_step <= previous_until_step:
                raise ValueError(
                    "vq_discarding_threshold_schedule until_step values must be strictly increasing"
                )
            parsed.append((threshold, until_step))
            previous_until_step = until_step

        return parsed

    def _get_vq_discarding_threshold(self, step: int) -> float:
        if self.vq_discarding_threshold_schedule is None:
            return self.vq_discarding_threshold
        for threshold, until_step in self.vq_discarding_threshold_schedule:
            if step < until_step:
                return threshold
        return self.vq_discarding_threshold_schedule[-1][0]

    def load(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        pt = torch.load(str(path), weights_only=False)
        pt = {k.replace("module.", ""): v for k, v in pt.items()}
        self.load_state_dict(pt, strict=False)

    @property
    def patch_height_width(self):
        return self._effective_grid_size

    @property
    def action_shape(self):
        """Returns (action_h, action_w) for reshaping latent codes."""
        return self._action_shape

    def _encode_frames(self, first_frame, rest_frames):
        """
        Encode frame pair through encoder projection and temporal transformer.

        Args:
            first_frame: First frame [B, C, 1, H, W]
            rest_frames: Second frame [B, C, 1, H, W]

        Returns:
            enc_first_frame_tokens: frame token grid [B, 1, h, w, d]
            enc_rest_frames_tokens: frame token grid [B, 1, h, w, d]
            first_tokens_packed: same grid packed to [B, h*w, d] for VQ
            last_tokens_packed: same grid packed to [B, h*w, d] for VQ
        """
        enc_first_frame_tokens = self.encoder_projection(first_frame)
        enc_rest_frames_tokens = self.encoder_projection(rest_frames)
        enc_tokens = torch.cat((enc_first_frame_tokens, enc_rest_frames_tokens), dim=1)

        first_tokens, last_tokens = self.encode(enc_tokens)

        first_tokens_packed, _ = pack([first_tokens], "b * d")
        last_tokens_packed, _ = pack([last_tokens], "b * d")

        return (
            enc_first_frame_tokens,
            enc_rest_frames_tokens,
            first_tokens_packed,
            last_tokens_packed,
        )

    def encode(self, tokens):
        """
        Encode frame token grids into latent frame token grids.

        Args:
            tokens: frame token grids [B, T, h, w, d], where each [h, w] location
                corresponds to one image patch region and `d` is the feature dim.

        Returns:
            first_tokens: latent token grid for frame 1 [B, 1, h, w, d]
            last_tokens: latent token grid for frame 2 [B, 1, h, w, d]
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        # Spatial transformer runs within each frame: [B, T, h, w, d] -> [(B*T), (h*w), d].
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        tokens = self.enc_spatial_transformer(
            tokens, attn_bias=attn_bias, video_shape=video_shape
        )

        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        # Temporal transformer runs per spatial cell across time: [(B*h*w), T, d].
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]

        return first_tokens, last_tokens

    def decode(
        self,
        tokens,
        actions,
    ):
        """
        Decodes latent actions + context frame into reconstructed video.

        This prefers the auxiliary decoder for visualization/inference and falls
        back to the pixel decoder when aux reconstruction is unavailable.

        Args:
            tokens: Continuous embeddings of the first frame (PIXEL CONTEXT) [B, 1, h, w, d]
            actions: Continuous embeddings of the latent action [B, 1, h', w', d].

        Returns:
            recon_video: Reconstructed pixel values [B, C, 1, H, W], or None if no
                reconstruction decoder is enabled
        """
        decoder = None
        projector = None
        decoder_actions = actions
        if self.aux_decoder is not None:
            decoder = self.aux_decoder
            projector = self.aux_to_pixels
            decoder_actions = actions.detach()
        elif self.pixel_decoder is not None:
            decoder = self.pixel_decoder
            projector = self.pixel_to_pixels
        else:
            return None

        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])
        tokens = _decode_spatial_tokens(
            decoder=decoder,
            context_tokens=tokens,
            action_tokens=decoder_actions,
            attn_bias=self.spatial_rel_pos_bias(h, w, device=tokens.device),
            batch_size=b,
            h_dec=h,
            w_dec=w,
        )
        if tuple(tokens.shape[:-1]) != video_shape:
            raise RuntimeError(
                "Decoded reconstruction tokens do not match the expected video shape"
            )
        recon_video = projector(tokens)

        return recon_video

    def _codebook_ids_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a frame pair and return only discrete code ids [B, code_seq_len]."""
        encoded = self._encode_video_pair(video)
        return self.vq.get_indices(encoded.first_tokens, encoded.last_tokens)

    def _inference_vq_outputs(
        self,
        video: torch.Tensor,
        *,
        user_action_token_num=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a frame pair and run VQ inference.

        Returns:
        - first_frame: [B, C, 1, H, W]
        - tokens: flat quantized tokens [B, code_seq_len, quant_dim]
        - indices: discrete ids [B, code_seq_len]
        """
        encoded = self._encode_video_pair(video)
        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(
                encoded.first_tokens,
                encoded.last_tokens,
                user_action_token_num=user_action_token_num,
            )
        else:
            tokens, indices = self.vq.inference(
                encoded.first_tokens, encoded.last_tokens
            )
        return encoded.first_frame, tokens, indices

    def _normalize_video_input(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize input to [B, C, T, H, W] and validate the configured image size."""
        if video.ndim not in {4, 5}:
            raise ValueError(f"Expected 4D or 5D input, got {video.ndim}D")
        if video.ndim == 4:
            video = rearrange(video, "b c h w -> b c 1 h w")
        if tuple(video.shape[3:]) != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {tuple(video.shape[3:])}"
            )
        return video

    def _encode_video_pair(self, video: torch.Tensor) -> EncodedFrames:
        """Encode a 2-frame video into frame features and packed latent tokens.

        The model first builds a frame token grid [B, 1, h, w, d] for each frame,
        then packs each grid to [B, h*w, d] so VQ can treat it as a token sequence.
        """
        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]
        enc_first_frame_tokens, enc_rest_frames_tokens, first_tokens, last_tokens = (
            self._encode_frames(first_frame, rest_frames)
        )
        # first_tokens / last_tokens are packed per-frame grids: [B, h*w, dim].
        return EncodedFrames(
            batch_size=int(video.shape[0]),
            device=video.device,
            first_frame=first_frame,
            rest_frames=rest_frames,
            enc_first_frame_tokens=enc_first_frame_tokens,
            enc_rest_frames_tokens=enc_rest_frames_tokens,
            first_tokens=first_tokens,
            last_tokens=last_tokens,
        )

    def _quantize_encoded_video(self, encoded: EncodedFrames) -> EncodedBatch:
        """Quantize packed frame latents into an action representation.

        Inputs:
        - `first_tokens` / `last_tokens`: packed frame latents [B, h*w, d]

        Outputs:
        - `tokens`: quantized action embeddings [B, code_seq_len, quant_dim]
        - `indices`: discrete code ids [B, code_seq_len]

        This action sequence is the compressed representation of the transition
        from the first frame to the second frame.
        """
        tokens, perplexity, _, indices = self.vq(
            encoded.first_tokens,
            encoded.last_tokens,
            codebook_training_only=False,
        )
        # VQ returns a flat action sequence [B, code_seq_len, quant_dim].
        return EncodedBatch(
            batch_size=encoded.batch_size,
            device=encoded.device,
            first_frame=encoded.first_frame,
            rest_frames=encoded.rest_frames,
            enc_first_frame_tokens=encoded.enc_first_frame_tokens,
            enc_rest_frames_tokens=encoded.enc_rest_frames_tokens,
            first_tokens=encoded.first_tokens,
            last_tokens=encoded.last_tokens,
            tokens=tokens,
            indices=indices,
            perplexity=perplexity,
        )

    def _prepare_action_tokens(self, encoded: EncodedBatch) -> torch.Tensor:
        """Reshape the flat action sequence into an action grid for decoder cross-attention.

        `tokens` starts as [B, code_seq_len, quant_dim]. We reshape it to
        [B, 1, ah, aw, quant_dim] so the decoder can treat the latent action as a
        small 2D layout of action slots rather than a flat list.
        """
        action_h, action_w = self.action_shape
        # Decoder branches consume a spatial action grid [B, 1, ah, aw, quant_dim].
        action_tokens = rearrange(
            encoded.tokens,
            "b (t h w) d -> b t h w d",
            h=action_h,
            w=action_w,
        )
        if self.latent_ablation == "permute_batch" and encoded.batch_size > 1:
            perm = torch.randperm(encoded.batch_size, device=action_tokens.device)
            action_tokens = action_tokens[perm]
        return action_tokens

    def _update_codebook_stats(self, step: int) -> CodebookStats:
        """Collect replacement-window stats and optionally refresh underused codebook entries."""
        step_i = int(step)
        current_vq_discarding_threshold = self._get_vq_discarding_threshold(step_i)
        window_total = self.vq.codebooks_used.sum().detach()
        unused_indices, used_indices, min_count_val = self.vq._get_replacement_indices(
            discarding_threshold=current_vq_discarding_threshold
        )

        codebook_replaced = 0.0
        replaced_count = float(unused_indices.shape[0])
        used_count = float(used_indices.shape[0])
        min_count = float(min_count_val)

        if step_i != 0 and self._should_replace_codebook(step_i):
            logger.debug("Replacing unused codebook entries at step %s", step_i)
            codebook_replaced = 1.0
            unused_c, used_c, total_c, min_c = self.vq.replace_unused_codebooks(
                discarding_threshold=current_vq_discarding_threshold
            )
            replaced_count = float(unused_c)
            used_count = float(used_c)
            min_count = float(min_c)
            window_total = window_total.new_tensor(float(total_c))
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[Codebook] step={step_i} replaced={int(replaced_count)} "
                    f"used={int(used_count)} total={int(total_c)} min_count={min_count:.4f} "
                    f"threshold={current_vq_discarding_threshold:.6f}"
                )

        return CodebookStats(
            current_threshold=current_vq_discarding_threshold,
            window_total=window_total,
            codebook_replaced=codebook_replaced,
            replaced_count=replaced_count,
            used_count=used_count,
            min_count=min_count,
        )

    def _initialize_metrics(
        self,
        *,
        perplexity: torch.Tensor,
        indices: torch.Tensor,
        step: int,
        codebook: CodebookStats,
    ) -> dict[str, object]:
        """Build the base per-step metrics dict before decoder-specific losses append to it."""
        metrics: dict[str, object] = {
            "code_usage_perplexity_in_batch": perplexity.detach(),
            "replacement_applied_this_step": codebook.codebook_replaced,
            "entries_below_usage_threshold_count_in_window": codebook.replaced_count,
            "entries_at_or_above_usage_threshold_count_in_window": codebook.used_count,
            "usage_count_threshold_in_window": codebook.min_count,
            "discarding_threshold_fraction_of_expected_usage": codebook.current_threshold,
            "code_assignments_in_window": codebook.window_total,
        }
        if (
            self.metrics_num_unique_codes_every_n_steps > 0
            and int(step) % int(self.metrics_num_unique_codes_every_n_steps) == 0
        ):
            metrics["unique_codes_in_batch"] = int(indices.unique().size(0))
        return metrics

    def _build_decoder_shared_inputs(
        self,
        *,
        first_frame: torch.Tensor,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor, Optional[torch.Tensor]]:
        """Prepare the decoder context grid built from the first frame.

        `pixel_context` is a frame token grid [B, 1, h, w, d]. Decoder branches use
        it as the spatial canvas for the predicted next-frame representation, while
        the action grid supplies the transition information via cross-attention.
        """
        h_dec, w_dec = self.patch_height_width
        attn_bias = self.spatial_rel_pos_bias(h_dec, w_dec, device=device)

        pixel_context = None
        if (
            self.pixel_decoder is not None
            or self.aux_decoder is not None
            or self.flow_decoder is not None
        ):
            if self.decoder_context_projection is None:
                raise RuntimeError(
                    "decoder_context_projection is required when a pixel-context decoder is enabled"
                )
            # Shared decoder context tokens: [B, 1, h, w, dim].
            pixel_context = self.decoder_context_projection(first_frame)
        return h_dec, w_dec, attn_bias, pixel_context

    def forward(
        self,
        video,
        step=0,
        return_recons_only=False,
        return_only_codebook_ids=False,
    ):
        """
        Forward pass for training.

        Args:
            video: Input frame pairs [B, C, 2, H, W] or [B, C, H, W] for single frame
            step: Training step (for codebook replacement scheduling)
            return_recons_only: If True, return only reconstructed frames
            return_only_codebook_ids: If True, return only codebook indices

        Returns:
            If return_recons_only: reconstructed frames [B, C, H, W]
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: (loss, metrics_dict)
        """
        video = self._normalize_video_input(video)

        if return_only_codebook_ids:
            return self._codebook_ids_from_video(video)

        encoded_frames = self._encode_video_pair(video)
        encoded = self._quantize_encoded_video(encoded_frames)
        codebook = self._update_codebook_stats(int(step))

        action_tokens = self._prepare_action_tokens(encoded)
        metrics = self._initialize_metrics(
            perplexity=encoded.perplexity,
            indices=encoded.indices,
            step=int(step),
            codebook=codebook,
        )
        total_loss = encoded.rest_frames.new_zeros((), requires_grad=True)

        h_dec, w_dec, attn_bias, pixel_context = self._build_decoder_shared_inputs(
            first_frame=encoded.first_frame,
            device=encoded.device,
        )

        dino_loss_term = compute_dino_loss(
            self,
            encoded=encoded,
            action_tokens=action_tokens,
            attn_bias=attn_bias,
            h_dec=h_dec,
            w_dec=w_dec,
            step=int(step),
            metrics=metrics,
        )
        if dino_loss_term is not None:
            total_loss = total_loss + dino_loss_term

        pixel_loss = compute_pixel_loss(
            self,
            rest_frames=encoded.rest_frames,
            action_tokens=action_tokens,
            pixel_context=pixel_context,
            attn_bias=attn_bias,
            h_dec=h_dec,
            w_dec=w_dec,
            batch_size=encoded.batch_size,
            metrics=metrics,
        )
        if pixel_loss is not None:
            total_loss = total_loss + pixel_loss

        aux_loss, recon_frames = compute_aux_outputs(
            self,
            rest_frames=encoded.rest_frames,
            action_tokens=action_tokens,
            pixel_context=pixel_context,
            attn_bias=attn_bias,
            h_dec=h_dec,
            w_dec=w_dec,
            batch_size=encoded.batch_size,
            return_recons_only=bool(return_recons_only),
            metrics=metrics,
        )
        if return_recons_only:
            if recon_frames is None and pixel_context is not None:
                recon_video = self.decode(pixel_context, action_tokens)
                if recon_video is not None:
                    recon_frames = rearrange(recon_video, "b c 1 h w -> b c h w")
            return recon_frames
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        flow_loss_term = compute_flow_loss_term(
            self,
            first_frame=encoded.first_frame,
            rest_frames=encoded.rest_frames,
            action_tokens=action_tokens,
            pixel_context=pixel_context,
            attn_bias=attn_bias,
            step=int(step),
            metrics=metrics,
        )
        if flow_loss_term is not None:
            total_loss = total_loss + flow_loss_term

        return total_loss, metrics

    def inference(
        self,
        video,
        return_only_codebook_ids=False,
        user_action_token_num=None,
    ):
        """
        Inference pass (no loss computation).

        Args:
            video: Input frame pairs [B, C, 2, H, W] or [B, C, H, W]
            return_only_codebook_ids: If True, return only codebook indices
            user_action_token_num: Optional override for action token selection

        Returns:
            If return_only_codebook_ids: codebook indices [B, code_seq_len]
            Otherwise: reconstructed frames [B, C, H, W], or None if no
            reconstruction decoder is enabled
        """
        video = self._normalize_video_input(video)
        first_frame, tokens, indices = self._inference_vq_outputs(
            video,
            user_action_token_num=user_action_token_num,
        )

        if return_only_codebook_ids:
            return indices

        if self.decoder_context_projection is None:
            return None

        action_h, action_w = self.action_shape
        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=action_h, w=action_w)

        # Decoder uses pixel projection context
        dec_first_frame_tokens = self.decoder_context_projection(first_frame)

        recon_video = self.decode(dec_first_frame_tokens, actions=tokens)
        if recon_video is None:
            return None
        recon_frames = rearrange(recon_video, "b c 1 h w -> b c h w")

        return recon_frames
