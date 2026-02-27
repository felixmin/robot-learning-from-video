from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence

import torch
import torch.nn.functional as F

from foundation.action_tokens import ActionTokenConfig
from foundation.backends.interfaces import BackendMode, FoundationBatch, LatentOutput, LossOutput
from foundation.image_adapters import oxe_first_frames_to_pil
from foundation.vla_inputs import ChatConfig


def _resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1.0) -> torch.Tensor:
    """GPU-accelerated image resize with aspect-ratio preserving padding.

    LeRobot-style resize that maintains aspect ratio and pads to target size.
    """
    if img.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got shape {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # Pad on left and top of image (LeRobot convention)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def _gpu_preprocess_images(
    frames: torch.Tensor, target_size: tuple[int, int] = (384, 384), normalize: bool = True
) -> torch.Tensor:
    """GPU-accelerated image preprocessing for SmolVLM.

    Args:
        frames: Tensor of shape (B, C, H, W) with values in [0, 255] (uint8) or [0, 1] (float)
        target_size: Target (height, width) for SigLIP
        normalize: If True, normalize from [0, 1] to [-1, 1] for SigLIP

    Returns:
        Preprocessed tensor on the same device
    """
    # Convert to float if needed (uint8 -> float in [0, 1])
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.dtype != torch.float32 and frames.dtype != torch.float16 and frames.dtype != torch.bfloat16:
        frames = frames.float()

    # Resize with aspect-ratio preserving padding
    img = _resize_with_pad(frames, target_size[1], target_size[0], pad_value=0.0)

    # Normalize from [0, 1] to [-1, 1] as expected by SigLIP
    if normalize:
        img = img * 2.0 - 1.0

    return img


def _masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)


def _infer_hidden_size(model: Any) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    raise AttributeError("Could not infer hidden size from model.config")


def _get_last_layer_module(model: Any, optimized: bool = False) -> torch.nn.Module | None:
    """Find the last transformer layer to attach a hook for capturing hidden states.

    Args:
        model: The VLM model
        optimized: If True, return the text_model's last layer (for direct text_model calls)
                   If False, return the full VLM's last layer (for full VLM forward)
    """
    if optimized:
        # For optimized path: we call text_model directly
        # SmolVLM: model.model.text_model.layers[-1]
        if hasattr(model, "model") and hasattr(model.model, "text_model"):
            text_model = model.model.text_model
            if hasattr(text_model, "layers"):
                return text_model.layers[-1]
        # Direct text_model
        if hasattr(model, "layers"):
            return model.layers[-1]
    else:
        # For original path: full VLM forward
        # SmolVLM2 uses model.language_model.model.layers[-1]
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers[-1]
        # Fallback paths
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[-1]
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[-1]
    return None


@dataclass
class SmolLatentHeadBackendConfig:
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4))
    # Optimization flags
    use_gpu_preprocessing: bool = True  # Use GPU-accelerated image preprocessing (48000x faster)
    image_size: tuple[int, int] = (384, 384)  # SigLIP expected size


class SmolLatentHeadBackend(torch.nn.Module):
    """
    SmolVLM latent-head backend (LeRobot `latent_smol`-style latent mode).

    - Conditions on frame t only (first frame) to avoid leakage from (t+Δ).
    - Predicts LAQ codes via CE loss from pooled prefix hidden state.
    - Uses forward hook to capture last hidden state efficiently (avoids storing all layers).
    """

    def __init__(
        self,
        *,
        config: SmolLatentHeadBackendConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        frames_to_images: Callable[[torch.Tensor], List[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.vlm = vlm
        self.processor = processor
        self.frames_to_images = frames_to_images or oxe_first_frames_to_pil

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)

        self.laq_head: torch.nn.Linear | None = None
        self._last_hidden_state: torch.Tensor | None = None
        self._hook_handle: Any = None

        # Cached components for optimized forward path
        self._vision_model: torch.nn.Module | None = None
        self._connector: torch.nn.Module | None = None
        self._text_model: torch.nn.Module | None = None
        self._text_embeddings: torch.nn.Module | None = None

    @staticmethod
    def _extract_frames(batch: FoundationBatch) -> torch.Tensor:
        streams = batch.image_streams
        if streams is None:
            raise ValueError("batch.image_streams is required")
        if "observation.images.rgb" not in streams:
            raise KeyError("batch.image_streams must include key 'observation.images.rgb'")
        return streams["observation.images.rgb"]

    @staticmethod
    def _extract_instructions(batch: FoundationBatch) -> list[str]:
        if batch.task_text is None:
            raise ValueError("batch.task_text is required")
        return [str(x) for x in batch.task_text]

    def _capture_hidden_state_hook(
        self, module: torch.nn.Module, input: Any, output: Any
    ) -> None:
        """Forward hook to capture hidden state from the last transformer layer."""
        # Output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._last_hidden_state = hidden

    def setup(self, *, device: torch.device) -> None:
        if self.vlm is None or self.processor is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.vlm = AutoModelForImageTextToText.from_pretrained(
                self.cfg.model_name,
                torch_dtype=self.cfg.torch_dtype,
                trust_remote_code=bool(self.cfg.trust_remote_code),
            )
            self.processor = AutoProcessor.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=bool(self.cfg.trust_remote_code),
            )

        self.vlm.to(device)
        self.vlm.train()  # Enable training mode for backbone

        # Register forward hook on the last transformer layer
        # Use optimized=True if GPU preprocessing is enabled (we call text_model directly)
        last_layer = _get_last_layer_module(self.vlm, optimized=self.cfg.use_gpu_preprocessing)
        if last_layer is not None and self._hook_handle is None:
            self._hook_handle = last_layer.register_forward_hook(self._capture_hidden_state_hook)

        if self.laq_head is None:
            hidden = _infer_hidden_size(self.vlm)
            try:
                vlm_dtype = next(self.vlm.parameters()).dtype
            except StopIteration:
                vlm_dtype = self.cfg.torch_dtype
            self.laq_head = torch.nn.Linear(hidden, self.code_seq_len * self.codebook_size).to(
                device=device, dtype=vlm_dtype
            )

    def _require_ready(self) -> tuple[torch.device, torch.nn.Module, Any, torch.nn.Linear]:
        if self.vlm is None or self.processor is None or self.laq_head is None:
            raise RuntimeError("Backend not initialized. Call setup(device=...) first.")
        try:
            device = next(self.vlm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return device, self.vlm, self.processor, self.laq_head

    def _build_texts(self, instructions: Sequence[str]) -> list[str]:
        proc = self.processor
        sys = self.cfg.chat.system_prompt

        apply_chat = getattr(proc, "apply_chat_template", None) if proc is not None else None
        if apply_chat is None:
            if sys:
                return [f"{sys}\n{instr}" for instr in instructions]
            return [str(instr) for instr in instructions]

        texts: list[str] = []
        for instr in instructions:
            messages: list[dict[str, Any]] = []
            if sys:
                messages.append({"role": "system", "content": [{"type": "text", "text": str(sys)}]})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": str(instr)},
                    ],
                }
            )
            texts.append(str(apply_chat(messages, tokenize=False, add_generation_prompt=False)))
        return texts

    def _forward_logits(self, batch: FoundationBatch) -> torch.Tensor:
        """Dispatch to optimized or original forward path based on config."""
        _device, _vlm, _processor, head = self._require_ready()
        pooled = self._forward_pooled(batch)
        if pooled.dtype != head.weight.dtype:
            pooled = pooled.to(dtype=head.weight.dtype)
        return head(pooled).view(-1, self.code_seq_len, self.codebook_size)

    def _forward_pooled(self, batch: FoundationBatch) -> torch.Tensor:
        """Return pooled trunk features before the latent action head."""
        if self.cfg.use_gpu_preprocessing:
            return self._forward_pooled_optimized(batch)
        return self._forward_pooled_original(batch)

    def _forward_pooled_original(self, batch: FoundationBatch) -> torch.Tensor:
        """Original forward path using HF processor (slow but reference implementation)."""
        device, vlm, processor, _head = self._require_ready()

        frames = self._extract_frames(batch)
        images_1 = self.frames_to_images(frames)
        # SmolVLMProcessor expects nested images: one sublist per sample.
        images = [[img] for img in images_1]
        texts = self._build_texts(self._extract_instructions(batch))

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            raise KeyError("processor output must include attention_mask")

        # Clear previous hidden state
        self._last_hidden_state = None

        # Run forward pass - hook will capture the last hidden state
        # Use output_hidden_states=False to avoid memory overhead
        if self._hook_handle is not None:
            # Hook path: efficient, only stores last layer output
            _out = vlm(**inputs, output_hidden_states=False, return_dict=True)
            if self._last_hidden_state is None:
                raise RuntimeError("Forward hook did not capture hidden state.")
            last = self._last_hidden_state
        else:
            # Fallback path: use output_hidden_states (less efficient)
            out = vlm(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states.")
            last = hidden_states[-1]

        attn = inputs["attention_mask"].to(dtype=torch.bool)
        pooled = _masked_mean_pool(last, attn)
        return pooled

    def _forward_pooled_optimized(self, batch: FoundationBatch) -> torch.Tensor:
        """Optimized forward path bypassing HF processor entirely (LeRobot-style).

        This approach:
        1. Preprocesses images on GPU (48000x faster than HF processor)
        2. Calls vision encoder directly (bypasses image splitting/patching)
        3. Uses tokenizer for text, not full processor
        4. Manually constructs embeddings and runs through text model

        The HF processor is slow because:
        - CPU-bound PIL image handling
        - Image splitting creates 17 patches even for small images
        - All operations run on CPU

        By calling model components directly, we get massive speedups.
        """
        device, vlm, processor, _head = self._require_ready()

        # Cache model components on first call
        if self._vision_model is None:
            self._cache_model_components(vlm)

        # Get first frame from batch - handle multiple layouts
        # OXE data can be:
        #   [B, 2, H, W, 3] - channels last (most common)
        #   [B, 2, 3, H, W] - channels first
        #   [B, 3, T, H, W] - LAQ video format
        frames = self._extract_frames(batch)
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with ndim=5, got {frames.ndim}")

        if frames.shape[-1] == 3:
            # [B, T, H, W, 3] -> [B, H, W, 3] -> [B, 3, H, W]
            first = frames[:, 0]  # [B, H, W, 3]
            first = first.permute(0, 3, 1, 2)  # [B, 3, H, W]
        elif frames.shape[2] == 3:
            # [B, T, 3, H, W] -> [B, 3, H, W]
            first = frames[:, 0]  # [B, 3, H, W]
        elif frames.shape[1] == 3:
            # [B, 3, T, H, W] -> [B, 3, H, W]
            first = frames[:, :, 0]  # [B, 3, H, W]
        else:
            raise ValueError(f"Unrecognized frame layout: {tuple(frames.shape)}")

        batch_size = first.shape[0]

        # ========================================
        # 1. GPU Image Preprocessing
        # ========================================
        # LeRobot-style: resize with padding, normalize to [-1, 1]
        # first is now [B, 3, H, W] in channels-first format
        pixel_values = _gpu_preprocess_images(
            first.to(device=device),
            target_size=self.cfg.image_size,
            normalize=True,
        ).to(dtype=self.cfg.torch_dtype)

        # ========================================
        # 2. Vision Encoding (Direct Call)
        # ========================================
        # Call vision encoder directly - bypasses HF processor entirely
        # This is 0.06ms vs 3080ms for HF processor!
        image_hidden_states = self._vision_model(
            pixel_values=pixel_values.to(dtype=self._vision_model.embeddings.patch_embedding.weight.dtype),
            patch_attention_mask=None,
        ).last_hidden_state

        # Apply connector (projector/resampler)
        image_embeds = self._connector(image_hidden_states)
        num_image_tokens = image_embeds.shape[1]

        # ========================================
        # 3. Text Tokenization (Tokenizer only)
        # ========================================
        # Build text with instruction
        texts = self._build_texts_simple(self._extract_instructions(batch))

        # Tokenize - use tokenizer directly, not processor
        tokenizer = processor.tokenizer
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = text_inputs["input_ids"].to(device)
        text_attention_mask = text_inputs["attention_mask"].to(device)

        # Embed text tokens
        text_embeds = self._text_embeddings(input_ids)

        # ========================================
        # 4. Build Combined Sequence
        # ========================================
        # SmolVLM expects: [image_tokens] [text_tokens]
        # We concatenate image embeddings with text embeddings
        # Note: LeRobot normalizes by sqrt(dim), but for standard SmolVLM this is not needed
        # as the model expects raw embeddings without additional scaling

        # Concatenate: [image] [text]
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Build attention mask: all image tokens + text attention mask
        image_attention_mask = torch.ones(batch_size, num_image_tokens, device=device, dtype=torch.bool)
        combined_attention_mask = torch.cat([image_attention_mask, text_attention_mask.bool()], dim=1)

        # Build position ids
        position_ids = torch.arange(combined_embeds.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)

        # ========================================
        # 5. Forward Through Text Model
        # ========================================
        # Clear previous hidden state
        self._last_hidden_state = None

        # Run through text model layers using our hook
        if self._hook_handle is not None:
            # Use the language_model.model for direct forward with embeddings
            # This is the LlamaModel inside SmolVLM
            outputs = self._text_model(
                inputs_embeds=combined_embeds.to(dtype=self._text_model.embed_tokens.weight.dtype),
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
                return_dict=True,
            )
            # Hook should have captured last hidden state
            if self._last_hidden_state is None:
                # Fallback: use output directly
                last = outputs.last_hidden_state
            else:
                last = self._last_hidden_state
        else:
            # Fallback without hook
            outputs = self._text_model(
                inputs_embeds=combined_embeds.to(dtype=self._text_model.embed_tokens.weight.dtype),
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            last = outputs.last_hidden_state

        # Pool and predict
        pooled = _masked_mean_pool(last, combined_attention_mask)
        return pooled

    def _cache_model_components(self, vlm: torch.nn.Module) -> None:
        """Cache references to model components for direct access."""
        # SmolVLM structure: vlm.model.{vision_model, connector, text_model}
        model = vlm.model if hasattr(vlm, "model") else vlm

        self._vision_model = model.vision_model
        self._connector = model.connector
        self._text_model = model.text_model
        self._text_embeddings = model.text_model.embed_tokens

    def _build_texts_simple(self, instructions: Sequence[str]) -> list[str]:
        """Build simple text prompts without chat template (for direct tokenization)."""
        sys = self.cfg.chat.system_prompt
        if sys:
            return [f"{sys}\n{instr}" for instr in instructions]
        return [str(instr) for instr in instructions]

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}")
        if batch.target_codes is None:
            raise ValueError("batch.target_codes is required for latent-head training.")

        logits = self._forward_logits(batch)
        codes = batch.target_codes.to(device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits.reshape(-1, self.codebook_size), codes.reshape(-1))
        return LossOutput(loss=loss, metrics={"loss": float(loss.detach().cpu().item())})

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}")
        logits = self._forward_logits(batch)
        tokens = logits.argmax(dim=-1)
        return LatentOutput(logits=logits, tokens=tokens, vector=None, meta=None)


@dataclass
class SmolFlowActionBackendConfig:
    model_name: str
    latent_vector_dim: int
    action_dim: int
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = False
    chat: ChatConfig = field(default_factory=lambda: ChatConfig(system_prompt=None))
    action_tokens: ActionTokenConfig = field(default_factory=lambda: ActionTokenConfig(codebook_size=8, code_seq_len=4))
    use_gpu_preprocessing: bool = True
    image_size: tuple[int, int] = (384, 384)
    flow_hidden_dim: int = 1024
    flow_steps: int = 8
    latent_loss_weight: float = 1.0
    action_loss_weight: float = 1.0


class SmolFlowActionBackend(SmolLatentHeadBackend):
    """
    Shared-trunk SmolVLM backend with two lightweight heads:
    - Flow-matching head for continuous LAQ codebook vectors
    - Action regression head for real robot actions
    """

    def __init__(
        self,
        *,
        config: SmolFlowActionBackendConfig,
        vlm: torch.nn.Module | None = None,
        processor: Any | None = None,
        frames_to_images: Callable[[torch.Tensor], List[Any]] | None = None,
    ) -> None:
        super().__init__(
            config=SmolLatentHeadBackendConfig(
                model_name=config.model_name,
                torch_dtype=config.torch_dtype,
                trust_remote_code=config.trust_remote_code,
                chat=config.chat,
                action_tokens=config.action_tokens,
                use_gpu_preprocessing=config.use_gpu_preprocessing,
                image_size=config.image_size,
            ),
            vlm=vlm,
            processor=processor,
            frames_to_images=frames_to_images,
        )
        self.flow_cfg = config
        self.latent_vector_dim = int(config.latent_vector_dim)
        self.action_dim = int(config.action_dim)

        self.flow_head: torch.nn.Module | None = None
        self.action_head: torch.nn.Linear | None = None

    def setup(self, *, device: torch.device) -> None:
        super().setup(device=device)
        if self.laq_head is None:
            raise RuntimeError("Smol latent head must be initialized before flow/action heads")

        hidden = int(self.laq_head.in_features)
        dtype = self.laq_head.weight.dtype
        if self.flow_head is None:
            self.flow_head = torch.nn.Sequential(
                torch.nn.Linear(hidden + self.latent_vector_dim + 1, int(self.flow_cfg.flow_hidden_dim)),
                torch.nn.SiLU(),
                torch.nn.Linear(int(self.flow_cfg.flow_hidden_dim), self.latent_vector_dim),
            ).to(device=device, dtype=dtype)
        if self.action_head is None:
            self.action_head = torch.nn.Linear(hidden, self.action_dim).to(device=device, dtype=dtype)

    def _require_multitask_ready(self) -> tuple[torch.device, torch.nn.Module, torch.nn.Linear]:
        if self.vlm is None or self.flow_head is None or self.action_head is None:
            raise RuntimeError("Backend not initialized. Call setup(device=...) first.")
        try:
            device = next(self.vlm.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return device, self.flow_head, self.action_head

    def _require_target_vector(self, batch: FoundationBatch) -> torch.Tensor:
        if batch.target_latent_vectors is None:
            raise ValueError("batch.target_latent_vectors is required for flow-matching latent training.")
        vec = batch.target_latent_vectors
        if vec.ndim == 3:
            vec = vec.reshape(vec.shape[0], -1)
        elif vec.ndim != 2:
            raise ValueError(f"Expected target_latent_vectors [B,S,D] or [B,D], got {tuple(vec.shape)}")
        if vec.shape[1] != self.latent_vector_dim:
            raise ValueError(
                f"target_latent_vectors dim mismatch: expected {self.latent_vector_dim}, got {int(vec.shape[1])}"
            )
        return vec

    def _require_target_actions(self, batch: FoundationBatch) -> torch.Tensor:
        if batch.target_actions is None:
            raise ValueError("batch.target_actions is required for real-action training.")
        actions = batch.target_actions
        if actions.ndim != 2:
            raise ValueError(f"Expected target_actions [B,A], got {tuple(actions.shape)}")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"target_actions dim mismatch: expected {self.action_dim}, got {int(actions.shape[1])}")
        return actions

    def _flow_matching_loss(self, pooled: torch.Tensor, target_vec: torch.Tensor, flow_head: torch.nn.Module) -> torch.Tensor:
        x_1 = target_vec
        x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0], 1), device=x_1.device, dtype=x_1.dtype)
        x_t = (1.0 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        v_pred = flow_head(torch.cat([pooled, x_t, t], dim=-1))
        return F.mse_loss(v_pred, v_target)

    def _flow_sample(self, pooled: torch.Tensor, flow_head: torch.nn.Module) -> torch.Tensor:
        steps = int(self.flow_cfg.flow_steps)
        if steps <= 0:
            raise ValueError("flow_steps must be > 0")
        batch_size = int(pooled.shape[0])
        x = torch.randn((batch_size, self.latent_vector_dim), device=pooled.device, dtype=pooled.dtype)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t_val = float(i) / float(steps)
            t = torch.full((batch_size, 1), t_val, device=pooled.device, dtype=pooled.dtype)
            v = flow_head(torch.cat([pooled, x, t], dim=-1))
            x = x + dt * v
        return x

    def loss_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LossOutput:
        if mode is BackendMode.CODES:
            return super().loss_from_batch(batch, mode=mode)
        if mode not in (BackendMode.ACTIONS, BackendMode.MULTITASK):
            raise NotImplementedError(f"{type(self).__name__} does not support mode={mode.value!r}")

        _device, flow_head, action_head = self._require_multitask_ready()
        pooled = self._forward_pooled(batch)
        if pooled.dtype != action_head.weight.dtype:
            pooled = pooled.to(dtype=action_head.weight.dtype)

        pred_actions = action_head(pooled)
        target_actions = self._require_target_actions(batch).to(device=pred_actions.device, dtype=pred_actions.dtype)
        action_loss = F.mse_loss(pred_actions, target_actions)

        if mode is BackendMode.ACTIONS:
            return LossOutput(
                loss=action_loss,
                metrics={
                    "loss": float(action_loss.detach().cpu().item()),
                    "action_loss": float(action_loss.detach().cpu().item()),
                },
            )

        target_vec = self._require_target_vector(batch).to(device=pooled.device, dtype=pooled.dtype)
        latent_loss = self._flow_matching_loss(pooled, target_vec, flow_head)
        total = float(self.flow_cfg.latent_loss_weight) * latent_loss + float(self.flow_cfg.action_loss_weight) * action_loss
        return LossOutput(
            loss=total,
            metrics={
                "loss": float(total.detach().cpu().item()),
                "latent_loss": float(latent_loss.detach().cpu().item()),
                "action_loss": float(action_loss.detach().cpu().item()),
            },
        )

    @torch.no_grad()
    def latent_from_batch(self, batch: FoundationBatch, *, mode: BackendMode) -> LatentOutput:
        if mode is BackendMode.CODES:
            return super().latent_from_batch(batch, mode=mode)
        if mode not in (BackendMode.ACTIONS, BackendMode.MULTITASK):
            raise NotImplementedError(f"{type(self).__name__} does not support mode={mode.value!r}")

        _device, flow_head, action_head = self._require_multitask_ready()
        pooled = self._forward_pooled(batch)
        if pooled.dtype != action_head.weight.dtype:
            pooled = pooled.to(dtype=action_head.weight.dtype)
        pred_actions = action_head(pooled)

        if mode is BackendMode.ACTIONS:
            return LatentOutput(logits=None, tokens=None, vector=None, actions=pred_actions, meta=None)

        vec = self._flow_sample(pooled, flow_head)
        return LatentOutput(logits=None, tokens=None, vector=vec, actions=pred_actions, meta=None)
