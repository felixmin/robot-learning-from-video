import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class DINOFeatureExtractor(nn.Module):
    """
    Wrapper for DINOv3 (and v2) models from Hugging Face.

    Handles:
    - Loading pretrained weights (e.g. facebook/dinov3-vits16-pretrain-lvd1689m)
    - Freezing parameters
    - Input resizing (to multiple of patch size)
    - Normalization (ImageNet)
    - Feature extraction (last layer or multi-layer)
    - Handling Register Tokens (DINOv3 has 4 registers + 1 CLS)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze: bool = True,
        target_size: int = 256,
    ):
        super().__init__()

        print(f"Loading DINO model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.config = self.model.config

        # Determine patch size
        self.patch_size = getattr(self.config, "patch_size", 16)

        # Determine registers
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 0)

        # Adjust target size to be divisible by patch size
        if target_size % self.patch_size != 0:
            patches = round(target_size / self.patch_size)
            self.target_size = patches * self.patch_size
            print(
                f"Adjusted DINO input size from {target_size} to {self.target_size} (multiple of patch {self.patch_size})"
            )
        else:
            self.target_size = target_size

        self.hidden_size = self.config.hidden_size

        # Normalization stats (ImageNet)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, images, output_hidden_states: bool = False, layer_indices: list = None
    ):
        """
        Args:
            images: [B, C, H, W] tensor, value range [0, 1] usually
            output_hidden_states: Whether to return all layer outputs
            layer_indices: List of specific layer indices to return (if output_hidden_states=True)

        Returns:
            If output_hidden_states=False:
                patch_features: [B, H_grid, W_grid, D]
            If output_hidden_states=True:
                List of tensors for specified layers
        """
        # 1. Resize if needed
        if images.shape[-1] != self.target_size or images.shape[-2] != self.target_size:
            images = F.interpolate(
                images,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        # 2. Normalize
        images = (images - self.mean) / self.std

        # 3. Forward
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Calculate expected spatial dims
        h = w = self.target_size // self.patch_size

        # Helper to extract spatial tokens
        def extract_spatial(feat):
            # Dynamic detection of register tokens
            # feat: [B, N, D]
            # Expected grid tokens: h * w
            n_tokens = feat.shape[1]
            n_spatial = h * w

            if n_tokens < n_spatial:
                raise ValueError(
                    f"Feature sequence length {n_tokens} is smaller than expected grid {h}x{w}={n_spatial}"
                )

            # Assuming structure: [CLS, (Registers...), Spatial...]
            # The spatial tokens are always at the end
            feat_spatial = feat[:, -n_spatial:, :]

            return feat_spatial.reshape(feat.shape[0], h, w, -1)

        # 4. Extract
        if output_hidden_states:
            if layer_indices is None:
                return outputs.hidden_states

            selected_features = []
            for idx in layer_indices:
                # Add 1 because hidden_states[0] is embeddings
                layer_idx = idx + 1 if idx >= 0 else len(outputs.hidden_states) + idx
                if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
                    raise ValueError(
                        f"Layer index {idx} out of bounds for model with {len(outputs.hidden_states)-1} layers"
                    )

                feat = outputs.hidden_states[layer_idx]
                selected_features.append(extract_spatial(feat))

            return selected_features
        else:
            return extract_spatial(outputs.last_hidden_state)


class DINOEncoder(nn.Module):
    """
    Encoder module that uses DINO features + Projection.
    Replaces the raw pixel projection layer.
    """

    def __init__(self, feature_extractor, out_dim, pool_to_grid=None):
        """
        Args:
            feature_extractor: DINOFeatureExtractor instance
            out_dim: Output dimension for projection
            pool_to_grid: If set (e.g., 8), pool features from DINO's grid to this size.
                         This reduces memory for downstream transformers (O(n²) attention).
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.in_dim = feature_extractor.hidden_size
        self.out_dim = out_dim
        self.pool_to_grid = pool_to_grid

        # Projection layer to match LAM model dimension
        self.proj = nn.Linear(self.in_dim, out_dim)
        # self.norm = nn.LayerNorm(out_dim)  # Removed to allow learnable scale

        # Pooling layer if needed
        if pool_to_grid is not None:
            dino_grid = feature_extractor.target_size // feature_extractor.patch_size
            if dino_grid != pool_to_grid:
                # Use adaptive avg pool to downsample
                self.pool = nn.AdaptiveAvgPool2d((pool_to_grid, pool_to_grid))
                print(
                    f"  - Pooling DINO features from {dino_grid}x{dino_grid} to {pool_to_grid}x{pool_to_grid}"
                )
            else:
                self.pool = None
        else:
            self.pool = None

    def forward(self, images):
        # Extract features [B, H, W, D_dino]
        features = self.feature_extractor(images)

        # Pool if needed (before projection to save compute)
        if self.pool is not None:
            # [B, H, W, D] -> [B, D, H, W] -> pool -> [B, D, H', W'] -> [B, H', W', D]
            features = features.permute(0, 3, 1, 2)
            features = self.pool(features)
            features = features.permute(0, 2, 3, 1)

        # Project [B, H, W, D_dino] -> [B, H, W, D_model]
        features = self.proj(features)
        # features = self.norm(features)

        # Add Time dim [B, 1, h, w, d] to match LAM expected format
        features = features.unsqueeze(1)

        return features

    @property
    def output_grid_size(self):
        """Return the output grid size after pooling."""
        if self.pool_to_grid is not None:
            return self.pool_to_grid
        return self.feature_extractor.target_size // self.feature_extractor.patch_size


class DINOWrapper(nn.Module):
    """Wrapper to handle [B, C, 1, H, W] -> [B, C, H, W] conversion."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # x: [B, C, 1, H, W] -> [B, C, H, W]
        if x.ndim == 5 and x.shape[2] == 1:
            x = x.squeeze(2)
        return self.encoder(x)
