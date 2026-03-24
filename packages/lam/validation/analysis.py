"""
Analysis strategies for LAM validation (latent space, clustering, histograms).
"""

from collections import Counter
from typing import Any, Dict, List
import io

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from einops import rearrange
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from PIL import Image

from .core import ValidationStrategy, ValidationCache
from .metrics import compute_entropy


def _has_reconstruction_decoder(pl_module: pl.LightningModule) -> bool:
    model = getattr(pl_module, "model", None)
    return bool(
        getattr(model, "aux_decoder", None) is not None
        or getattr(model, "pixel_decoder", None) is not None
    )


class LatentTransferStrategy(ValidationStrategy):
    """
    Test if latent actions transfer between different scenes.

    For pairs (s_a, s_a') and (s_b, s_b'):
    1. Encode z_a = E(s_a, s_a')
    2. Apply z_a to s_b: s_b'_pred = D(s_b, z_a)
    3. Compare s_b'_pred with actual s_b' (should be different)
       and with s_a' (should be similar in "action" applied)

    This measures how "action-like" vs "state-specific" the latents are.
    Useful for comparing IID vs holdout buckets to see if actions generalize.
    """

    def __init__(
        self,
        name: str = "latent_transfer",
        enabled: bool = True,
        every_n_validations: int = 10,
        num_pairs: int = 256,
        min_samples: int = 4,  # Need at least 4 samples for 2 pairs
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_pairs = num_pairs

    def needs_caching(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Run latent transfer analysis."""
        metrics = {}

        if not _has_reconstruction_decoder(pl_module):
            return self.no_output("reconstruction_decoder_unavailable")

        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) < 4:
            return self.no_output("insufficient_frames")

        # Sample pairs
        n = min(self.num_pairs, len(all_frames) // 2)
        indices = torch.randperm(len(all_frames))[: n * 2]

        # Split into source and target pairs
        source_frames = all_frames[indices[:n]]  # (s_a, s_a')
        target_frames = all_frames[indices[n:]]  # (s_b, s_b')

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            device = pl_module.device
            source_frames = source_frames.to(device)
            target_frames = target_frames.to(device)

            # Encode source pairs to get latent actions
            # Use task helper which handles raw pixels -> quantized latents
            source_latents, source_indices = pl_module.encode_latents(
                source_frames
            )  # z_a

            # Get target initial frames
            target_s0 = target_frames[:, :, 0:1]  # s_b (keep dim for concat)

            # Also encode target frames to get their true indices (for comparison)
            _, target_indices = pl_module.encode_latents(target_frames)

            # Decode: apply source latent to target initial frame
            # Use task helper which handles embedding and reshaping
            transferred_recons = pl_module.decode_with_latents(
                target_s0,
                source_latents,
            )  # s_b'_pred

            # Remove time dim if present [B, C, 1, H, W] -> [B, C, H, W]
            if transferred_recons.ndim == 5:
                transferred_recons = transferred_recons.squeeze(2)

            # Get ground truth
            target_s1_true = target_frames[:, :, 1]  # s_b' (true)
            # Compute transfer error (pred vs true target)
            transfer_mse = F.mse_loss(transferred_recons, target_s1_true)

            # Compute self-reconstruction error (for reference)
            self_recons = pl_module.model(target_frames, return_recons_only=True)
            self_mse = F.mse_loss(self_recons, target_s1_true)
        pl_module.train(was_training)

        # Use metric_suffix for bucket-specific logging
        metrics[f"val/latent_transfer_mse{metric_suffix}"] = transfer_mse.item()
        metrics[f"val/self_recon_mse{metric_suffix}"] = self_mse.item()
        metrics[f"val/transfer_ratio{metric_suffix}"] = transfer_mse.item() / (
            self_mse.item() + 1e-8
        )

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)

        # Visualize some transfers
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_transfers(
                source_frames[:4].cpu(),
                target_frames[:4].cpu(),
                transferred_recons[:4].cpu(),
                self_recons[:4].cpu(),  # True reconstruction for comparison
                source_indices[:4].cpu(),
                target_indices[:4].cpu(),
                wandb_logger,
                trainer.global_step,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(n), metrics=metrics)

    def _visualize_transfers(
        self,
        source_frames: torch.Tensor,
        target_frames: torch.Tensor,
        transferred: torch.Tensor,
        true_recon: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ):
        """
        Visualize latent transfer results with true reconstruction comparison.

        Grid columns:
        1. s_a: Source first frame
        2. s_a': Source second frame (ground truth action result)
        3. s_b: Target first frame
        4. D(s_b, z_a): Transfer reconstruction (using source's latent)
        5. D(s_b, z_b): True reconstruction (using target's own latent)
        6. s_b': Target second frame (ground truth)
        """
        s_a = source_frames[:, :, 0]  # Source first frame
        s_a_prime = source_frames[:, :, 1]  # Source action result (GT)
        s_b = target_frames[:, :, 0]  # Target first frame
        s_b_recon_true = true_recon  # D(s_b, z_b) - true recon
        s_b_recon_transfer = transferred  # D(s_b, z_a) - transfer recon
        s_b_prime_true = target_frames[:, :, 1]  # Target GT

        num_samples = len(s_a)
        if num_samples == 0:
            return

        # Create figure
        # 6 columns of images
        fig, axes = plt.subplots(
            num_samples, 6, figsize=(20, 3.5 * num_samples), squeeze=False
        )

        # Column titles
        col_titles = [
            "s_a",
            "s_a'",
            "s_b",
            "D(s_b, z_a)\n(Transfer)",
            "D(s_b, z_b)\n(Self)",
            "s_b'",
        ]

        for i in range(num_samples):
            # Row images (swapped 4 and 5 as requested)
            # 4: Transfer (z_a applied to s_b)
            # 5: Self (z_b applied to s_b)
            imgs = [
                s_a[i],
                s_a_prime[i],
                s_b[i],
                s_b_recon_transfer[i],
                s_b_recon_true[i],
                s_b_prime_true[i],
            ]

            # Prepare token string
            tokens_a = str(source_indices[i].tolist())
            tokens_b = str(target_indices[i].tolist())

            row_text = (
                f"Row {i}\nDet(a): {tokens_a}\nDet(b): {tokens_b}\nApp(b): {tokens_a}"
            )

            for j, img in enumerate(imgs):
                ax = axes[i, j]

                # Convert [C, H, W] to [H, W, C] for imshow and clamp
                img_np = img.permute(1, 2, 0).clamp(0.0, 1.0).numpy()

                ax.imshow(img_np)
                ax.axis("off")

                if i == 0:
                    ax.set_title(col_titles[j], fontsize=12)

                if j == 0:
                    # Add text to the left of the first image of the row
                    ax.text(
                        -0.2,
                        0.5,
                        row_text,
                        transform=ax.transAxes,
                        va="center",
                        ha="right",
                        fontsize=10,
                        rotation=0,
                        family="monospace",
                    )

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)

        wandb_logger.log_image(
            key=f"val/latent_transfer{metric_suffix}",
            images=[img],
            caption=[f"Step {global_step}"],
        )
        plt.close(fig)


class PermutedLatentVisualizationStrategy(ValidationStrategy):
    """Compare self reconstruction with a reconstruction from batch-permuted latents."""

    def __init__(
        self,
        name: str = "permuted_latent_visualization",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 8,
        min_samples: int = 2,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,
        )
        self.num_samples = num_samples

    def needs_caching(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        if not _has_reconstruction_decoder(pl_module):
            return self.no_output("reconstruction_decoder_unavailable")

        all_frames = cache.get_all_frames()
        if all_frames is None or len(all_frames) < 2:
            return self.no_output("insufficient_frames")

        n = min(int(self.num_samples), len(all_frames))
        indices = torch.randperm(len(all_frames))[:n]
        frames = all_frames[indices]

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            device = pl_module.device
            frames = frames.to(device)
            self_recons = pl_module.model(frames, return_recons_only=True)
            if self_recons is None:
                pl_module.train(was_training)
                return self.no_output("reconstruction_unavailable")

            latents, latent_indices = pl_module.encode_latents(frames)
            if n == 2:
                perm = torch.tensor([1, 0], device=latents.device)
            else:
                shift = int(torch.randint(1, n, size=(1,), device=latents.device).item())
                perm = torch.roll(torch.arange(n, device=latents.device), shifts=shift)
            permuted_recons = pl_module.decode_with_latents(
                frames[:, :, 0:1],
                latents[perm],
            )
            if permuted_recons is None:
                pl_module.train(was_training)
                return self.no_output("reconstruction_unavailable")
            if permuted_recons.ndim == 5:
                permuted_recons = permuted_recons.squeeze(2)
        pl_module.train(was_training)

        target_frames = frames[:, :, 1]
        self_mse = F.mse_loss(self_recons, target_frames)
        permuted_mse = F.mse_loss(permuted_recons, target_frames)
        metrics = {
            f"val/permuted_latent_self_recon_mse{metric_suffix}": self_mse.item(),
            f"val/permuted_latent_mse{metric_suffix}": permuted_mse.item(),
            f"val/permuted_latent_ratio{metric_suffix}": permuted_mse.item()
            / (self_mse.item() + 1e-8),
        }
        pl_module.log_dict(metrics, sync_dist=True)

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_permuted_latents(
                frames=frames.cpu(),
                self_recons=self_recons.cpu(),
                permuted_recons=permuted_recons.cpu(),
                latent_indices=latent_indices.cpu(),
                perm=perm.cpu(),
                wandb_logger=wandb_logger,
                global_step=trainer.global_step,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(n), metrics=metrics)

    def _visualize_permuted_latents(
        self,
        *,
        frames: torch.Tensor,
        self_recons: torch.Tensor,
        permuted_recons: torch.Tensor,
        latent_indices: torch.Tensor,
        perm: torch.Tensor,
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ) -> None:
        num_samples = len(frames)
        if num_samples == 0:
            return

        fig, axes = plt.subplots(
            num_samples, 4, figsize=(14, 3.5 * num_samples), squeeze=False
        )
        col_titles = [
            "s_t",
            "s_t+1",
            "D(s_t, z_self)",
            "D(s_t, z_perm)",
        ]

        for i in range(num_samples):
            imgs = [
                frames[i, :, 0],
                frames[i, :, 1],
                self_recons[i],
                permuted_recons[i],
            ]
            perm_source = int(perm[i].item())

            for j, img in enumerate(imgs):
                ax = axes[i, j]
                img_np = img.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
                ax.imshow(img_np)
                ax.axis("off")
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=12)
                if j == 0:
                    ax.text(
                        -0.15,
                        0.5,
                        "\n".join(
                            [
                                f"perm<-{perm_source}",
                                f"self: {latent_indices[i].tolist()}",
                                f"perm: {latent_indices[perm_source].tolist()}",
                            ]
                        ),
                        transform=ax.transAxes,
                        va="center",
                        ha="right",
                        fontsize=9,
                        family="monospace",
                    )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)

        wandb_logger.log_image(
            key=f"val/permuted_latent_visualization{metric_suffix}",
            images=[img],
            caption=[f"Step {global_step}"],
        )
        plt.close(fig)


class CodebookHistogramStrategy(ValidationStrategy):
    """
    Visualize codebook usage distribution as a histogram.

    Shows which codebook entries are used most/least frequently,
    helping identify if the codebook is being utilized effectively.
    """

    def __init__(
        self,
        name: str = "codebook_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True  # Need codes for histogram

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate codebook usage histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return self.no_output("no_cached_codes")

        # Flatten codes to get all codebook indices used
        # all_codes shape: [N, code_seq_len] where values are codebook indices
        codes_flat = all_codes.flatten()

        # Get codebook size from model (NSVQ uses num_embeddings)
        codebook_size = pl_module.model.vq.num_embeddings

        # Count usage per codebook entry
        counts = torch.bincount(codes_flat.long(), minlength=codebook_size)

        # Compute metrics with suffix
        used_codes = (counts > 0).sum().item()
        metrics[f"val/codebook_entry_utilization_val_cache_all{metric_suffix}"] = (
            used_codes / codebook_size
        )
        metrics[f"val/codebook_usage_entropy_val_cache_all{metric_suffix}"] = (
            compute_entropy(counts.float())
        )

        # Log to trainer
        pl_module.log_dict(metrics, sync_dist=True)

        # Create histogram visualization
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(
                counts,
                wandb_logger,
                trainer.global_step,
                codebook_size,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(len(all_codes)), metrics=metrics)

    def _create_histogram(
        self,
        counts: torch.Tensor,
        wandb_logger,
        global_step: int,
        codebook_size: int,
        metric_suffix: str = "",
    ):
        """Create and log histogram of codebook usage."""
        try:
            fig, ax = plt.subplots(figsize=(10, 4))

            x = range(codebook_size)
            ax.bar(x, counts.cpu().numpy(), color="steelblue", alpha=0.8)
            ax.set_xlabel("Codebook Index")
            ax.set_ylabel("Usage Count")
            ax.set_title(f"Codebook Usage Distribution (Step {global_step})")

            # Add statistics text
            used = (counts > 0).sum().item()
            ax.text(
                0.95,
                0.95,
                f"Used: {used}/{codebook_size}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/codebook_histogram{metric_suffix}",
                images=[img],
                caption=[f"Step {global_step}"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: codebook_histogram visualization failed: {e}")


class LatentSequenceHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of latent token sequences (combinations).

    Since the combination space can be large (e.g., 8^4 = 4096), this strategy
    plots the top-N most frequent sequences to show if the model collapses
    to a few specific action patterns.
    """

    def __init__(
        self,
        name: str = "sequence_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_top_sequences: int = 50,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )
        self.num_top_sequences = num_top_sequences

    def needs_caching(self) -> bool:
        return True  # Need codes for sequence analysis

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate sequence usage histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return self.no_output("no_cached_codes")

        # all_codes shape: [N, code_seq_len]
        # Convert to list of tuples for counting
        sequences = [tuple(c.tolist()) for c in all_codes]

        counter = Counter(sequences)

        # Metrics
        unique_seqs = len(counter)
        total_samples = len(all_codes)

        # Calculate entropy of sequence distribution
        counts = torch.tensor(list(counter.values()), dtype=torch.float)
        metrics[f"val/latent_sequence_entropy_val_cache_all{metric_suffix}"] = (
            compute_entropy(counts)
        )
        metrics[f"val/latent_sequence_unique_count_val_cache_all{metric_suffix}"] = (
            unique_seqs
        )
        metrics[f"val/latent_sequence_sample_count_val_cache_all{metric_suffix}"] = (
            total_samples
        )

        pl_module.log_dict(metrics, sync_dist=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_histogram(
                counter, wandb_logger, trainer.global_step, metric_suffix=metric_suffix
            )

        return self.success(produced=int(len(all_codes)), metrics=metrics)

    def _create_histogram(
        self,
        counter,
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ):
        """Create and log histogram of top sequence usage."""
        try:
            # Get top N most common
            most_common = counter.most_common(self.num_top_sequences)
            if not most_common:
                return

            labels, values = zip(*most_common)
            # Convert tuple labels to strings "1-2-3-4"
            str_labels = ["-".join(map(str, label)) for label in labels]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(values))
            ax.bar(x, values, color="mediumpurple", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(str_labels, rotation=90, fontsize=8)
            ax.set_xlabel("Token Sequence")
            ax.set_ylabel("Count")
            ax.set_title(f"Top {len(values)} Latent Sequences (Step {global_step})")

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/sequence_histogram{metric_suffix}",
                images=[img],
                caption=[
                    f"Step {global_step}: Distribution of top {len(values)} sequences"
                ],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: sequence_histogram visualization failed: {e}")


class AllSequencesHistogramStrategy(ValidationStrategy):
    """
    Visualize the distribution of ALL latent token sequences (sorted frequency).

    This shows the "long tail" of the distribution.
    X-axis: Sequence rank (1 to N)
    Y-axis: Frequency count
    No labels on X-axis to avoid clutter.
    """

    def __init__(
        self,
        name: str = "all_sequences_histogram",
        enabled: bool = True,
        every_n_validations: int = 1,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, etc.
        )

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate all sequences histogram."""
        metrics = {}

        # Use all_codes for true distribution across all validation samples
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            return self.no_output("no_cached_codes")

        # all_codes shape: [N, code_seq_len]
        sequences = [tuple(c.tolist()) for c in all_codes]
        counter = Counter(sequences)

        # Sort counts descending
        sorted_counts = sorted(counter.values(), reverse=True)

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_plot(
                sorted_counts,
                wandb_logger,
                trainer.global_step,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(len(sorted_counts)), metrics=metrics)

    def _create_plot(
        self,
        counts: List[int],
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ):
        """Create and log plot of all sequence counts."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(counts))
            ax.bar(x, counts, color="teal", width=1.0, alpha=0.8)
            # Alternatively use plot/fill_between for very dense data
            # ax.plot(x, counts, color='teal')
            # ax.fill_between(x, counts, color='teal', alpha=0.3)

            ax.set_xlabel("Sequence Rank")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Distribution of All {len(counts)} Unique Sequences (Step {global_step})"
            )
            ax.set_yscale("log")  # Log scale helps see the tail

            # Add stats
            ax.text(
                0.95,
                0.95,
                f"Total Unique: {len(counts)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/all_sequences_histogram{metric_suffix}",
                images=[img],
                caption=[f"Step {global_step}: Long tail distribution (log scale)"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: all_sequences_histogram visualization failed: {e}")


class CodebookEmbeddingStrategy(ValidationStrategy):
    """
    Visualize codebook geometry via t-SNE dimensionality reduction.

    Projects the learned codebook embedding vectors to 2D to reveal
    whether the codebook has learned distinct semantic clusters or
    is just a uniform sphere (which might imply poor learning).

    Each point is a codebook entry (one row of vq.codebooks).
    Points are colored by usage frequency from cached validation codes.
    """

    def __init__(
        self,
        name: str = "codebook_embedding",
        enabled: bool = True,
        every_n_validations: int = 10,
        method: str = "tsne",  # "tsne" or "umap" (umap requires umap-learn)
        perplexity: int = 30,  # t-SNE perplexity parameter
        pca_components: int = 50,  # PCA preprocessing for speed/stability (0 to disable)
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,
        )
        self.method = method
        self.perplexity = perplexity
        self.pca_components = pca_components

    def needs_caching(self) -> bool:
        return True  # Need codes to compute usage frequencies

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate codebook embedding visualization."""
        metrics = {}

        # Get codebook embeddings from model
        # Shape: [num_embeddings, embedding_dim] e.g. [256, 32]
        codebook = pl_module.model.vq.codebooks.detach().cpu().numpy()
        num_embeddings = codebook.shape[0]
        embedding_dim = codebook.shape[1]

        # Compute usage counts from cached codes
        all_codes = cache.get_all_codes()
        if all_codes is None or len(all_codes) == 0:
            # No codes to compute usage, use zeros (unknown usage)
            usage_counts = torch.zeros(num_embeddings)
        else:
            # Flatten all codes and count occurrences per codebook entry
            codes_flat = all_codes.flatten().long()
            usage_counts = torch.bincount(codes_flat, minlength=num_embeddings).float()

        usage_counts_np = usage_counts.numpy()

        # Apply dimensionality reduction
        try:
            embeddings_2d = self._reduce_dimensions(codebook, embedding_dim)
        except Exception as e:
            print(f"Warning: codebook_embedding dimensionality reduction failed: {e}")
            return self.no_output("dimensionality_reduction_failed")

        # Create visualization
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_scatter(
                embeddings_2d,
                usage_counts_np,
                wandb_logger,
                trainer.global_step,
                num_embeddings,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(num_embeddings), metrics=metrics)

    def _reduce_dimensions(self, codebook, embedding_dim):
        """Apply t-SNE (or UMAP) with optional PCA preprocessing."""

        n_samples = len(codebook)

        # t-SNE requires at least 2 samples and perplexity < n_samples
        # Skip dimensionality reduction for very small codebooks
        if n_samples < 2:
            raise ValueError(
                f"Codebook too small for visualization: {n_samples} entries"
            )

        data = codebook

        # PCA preprocessing for speed/stability when embedding_dim is large
        if self.pca_components > 0 and embedding_dim > self.pca_components:
            from sklearn.decomposition import PCA

            n_components = min(self.pca_components, n_samples - 1, embedding_dim)
            if n_components > 0:
                pca = PCA(n_components=n_components, random_state=42)
                data = pca.fit_transform(data)

        if self.method == "tsne":
            from sklearn.manifold import TSNE

            # t-SNE requires perplexity < n_samples and perplexity >= 1
            # For very small codebooks, fall back to PCA if t-SNE can't run
            max_perplexity = max(1, n_samples - 1)
            effective_perplexity = min(self.perplexity, max_perplexity)
            if effective_perplexity < 5 and n_samples < 10:
                # t-SNE won't work well, use simple PCA projection
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(codebook)
            else:
                effective_perplexity = max(5, effective_perplexity)
                tsne = TSNE(
                    n_components=2,
                    perplexity=effective_perplexity,
                    random_state=42,
                    n_iter=1000,
                )
                embeddings_2d = tsne.fit_transform(data)
        elif self.method == "umap":
            try:
                from umap import UMAP

                # UMAP also needs n_neighbors < n_samples
                n_neighbors = min(15, n_samples - 1)
                if n_neighbors < 2:
                    # Fall back to PCA
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=2, random_state=42)
                    embeddings_2d = pca.fit_transform(codebook)
                else:
                    reducer = UMAP(
                        n_components=2, n_neighbors=n_neighbors, random_state=42
                    )
                    embeddings_2d = reducer.fit_transform(data)
            except ImportError:
                print("Warning: umap-learn not installed, falling back to t-SNE")
                # Temporarily switch method to avoid recursion
                original_method = self.method
                self.method = "tsne"
                try:
                    embeddings_2d = self._reduce_dimensions(codebook, embedding_dim)
                finally:
                    self.method = original_method
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'tsne' or 'umap'.")

        return embeddings_2d

    def _create_scatter(
        self,
        embeddings_2d,
        usage_counts,
        wandb_logger,
        global_step: int,
        num_embeddings: int,
        metric_suffix: str = "",
    ):
        """Create and log scatter plot of codebook embeddings."""
        import numpy as np

        try:
            fig, ax = plt.subplots(figsize=(10, 10))

            # Use log1p for visibility (many tokens may have 0 usage)
            colors = np.log1p(usage_counts)

            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=colors,
                cmap="viridis",
                alpha=0.7,
                s=20,
                edgecolors="none",
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("log(usage + 1)")

            # Add statistics
            used_tokens = (usage_counts > 0).sum()
            total_usage = usage_counts.sum()
            ax.set_title(
                f"Codebook Embedding Space ({self.method.upper()})\n"
                f"Step {global_step} | Used: {used_tokens}/{num_embeddings} tokens"
            )
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/codebook_embedding{metric_suffix}",
                images=[img],
                caption=[
                    f"Step {global_step}: {self.method.upper()} of {num_embeddings} codebook vectors, "
                    f"colored by usage ({int(total_usage)} total codes)"
                ],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: codebook_embedding visualization failed: {e}")


class SequenceExamplesStrategy(ValidationStrategy):
    """
    Visualize frame pairs grouped by exact code sequence identity.

    Replaces the flawed ClusteringStrategy approach (k-means on token indices)
    with mathematically sound grouping: samples are bucketed by their
    exact token sequence (e.g., all samples with code [3, 1, 2] together).

    This answers: "What does sequence X represent?" by showing the actual
    frame transitions that produced that sequence.
    """

    def __init__(
        self,
        name: str = "sequence_examples",
        enabled: bool = True,
        every_n_validations: int = 3,
        top_k_sequences: int = 16,  # Number of most frequent sequences to visualize
        examples_per_sequence: int = 4,  # Frame pairs to show per sequence
        min_samples: int = 16,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,
        )
        self.top_k_sequences = top_k_sequences
        self.examples_per_sequence = examples_per_sequence

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate sequence examples visualization."""
        metrics = {}

        # Get bounded codes and frames (they have correspondence)
        codes = cache.get_codes()
        frames = cache.get_all_frames()

        if codes is None or frames is None:
            return self.no_output("missing_codes_or_frames")

        if len(codes) < self.min_samples:
            return self.no_output("insufficient_samples")

        # Convert codes to sequence tuples for exact matching
        # codes shape: [N, code_seq_len] where values are codebook indices
        sequences = [tuple(c.tolist()) for c in codes]

        # Count sequence frequencies from bounded codes only
        # This ensures consistency: the count shown matches what's visualizable
        counter = Counter(sequences)

        # Get top-K most frequent sequences
        top_sequences = counter.most_common(self.top_k_sequences)

        if not top_sequences:
            return self.no_output("no_top_sequences")

        # Visualize
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_sequences(
                frames,
                sequences,
                top_sequences,
                wandb_logger,
                trainer.global_step,
                metric_suffix=metric_suffix,
            )

        return self.success(produced=int(len(top_sequences)), metrics=metrics)

    def _visualize_sequences(
        self,
        frames: torch.Tensor,
        sequences: List[tuple],
        top_sequences: List[tuple],
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ):
        """Create visualization grid for top sequences."""
        grids = []
        captions = []

        for seq, count in top_sequences:
            # Find all samples with this exact sequence
            indices = [i for i, s in enumerate(sequences) if s == seq]

            if len(indices) == 0:
                # Sequence from all_codes not present in bounded cache
                continue

            # Sample up to examples_per_sequence frames
            n_examples = min(self.examples_per_sequence, len(indices))
            sample_indices = indices[:n_examples]

            # Get frame pairs for this sequence
            example_frames = frames[sample_indices]  # [n_examples, C, 2, H, W]

            # Create mini-grid: [frame_t, frame_t+offset] for each example
            frame_t = example_frames[:, :, 0]  # [n_examples, C, H, W]
            frame_t_plus = example_frames[:, :, 1]  # [n_examples, C, H, W]

            # Interleave: [t0, t0+, t1, t1+, ...]
            imgs = torch.stack([frame_t, frame_t_plus], dim=1)  # [n, 2, C, H, W]
            imgs = rearrange(imgs, "b r c h w -> (b r) c h w")
            imgs = imgs.clamp(0.0, 1.0)

            grid = make_grid(imgs, nrow=2, normalize=False, padding=2)
            grids.append(grid)

            # Format sequence as string: "3-1-2" for readability
            seq_str = "-".join(map(str, seq))
            captions.append(f"[{seq_str}] (n={count})")

        if not grids:
            return

        # Create figure with individual sequence grids
        try:
            n_sequences = len(grids)
            # Determine layout: aim for roughly square grid of sequence panels
            n_cols = min(4, n_sequences)
            n_rows = (n_sequences + n_cols - 1) // n_cols

            # Each grid has shape [C, H, W] after make_grid
            # We'll create a subplot for each sequence
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 5 * n_rows),
                squeeze=False,
            )

            for idx, (grid, caption) in enumerate(zip(grids, captions)):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                # Convert grid tensor to numpy for imshow
                grid_np = grid.permute(1, 2, 0).numpy()  # [H, W, C]
                ax.imshow(grid_np)
                ax.set_title(caption, fontsize=10, family="monospace")
                ax.axis("off")

            # Hide empty subplots
            for idx in range(len(grids), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis("off")

            plt.suptitle(
                f"Top {len(grids)} Sequences by Frequency (Step {global_step})",
                fontsize=12,
            )
            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/sequence_examples{metric_suffix}",
                images=[img],
                caption=[
                    f"Step {global_step}: Frame pairs grouped by exact code sequence"
                ],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: sequence_examples visualization failed: {e}")


class TopSequenceApplicationStrategy(ValidationStrategy):
    """Apply the most frequent latent token sequences to one anchor frame."""

    def __init__(
        self,
        name: str = "top_sequence_applications",
        enabled: bool = True,
        every_n_validations: int = 1,
        top_k_sequences: int = 12,
        min_samples: int = 16,
        anchor_index: int = 0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            min_samples=min_samples,
            **kwargs,
        )
        self.top_k_sequences = top_k_sequences
        self.anchor_index = anchor_index

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        if not _has_reconstruction_decoder(pl_module):
            return self.no_output("reconstruction_decoder_unavailable")

        all_codes = cache.get_all_codes()
        frames = cache.get_all_frames()
        if all_codes is None or frames is None:
            return self.no_output("missing_codes_or_frames")
        if len(all_codes) < self.min_samples or len(frames) == 0:
            return self.no_output("insufficient_samples")

        sequences = [tuple(c.tolist()) for c in all_codes]
        counter = Counter(sequences)
        top_sequences = counter.most_common(self.top_k_sequences)
        if not top_sequences:
            return self.no_output("no_top_sequences")

        anchor_idx = min(max(int(self.anchor_index), 0), len(frames) - 1)
        anchor_pair = frames[anchor_idx : anchor_idx + 1]
        anchor_first = anchor_pair[:, :, 0:1]

        with torch.no_grad():
            device = pl_module.device
            seq_tensor = torch.tensor(
                [list(seq) for seq, _count in top_sequences],
                device=device,
                dtype=torch.long,
            )
            raw_latents = pl_module.model.vq.codebooks[seq_tensor]
            action_latents = pl_module.model.vq.project_out(raw_latents)
            repeated_anchor = anchor_first.to(device).repeat(len(top_sequences), 1, 1, 1, 1)
            decoded = pl_module.decode_with_latents(repeated_anchor, action_latents)
            if decoded is None:
                return self.no_output("reconstruction_unavailable")
            if decoded.ndim == 5:
                decoded = decoded.squeeze(2)

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._visualize_top_sequence_applications(
                anchor_pair=anchor_pair.cpu(),
                decoded=decoded.cpu(),
                top_sequences=top_sequences,
                wandb_logger=wandb_logger,
                global_step=trainer.global_step,
                metric_suffix=metric_suffix,
            )

        metrics = {
            f"val/top_sequence_application_count{metric_suffix}": int(len(top_sequences))
        }
        pl_module.log_dict(metrics, sync_dist=True)
        return self.success(produced=int(len(top_sequences)), metrics=metrics)

    def _visualize_top_sequence_applications(
        self,
        *,
        anchor_pair: torch.Tensor,
        decoded: torch.Tensor,
        top_sequences: List[tuple],
        wandb_logger,
        global_step: int,
        metric_suffix: str = "",
    ) -> None:
        num_rows = len(top_sequences)
        if num_rows == 0:
            return

        anchor_frame = anchor_pair[0, :, 0]
        true_next = anchor_pair[0, :, 1]

        fig, axes = plt.subplots(
            num_rows,
            3,
            figsize=(10, 3.2 * num_rows),
            squeeze=False,
        )
        col_titles = ["anchor", "true_next", "applied_sequence"]

        for row, ((seq, count), pred) in enumerate(zip(top_sequences, decoded)):
            imgs = [anchor_frame, true_next, pred]
            for col, img in enumerate(imgs):
                ax = axes[row, col]
                ax.imshow(img.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=11)
                if col == 0:
                    ax.set_ylabel(
                        f"{list(seq)}\ncount={count}",
                        fontsize=9,
                        family="monospace",
                        rotation=0,
                        labelpad=40,
                        va="center",
                    )

        plt.suptitle(
            f"Top {num_rows} latent sequences on one anchor frame (Step {global_step})",
            fontsize=12,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)

        wandb_logger.log_image(
            key=f"val/top_sequence_applications{metric_suffix}",
            images=[img],
            caption=[
                f"Step {global_step}: one row per exact latent token sequence, applied to a shared anchor frame"
            ],
        )
        plt.close(fig)
