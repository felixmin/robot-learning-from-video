"""
Scatter plot strategies for LAM validation.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import warnings
import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lightning.pytorch as pl

from .core import ValidationStrategy, ValidationCache


class MetadataScatterStrategy(ValidationStrategy):
    """
    Base class for scatter plot strategies that visualize metadata (actions, states)
    colored by codebook tokens or sequences.

    Provides common functionality:
    - Sample filtering by metadata requirements
    - Sample limiting with random selection
    - Matplotlib figure creation and cleanup

    Note: Dataset filtering is handled by bucket bindings, not this class.
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
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
        self.num_samples = num_samples

    def needs_caching(self) -> bool:
        return True

    def needs_codes(self) -> bool:
        return True

    def _filter_samples_with_metadata(
        self,
        cache: ValidationCache,
        required_keys: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Any], List[int]]:
        """
        Filter samples that have required metadata keys.

        Args:
            cache: Validation cache
            required_keys: List of metadata keys required (e.g., ["action"])

        Returns:
            Tuple of (filtered_metadata, codes_list, valid_indices)
        """
        all_metadata = cache.get_all_metadata()
        all_codes = cache.get_codes()  # Use bounded codes for metadata correspondence

        if not all_metadata or all_codes is None:
            return [], [], []

        filtered_meta = []
        codes_list = []
        valid_indices = []

        for i, meta in enumerate(all_metadata):
            # Check all required keys exist
            has_all = True
            for key in required_keys:
                val = meta.get(key)
                if val is None:
                    has_all = False
                    break
                # Need at least 2D for scatter plots
                if isinstance(val, (list, tuple)) and len(val) < 2:
                    has_all = False
                    break

            if has_all and i < len(all_codes):
                filtered_meta.append(meta)
                codes_list.append(all_codes[i])
                valid_indices.append(i)

        return filtered_meta, codes_list, valid_indices

    def _limit_samples(
        self,
        *arrays,
        n: Optional[int] = None,
    ) -> Tuple:
        """
        Randomly limit samples to n (or self.num_samples).

        Args:
            *arrays: Variable number of lists/tensors to sample from
            n: Number of samples (defaults to self.num_samples)

        Returns:
            Tuple of sampled arrays in same order as input
        """
        if not arrays or len(arrays[0]) == 0:
            return tuple([] for _ in arrays)

        n = n or self.num_samples
        total = len(arrays[0])
        n = min(n, total)

        # Use numpy for random choice if not using torch tensors
        import torch

        indices = torch.randperm(total)[:n].tolist()

        result = []
        for arr in arrays:
            if isinstance(arr, torch.Tensor):
                result.append(arr[indices])
            elif isinstance(arr, np.ndarray):
                result.append(arr[indices])
            else:
                result.append([arr[i] for i in indices])

        return tuple(result)

    def _create_scatter_figure(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar_label: str = "Token ID",
        num_colors: int = 8,
        figsize: Tuple[int, int] = (8, 8),
        stats_text: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Create a scatter plot figure.

        Args:
            x, y: Coordinates for scatter points
            colors: Values for coloring points
            title: Plot title
            xlabel, ylabel: Axis labels
            colorbar_label: Label for colorbar
            num_colors: Number of distinct colors in colormap
            figsize: Figure size tuple
            stats_text: Optional text to show in corner

        Returns:
            PIL Image of the figure, or None on error
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)

            cmap = plt.cm.get_cmap("tab20", num_colors)
            scatter = ax.scatter(
                x,
                y,
                c=colors,
                cmap=cmap,
                alpha=0.6,
                s=20,
                vmin=0,
                vmax=max(num_colors - 1, colors.max()),
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            plt.colorbar(scatter, ax=ax, label=colorbar_label)

            if stats_text:
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            plt.close(fig)
            return img
        except Exception as e:
            print(f"Warning: scatter plot creation failed: {e}")
            return None


class ActionTokenScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of 2D actions colored by their assigned codebook tokens.

    Only runs when samples have 'action' metadata with 2D values.
    This helps visualize how the codebook discretizes continuous action space.
    """

    def __init__(
        self,
        name: str = "action_token_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate action-token scatter plot."""
        # Filter samples with action metadata
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(
            cache, ["action"]
        )

        if len(filtered_meta) < self.min_samples:
            return self.no_output("insufficient_action_metadata")

        # Extract actions and first token from codes
        actions = [meta["action"][:2] for meta in filtered_meta]
        tokens = [c[0].item() if c.ndim > 0 else c.item() for c in codes_list]

        # Limit samples
        actions, tokens = self._limit_samples(actions, tokens)

        if not actions:
            return self.no_output("no_actions_after_sampling")

        # Create scatter plot
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            actions_np = np.array(actions)
            tokens_np = np.array(tokens)
            codebook_size = pl_module.model.vq.num_embeddings

            img = self._create_scatter_figure(
                x=actions_np[:, 0],
                y=actions_np[:, 1],
                colors=tokens_np,
                title=f"2D Actions Colored by Codebook Token (Step {trainer.global_step})",
                xlabel="Action X (cumulative dx)",
                ylabel="Action Y (cumulative dy)",
                colorbar_label="Token ID",
                num_colors=codebook_size,
                stats_text=f"Samples: {len(actions)}\nUnique tokens: {len(set(tokens))}",
            )

            if img:
                wandb_logger.log_image(
                    key=f"val/action_token_scatter{metric_suffix}",
                    images=[img],
                    caption=[
                        f"Step {trainer.global_step}: 2D actions colored by assigned token"
                    ],
                )

        return self.success(produced=int(len(actions)))


class ActionSequenceScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of 2D actions colored by their assigned FULL token sequence.

    This visualizes if specific action trajectories (dx, dy) map consistently
    to specific latent code sequences.
    """

    def __init__(
        self,
        name: str = "action_sequence_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate action-sequence scatter plot."""
        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(
            cache, ["action"]
        )

        if len(filtered_meta) < self.min_samples:
            return self.no_output("insufficient_action_metadata")

        # Map unique sequences to IDs
        sequences = [tuple(c.tolist()) for c in codes_list]
        unique_seqs = list(set(sequences))
        seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
        num_unique_seqs = len(unique_seqs)

        # Extract actions and sequence IDs
        actions = [meta["action"][:2] for meta in filtered_meta]
        seq_ids = [seq_to_id[seq] for seq in sequences]

        # Use base class sample limiting
        actions, seq_ids = self._limit_samples(actions, seq_ids)

        if not actions:
            return self.no_output("no_actions_after_sampling")

        # Create scatter plot
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_sequence_scatter(
                actions, seq_ids, wandb_logger, trainer.global_step, num_unique_seqs
            )

        return self.success(produced=int(len(actions)))

    def _create_sequence_scatter(
        self,
        actions: List[List[float]],
        seq_ids: List[int],
        wandb_logger,
        global_step: int,
        num_unique: int,
    ):
        """Create scatter plot colored by sequence ID."""
        try:
            if num_unique > 100:
                warnings.warn(
                    f"ActionSequenceScatterStrategy: {num_unique} unique sequences. Colors may be indistinguishable."
                )

            actions_np = np.array(actions)
            ids_np = np.array(seq_ids)

            fig, ax = plt.subplots(figsize=(10, 8))

            cmap = plt.cm.get_cmap("nipy_spectral", num_unique)
            markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]

            # Plot each sequence ID with unique marker and color
            for uid in np.unique(ids_np):
                mask = ids_np == uid
                marker = markers[uid % len(markers)]
                color = cmap(uid / max(1, num_unique - 1))
                ax.scatter(
                    actions_np[mask, 0],
                    actions_np[mask, 1],
                    color=color,
                    marker=marker,
                    alpha=0.6,
                    s=30,
                )

            ax.set_xlabel("Action X (cumulative dx)")
            ax.set_ylabel("Action Y (cumulative dy)")
            ax.set_title(f"2D Actions Colored by Full Sequence (Step {global_step})")
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            norm = plt.Normalize(vmin=0, vmax=num_unique - 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Sequence ID")

            ax.text(
                0.02,
                0.98,
                f"Samples: {len(actions)}\nUnique Seqs: {num_unique}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/action_sequence_scatter{metric_suffix}",
                images=[img],
                caption=[f"Step {global_step}: Colored by unique sequence ID"],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: action_sequence_scatter visualization failed: {e}")


class TopSequencesScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot highlighting ONLY the top N most frequent latent sequences.

    Top sequences get distinct high-contrast colors.
    All other sequences are plotted in grey.
    This helps visualize if the most common modes correspond to specific
    actions (e.g., "move forward", "stop") or are scattered noise.
    """

    def __init__(
        self,
        name: str = "top_sequences_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        num_top_sequences: int = 5,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )
        self.num_top_sequences = num_top_sequences

    def required_metadata(self) -> List[str]:
        return ["action"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate top sequences scatter plot."""
        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(
            cache, ["action"]
        )

        if len(filtered_meta) < self.min_samples:
            return self.no_output("insufficient_action_metadata")

        # Extract actions and sequences
        actions = [meta["action"][:2] for meta in filtered_meta]
        sequences = [tuple(c.tolist()) for c in codes_list]

        # Find top N sequences
        counter = Counter(sequences)
        top_seqs_counts = counter.most_common(self.num_top_sequences)
        top_seqs = {seq for seq, _ in top_seqs_counts}
        seq_to_cat = {seq: i for i, (seq, _) in enumerate(top_seqs_counts)}

        # Map to categories: 0..N-1 for top, -1 for others
        categories = [seq_to_cat.get(seq, -1) for seq in sequences]

        # Use base class sample limiting
        actions, categories = self._limit_samples(actions, categories)

        if not actions:
            return self.no_output("no_actions_after_sampling")

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_top_scatter(
                actions, categories, top_seqs_counts, wandb_logger, trainer.global_step
            )

        return self.success(produced=int(len(actions)))

    def _create_top_scatter(
        self,
        actions: List[List[float]],
        categories: List[int],
        top_seqs_counts: List[Tuple[Tuple[int, ...], int]],
        wandb_logger,
        global_step: int,
    ):
        """Create scatter plot with top sequences highlighted."""
        try:
            actions_np = np.array(actions)
            cats_np = np.array(categories)

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot "Other" (Grey) first
            mask_other = cats_np == -1
            if np.any(mask_other):
                ax.scatter(
                    actions_np[mask_other, 0],
                    actions_np[mask_other, 1],
                    c="lightgrey",
                    alpha=0.5,
                    s=15,
                    label=f"Others ({np.sum(mask_other)})",
                    zorder=1,
                )

            # Plot Top N with distinct colors
            colors = plt.cm.tab10.colors
            for i, (seq, count) in enumerate(top_seqs_counts):
                mask = cats_np == i
                if np.any(mask):
                    ax.scatter(
                        actions_np[mask, 0],
                        actions_np[mask, 1],
                        c=[colors[i % len(colors)]],
                        alpha=0.9,
                        s=30,
                        label=f"{seq}: {count}",
                        zorder=2,
                    )

            ax.set_xlabel("Action X (cumulative dx)")
            ax.set_ylabel("Action Y (cumulative dy)")
            ax.set_title(
                f"Top {self.num_top_sequences} Sequences vs Action (Step {global_step})"
            )
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/top_sequences_scatter{metric_suffix}",
                images=[img],
                caption=[
                    f"Step {global_step}: Top {self.num_top_sequences} sequences highlighted"
                ],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: top_sequences_scatter visualization failed: {e}")


class StateSequenceScatterStrategy(MetadataScatterStrategy):
    """
    Scatter plot of ROBOT STATE (x, y) colored by assigned token sequence.

    Visualizes how latent sequences distribute across the state space.
    Highlights the top N most frequent sequences to check for spatial clusters.
    Requires 'initial_state' in metadata.
    """

    def __init__(
        self,
        name: str = "state_sequence_scatter",
        enabled: bool = True,
        every_n_validations: int = 1,
        num_samples: int = 1000,
        num_top_sequences: int = 20,
        min_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            enabled=enabled,
            every_n_validations=every_n_validations,
            num_samples=num_samples,
            min_samples=min_samples,
            **kwargs,  # Pass buckets, dataset_filter, etc.
        )
        self.num_top_sequences = num_top_sequences

    def required_metadata(self) -> List[str]:
        return ["initial_state"]

    def run(
        self,
        cache: ValidationCache,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        metric_suffix: str = "",
    ) -> Dict[str, Any]:
        """Generate state-sequence scatter plot."""
        # Use base class filtering
        filtered_meta, codes_list, _ = self._filter_samples_with_metadata(
            cache, ["initial_state"]
        )

        if len(filtered_meta) < self.min_samples:
            return self.no_output("insufficient_state_metadata")

        # Extract states and sequences
        states = [meta["initial_state"][:2] for meta in filtered_meta]
        sequences = [tuple(c.tolist()) for c in codes_list]

        # Find top N sequences
        counter = Counter(sequences)
        top_seqs_counts = counter.most_common(self.num_top_sequences)
        seq_to_cat = {seq: i for i, (seq, _) in enumerate(top_seqs_counts)}

        # Map to categories: 0..N-1 for top, -1 for others
        categories = [seq_to_cat.get(seq, -1) for seq in sequences]

        # Use base class sample limiting
        states, categories = self._limit_samples(states, categories)

        if not states:
            return self.no_output("no_states_after_sampling")

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is not None:
            self._create_state_scatter(
                states, categories, top_seqs_counts, wandb_logger, trainer.global_step
            )

        return self.success(produced=int(len(states)))

    def _create_state_scatter(
        self,
        states: List[List[float]],
        categories: List[int],
        top_seqs_counts: List[Tuple[Tuple[int, ...], int]],
        wandb_logger,
        global_step: int,
    ):
        """Create scatter plot."""
        try:
            states_np = np.array(states)
            # Add jitter to reveal overlaps
            jitter = np.random.normal(0, 0.005, size=states_np.shape)
            states_np = states_np + jitter
            cats_np = np.array(categories)

            fig, ax = plt.subplots(figsize=(10, 8))

            cmap = plt.cm.get_cmap("tab20", self.num_top_sequences)
            markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]

            # Plot "Other" (Grey) first
            mask_other = cats_np == -1
            if np.any(mask_other):
                ax.scatter(
                    states_np[mask_other, 0],
                    states_np[mask_other, 1],
                    c="lightgrey",
                    marker=".",
                    alpha=0.3,
                    s=15,
                    label=f"Others ({np.sum(mask_other)})",
                    zorder=1,
                )

            # Plot Top N with markers
            for i, (seq, count) in enumerate(top_seqs_counts):
                mask = cats_np == i
                if np.any(mask):
                    ax.scatter(
                        states_np[mask, 0],
                        states_np[mask, 1],
                        color=cmap(i),
                        marker=markers[i % len(markers)],
                        alpha=0.8,
                        s=40,
                        label=f"{seq}: {count}",
                        zorder=2,
                    )

            ax.set_xlabel("State X")
            ax.set_ylabel("State Y")
            ax.set_title(
                f"State vs Top {self.num_top_sequences} Sequences (Step {global_step})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger.log_image(
                key=f"val/state_sequence_scatter{metric_suffix}",
                images=[img],
                caption=[
                    f"Step {global_step}: Top {self.num_top_sequences} sequences by state"
                ],
            )

            plt.close(fig)
        except Exception as e:
            print(f"Warning: state_sequence_scatter visualization failed: {e}")
