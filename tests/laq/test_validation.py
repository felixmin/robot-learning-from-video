"""
Tests for LAQ validation strategies.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from laq.validation import (
    ValidationCache,
    ValidationStrategy,
    BasicVisualizationStrategy,
    FlowVisualizationStrategy,
    LatentTransferStrategy,
    AllSequencesHistogramStrategy,
    CodebookEmbeddingStrategy,
    SequenceExamplesStrategy,
    create_validation_strategies,
    prune_metadata,
    ESSENTIAL_METADATA_KEYS,
    STRATEGY_REGISTRY,
)
from laq.callbacks import ValidationStrategyCallback, TrainPreviewBufferCallback


class TestValidationCache:
    """Test ValidationCache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = ValidationCache()
        assert len(cache.frames) == 0
        assert len(cache.latents) == 0
        assert len(cache.codes) == 0
        assert cache.fixed_frames is None
        assert len(cache.metadata) == 0

    def test_cache_append_and_get(self):
        """Test appending and retrieving cached data."""
        cache = ValidationCache()

        # Add some fake data
        frames1 = torch.randn(4, 3, 2, 64, 64)
        frames2 = torch.randn(4, 3, 2, 64, 64)
        cache.frames.append(frames1)
        cache.frames.append(frames2)

        all_frames = cache.get_all_frames()
        assert all_frames.shape == (8, 3, 2, 64, 64)

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ValidationCache()
        cache.frames.append(torch.randn(4, 3, 2, 64, 64))
        cache.latents.append(torch.randn(4, 32))

        cache.clear()
        assert len(cache.frames) == 0
        assert len(cache.latents) == 0
    
    def test_cache_metadata(self):
        """Test metadata storage and retrieval."""
        cache = ValidationCache()

        # Add metadata batches using standardized keys
        meta1 = [{"dataset_name": "youtube"}, {"dataset_name": "youtube"}]
        meta2 = [{"dataset_name": "bridge"}, {"dataset_name": "bridge"}]
        cache.metadata.append(meta1)
        cache.metadata.append(meta2)

        all_meta = cache.get_all_metadata()
        assert len(all_meta) == 4
        assert all_meta[0]["dataset_name"] == "youtube"
        assert all_meta[2]["dataset_name"] == "bridge"

    def test_get_frames_by_dataset_name(self):
        """Test filtering frames by dataset name."""
        cache = ValidationCache()

        # Add frames with metadata using standardized keys
        frames = torch.randn(4, 3, 2, 64, 64)
        cache.frames.append(frames)
        cache.metadata.append([
            {"dataset_name": "youtube"},
            {"dataset_name": "bridge"},
            {"dataset_name": "youtube"},
            {"dataset_name": "bridge"},
        ])

        youtube_frames = cache.get_frames_by_dataset_name("youtube")
        bridge_frames = cache.get_frames_by_dataset_name("bridge")

        assert youtube_frames.shape[0] == 2
        assert bridge_frames.shape[0] == 2


class TestValidationStrategy:
    """Test ValidationStrategy base class."""

    def test_should_run_every_n(self):
        """Test should_run logic with every_n_validations."""
        strategy = BasicVisualizationStrategy(
            enabled=True,
        )

        # Should run on first validation
        assert strategy.should_run()

        # Increment counter
        strategy.increment_count()

        # Should still run (every_n_validations=1)
        assert strategy.should_run()

    def test_should_run_disabled(self):
        """Test disabled strategy never runs."""
        strategy = BasicVisualizationStrategy(enabled=False)
        assert not strategy.should_run()

    def test_should_run_periodic(self):
        """Test periodic strategy runs at correct intervals."""
        strategy = LatentTransferStrategy(
            enabled=True,
            every_n_validations=5,
        )

        # Should run on first validation (count=0)
        assert strategy.should_run()

        # Skip next 4
        for i in range(4):
            strategy.increment_count()
            assert not strategy.should_run()

        # Should run again on 5th
        strategy.increment_count()  # count=5
        assert strategy.should_run()


class TestCreateValidationStrategies:
    """Test strategy creation from config."""

    def test_create_all_strategies(self):
        """Test creating all strategy types."""
        config = {
            "basic": {"enabled": True},
            "latent_transfer": {"enabled": True, "every_n_validations": 5},
            "sequence_examples": {"enabled": True, "every_n_validations": 3},
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 3
        # Names come from instance keys, not default strategy names
        names = {s.name for s in strategies}
        assert "basic" in names
        assert "latent_transfer" in names
        assert "sequence_examples" in names

    def test_create_only_basic(self):
        """Test creating only basic strategy."""
        config = {
            "basic": {"enabled": True},
            "latent_transfer": {"enabled": False},
            "sequence_examples": {"enabled": False},
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 1
        assert strategies[0].name == "basic"  # Name from instance key

    def test_empty_config(self):
        """Test with empty config."""
        strategies = create_validation_strategies({})
        assert len(strategies) == 0


class TestBasicVisualizationStrategy:
    """Test BasicVisualizationStrategy."""

    def test_needs_caching(self):
        """Test that basic strategy needs caching."""
        strategy = BasicVisualizationStrategy()
        assert strategy.needs_caching()

    def test_run_with_empty_cache(self):
        """Test run with empty cache returns empty metrics."""
        strategy = BasicVisualizationStrategy()
        cache = ValidationCache()

        # Create mock pl_module and trainer
        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}

    def test_train_visualization_uses_preview_buffer_callback(self):
        strategy = BasicVisualizationStrategy(
            visualize_train=True,
            visualize_val=False,
            num_train_samples=2,
        )
        cache = ValidationCache()
        pl_module = MagicMock()
        wandb_logger = MagicMock()
        strategy._get_wandb_logger = lambda _trainer: wandb_logger
        strategy._create_recon_grid = MagicMock(return_value=torch.zeros(3, 16, 16))

        cb = TrainPreviewBufferCallback(
            enabled=True,
            max_samples=16,
            samples_per_batch=2,
        )
        batch = {
            "frames": torch.rand(4, 3, 2, 8, 8),
            "dataset_name": ["bridge", "kuka", "rt1", "bridge"],
            "dataset_type": ["bridge", "kuka", "rt1", "bridge"],
        }
        cb.on_train_batch_end(MagicMock(), pl_module, None, batch, 0)

        trainer = MagicMock()
        trainer.global_step = 10
        trainer.callbacks = [cb]

        strategy.run(cache, pl_module, trainer)

        logged_keys = [call.kwargs["key"] for call in wandb_logger.log_image.call_args_list]
        assert "train/fixed_reconstructions" in logged_keys
        assert "train/random_reconstructions" in logged_keys

    def test_train_visualization_does_not_touch_train_dataloader(self):
        strategy = BasicVisualizationStrategy(
            visualize_train=True,
            visualize_val=False,
            num_train_samples=2,
        )
        cache = ValidationCache()
        pl_module = MagicMock()
        wandb_logger = MagicMock()
        strategy._create_recon_grid = MagicMock(return_value=torch.zeros(3, 16, 16))

        cb = TrainPreviewBufferCallback(
            enabled=True,
            max_samples=8,
            samples_per_batch=2,
        )
        batch = {
            "frames": torch.rand(4, 3, 2, 8, 8),
            "dataset_name": ["bridge", "kuka", "rt1", "bridge"],
        }
        cb.on_train_batch_end(MagicMock(), pl_module, None, batch, 0)

        class _Trainer:
            def __init__(self):
                self.global_step = 10
                self.callbacks = [cb]

            @property
            def train_dataloader(self):
                raise AssertionError("train_dataloader should not be accessed")

        trainer = _Trainer()
        strategy._visualize_training_samples(cache, pl_module, trainer, wandb_logger)


class TestTrainPreviewBufferCallback:
    def test_collect_and_sample(self):
        cb = TrainPreviewBufferCallback(
            enabled=True,
            max_samples=3,
            samples_per_batch=2,
        )
        batch = {
            "frames": torch.rand(4, 3, 2, 8, 8),
            "dataset_name": ["a", "b", "c", "d"],
            "dataset_type": ["x", "y", "z", "w"],
            "language": ["l1", "l2", "l3", "l4"],
        }

        cb.on_train_batch_end(MagicMock(), MagicMock(), None, batch, 0)
        cb.on_train_batch_end(MagicMock(), MagicMock(), None, batch, 1)

        frames, metadata = cb.sample(3)
        assert frames is not None
        assert metadata is not None
        assert frames.shape[0] == 3
        assert len(metadata) == 3
        assert all("dataset_name" in m for m in metadata)


class TestFlowVisualizationStrategy:
    """Test FlowVisualizationStrategy helpers."""

    def test_direction_panel_shape_and_range(self):
        strategy = FlowVisualizationStrategy()
        panel = strategy._create_direction_panel(
            gt_dx=3.0,
            gt_dy=-1.5,
            pred_dx=-2.0,
            pred_dy=0.5,
            height=64,
            width=64,
        )

        assert panel.shape == (3, 64, 64)
        assert torch.isfinite(panel).all()
        assert panel.min() >= 0.0
        assert panel.max() <= 1.0

    def test_direction_panel_handles_static_vectors(self):
        strategy = FlowVisualizationStrategy()
        panel = strategy._create_direction_panel(
            gt_dx=0.0,
            gt_dy=0.0,
            pred_dx=0.0,
            pred_dy=0.0,
            height=48,
            width=80,
        )

        assert panel.shape == (3, 48, 80)
        assert torch.isfinite(panel).all()


class TestLatentTransferStrategy:
    """Test LatentTransferStrategy."""

    def test_needs_caching(self):
        """Test that latent transfer strategy needs caching."""
        strategy = LatentTransferStrategy()
        assert strategy.needs_caching()

    def test_run_with_insufficient_data(self):
        """Test run with insufficient data returns empty metrics."""
        strategy = LatentTransferStrategy(num_pairs=10)
        cache = ValidationCache()

        # Only add 2 frames (need at least 4)
        cache.frames.append(torch.randn(2, 3, 2, 64, 64))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}


class TestCodebookEmbeddingStrategy:
    """Test CodebookEmbeddingStrategy."""

    def test_needs_caching(self):
        """Test that codebook embedding strategy needs caching."""
        strategy = CodebookEmbeddingStrategy()
        assert strategy.needs_caching()

    def test_needs_codes(self):
        """Test that codebook embedding strategy needs codes."""
        strategy = CodebookEmbeddingStrategy()
        assert strategy.needs_codes()

    def test_default_params(self):
        """Test default parameters."""
        strategy = CodebookEmbeddingStrategy()
        assert strategy.method == "tsne"
        assert strategy.perplexity == 30
        assert strategy.pca_components == 50
        assert strategy.every_n_validations == 10


class TestValidationStrategyCallbackBucketSuffix:
    def test_bucket_bound_strategy_gets_suffix(self):
        class DummyBucketStrategy(ValidationStrategy):
            def __init__(self):
                super().__init__(
                    name="dummy_bucket",
                    enabled=True,
                    every_n_validations=1,
                    min_samples=0,
                    buckets=["bridge"],
                )
                self.seen_suffix = None

            def needs_caching(self) -> bool:
                return False

            def run(self, cache, pl_module, trainer, metric_suffix: str = ""):
                self.seen_suffix = metric_suffix
                return {}

        strategy = DummyBucketStrategy()
        cb = ValidationStrategyCallback(
            strategies=[strategy],
            bucket_configs={
                "bridge": {"filters": {"dataset_name": "bridge"}, "max_samples": 1},
            },
            num_fixed_samples=0,
            num_random_samples=0,
            max_cached_samples=0,
            run_gc_after_validation=False,
        )

        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_validation_epoch_end(trainer, pl_module)

        assert strategy.seen_suffix == "_bridge"

    def test_holdout_bucket_adds_holdout_suffix(self):
        class DummyBucketStrategy(ValidationStrategy):
            def __init__(self):
                super().__init__(
                    name="dummy_bucket",
                    enabled=True,
                    every_n_validations=1,
                    min_samples=0,
                    buckets=["bridge"],
                )
                self.seen_suffix = None

            def needs_caching(self) -> bool:
                return False

            def run(self, cache, pl_module, trainer, metric_suffix: str = ""):
                self.seen_suffix = metric_suffix
                return {}

        strategy = DummyBucketStrategy()
        cb = ValidationStrategyCallback(
            strategies=[strategy],
            bucket_configs={
                "bridge": {
                    "filters": {"dataset_name": "bridge"},
                    "max_samples": 1,
                    "is_holdout": True,
                },
            },
            num_fixed_samples=0,
            num_random_samples=0,
            max_cached_samples=0,
            run_gc_after_validation=False,
        )

        trainer = MagicMock()
        pl_module = MagicMock()
        cb.on_validation_epoch_end(trainer, pl_module)

        assert strategy.seen_suffix == "_bridge_holdout"

    def test_custom_params(self):
        """Test custom parameters."""
        strategy = CodebookEmbeddingStrategy(
            method="umap",
            perplexity=50,
            pca_components=0,
            every_n_validations=5,
        )
        assert strategy.method == "umap"
        assert strategy.perplexity == 50
        assert strategy.pca_components == 0
        assert strategy.every_n_validations == 5


class TestSequenceExamplesStrategy:
    """Test SequenceExamplesStrategy."""

    def test_needs_caching(self):
        """Test that sequence examples strategy needs caching."""
        strategy = SequenceExamplesStrategy()
        assert strategy.needs_caching()

    def test_needs_codes(self):
        """Test that sequence examples strategy needs codes."""
        strategy = SequenceExamplesStrategy()
        assert strategy.needs_codes()

    def test_default_params(self):
        """Test default parameters."""
        strategy = SequenceExamplesStrategy()
        assert strategy.top_k_sequences == 16
        assert strategy.examples_per_sequence == 4
        assert strategy.every_n_validations == 3

    def test_run_with_insufficient_data(self):
        """Test run with insufficient data returns empty metrics."""
        strategy = SequenceExamplesStrategy(min_samples=16)
        cache = ValidationCache()

        # Only add 5 codes (need at least min_samples)
        cache.codes.append(torch.randint(0, 8, (5, 4)))
        cache.frames.append(torch.randn(5, 3, 2, 64, 64))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.loggers = []

        metrics = strategy.run(cache, pl_module, trainer)
        assert metrics == {}

    def test_run_forwards_metric_suffix_to_visualizer(self):
        """Test metric suffix is forwarded to _visualize_sequences."""
        strategy = SequenceExamplesStrategy(min_samples=2)
        strategy._visualize_sequences = MagicMock()

        cache = ValidationCache()
        cache.codes.append(torch.tensor([[1, 2], [1, 2]]))
        cache.frames.append(torch.randn(2, 3, 2, 16, 16))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.global_step = 42

        strategy._get_wandb_logger = lambda _trainer: MagicMock()

        strategy.run(cache, pl_module, trainer, metric_suffix="_bridge_holdout")

        strategy._visualize_sequences.assert_called_once()
        assert strategy._visualize_sequences.call_args.kwargs["metric_suffix"] == "_bridge_holdout"


class TestAllSequencesHistogramStrategy:
    """Test AllSequencesHistogramStrategy."""

    def test_run_forwards_metric_suffix_to_plotter(self):
        """Test metric suffix is forwarded to _create_plot."""
        strategy = AllSequencesHistogramStrategy()
        strategy._create_plot = MagicMock()

        cache = ValidationCache()
        cache.all_codes.append(torch.tensor([[1, 2], [1, 2], [3, 4]]))

        pl_module = MagicMock()
        trainer = MagicMock()
        trainer.global_step = 7

        strategy._get_wandb_logger = lambda _trainer: MagicMock()

        strategy.run(cache, pl_module, trainer, metric_suffix="_bridge_holdout")

        strategy._create_plot.assert_called_once()
        assert strategy._create_plot.call_args.kwargs["metric_suffix"] == "_bridge_holdout"


class TestMetadataPruning:
    """Test metadata pruning for RAM safety."""

    def test_prune_metadata_keeps_essential_keys(self):
        """Test that pruning keeps only essential metadata keys."""
        full_metadata = {
            "dataset_name": "bridge",
            "action": [0.1, 0.2, 0.3],
            "initial_state": [1.0, 2.0],
            "language": "pick up the block",  # Essential - should be kept
            "episode_id": "12345",  # Should be removed (not essential for caching)
            "frame_idx": 42,  # Should be removed
            "raw_observation": {"large": "dict"},  # Should be removed
        }

        pruned = prune_metadata(full_metadata)

        assert "dataset_name" in pruned
        assert "action" in pruned
        assert "initial_state" in pruned
        assert "language" in pruned  # Now kept as essential
        assert "episode_id" not in pruned
        assert "frame_idx" not in pruned
        assert "raw_observation" not in pruned

    def test_prune_metadata_empty_input(self):
        """Test pruning with empty metadata."""
        pruned = prune_metadata({})
        assert pruned == {}

    def test_cache_add_sample_prunes_by_default(self):
        """Test that add_sample prunes metadata by default."""
        cache = ValidationCache()

        frame = torch.randn(3, 2, 64, 64)
        full_meta = {
            "dataset_name": "bridge",
            "episode_id": "should be removed",
            "extra_field": "also removed",
        }

        cache.add_sample(frame, full_meta, prune=True)

        stored_meta = cache.get_all_metadata()[0]
        assert "dataset_name" in stored_meta
        assert "episode_id" not in stored_meta
        assert "extra_field" not in stored_meta

    def test_cache_add_sample_no_prune(self):
        """Test that add_sample can skip pruning."""
        cache = ValidationCache()

        frame = torch.randn(3, 2, 64, 64)
        full_meta = {
            "dataset_name": "bridge",
            "episode_id": "should be kept",
        }

        cache.add_sample(frame, full_meta, prune=False)

        stored_meta = cache.get_all_metadata()[0]
        assert "dataset_name" in stored_meta
        assert "episode_id" in stored_meta

    def test_cache_add_batch_prunes_by_default(self):
        """Test that add_batch prunes metadata by default."""
        cache = ValidationCache()

        frames = torch.randn(2, 3, 2, 64, 64)
        metadata_list = [
            {"dataset_name": "bridge", "episode_id": "remove1"},
            {"dataset_name": "youtube", "episode_id": "remove2"},
        ]

        cache.add_batch(frames, metadata_list, prune=True)

        all_meta = cache.get_all_metadata()
        assert len(all_meta) == 2
        for meta in all_meta:
            assert "dataset_name" in meta
            assert "episode_id" not in meta


class TestStrategyBuckets:
    """Test bucket configuration in strategies."""

    def test_strategy_default_empty_buckets(self):
        """Test that strategies have empty buckets by default."""
        strategy = BasicVisualizationStrategy()
        assert strategy.buckets == []

    def test_strategy_with_buckets(self):
        """Test creating strategy with bucket configuration."""
        strategy = LatentTransferStrategy(
            buckets=["language_table", "bridge"],
        )
        assert strategy.buckets == ["language_table", "bridge"]

    def test_strategy_single_bucket(self):
        """Test strategy with single bucket."""
        strategy = SequenceExamplesStrategy(
            buckets=["language_table"],
        )
        assert strategy.buckets == ["language_table"]


class TestCompositionPattern:
    """Test the composition pattern for strategy creation."""

    def test_create_with_type_field(self):
        """Test creating multiple instances of same type using 'type' field."""
        config = {
            "transfer_lt": {
                "type": "latent_transfer",
                "buckets": ["language_table"],
                "every_n_validations": 2,
            },
            "transfer_bridge": {
                "type": "latent_transfer",
                "buckets": ["bridge"],
                "every_n_validations": 5,
            },
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 2

        # Check instance names (not type names)
        names = {s.name for s in strategies}
        assert "transfer_lt" in names
        assert "transfer_bridge" in names

        # Check each has correct buckets
        lt_strategy = next(s for s in strategies if s.name == "transfer_lt")
        bridge_strategy = next(s for s in strategies if s.name == "transfer_bridge")

        assert lt_strategy.buckets == ["language_table"]
        assert lt_strategy.every_n_validations == 2
        assert bridge_strategy.buckets == ["bridge"]
        assert bridge_strategy.every_n_validations == 5

    def test_backwards_compat_no_type_field(self):
        """Test backwards compatibility when type field is omitted."""
        config = {
            "basic": {"enabled": True},
            "latent_transfer": {"enabled": True, "every_n_validations": 3},
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 2
        # When no type field, instance name is used as type
        names = {s.name for s in strategies}
        assert "basic" in names
        assert "latent_transfer" in names

    def test_skip_disabled_strategies(self):
        """Test that disabled strategies are skipped."""
        config = {
            "transfer_lt": {
                "type": "latent_transfer",
                "enabled": False,
            },
            "transfer_bridge": {
                "type": "latent_transfer",
                "enabled": True,
            },
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 1
        assert strategies[0].name == "transfer_bridge"

    def test_unknown_type_warning(self, capsys):
        """Test that unknown strategy type prints warning."""
        config = {
            "my_custom": {
                "type": "nonexistent_strategy",
            },
        }
        strategies = create_validation_strategies(config)

        assert len(strategies) == 0
        captured = capsys.readouterr()
        assert "Unknown strategy type" in captured.out

    def test_strategy_registry_completeness(self):
        """Test that all expected strategy types are in registry."""
        expected_types = [
            "basic",
            "basic_visualization",
            "latent_transfer",
            "codebook_histogram",
            "sequence_histogram",
            "all_sequences_histogram",
            "codebook_embedding",
            "sequence_examples",
            "action_token_scatter",
            "action_sequence_scatter",
            "top_sequences_scatter",
            "state_sequence_scatter",
            "flow_visualization",
        ]
        for strategy_type in expected_types:
            assert strategy_type in STRATEGY_REGISTRY, f"Missing: {strategy_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
