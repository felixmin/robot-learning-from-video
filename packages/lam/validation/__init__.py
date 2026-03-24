"""
LAM Validation Package.

Exports core infrastructure and strategy factory.
"""

from .core import (
    ValidationCache,
    BucketConfig,
    ValidationStrategy,
    ESSENTIAL_METADATA_KEYS,
    prune_metadata,
)
from .metrics import compute_entropy
from .factory import create_validation_strategies, STRATEGY_REGISTRY
from .visualization import BasicVisualizationStrategy
from .analysis import (
    LatentTransferStrategy,
    PermutedLatentVisualizationStrategy,
    TopSequenceApplicationStrategy,
    CodebookHistogramStrategy,
    LatentSequenceHistogramStrategy,
    AllSequencesHistogramStrategy,
    CodebookEmbeddingStrategy,
    SequenceExamplesStrategy,
)
from .scatter import (
    MetadataScatterStrategy,
    ActionTokenScatterStrategy,
    ActionSequenceScatterStrategy,
    TopSequencesScatterStrategy,
    StateSequenceScatterStrategy,
)
from .flow import FlowVisualizationStrategy

__all__ = [
    # Core
    "ValidationCache",
    "BucketConfig",
    "ValidationStrategy",
    "ESSENTIAL_METADATA_KEYS",
    "prune_metadata",
    # Metrics
    "compute_entropy",
    # Factory
    "create_validation_strategies",
    "STRATEGY_REGISTRY",
    # Strategies
    "BasicVisualizationStrategy",
    "LatentTransferStrategy",
    "PermutedLatentVisualizationStrategy",
    "TopSequenceApplicationStrategy",
    "CodebookHistogramStrategy",
    "LatentSequenceHistogramStrategy",
    "AllSequencesHistogramStrategy",
    "CodebookEmbeddingStrategy",
    "SequenceExamplesStrategy",
    "MetadataScatterStrategy",
    "ActionTokenScatterStrategy",
    "ActionSequenceScatterStrategy",
    "TopSequencesScatterStrategy",
    "StateSequenceScatterStrategy",
    "FlowVisualizationStrategy",
]
