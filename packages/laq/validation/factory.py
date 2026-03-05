"""
Factory for creating validation strategies from configuration.
"""

from typing import Any, Dict, List, Optional, Type

from .core import ValidationStrategy
from .visualization import BasicVisualizationStrategy
from .analysis import (
    LatentTransferStrategy,
    CodebookHistogramStrategy,
    LatentSequenceHistogramStrategy,
    AllSequencesHistogramStrategy,
    CodebookEmbeddingStrategy,
    SequenceExamplesStrategy,
)
from .scatter import (
    ActionTokenScatterStrategy,
    ActionSequenceScatterStrategy,
    TopSequencesScatterStrategy,
    StateSequenceScatterStrategy,
)
from .flow import FlowVisualizationStrategy


# Strategy type registry
STRATEGY_REGISTRY: Dict[str, Type[ValidationStrategy]] = {
    "basic": BasicVisualizationStrategy,
    "basic_visualization": BasicVisualizationStrategy,
    "latent_transfer": LatentTransferStrategy,
    "codebook_histogram": CodebookHistogramStrategy,
    "sequence_histogram": LatentSequenceHistogramStrategy,
    "all_sequences_histogram": AllSequencesHistogramStrategy,
    "codebook_embedding": CodebookEmbeddingStrategy,
    "sequence_examples": SequenceExamplesStrategy,
    "action_token_scatter": ActionTokenScatterStrategy,
    "action_sequence_scatter": ActionSequenceScatterStrategy,
    "top_sequences_scatter": TopSequencesScatterStrategy,
    "state_sequence_scatter": StateSequenceScatterStrategy,
    "flow_visualization": FlowVisualizationStrategy,
}


def create_validation_strategies(
    config: Dict[str, Any],
    bucket_filters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ValidationStrategy]:
    """
    Create validation strategies from config.

    Args:
        config: validation.strategies config dict
        bucket_filters: Optional dict of bucket_name -> filter dict (for visualization grouping)

    Returns:
        List of ValidationStrategy instances
    """
    strategies = []

    if not config:
        return strategies

    for instance_name, instance_config in config.items():
        if not instance_config.get("enabled", True):
            continue

        if "type" not in instance_config:
            raise ValueError(
                f"Validation strategy '{instance_name}' must define an explicit 'type'."
            )
        strategy_type = instance_config["type"]

        if strategy_type not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy type '{strategy_type}' for instance '{instance_name}'."
            )

        strategy_class = STRATEGY_REGISTRY[strategy_type]

        # Build kwargs, excluding 'type' which is for routing
        kwargs = {k: v for k, v in instance_config.items() if k != "type"}
        kwargs["name"] = instance_name  # Use instance name, not type

        # Pass bucket filters to basic visualization grouping.
        if strategy_type in ("basic", "basic_visualization") and bucket_filters:
            kwargs["bucket_filters"] = bucket_filters

        try:
            strategies.append(strategy_class(**kwargs))
        except TypeError as e:
            raise TypeError(
                f"Error creating strategy '{instance_name}' ({strategy_type}): {e}"
            ) from e

    return strategies
