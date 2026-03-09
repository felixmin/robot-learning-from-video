"""
Unified filter logic for matching metadata against filter conditions.

Used by both data loading (SceneFilter) and validation (bucket filtering).
"""

from typing import Any, Callable, Dict, Union


def matches_filters(
    data: Dict[str, Any],
    filters: Dict[str, Any],
    get_value: Callable[[Dict[str, Any], str], Any] = lambda d, k: d.get(k),
) -> bool:
    """
    Check if data matches all filter conditions.

    Args:
        data: Dictionary of metadata to check
        filters: Dictionary of filter conditions
        get_value: Function to extract value for a key from data.
                   Default uses dict.get(). For SceneMetadata, pass a custom
                   function that checks attributes and extras.

    Supported conditions (with YAML examples):
        - Equality: `label: "static"` or `has_hands: true`
        - Comparison: `max_trans: [">", 10.0]` or `num_frames: [">=", 100]`
        - Exclusion: `label: ["!=", "static"]`
        - Not null: `action: ["not_null", true]`
        - Membership: `dataset_name: ["in", ["youtube", "bridge"]]`
        - Multiple values: `task: ["pnp", "stack"]` (value in list)
        - Callable (Python only): `{"num_frames": lambda x: x > 100}`

    Operators: ">", ">=", "<", "<=", "!=", "==", "in", "not_null"

    Returns:
        True if data matches all conditions, False otherwise.
        If a filter key is not found in data, returns False.
    """
    if not filters:
        return True

    for key, condition in filters.items():
        value = get_value(data, key)

        # Callable condition (Python-only, not from YAML)
        if callable(condition):
            if value is None or not condition(value):
                return False
            continue

        # Operator-based conditions: [op, target]
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            first_elem = condition[0]
            if isinstance(first_elem, str) and first_elem in (
                ">",
                ">=",
                "<",
                "<=",
                "!=",
                "==",
                "in",
                "not_null",
            ):
                op, target = condition
                if not _apply_operator(op, value, target):
                    return False
                continue

        # List of allowed values (membership check)
        if isinstance(condition, (list, tuple)):
            if value not in condition:
                return False
            continue

        # Direct equality
        if value != condition:
            return False

    return True


def _apply_operator(op: str, value: Any, target: Any) -> bool:
    """Apply a comparison operator."""
    if op == "not_null":
        return value is not None
    if value is None:
        return False  # Can't compare None with operators

    if op == ">":
        return value > target
    elif op == ">=":
        return value >= target
    elif op == "<":
        return value < target
    elif op == "<=":
        return value <= target
    elif op == "!=":
        return value != target
    elif op == "==":
        return value == target
    elif op == "in":
        return value in target

    return False  # Unknown operator
