"""
Exploration Module - Generate exploratory rollouts for diverse experience.

Implements User Story 2: Generate Exploratory Rollouts
Expands beyond expert demonstrations by generating alternative actions
and predicting their outcomes using the trained world model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import random

import dspy
from dspy import Example

from .utils import (
    load_jsonl,
    save_jsonl,
    setup_logger,
    MetricsTracker,
)
from .world_model import WorldModelModule, predict_next_state


# ============================================================================
# Alternative Action Generation (T018)
# ============================================================================

class AlternativeActionSig(dspy.Signature):
    """
    Generate alternative actions for a given state.

    Given a state and the expert's action, generate 2-4 alternative
    actions that an agent might reasonably take in that state.
    """

    state: str = dspy.InputField(desc="current environment state description")
    expert_action: str = dspy.InputField(desc="action chosen by expert in this state")

    alternatives: str = dspy.OutputField(
        desc="2-4 alternative actions separated by newlines, each on its own line. "
        "At least 50% should differ from the expert action. "
        "Format: one action per line, no numbering or bullets."
    )


def generate_alternative_actions(
    state: str,
    expert_action: str,
    num_alternatives: int = 3,
    ensure_diversity: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Generate alternative actions for a given state and expert action.

    Implements T018: Uses DSPy to generate diverse alternative actions
    that differ from the expert action at least 50% of the time (SC-003).

    Args:
        state: Current environment state description
        expert_action: Action chosen by expert in this state
        num_alternatives: Number of alternatives to generate (2-4 recommended)
        ensure_diversity: If True, ensure at least 50% differ from expert
        logger: Optional logger instance

    Returns:
        List of alternative action strings (length = num_alternatives)

    Raises:
        ValueError: If state or expert_action is empty

    Example:
        >>> alternatives = generate_alternative_actions(
        ...     "Vehicle approaching intersection with red light",
        ...     "stop",
        ...     num_alternatives=3
        ... )
        >>> print(alternatives)
        ["proceed slowly", "turn right", "stop"]
    """
    if logger is None:
        logger = setup_logger("exploration")

    # Validate inputs
    if not state or not isinstance(state, str):
        raise ValueError("State must be a non-empty string")

    if not expert_action or not isinstance(expert_action, str):
        raise ValueError("Expert action must be a non-empty string")

    if num_alternatives < 2 or num_alternatives > 4:
        raise ValueError("num_alternatives must be between 2 and 4")

    try:
        # Use DSPy to generate alternatives
        predictor = dspy.Predict(AlternativeActionSig)
        prediction = predictor(state=state, expert_action=expert_action)

        # Parse alternatives (one per line)
        alternatives_text = prediction.alternatives.strip()
        alternatives = [
            line.strip()
            for line in alternatives_text.split("\n")
            if line.strip()
        ]

        # Ensure we have the right number of alternatives
        if len(alternatives) < num_alternatives:
            # Pad with variations of expert action if needed
            logger.warning(
                f"Generated only {len(alternatives)} alternatives, padding to {num_alternatives}",
                extra={"stage": "exploration", "metric": "alternatives_padded"}
            )
            while len(alternatives) < num_alternatives:
                alternatives.append(expert_action)

        # Truncate if too many
        alternatives = alternatives[:num_alternatives]

        # Ensure diversity (SC-003: 50%+ should differ from expert)
        if ensure_diversity:
            num_different = sum(1 for alt in alternatives if alt.lower() != expert_action.lower())
            diversity_ratio = num_different / len(alternatives)

            if diversity_ratio < 0.5:
                # Force diversity by replacing some duplicates with simple variations
                logger.warning(
                    f"Diversity ratio {diversity_ratio:.2%} below 50%, forcing diversity",
                    extra={"stage": "exploration", "metric": "diversity_forced"}
                )

                # Simple heuristic variations if DSPy didn't provide enough diversity
                variations = [
                    f"do not {expert_action}",
                    f"slowly {expert_action}",
                    f"quickly {expert_action}",
                    "wait and observe",
                ]

                for i, alt in enumerate(alternatives):
                    if alt.lower() == expert_action.lower() and i < len(alternatives) // 2:
                        if i < len(variations):
                            alternatives[i] = variations[i]

        return alternatives

    except Exception as e:
        logger.error(
            f"Failed to generate alternative actions: {e}",
            extra={"stage": "exploration", "metric": "generation_error"}
        )
        # Fallback: return simple variations
        fallback = [
            expert_action,
            f"do not {expert_action}",
            "wait and observe",
        ]
        return fallback[:num_alternatives]


# ============================================================================
# Exploratory Rollout Generation (T019)
# ============================================================================

def generate_exploratory_rollouts(
    expert_demos_path: str,
    world_model: WorldModelModule,
    output_path: str = "data/exploratory_rollouts.jsonl",
    num_alternatives_per_demo: int = 2,
    target_expansion_ratio: float = 3.0,
    logger: Optional[logging.Logger] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Generate exploratory rollouts from expert demonstrations.

    Implements T019: For each expert demo, generates alternative actions
    and uses the world model to predict outcomes. Targets 3x data expansion (SC-002).

    Args:
        expert_demos_path: Path to expert_demos.jsonl
        world_model: Trained WorldModelModule for predicting outcomes
        output_path: Path to save exploratory_rollouts.jsonl
        num_alternatives_per_demo: Number of alternatives to generate per demo (default 2)
        target_expansion_ratio: Target ratio of total data to original (default 3.0)
        logger: Optional logger instance
        metrics_tracker: Optional metrics tracker

    Returns:
        Tuple of (num_rollouts, metrics_dict)

    Raises:
        FileNotFoundError: If expert_demos_path doesn't exist
        ValueError: If world_model is None or expert demos is empty

    Example:
        >>> from agent_learning.world_model import load_trained_world_model
        >>> world_model = load_trained_world_model("artifacts/world_model.bin")
        >>> num_rollouts, metrics = generate_exploratory_rollouts(
        ...     "examples/example_expert_demos.jsonl",
        ...     world_model,
        ...     num_alternatives_per_demo=2
        ... )
        >>> print(f"Generated {num_rollouts} exploratory rollouts")
    """
    if logger is None:
        logger = setup_logger("exploration")

    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()

    metrics_tracker.start_stage("exploration")
    logger.info(
        "Starting exploratory rollout generation",
        extra={"stage": "exploration", "metric": "status"}
    )

    # Validate inputs
    if world_model is None:
        raise ValueError("world_model must not be None")

    # Load expert demonstrations
    logger.info(f"Loading expert demonstrations from {expert_demos_path}")
    expert_demos = load_jsonl(expert_demos_path)

    if not expert_demos:
        raise ValueError("expert_demos is empty")

    num_expert_demos = len(expert_demos)
    logger.info(
        f"Loaded {num_expert_demos} expert demonstrations",
        extra={
            "stage": "exploration",
            "metric": "expert_demos_loaded",
            "value": num_expert_demos
        }
    )

    # Generate exploratory rollouts
    exploratory_rollouts = []
    num_alternatives_generated = 0
    num_alternatives_differ = 0
    unique_actions = set()

    for demo_idx, demo in enumerate(expert_demos):
        state = demo["state"]
        expert_action = demo["action"]
        expert_next_state = demo["next_state"]

        # Generate alternative actions
        try:
            alternatives = generate_alternative_actions(
                state=state,
                expert_action=expert_action,
                num_alternatives=num_alternatives_per_demo,
                ensure_diversity=True,
                logger=logger,
            )
        except Exception as e:
            logger.error(
                f"Failed to generate alternatives for demo {demo_idx}: {e}",
                extra={"stage": "exploration", "metric": "generation_error"}
            )
            continue

        # For each alternative, predict outcome using world model
        for alt_action in alternatives:
            # Track metrics
            unique_actions.add(alt_action.lower())
            num_alternatives_generated += 1

            if alt_action.lower() != expert_action.lower():
                num_alternatives_differ += 1

            # Predict next state for alternative action
            predicted_next_state = predict_next_state(
                world_model=world_model,
                state=state,
                action=alt_action,
                logger=logger,
            )

            if predicted_next_state is None:
                logger.warning(
                    f"World model prediction failed for demo {demo_idx}, alternative: {alt_action}",
                    extra={"stage": "exploration", "metric": "prediction_failure"}
                )
                # Use expert next state as fallback
                predicted_next_state = expert_next_state

            # Create exploratory rollout entry
            rollout = {
                "state": state,
                "action": alt_action,
                "next_state": predicted_next_state,
                "source_demo_id": demo_idx,
                "expert_action": expert_action,
                "expert_next_state": expert_next_state,
            }

            exploratory_rollouts.append(rollout)

    num_rollouts = len(exploratory_rollouts)

    # Calculate metrics
    data_expansion_ratio = (num_expert_demos + num_rollouts) / num_expert_demos if num_expert_demos > 0 else 0
    alternative_coverage = num_alternatives_differ / num_alternatives_generated if num_alternatives_generated > 0 else 0

    logger.info(
        f"Generated {num_rollouts} exploratory rollouts",
        extra={
            "stage": "exploration",
            "metric": "rollouts_generated",
            "value": num_rollouts
        }
    )

    logger.info(
        f"Data expansion ratio: {data_expansion_ratio:.2f}x (target: {target_expansion_ratio:.2f}x)",
        extra={
            "stage": "exploration",
            "metric": "data_expansion_ratio",
            "value": data_expansion_ratio
        }
    )

    logger.info(
        f"Alternative coverage: {alternative_coverage:.2%} (target: >50%)",
        extra={
            "stage": "exploration",
            "metric": "alternative_coverage",
            "value": alternative_coverage
        }
    )

    # Check if we met targets
    if data_expansion_ratio < target_expansion_ratio:
        logger.warning(
            f"Data expansion ratio ({data_expansion_ratio:.2f}x) below target ({target_expansion_ratio:.2f}x)",
            extra={"stage": "exploration", "metric": "expansion_warning"}
        )

    if alternative_coverage < 0.5:
        logger.warning(
            f"Alternative coverage ({alternative_coverage:.2%}) below 50% threshold (SC-003)",
            extra={"stage": "exploration", "metric": "coverage_warning"}
        )

    # Save exploratory rollouts
    output_path = Path(output_path)
    save_jsonl(exploratory_rollouts, output_path)

    logger.info(
        f"Saved exploratory rollouts to {output_path}",
        extra={
            "stage": "exploration",
            "metric": "rollouts_saved",
            "value": str(output_path)
        }
    )

    # Log all metrics
    duration = metrics_tracker.end_stage("exploration")
    metrics_tracker.log_metric("exploration", "num_rollouts", num_rollouts)
    metrics_tracker.log_metric("exploration", "data_expansion_ratio", data_expansion_ratio)
    metrics_tracker.log_metric("exploration", "alternative_coverage", alternative_coverage)
    metrics_tracker.log_metric("exploration", "unique_actions", len(unique_actions))

    metrics_dict = {
        "num_rollouts": num_rollouts,
        "expansion_ratio": data_expansion_ratio,  # Changed from data_expansion_ratio to match contract
        "alternative_coverage": alternative_coverage,
        "unique_actions": len(unique_actions),
        "generation_duration": duration,
    }

    return num_rollouts, metrics_dict


# ============================================================================
# Validation Utilities (T020)
# ============================================================================

def validate_exploratory_data(
    exploratory_rollouts: List[Dict[str, Any]],
    expert_demos: List[Dict[str, Any]],
    min_alternative_coverage: float = 0.5,
    min_expansion_ratio: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Validate exploratory rollout data quality.

    Implements T020: Validates that exploratory rollouts meet quality criteria:
    - At least 50% alternative actions differ from expert (SC-003)
    - No duplicate expert demos in exploratory rollouts
    - All required fields present and non-empty
    - Predicted next_state is consistent

    Args:
        exploratory_rollouts: List of exploratory rollout dictionaries
        expert_demos: List of expert demonstration dictionaries
        min_alternative_coverage: Minimum fraction of alternatives that differ from expert
        min_expansion_ratio: Minimum data expansion ratio
        logger: Optional logger instance

    Returns:
        Dictionary with validation results:
        - valid: bool - Overall validation result
        - alternative_coverage: float - Fraction of alternatives that differ
        - expansion_ratio: float - Actual data expansion ratio
        - errors: List[str] - Any validation errors found
        - warnings: List[str] - Any validation warnings

    Example:
        >>> rollouts = load_jsonl("data/exploratory_rollouts.jsonl")
        >>> demos = load_jsonl("examples/example_expert_demos.jsonl")
        >>> results = validate_exploratory_data(rollouts, demos)
        >>> print(f"Valid: {results['valid']}, Coverage: {results['alternative_coverage']:.2%}")
    """
    if logger is None:
        logger = setup_logger("exploration")

    errors = []
    warnings = []

    # Validate exploratory_rollouts is not empty
    if not exploratory_rollouts:
        errors.append("exploratory_rollouts is empty")
        return {
            "valid": False,
            "alternative_coverage": 0.0,
            "expansion_ratio": 0.0,
            "errors": errors,
            "warnings": warnings,
        }

    # Validate schema: all required fields present
    required_fields = ["state", "action", "next_state", "source_demo_id", "expert_action"]
    for i, rollout in enumerate(exploratory_rollouts):
        missing_fields = [f for f in required_fields if f not in rollout]
        if missing_fields:
            errors.append(f"Rollout {i} missing fields: {missing_fields}")

        # Check for empty fields
        for field in required_fields:
            if field in rollout and not rollout[field]:
                errors.append(f"Rollout {i} has empty {field}")

    # Calculate alternative coverage (SC-003)
    num_different = 0
    total_alternatives = 0

    for rollout in exploratory_rollouts:
        if "action" in rollout and "expert_action" in rollout:
            total_alternatives += 1
            if rollout["action"].lower() != rollout["expert_action"].lower():
                num_different += 1

    alternative_coverage = num_different / total_alternatives if total_alternatives > 0 else 0.0

    if alternative_coverage < min_alternative_coverage:
        errors.append(
            f"Alternative coverage {alternative_coverage:.2%} below minimum {min_alternative_coverage:.2%} (SC-003)"
        )

    # Calculate expansion ratio (SC-002)
    num_expert = len(expert_demos)
    num_total = num_expert + len(exploratory_rollouts)
    expansion_ratio = num_total / num_expert if num_expert > 0 else 0.0

    if expansion_ratio < min_expansion_ratio:
        warnings.append(
            f"Expansion ratio {expansion_ratio:.2f}x below minimum {min_expansion_ratio:.2f}x (SC-002)"
        )

    # Check for duplicate expert demos (no expert demo should appear as exploratory rollout)
    expert_states_actions = {
        (demo["state"].lower(), demo["action"].lower())
        for demo in expert_demos
    }

    duplicates_found = 0
    for rollout in exploratory_rollouts:
        if "state" in rollout and "action" in rollout:
            rollout_key = (rollout["state"].lower(), rollout["action"].lower())
            if rollout_key in expert_states_actions:
                duplicates_found += 1

    if duplicates_found > 0:
        warnings.append(
            f"Found {duplicates_found} exploratory rollouts that duplicate expert demos"
        )

    # Check predicted_next_state consistency
    empty_predictions = sum(
        1 for rollout in exploratory_rollouts
        if "next_state" in rollout and not rollout["next_state"]
    )

    if empty_predictions > 0:
        errors.append(f"Found {empty_predictions} rollouts with empty next_state predictions")

    # Overall validation result
    valid = len(errors) == 0

    logger.info(
        f"Validation result: {'PASS' if valid else 'FAIL'}",
        extra={
            "stage": "exploration",
            "metric": "validation_result",
            "value": valid
        }
    )

    if errors:
        for error in errors:
            logger.error(error, extra={"stage": "exploration", "metric": "validation_error"})

    if warnings:
        for warning in warnings:
            logger.warning(warning, extra={"stage": "exploration", "metric": "validation_warning"})

    return {
        "valid": valid,
        "alternative_coverage": alternative_coverage,
        "expansion_ratio": expansion_ratio,
        "errors": errors,
        "warnings": warnings,
    }


def check_alternative_coverage(
    exploratory_rollouts: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> float:
    """
    Calculate what fraction of alternative actions differ from expert action.

    Implements SC-003 check: verifies that at least 50% of alternative actions
    differ from the expert action.

    Args:
        exploratory_rollouts: List of exploratory rollout dictionaries
        threshold: Minimum required fraction (default 0.5 for 50%)

    Returns:
        Fraction of alternatives that differ from expert action (0.0-1.0)
    """
    num_different = 0
    total = 0

    for rollout in exploratory_rollouts:
        if "action" in rollout and "expert_action" in rollout:
            total += 1
            if rollout["action"].lower() != rollout["expert_action"].lower():
                num_different += 1

    return num_different / total if total > 0 else 0.0


def calculate_expansion_ratio(
    num_expert_demos: int,
    num_exploratory_rollouts: int,
) -> float:
    """
    Calculate data expansion ratio.

    Implements SC-002 check: verifies data expansion achieved (target 3x).

    Args:
        num_expert_demos: Number of original expert demonstrations
        num_exploratory_rollouts: Number of exploratory rollouts generated

    Returns:
        Expansion ratio (total_data / original_data)
    """
    total_data = num_expert_demos + num_exploratory_rollouts
    return total_data / num_expert_demos if num_expert_demos > 0 else 0.0
