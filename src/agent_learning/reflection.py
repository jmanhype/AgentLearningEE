"""
Reflection Module - Generate structured EE-style reasoning data.

Implements User Story 3: Generate self-reflection training data
Creates 4-section reasoning comparing expert action against alternatives.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import dspy
from dspy import Example

from .utils import (
    load_jsonl,
    save_jsonl,
    setup_logger,
    MetricsTracker,
)


# ============================================================================
# DSPy Signature for Reflection Generation
# ============================================================================

class ReflectionSig(dspy.Signature):
    """
    Generate structured 4-section EE-style reasoning for state-action pair.

    Compares expert action against alternatives to create training data
    for policy learning.
    """

    # Inputs
    state: str = dspy.InputField(
        desc="current environment state description"
    )
    expert_action: str = dspy.InputField(
        desc="action chosen by expert demonstrator"
    )
    expert_next_state: str = dspy.InputField(
        desc="resulting state after expert action"
    )
    alternative_action: str = dspy.InputField(
        desc="alternative action to compare against expert"
    )
    alternative_next_state: str = dspy.InputField(
        desc="predicted state after alternative action"
    )

    # Output
    reasoning: str = dspy.OutputField(
        desc=(
            "structured self-reflection reasoning following 4-section EE template: "
            "1) Situation Analysis - analyze current state and key factors; "
            "2) Expert Action Evaluation - explain expert's action and rationale; "
            "3) Alternative Actions Analysis - compare alternatives with benefits/drawbacks; "
            "4) Conclusion - final decision with justification. "
            "Must be detailed, analytical, and compare expert vs alternatives."
        )
    )


# ============================================================================
# Reflection Generation (T025)
# ============================================================================

def generate_reflection_data(
    exploratory_rollouts_path: str,
    output_path: str = "artifacts/reflection_data.jsonl",
    max_reflections: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Generate reflection data with EE-style reasoning from exploratory rollouts.

    Implements User Story 3 acceptance criteria:
    - Load exploratory rollouts (state, expert_action, alternative_action, outcomes)
    - For each rollout, generate 4-section reasoning
    - Create training examples: (state, reasoning, action)
    - Save to reflection_data.jsonl
    - Validate reasoning structure (SC-005: contains all 4 sections)

    Args:
        exploratory_rollouts_path: Path to exploratory_rollouts.jsonl
        output_path: Path to save reflection data
        max_reflections: Optional limit on reflections to generate
        logger: Optional logger instance
        metrics_tracker: Optional metrics tracker

    Returns:
        Tuple of (num_reflections, metrics_dict)

    Raises:
        ValueError: If insufficient rollouts or invalid data
        FileNotFoundError: If exploratory_rollouts_path doesn't exist
    """
    if logger is None:
        logger = setup_logger("reflection")

    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()

    metrics_tracker.start_stage("reflection_generation")
    logger.info(
        "Starting reflection data generation",
        extra={"stage": "reflection", "metric": "status"},
    )

    # Load exploratory rollouts
    rollouts = load_jsonl(exploratory_rollouts_path)
    num_rollouts = len(rollouts)

    logger.info(
        f"Loaded {num_rollouts} exploratory rollouts",
        extra={
            "stage": "reflection",
            "metric": "rollouts_loaded",
            "value": num_rollouts,
        },
    )

    # Validate minimum rollouts
    if num_rollouts < 10:
        raise ValueError(
            f"Insufficient exploratory rollouts: need at least 10, received {num_rollouts}"
        )

    # Limit reflections if specified
    if max_reflections is not None:
        rollouts = rollouts[:max_reflections]
        logger.info(
            f"Limited to {len(rollouts)} reflections (max_reflections={max_reflections})",
            extra={
                "stage": "reflection",
                "metric": "reflections_limited",
                "value": len(rollouts),
            },
        )

    # Initialize reflection generator
    reflection_generator = dspy.ChainOfThought(ReflectionSig)

    # Generate reflection data
    reflection_data = []
    successful = 0
    failed = 0

    for i, rollout in enumerate(rollouts):
        # Validate rollout fields
        required_fields = [
            "state",
            "expert_action",
            "expert_next_state",
            "action",
            "next_state",
        ]
        if not all(field in rollout for field in required_fields):
            logger.warning(
                f"Rollout {i} missing required fields, skipping",
                extra={
                    "stage": "reflection",
                    "metric": "rollout_skipped",
                    "value": i,
                },
            )
            failed += 1
            continue

        try:
            # Generate structured reasoning
            prediction = reflection_generator(
                state=rollout["state"],
                expert_action=rollout["expert_action"],
                expert_next_state=rollout["expert_next_state"],
                alternative_action=rollout["action"],
                alternative_next_state=rollout["next_state"],
            )

            reasoning = prediction.reasoning

            # Validate reasoning structure (SC-005)
            if not validate_reasoning_structure(reasoning):
                logger.warning(
                    f"Rollout {i} generated invalid reasoning structure, skipping",
                    extra={
                        "stage": "reflection",
                        "metric": "invalid_structure",
                        "value": i,
                    },
                )
                failed += 1
                continue

            # Create reflection example
            # Use expert action as the "correct" action for training
            reflection_example = {
                "state": rollout["state"],
                "reasoning": reasoning,
                "action": rollout["expert_action"],
                "source_rollout_id": i,
                "expert_action": rollout["expert_action"],
                "alternative_action": rollout["action"],
            }

            reflection_data.append(reflection_example)
            successful += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Generated {i + 1}/{len(rollouts)} reflections",
                    extra={
                        "stage": "reflection",
                        "metric": "progress",
                        "value": i + 1,
                    },
                )

        except Exception as e:
            logger.warning(
                f"Failed to generate reflection for rollout {i}: {e}",
                extra={
                    "stage": "reflection",
                    "metric": "generation_error",
                    "value": i,
                },
            )
            failed += 1
            continue

    # Save reflection data
    save_jsonl(reflection_data, output_path)

    logger.info(
        f"Generated {successful} reflections, {failed} failed",
        extra={
            "stage": "reflection",
            "metric": "generation_complete",
            "value": f"{successful}/{failed}",
        },
    )

    # Calculate metrics
    duration = metrics_tracker.end_stage("reflection_generation")
    success_rate = successful / len(rollouts) if len(rollouts) > 0 else 0.0

    metrics_tracker.log_metric("reflection", "reflections_generated", successful)
    metrics_tracker.log_metric("reflection", "success_rate", success_rate)
    metrics_tracker.log_metric("reflection", "failed_generations", failed)

    logger.info(
        f"Reflection data saved to {output_path}",
        extra={
            "stage": "reflection",
            "metric": "data_saved",
            "value": output_path,
        },
    )

    metrics_dict = {
        "reflections_generated": successful,
        "failed_generations": failed,
        "success_rate": success_rate,
        "generation_duration": duration,
    }

    return successful, metrics_dict


# ============================================================================
# Validation Utilities (T028)
# ============================================================================

def validate_reasoning_structure(reasoning: str) -> bool:
    """
    Validate that reasoning contains all 4 required sections (SC-005).

    Args:
        reasoning: Reasoning text to validate

    Returns:
        True if all sections present, False otherwise
    """
    if not reasoning or not isinstance(reasoning, str):
        return False

    reasoning_lower = reasoning.lower()

    # Check for section indicators
    required_sections = [
        "situation",  # Section 1: Situation Analysis
        "expert",  # Section 2: Expert Action Evaluation
        "alternative",  # Section 3: Alternative Actions Analysis
        "conclusion",  # Section 4: Conclusion
    ]

    sections_found = [section in reasoning_lower for section in required_sections]

    return all(sections_found)


def validate_reflection_data(
    reflection_data_path: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate reflection data file for completeness and quality.

    Validates:
    - File exists and is readable
    - All entries have required fields
    - Reasoning structure is valid (SC-005)
    - No duplicate states
    - Actions are non-empty

    Args:
        reflection_data_path: Path to reflection_data.jsonl
        logger: Optional logger instance

    Returns:
        Tuple of (is_valid, validation_report)
    """
    if logger is None:
        logger = setup_logger("reflection")

    # Check file exists
    if not Path(reflection_data_path).exists():
        return False, {"error": "File not found"}

    # Load data
    try:
        reflection_data = load_jsonl(reflection_data_path)
    except Exception as e:
        return False, {"error": f"Failed to load file: {e}"}

    if len(reflection_data) == 0:
        return False, {"error": "File is empty"}

    # Validation checks
    issues = []
    valid_count = 0
    invalid_structure_count = 0
    seen_states = set()
    duplicate_count = 0

    for i, item in enumerate(reflection_data):
        # Check required fields
        required_fields = ["state", "reasoning", "action"]
        missing_fields = [f for f in required_fields if f not in item]
        if missing_fields:
            issues.append(f"Item {i}: Missing fields {missing_fields}")
            continue

        # Check non-empty
        if not item["state"] or not item["reasoning"] or not item["action"]:
            issues.append(f"Item {i}: Contains empty fields")
            continue

        # Check reasoning structure
        if not validate_reasoning_structure(item["reasoning"]):
            invalid_structure_count += 1
            issues.append(f"Item {i}: Invalid reasoning structure (missing sections)")
            continue

        # Check for duplicates
        state_key = item["state"]
        if state_key in seen_states:
            duplicate_count += 1
            issues.append(f"Item {i}: Duplicate state")
        else:
            seen_states.add(state_key)

        valid_count += 1

    # Calculate validation metrics
    total_count = len(reflection_data)
    valid_ratio = valid_count / total_count if total_count > 0 else 0.0
    structure_quality = 1.0 - (invalid_structure_count / total_count) if total_count > 0 else 0.0

    validation_report = {
        "total_items": total_count,
        "valid_items": valid_count,
        "invalid_items": total_count - valid_count,
        "valid_ratio": valid_ratio,
        "structure_quality": structure_quality,
        "duplicate_count": duplicate_count,
        "issues": issues[:10],  # Return first 10 issues
        "total_issues": len(issues),
    }

    is_valid = valid_ratio >= 0.9 and structure_quality >= 0.75

    logger.info(
        f"Validation complete: {valid_count}/{total_count} valid ({valid_ratio:.1%})",
        extra={
            "stage": "reflection",
            "metric": "validation_complete",
            "value": valid_ratio,
        },
    )

    return is_valid, validation_report


def check_reasoning_quality(
    reflection_data_path: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Calculate quality metrics for reasoning in reflection data.

    Metrics:
    - Structure completeness: % with all 4 sections
    - Average reasoning length
    - Section balance: variance in section lengths

    Args:
        reflection_data_path: Path to reflection_data.jsonl
        logger: Optional logger instance

    Returns:
        Dictionary of quality metrics
    """
    if logger is None:
        logger = setup_logger("reflection")

    reflection_data = load_jsonl(reflection_data_path)

    if len(reflection_data) == 0:
        return {
            "structure_completeness": 0.0,
            "avg_reasoning_length": 0.0,
            "section_balance": 0.0,
        }

    # Calculate metrics
    complete_structure_count = 0
    reasoning_lengths = []

    for item in reflection_data:
        if "reasoning" not in item:
            continue

        reasoning = item["reasoning"]
        reasoning_lengths.append(len(reasoning))

        if validate_reasoning_structure(reasoning):
            complete_structure_count += 1

    structure_completeness = complete_structure_count / len(reflection_data)
    avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0.0

    # Calculate section balance (lower variance = better balance)
    section_lengths = []
    for item in reflection_data:
        if "reasoning" not in item:
            continue

        reasoning = item["reasoning"].lower()
        sections = ["situation", "expert", "alternative", "conclusion"]

        for section in sections:
            if section in reasoning:
                # Rough estimate of section length
                start_idx = reasoning.find(section)
                # Find next section or end
                next_idx = len(reasoning)
                for other_section in sections:
                    if other_section != section:
                        other_idx = reasoning.find(other_section, start_idx + 1)
                        if other_idx != -1 and other_idx < next_idx:
                            next_idx = other_idx
                section_length = next_idx - start_idx
                section_lengths.append(section_length)

    if section_lengths:
        import statistics
        mean_length = statistics.mean(section_lengths)
        if len(section_lengths) > 1:
            variance = statistics.variance(section_lengths)
            # Normalize variance by mean for comparison
            coefficient_of_variation = (variance ** 0.5) / mean_length if mean_length > 0 else 0
            section_balance = max(0.0, 1.0 - coefficient_of_variation)
        else:
            section_balance = 1.0
    else:
        section_balance = 0.0

    metrics = {
        "structure_completeness": structure_completeness,
        "avg_reasoning_length": avg_reasoning_length,
        "section_balance": section_balance,
    }

    logger.info(
        f"Reasoning quality - completeness: {structure_completeness:.1%}, "
        f"avg length: {avg_reasoning_length:.0f}, balance: {section_balance:.2f}",
        extra={
            "stage": "reflection",
            "metric": "quality_analysis",
            "value": metrics,
        },
    )

    return metrics
