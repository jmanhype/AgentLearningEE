"""
Pipeline Module - Complete end-to-end pipeline for agent learning.

Implements User Story 4: Orchestrate complete pipeline from expert demos to trained policy
Integrates all components: world model → exploratory rollouts → reflection data → policy
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import shutil
from datetime import datetime

from .world_model import train_world_model
from .exploration import generate_exploratory_rollouts
from .reflection import generate_reflection_data, validate_reflection_data
from .policy import train_policy
from .utils import (
    load_jsonl,
    setup_logger,
    MetricsTracker,
)


# ============================================================================
# Complete Pipeline Execution (T032)
# ============================================================================

def run_complete_pipeline(
    expert_demos_path: str,
    output_dir: str = "artifacts",
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """
    Execute complete end-to-end pipeline from expert demos to trained policy.

    Implements User Story 4 acceptance criteria:
    - Orchestrate: expert_demos → world_model → exploratory_rollouts → reflection_data → policy
    - Process without reward signals (reward-free learning)
    - Log progress at each stage
    - Handle errors gracefully with cleanup
    - Return all artifact paths and metrics

    Pipeline stages:
    1. Train world model from expert demonstrations
    2. Generate exploratory rollouts using world model
    3. Generate reflection data with EE-style reasoning
    4. Train policy with structured reasoning
    5. Validate all outputs

    Args:
        expert_demos_path: Path to expert_demos.jsonl
        output_dir: Directory to save all artifacts (default: "artifacts")
        config: Optional configuration dict with stage-specific settings:
            - world_model: dict with training config (test_split, metric_threshold, etc.)
            - exploration: dict with exploration config (num_rollouts, exploration_rate, etc.)
            - reflection: dict with reflection config (max_reflections, etc.)
            - policy: dict with policy config (test_split, metric_threshold, etc.)
        logger: Optional logger instance
        metrics_tracker: Optional metrics tracker

    Returns:
        Dictionary containing:
        - artifacts: dict with paths to all generated files
        - metrics: dict with metrics from each stage
        - success: bool indicating if pipeline completed successfully
        - stage_completed: str indicating last completed stage
        - error: Optional error message if pipeline failed

    Raises:
        ValueError: If expert_demos_path doesn't exist
        RuntimeError: If any pipeline stage fails critically

    Example:
        >>> result = run_complete_pipeline(
        ...     expert_demos_path="data/expert_demos.jsonl",
        ...     output_dir="artifacts",
        ...     config={
        ...         "world_model": {"test_split": 0.2},
        ...         "exploration": {"num_rollouts": 50},
        ...         "policy": {"metric_threshold": 0.70}
        ...     }
        ... )
        >>> print(f"Success: {result['success']}")
        >>> print(f"Policy path: {result['artifacts']['policy']}")
    """
    if logger is None:
        logger = setup_logger("pipeline")

    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()

    # Initialize config with defaults
    if config is None:
        config = {}

    world_model_config = config.get("world_model", {})
    exploration_config = config.get("exploration", {})
    reflection_config = config.get("reflection", {})
    policy_config = config.get("policy", {})

    # Validate inputs
    if not Path(expert_demos_path).exists():
        raise ValueError(f"Expert demos file not found: {expert_demos_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize result structure
    result = {
        "artifacts": {},
        "metrics": {},
        "success": False,
        "stage_completed": None,
        "error": None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    metrics_tracker.start_stage("complete_pipeline")
    logger.info(
        "Starting complete pipeline execution",
        extra={
            "stage": "pipeline",
            "metric": "status",
            "expert_demos": expert_demos_path,
        },
    )

    try:
        # ====================================================================
        # Stage 1: Train World Model
        # ====================================================================
        logger.info(
            "Stage 1/4: Training world model",
            extra={"stage": "pipeline", "metric": "stage_start", "value": 1},
        )

        world_model_path = str(output_path / "world_model.pkl")
        trained_world_model, wm_metrics = train_world_model(
            expert_demos_path=expert_demos_path,
            output_path=world_model_path,
            logger=logger,
            metrics_tracker=metrics_tracker,
            **world_model_config,
        )

        result["artifacts"]["world_model"] = world_model_path
        result["metrics"]["world_model"] = wm_metrics
        result["stage_completed"] = "world_model"

        logger.info(
            f"Stage 1/4 complete: World model trained (accuracy: {wm_metrics.get('accuracy', 0):.2%})",
            extra={
                "stage": "pipeline",
                "metric": "stage_complete",
                "value": "world_model",
            },
        )

        # ====================================================================
        # Stage 2: Generate Exploratory Rollouts
        # ====================================================================
        logger.info(
            "Stage 2/4: Generating exploratory rollouts",
            extra={"stage": "pipeline", "metric": "stage_start", "value": 2},
        )

        rollouts_path = str(output_path / "exploratory_rollouts.jsonl")
        num_rollouts, exploration_metrics = generate_exploratory_rollouts(
            world_model=trained_world_model,
            expert_demos_path=expert_demos_path,
            output_path=rollouts_path,
            logger=logger,
            metrics_tracker=metrics_tracker,
            **exploration_config,
        )

        result["artifacts"]["exploratory_rollouts"] = rollouts_path
        result["metrics"]["exploration"] = exploration_metrics
        result["stage_completed"] = "exploration"

        logger.info(
            f"Stage 2/4 complete: {num_rollouts} exploratory rollouts generated",
            extra={
                "stage": "pipeline",
                "metric": "stage_complete",
                "value": "exploration",
            },
        )

        # ====================================================================
        # Stage 3: Generate Reflection Data
        # ====================================================================
        logger.info(
            "Stage 3/4: Generating reflection data",
            extra={"stage": "pipeline", "metric": "stage_start", "value": 3},
        )

        reflection_path = str(output_path / "reflection_data.jsonl")
        num_reflections, reflection_metrics = generate_reflection_data(
            exploratory_rollouts_path=rollouts_path,
            output_path=reflection_path,
            logger=logger,
            metrics_tracker=metrics_tracker,
            **reflection_config,
        )

        result["artifacts"]["reflection_data"] = reflection_path
        result["metrics"]["reflection"] = reflection_metrics
        result["stage_completed"] = "reflection"

        logger.info(
            f"Stage 3/4 complete: {num_reflections} reflections generated "
            f"(success rate: {reflection_metrics.get('success_rate', 0):.2%})",
            extra={
                "stage": "pipeline",
                "metric": "stage_complete",
                "value": "reflection",
            },
        )

        # Validate reflection data quality
        is_valid, validation_report = validate_reflection_data(
            reflection_path, logger=logger
        )
        result["metrics"]["reflection"]["validation"] = validation_report

        if not is_valid:
            logger.warning(
                f"Reflection data validation issues: {validation_report.get('total_issues', 0)} issues found",
                extra={
                    "stage": "pipeline",
                    "metric": "validation_warning",
                    "value": validation_report,
                },
            )

        # ====================================================================
        # Stage 4: Train Policy
        # ====================================================================
        logger.info(
            "Stage 4/4: Training policy",
            extra={"stage": "pipeline", "metric": "stage_start", "value": 4},
        )

        policy_path = str(output_path / "policy.pkl")
        trained_policy, policy_metrics = train_policy(
            reflection_data_path=reflection_path,
            output_path=policy_path,
            logger=logger,
            metrics_tracker=metrics_tracker,
            **policy_config,
        )

        result["artifacts"]["policy"] = policy_path
        result["metrics"]["policy"] = policy_metrics
        result["stage_completed"] = "policy"

        logger.info(
            f"Stage 4/4 complete: Policy trained "
            f"(accuracy: {policy_metrics.get('accuracy', 0):.2%}, "
            f"reasoning quality: {policy_metrics.get('reasoning_quality', 0):.2%})",
            extra={
                "stage": "pipeline",
                "metric": "stage_complete",
                "value": "policy",
            },
        )

        # ====================================================================
        # Pipeline Complete
        # ====================================================================
        result["success"] = True
        duration = metrics_tracker.end_stage("complete_pipeline")
        result["metrics"]["pipeline_duration"] = duration

        logger.info(
            f"Pipeline completed successfully in {duration:.2f}s",
            extra={
                "stage": "pipeline",
                "metric": "pipeline_complete",
                "value": duration,
            },
        )

        return result

    except Exception as e:
        # Handle pipeline failure
        result["success"] = False
        result["error"] = str(e)

        logger.error(
            f"Pipeline failed at stage {result['stage_completed']}: {e}",
            extra={
                "stage": "pipeline",
                "metric": "pipeline_error",
                "value": str(e),
            },
            exc_info=True,
        )

        # Attempt cleanup of partial artifacts
        try:
            cleanup_partial_artifacts(result["artifacts"], logger=logger)
        except Exception as cleanup_error:
            logger.warning(
                f"Cleanup failed: {cleanup_error}",
                extra={"stage": "pipeline", "metric": "cleanup_error"},
            )

        raise RuntimeError(
            f"Pipeline execution failed at stage {result['stage_completed']}: {e}"
        ) from e


# ============================================================================
# Pipeline Validation Utilities (T033)
# ============================================================================

def validate_pipeline_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate pipeline configuration for completeness and correctness.

    Validates:
    - All stage configs are dicts (if present)
    - Numeric parameters are in valid ranges
    - Required parameters are present
    - No conflicting settings

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> config = {
        ...     "world_model": {"test_split": 0.2},
        ...     "exploration": {"num_rollouts": 50},
        ...     "policy": {"metric_threshold": 0.70}
        ... }
        >>> is_valid, issues = validate_pipeline_config(config)
        >>> print(f"Valid: {is_valid}, Issues: {len(issues)}")
    """
    issues = []

    if not isinstance(config, dict):
        issues.append("Config must be a dictionary")
        return False, issues

    # Validate world_model config
    if "world_model" in config:
        wm_config = config["world_model"]
        if not isinstance(wm_config, dict):
            issues.append("world_model config must be a dictionary")
        else:
            # Validate test_split
            if "test_split" in wm_config:
                test_split = wm_config["test_split"]
                if not isinstance(test_split, (int, float)) or not (0.0 < test_split < 1.0):
                    issues.append(
                        f"world_model.test_split must be float in (0, 1), got {test_split}"
                    )

            # Validate metric_threshold
            if "metric_threshold" in wm_config:
                threshold = wm_config["metric_threshold"]
                if threshold is not None and (
                    not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0)
                ):
                    issues.append(
                        f"world_model.metric_threshold must be float in [0, 1] or None, got {threshold}"
                    )

    # Validate exploration config
    if "exploration" in config:
        exp_config = config["exploration"]
        if not isinstance(exp_config, dict):
            issues.append("exploration config must be a dictionary")
        else:
            # Validate num_rollouts
            if "num_rollouts" in exp_config:
                num_rollouts = exp_config["num_rollouts"]
                if not isinstance(num_rollouts, int) or num_rollouts < 1:
                    issues.append(
                        f"exploration.num_rollouts must be positive int, got {num_rollouts}"
                    )

            # Validate exploration_rate
            if "exploration_rate" in exp_config:
                exp_rate = exp_config["exploration_rate"]
                if not isinstance(exp_rate, (int, float)) or not (0.0 <= exp_rate <= 1.0):
                    issues.append(
                        f"exploration.exploration_rate must be float in [0, 1], got {exp_rate}"
                    )

    # Validate reflection config
    if "reflection" in config:
        ref_config = config["reflection"]
        if not isinstance(ref_config, dict):
            issues.append("reflection config must be a dictionary")
        else:
            # Validate max_reflections
            if "max_reflections" in ref_config:
                max_ref = ref_config["max_reflections"]
                if max_ref is not None and (not isinstance(max_ref, int) or max_ref < 1):
                    issues.append(
                        f"reflection.max_reflections must be positive int or None, got {max_ref}"
                    )

    # Validate policy config
    if "policy" in config:
        pol_config = config["policy"]
        if not isinstance(pol_config, dict):
            issues.append("policy config must be a dictionary")
        else:
            # Validate test_split
            if "test_split" in pol_config:
                test_split = pol_config["test_split"]
                if not isinstance(test_split, (int, float)) or not (0.0 < test_split < 1.0):
                    issues.append(
                        f"policy.test_split must be float in (0, 1), got {test_split}"
                    )

            # Validate metric_threshold
            if "metric_threshold" in pol_config:
                threshold = pol_config["metric_threshold"]
                if threshold is not None and (
                    not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0)
                ):
                    issues.append(
                        f"policy.metric_threshold must be float in [0, 1] or None, got {threshold}"
                    )

            # Validate max_bootstrapped_demos
            if "max_bootstrapped_demos" in pol_config:
                max_demos = pol_config["max_bootstrapped_demos"]
                if not isinstance(max_demos, int) or max_demos < 1:
                    issues.append(
                        f"policy.max_bootstrapped_demos must be positive int, got {max_demos}"
                    )

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_pipeline_artifacts(
    artifacts: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that all pipeline artifacts exist and are readable.

    Validates:
    - All artifact files exist
    - Files are non-empty
    - Files are readable JSONL or binary format
    - Basic schema validation for JSONL files

    Args:
        artifacts: Dictionary of artifact_name -> file_path
        logger: Optional logger instance

    Returns:
        Tuple of (is_valid, validation_report)

    Example:
        >>> artifacts = {
        ...     "world_model": "artifacts/world_model.bin",
        ...     "exploratory_rollouts": "artifacts/exploratory_rollouts.jsonl",
        ...     "reflection_data": "artifacts/reflection_data.jsonl",
        ...     "policy": "artifacts/policy.bin"
        ... }
        >>> is_valid, report = validate_pipeline_artifacts(artifacts)
        >>> print(f"Valid: {is_valid}")
    """
    if logger is None:
        logger = setup_logger("pipeline")

    issues = []
    artifact_status = {}

    for artifact_name, artifact_path in artifacts.items():
        status = {"exists": False, "readable": False, "size": 0, "issues": []}

        # Check file exists
        path = Path(artifact_path)
        if not path.exists():
            status["issues"].append("File not found")
            artifact_status[artifact_name] = status
            issues.append(f"{artifact_name}: File not found at {artifact_path}")
            continue

        status["exists"] = True
        status["size"] = path.stat().st_size

        # Check non-empty
        if status["size"] == 0:
            status["issues"].append("File is empty")
            issues.append(f"{artifact_name}: File is empty")
            continue

        # Check readable
        try:
            if artifact_path.endswith(".jsonl"):
                # Validate JSONL format
                data = load_jsonl(artifact_path)
                status["readable"] = True
                status["num_records"] = len(data)

                # Basic schema validation
                if len(data) == 0:
                    status["issues"].append("JSONL file contains no records")
                    issues.append(f"{artifact_name}: JSONL file is empty")
                elif artifact_name == "exploratory_rollouts":
                    # Check for required fields
                    required_fields = ["state", "action", "next_state"]
                    if not all(f in data[0] for f in required_fields):
                        status["issues"].append("Missing required fields")
                        issues.append(
                            f"{artifact_name}: Missing required fields {required_fields}"
                        )
                elif artifact_name == "reflection_data":
                    # Check for required fields
                    required_fields = ["state", "reasoning", "action"]
                    if not all(f in data[0] for f in required_fields):
                        status["issues"].append("Missing required fields")
                        issues.append(
                            f"{artifact_name}: Missing required fields {required_fields}"
                        )

            elif artifact_path.endswith((".bin", ".pkl")):
                # Just check file is readable
                with open(artifact_path, "rb") as f:
                    f.read(1)  # Try to read at least 1 byte
                status["readable"] = True

            else:
                status["issues"].append("Unknown file format")
                issues.append(f"{artifact_name}: Unknown file format")

        except Exception as e:
            status["issues"].append(f"Read error: {e}")
            issues.append(f"{artifact_name}: Failed to read file - {e}")

        artifact_status[artifact_name] = status

    is_valid = len(issues) == 0

    validation_report = {
        "is_valid": is_valid,
        "artifacts_checked": len(artifacts),
        "artifacts_valid": sum(
            1 for s in artifact_status.values() if len(s["issues"]) == 0
        ),
        "artifact_status": artifact_status,
        "issues": issues,
    }

    if is_valid:
        logger.info(
            f"All {len(artifacts)} pipeline artifacts validated successfully",
            extra={"stage": "pipeline", "metric": "validation_success"},
        )
    else:
        logger.warning(
            f"Pipeline artifact validation failed: {len(issues)} issues found",
            extra={
                "stage": "pipeline",
                "metric": "validation_failure",
                "value": len(issues),
            },
        )

    return is_valid, validation_report


def cleanup_partial_artifacts(
    artifacts: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Clean up partial artifacts after pipeline failure.

    Removes any artifact files that were created during failed pipeline run.
    Used for rollback after errors.

    Args:
        artifacts: Dictionary of artifact_name -> file_path
        logger: Optional logger instance

    Example:
        >>> artifacts = {"world_model": "artifacts/world_model.bin"}
        >>> cleanup_partial_artifacts(artifacts)
    """
    if logger is None:
        logger = setup_logger("pipeline")

    for artifact_name, artifact_path in artifacts.items():
        try:
            path = Path(artifact_path)
            if path.exists():
                path.unlink()
                logger.info(
                    f"Cleaned up partial artifact: {artifact_name}",
                    extra={"stage": "pipeline", "metric": "cleanup", "value": artifact_name},
                )
        except Exception as e:
            logger.warning(
                f"Failed to cleanup {artifact_name}: {e}",
                extra={"stage": "pipeline", "metric": "cleanup_error"},
            )


def get_pipeline_summary(result: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of pipeline execution.

    Args:
        result: Pipeline result dictionary from run_complete_pipeline()

    Returns:
        Formatted string summary

    Example:
        >>> result = run_complete_pipeline(...)
        >>> print(get_pipeline_summary(result))
    """
    lines = []
    lines.append("=" * 70)
    lines.append("AGENT LEARNING PIPELINE SUMMARY")
    lines.append("=" * 70)

    # Status
    status = "SUCCESS" if result["success"] else "FAILED"
    lines.append(f"\nStatus: {status}")
    lines.append(f"Stage Completed: {result.get('stage_completed', 'none')}")
    lines.append(f"Timestamp: {result.get('timestamp', 'unknown')}")

    if result.get("error"):
        lines.append(f"\nError: {result['error']}")

    # Metrics
    if result.get("metrics"):
        lines.append("\nMetrics:")

        if "world_model" in result["metrics"]:
            wm = result["metrics"]["world_model"]
            lines.append(
                f"  World Model: accuracy={wm.get('accuracy', 0):.2%}, "
                f"examples={wm.get('examples_trained', 0)}"
            )

        if "exploration" in result["metrics"]:
            exp = result["metrics"]["exploration"]
            lines.append(
                f"  Exploration: rollouts={exp.get('rollouts_generated', 0)}, "
                f"diversity={exp.get('diversity_rate', 0):.2%}"
            )

        if "reflection" in result["metrics"]:
            ref = result["metrics"]["reflection"]
            lines.append(
                f"  Reflection: reflections={ref.get('reflections_generated', 0)}, "
                f"success_rate={ref.get('success_rate', 0):.2%}"
            )

        if "policy" in result["metrics"]:
            pol = result["metrics"]["policy"]
            lines.append(
                f"  Policy: accuracy={pol.get('accuracy', 0):.2%}, "
                f"reasoning_quality={pol.get('reasoning_quality', 0):.2%}"
            )

        if "pipeline_duration" in result["metrics"]:
            duration = result["metrics"]["pipeline_duration"]
            lines.append(f"\nTotal Duration: {duration:.2f}s")

    # Artifacts
    if result.get("artifacts"):
        lines.append("\nArtifacts:")
        for artifact_name, artifact_path in result["artifacts"].items():
            lines.append(f"  {artifact_name}: {artifact_path}")

    lines.append("=" * 70)
    return "\n".join(lines)
