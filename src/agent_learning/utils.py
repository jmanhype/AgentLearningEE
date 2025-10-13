"""
Utility functions for Agent Learning via Early Experience.

Provides JSONL loading/saving, model serialization, logging, metrics tracking,
and DSPy LM configuration per contracts/module_signatures.yaml.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import dspy


# ============================================================================
# JSONL Loading and Saving (T006)
# ============================================================================

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file and return list of dictionaries.

    Args:
        file_path: Path to JSONL file (one JSON object per line)

    Returns:
        List of dictionaries, one per line

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid JSON

    Example:
        >>> demos = load_jsonl("data/expert_demos.jsonl")
        >>> print(f"Loaded {len(demos)} demonstrations")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {file_path}: {e}"
                )

    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output path for JSONL file

    Raises:
        ValueError: If data is empty or contains non-dict items

    Example:
        >>> reflections = [{"state": "...", "reasoning": "...", "action": "..."}]
        >>> save_jsonl(reflections, "data/reflection_data.jsonl")
    """
    if not data:
        raise ValueError("Cannot save empty data to JSONL")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"All items must be dictionaries, got {type(item)}")
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================================
# Model Serialization (T007)
# ============================================================================

def save_module(
    module: dspy.Module,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save DSPy module to binary file with optional metadata.

    Uses DSPy 2.0 module.save() instance method for serialization.

    Args:
        module: Trained DSPy module (Predict, ChainOfThought, etc.)
        file_path: Output path for .pkl file
        metadata: Optional metadata dict (training_data, accuracy, timestamp, etc.)

    Example:
        >>> world_model = dspy.Predict(WorldModelSig)
        >>> # ... train world_model ...
        >>> save_module(
        ...     world_model,
        ...     "artifacts/world_model.pkl",
        ...     metadata={"accuracy": 0.75, "timestamp": datetime.now().isoformat()}
        ... )
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save module using DSPy 2.0 instance method
    module.save(str(file_path))

    # Save metadata separately if provided
    if metadata:
        metadata_path = file_path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_module(file_path: Union[str, Path]) -> dspy.Module:
    """
    Load DSPy module from binary file.

    Uses pickle to load modules saved with DSPy 2.0 module.save().

    Args:
        file_path: Path to .pkl file

    Returns:
        Loaded DSPy module

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> world_model = load_module("artifacts/world_model.pkl")
        >>> prediction = world_model(state="...", action="...")
    """
    import pickle

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")

    # DSPy 2.0 modules are saved as pickle files
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_metadata(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a saved module.

    Args:
        file_path: Path to .pkl file (will look for .meta.json)

    Returns:
        Metadata dictionary if exists, None otherwise
    """
    file_path = Path(file_path)
    metadata_path = file_path.with_suffix(".meta.json")

    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)

    return None


# ============================================================================
# Logging Utilities (T008)
# ============================================================================

def setup_logger(
    name: str = "agent_learning",
    level: int = logging.INFO,
    json_format: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create JSON-formatted logger per contracts/observability requirements.

    Args:
        name: Logger name
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        json_format: If True, format logs as JSON with timestamp/stage/metric/value
        log_file: Optional file path for log output (console only if None)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("world_model", level=logging.DEBUG)
        >>> logger.info("Training started", extra={"stage": "world_model", "metric": "status"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs logs as JSON with fields: timestamp, stage, level, message, metric, value, unit
    per contracts/module_signatures.yaml observability requirements.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields if present (stage, metric, value, unit)
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "metric"):
            log_data["metric"] = record.metric
        if hasattr(record, "value"):
            log_data["value"] = record.value
        if hasattr(record, "unit"):
            log_data["unit"] = record.unit

        return json.dumps(log_data)


# ============================================================================
# Metrics Tracking (T009)
# ============================================================================

class MetricsTracker:
    """
    Track metrics for each training stage per contracts/observability.

    Stores metrics with timestamps and provides aggregation methods.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.log_metric("world_model", "accuracy", 0.75)
        >>> tracker.log_metric("world_model", "training_duration", 120.5, unit="seconds")
        >>> metrics = tracker.get_stage_metrics("world_model")
    """

    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.stage_start_times: Dict[str, float] = {}

    def start_stage(self, stage: str) -> None:
        """Mark the start of a training stage."""
        self.stage_start_times[stage] = time.time()
        if stage not in self.metrics:
            self.metrics[stage] = []

    def end_stage(self, stage: str) -> float:
        """Mark the end of a training stage and return duration."""
        if stage not in self.stage_start_times:
            raise ValueError(f"Stage {stage} was never started")

        duration = time.time() - self.stage_start_times[stage]
        self.log_metric(stage, "stage_duration", duration, unit="seconds")
        return duration

    def log_metric(
        self,
        stage: str,
        metric: str,
        value: Union[float, int, str],
        unit: Optional[str] = None,
    ) -> None:
        """
        Log a metric for a specific stage.

        Args:
            stage: Stage name (world_model, exploratory, reflection, policy, pipeline)
            metric: Metric name (accuracy, duration, expansion_ratio, etc.)
            value: Metric value
            unit: Optional unit (seconds, percent, count, etc.)
        """
        if stage not in self.metrics:
            self.metrics[stage] = []

        metric_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metric": metric,
            "value": value,
        }

        if unit:
            metric_entry["unit"] = unit

        self.metrics[stage].append(metric_entry)

    def get_stage_metrics(self, stage: str) -> List[Dict[str, Any]]:
        """Get all metrics for a specific stage."""
        return self.metrics.get(stage, [])

    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics across all stages."""
        return self.metrics

    def get_latest_metric(self, stage: str, metric: str) -> Optional[Any]:
        """Get the most recent value for a specific metric in a stage."""
        stage_metrics = self.get_stage_metrics(stage)

        for entry in reversed(stage_metrics):
            if entry["metric"] == metric:
                return entry["value"]

        return None

    def print_summary(self) -> None:
        """Print a human-readable summary of all metrics."""
        print("\n=== Metrics Summary ===\n")

        for stage, metrics in self.metrics.items():
            print(f"Stage: {stage}")
            for entry in metrics:
                unit_str = f" {entry['unit']}" if "unit" in entry else ""
                print(f"  {entry['metric']}: {entry['value']}{unit_str}")
            print()


# ============================================================================
# DSPy LM Configuration (T010)
# ============================================================================

def configure_lm(
    model_name: str = "gpt-3.5-turbo",
    provider: str = "openai",
    **kwargs,
) -> dspy.LM:
    """
    Configure DSPy language model for training and inference.

    Args:
        model_name: Model name (e.g., "gpt-3.5-turbo", "gpt-4", "meta-llama/Llama-2-7b")
        provider: Provider type ("openai", "local", "anthropic", etc.)
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured DSPy LM instance

    Raises:
        ValueError: If provider is unsupported or configuration is invalid

    Example:
        >>> # OpenAI
        >>> lm = configure_lm("gpt-3.5-turbo", provider="openai")
        >>> dspy.settings.configure(lm=lm)
        >>>
        >>> # Local model
        >>> lm = configure_lm(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     provider="local",
        ...     device="cuda"
        ... )
    """
    if provider == "openai":
        lm = dspy.OpenAI(model=model_name, **kwargs)

    elif provider == "local":
        # For local HuggingFace models
        lm = dspy.HFModel(model=model_name, **kwargs)

    elif provider == "anthropic":
        lm = dspy.Claude(model=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: openai, local, anthropic"
        )

    # Configure DSPy to use this LM
    dspy.settings.configure(lm=lm)

    return lm


def get_default_lm_config() -> Dict[str, Any]:
    """
    Get default LM configuration for CPU-based inference.

    Returns recommended settings for training without GPU.

    Returns:
        Dictionary with recommended LM configuration
    """
    return {
        "model_name": "gpt-3.5-turbo",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 500,
        "cache": True,  # Enable DSPy caching for faster iteration
    }
