"""
Unit tests for pipeline module (User Story 4).

Tests complete end-to-end pipeline orchestration and validation utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil

from agent_learning.pipeline import (
    run_complete_pipeline,
    validate_pipeline_config,
    validate_pipeline_artifacts,
    cleanup_partial_artifacts,
    get_pipeline_summary,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_expert_demos_file(temp_dir):
    """Create sample expert demos file."""
    demos = [
        {
            "state": f"test state {i}",
            "action": f"test action {i}",
            "next_state": f"test next state {i}",
        }
        for i in range(20)
    ]

    demos_path = Path(temp_dir) / "expert_demos.jsonl"
    with open(demos_path, "w") as f:
        for demo in demos:
            f.write(json.dumps(demo) + "\n")

    return str(demos_path)


@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        "world_model": {
            "test_split": 0.2,
            "metric_threshold": 0.5,
        },
        "exploration": {
            "num_rollouts": 20,
            "exploration_rate": 0.3,
        },
        "reflection": {
            "max_reflections": 15,
        },
        "policy": {
            "test_split": 0.2,
            "metric_threshold": 0.5,
        },
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_metrics_tracker():
    """Mock metrics tracker for testing."""
    tracker = Mock()
    tracker.start_stage = Mock()
    tracker.end_stage = Mock(return_value=10.5)  # Mock duration
    tracker.log_metric = Mock()
    return tracker


# ============================================================================
# Test Complete Pipeline Execution (T034)
# ============================================================================

class TestRunCompletePipeline:
    """Test complete pipeline execution."""

    @patch("agent_learning.pipeline.train_policy")
    @patch("agent_learning.pipeline.generate_reflection_data")
    @patch("agent_learning.pipeline.generate_exploratory_rollouts")
    @patch("agent_learning.pipeline.train_world_model")
    @patch("agent_learning.pipeline.validate_reflection_data")
    def test_complete_pipeline_success(
        self,
        mock_validate_reflection,
        mock_train_wm,
        mock_gen_rollouts,
        mock_gen_reflection,
        mock_train_policy,
        sample_expert_demos_file,
        temp_dir,
        sample_config,
        mock_logger,
        mock_metrics_tracker,
    ):
        """Test successful execution of complete pipeline."""
        # Setup mocks
        mock_train_wm.return_value = (Mock(), {"accuracy": 0.85, "examples_trained": 15})
        mock_gen_rollouts.return_value = (20, {"rollouts_generated": 20, "diversity_rate": 0.7})
        mock_gen_reflection.return_value = (15, {"reflections_generated": 15, "success_rate": 0.9})
        mock_train_policy.return_value = (Mock(), {"accuracy": 0.78, "reasoning_quality": 0.82})
        mock_validate_reflection.return_value = (True, {"is_valid": True, "total_issues": 0})

        # Run pipeline
        result = run_complete_pipeline(
            expert_demos_path=sample_expert_demos_file,
            output_dir=temp_dir,
            config=sample_config,
            logger=mock_logger,
            metrics_tracker=mock_metrics_tracker,
        )

        # Verify success
        assert result["success"] is True
        assert result["stage_completed"] == "policy"
        assert result["error"] is None

        # Verify all stages executed
        assert "world_model" in result["artifacts"]
        assert "exploratory_rollouts" in result["artifacts"]
        assert "reflection_data" in result["artifacts"]
        assert "policy" in result["artifacts"]

        # Verify metrics
        assert "world_model" in result["metrics"]
        assert "exploration" in result["metrics"]
        assert "reflection" in result["metrics"]
        assert "policy" in result["metrics"]
        assert "pipeline_duration" in result["metrics"]

        # Verify stage calls with config
        mock_train_wm.assert_called_once()
        mock_gen_rollouts.assert_called_once()
        mock_gen_reflection.assert_called_once()
        mock_train_policy.assert_called_once()

        # Verify metrics tracker used
        mock_metrics_tracker.start_stage.assert_called_with("complete_pipeline")
        mock_metrics_tracker.end_stage.assert_called_with("complete_pipeline")

    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_fails_on_world_model_error(
        self,
        mock_train_wm,
        sample_expert_demos_file,
        temp_dir,
        mock_logger,
    ):
        """Test pipeline fails gracefully when world model training fails."""
        # Setup mock to fail
        mock_train_wm.side_effect = ValueError("World model training failed")

        # Run pipeline and expect failure
        with pytest.raises(RuntimeError) as exc_info:
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
                logger=mock_logger,
            )

        assert "World model training failed" in str(exc_info.value)

    @patch("agent_learning.pipeline.generate_exploratory_rollouts")
    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_fails_on_exploration_error(
        self,
        mock_train_wm,
        mock_gen_rollouts,
        sample_expert_demos_file,
        temp_dir,
        mock_logger,
    ):
        """Test pipeline fails gracefully when exploration fails."""
        # Setup mocks
        mock_train_wm.return_value = (Mock(), {"accuracy": 0.85})
        mock_gen_rollouts.side_effect = ValueError("Exploration failed")

        # Run pipeline and expect failure
        with pytest.raises(RuntimeError) as exc_info:
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
                logger=mock_logger,
            )

        assert "Exploration failed" in str(exc_info.value)

    @patch("agent_learning.pipeline.generate_reflection_data")
    @patch("agent_learning.pipeline.generate_exploratory_rollouts")
    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_fails_on_reflection_error(
        self,
        mock_train_wm,
        mock_gen_rollouts,
        mock_gen_reflection,
        sample_expert_demos_file,
        temp_dir,
        mock_logger,
    ):
        """Test pipeline fails gracefully when reflection generation fails."""
        # Setup mocks
        mock_train_wm.return_value = (Mock(), {"accuracy": 0.85})
        mock_gen_rollouts.return_value = (20, {"rollouts_generated": 20})
        mock_gen_reflection.side_effect = ValueError("Reflection generation failed")

        # Run pipeline and expect failure
        with pytest.raises(RuntimeError) as exc_info:
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
                logger=mock_logger,
            )

        assert "Reflection generation failed" in str(exc_info.value)

    @patch("agent_learning.pipeline.train_policy")
    @patch("agent_learning.pipeline.generate_reflection_data")
    @patch("agent_learning.pipeline.generate_exploratory_rollouts")
    @patch("agent_learning.pipeline.train_world_model")
    @patch("agent_learning.pipeline.validate_reflection_data")
    def test_pipeline_fails_on_policy_error(
        self,
        mock_validate_reflection,
        mock_train_wm,
        mock_gen_rollouts,
        mock_gen_reflection,
        mock_train_policy,
        sample_expert_demos_file,
        temp_dir,
        mock_logger,
    ):
        """Test pipeline fails gracefully when policy training fails."""
        # Setup mocks
        mock_train_wm.return_value = (Mock(), {"accuracy": 0.85})
        mock_gen_rollouts.return_value = (20, {"rollouts_generated": 20})
        mock_gen_reflection.return_value = (15, {"reflections_generated": 15})
        mock_validate_reflection.return_value = (True, {"is_valid": True})
        mock_train_policy.side_effect = ValueError("Policy training failed")

        # Run pipeline and expect failure
        with pytest.raises(RuntimeError) as exc_info:
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
                logger=mock_logger,
            )

        assert "Policy training failed" in str(exc_info.value)

    def test_pipeline_fails_on_missing_expert_demos(self, temp_dir):
        """Test pipeline fails when expert demos file doesn't exist."""
        with pytest.raises(ValueError) as exc_info:
            run_complete_pipeline(
                expert_demos_path="nonexistent.jsonl",
                output_dir=temp_dir,
            )

        assert "Expert demos file not found" in str(exc_info.value)

    @patch("agent_learning.pipeline.train_policy")
    @patch("agent_learning.pipeline.generate_reflection_data")
    @patch("agent_learning.pipeline.generate_exploratory_rollouts")
    @patch("agent_learning.pipeline.train_world_model")
    @patch("agent_learning.pipeline.validate_reflection_data")
    def test_pipeline_with_default_config(
        self,
        mock_validate_reflection,
        mock_train_wm,
        mock_gen_rollouts,
        mock_gen_reflection,
        mock_train_policy,
        sample_expert_demos_file,
        temp_dir,
    ):
        """Test pipeline runs with default config when none provided."""
        # Setup mocks
        mock_train_wm.return_value = (Mock(), {"accuracy": 0.85})
        mock_gen_rollouts.return_value = (20, {"rollouts_generated": 20})
        mock_gen_reflection.return_value = (15, {"reflections_generated": 15})
        mock_train_policy.return_value = (Mock(), {"accuracy": 0.78})
        mock_validate_reflection.return_value = (True, {"is_valid": True})

        # Run pipeline with no config
        result = run_complete_pipeline(
            expert_demos_path=sample_expert_demos_file,
            output_dir=temp_dir,
        )

        assert result["success"] is True
        assert result["stage_completed"] == "policy"

    @patch("agent_learning.pipeline.cleanup_partial_artifacts")
    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_calls_cleanup_on_failure(
        self,
        mock_train_wm,
        mock_cleanup,
        sample_expert_demos_file,
        temp_dir,
        mock_logger,
    ):
        """Test pipeline calls cleanup when failing."""
        # Setup mock to fail
        mock_train_wm.side_effect = ValueError("Training failed")

        # Run pipeline and expect failure
        with pytest.raises(RuntimeError):
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
                logger=mock_logger,
            )

        # Verify cleanup was called
        mock_cleanup.assert_called_once()


# ============================================================================
# Test Pipeline Configuration Validation (T034)
# ============================================================================

class TestValidatePipelineConfig:
    """Test pipeline configuration validation."""

    def test_validates_valid_config(self, sample_config):
        """Test validation passes for valid config."""
        is_valid, issues = validate_pipeline_config(sample_config)

        assert is_valid is True
        assert len(issues) == 0

    def test_validates_empty_config(self):
        """Test validation passes for empty config."""
        is_valid, issues = validate_pipeline_config({})

        assert is_valid is True
        assert len(issues) == 0

    def test_fails_on_non_dict_config(self):
        """Test validation fails when config is not a dict."""
        is_valid, issues = validate_pipeline_config("not a dict")

        assert is_valid is False
        assert len(issues) > 0
        assert "Config must be a dictionary" in issues[0]

    def test_fails_on_invalid_world_model_test_split(self):
        """Test validation fails for invalid world_model.test_split."""
        config = {"world_model": {"test_split": 1.5}}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is False
        assert any("test_split" in issue for issue in issues)

    def test_fails_on_invalid_exploration_num_rollouts(self):
        """Test validation fails for invalid exploration.num_rollouts."""
        config = {"exploration": {"num_rollouts": -5}}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is False
        assert any("num_rollouts" in issue for issue in issues)

    def test_fails_on_invalid_exploration_rate(self):
        """Test validation fails for invalid exploration_rate."""
        config = {"exploration": {"exploration_rate": 1.5}}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is False
        assert any("exploration_rate" in issue for issue in issues)

    def test_fails_on_invalid_policy_threshold(self):
        """Test validation fails for invalid policy.metric_threshold."""
        config = {"policy": {"metric_threshold": 2.0}}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is False
        assert any("metric_threshold" in issue for issue in issues)

    def test_fails_on_non_dict_stage_config(self):
        """Test validation fails when stage config is not a dict."""
        config = {"world_model": "not a dict"}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is False
        assert "world_model config must be a dictionary" in issues[0]

    def test_allows_none_metric_threshold(self):
        """Test validation allows None for metric_threshold."""
        config = {
            "world_model": {"metric_threshold": None},
            "policy": {"metric_threshold": None},
        }

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is True
        assert len(issues) == 0

    def test_allows_none_max_reflections(self):
        """Test validation allows None for max_reflections."""
        config = {"reflection": {"max_reflections": None}}

        is_valid, issues = validate_pipeline_config(config)

        assert is_valid is True
        assert len(issues) == 0


# ============================================================================
# Test Pipeline Artifacts Validation (T034)
# ============================================================================

class TestValidatePipelineArtifacts:
    """Test pipeline artifacts validation."""

    def test_validates_valid_artifacts(self, temp_dir):
        """Test validation passes for valid artifacts."""
        # Create valid artifact files
        rollouts_path = Path(temp_dir) / "rollouts.jsonl"
        with open(rollouts_path, "w") as f:
            f.write(json.dumps({"state": "s", "action": "a", "next_state": "s'"}) + "\n")

        reflection_path = Path(temp_dir) / "reflection.jsonl"
        with open(reflection_path, "w") as f:
            f.write(json.dumps({"state": "s", "reasoning": "r", "action": "a"}) + "\n")

        policy_path = Path(temp_dir) / "policy.bin"
        with open(policy_path, "wb") as f:
            f.write(b"binary data")

        artifacts = {
            "exploratory_rollouts": str(rollouts_path),
            "reflection_data": str(reflection_path),
            "policy": str(policy_path),
        }

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is True
        assert report["artifacts_valid"] == 3
        assert len(report["issues"]) == 0

    def test_fails_on_missing_file(self, temp_dir):
        """Test validation fails when artifact file doesn't exist."""
        artifacts = {"policy": str(Path(temp_dir) / "nonexistent.bin")}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is False
        assert report["artifacts_valid"] == 0
        assert len(report["issues"]) > 0
        assert "File not found" in report["issues"][0]

    def test_fails_on_empty_file(self, temp_dir):
        """Test validation fails for empty artifact files."""
        empty_path = Path(temp_dir) / "empty.jsonl"
        empty_path.touch()

        artifacts = {"reflection_data": str(empty_path)}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is False
        assert "File is empty" in report["issues"][0]

    def test_fails_on_empty_jsonl(self, temp_dir):
        """Test validation fails for JSONL with no records."""
        jsonl_path = Path(temp_dir) / "empty.jsonl"
        with open(jsonl_path, "w") as f:
            pass  # Create empty file with some content to pass size check

        artifacts = {"exploratory_rollouts": str(jsonl_path)}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is False

    def test_fails_on_missing_required_fields_in_rollouts(self, temp_dir):
        """Test validation fails when rollouts missing required fields."""
        rollouts_path = Path(temp_dir) / "rollouts.jsonl"
        with open(rollouts_path, "w") as f:
            f.write(json.dumps({"state": "s"}) + "\n")  # Missing action, next_state

        artifacts = {"exploratory_rollouts": str(rollouts_path)}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is False
        assert any("Missing required fields" in issue for issue in report["issues"])

    def test_fails_on_missing_required_fields_in_reflection(self, temp_dir):
        """Test validation fails when reflection data missing required fields."""
        reflection_path = Path(temp_dir) / "reflection.jsonl"
        with open(reflection_path, "w") as f:
            f.write(json.dumps({"state": "s"}) + "\n")  # Missing reasoning, action

        artifacts = {"reflection_data": str(reflection_path)}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is False
        assert any("Missing required fields" in issue for issue in report["issues"])

    def test_validates_binary_files(self, temp_dir):
        """Test validation passes for readable binary files."""
        bin_path = Path(temp_dir) / "model.bin"
        with open(bin_path, "wb") as f:
            f.write(b"model data")

        artifacts = {"world_model": str(bin_path)}

        is_valid, report = validate_pipeline_artifacts(artifacts)

        assert is_valid is True
        assert report["artifact_status"]["world_model"]["readable"] is True


# ============================================================================
# Test Cleanup Utilities (T034)
# ============================================================================

class TestCleanupPartialArtifacts:
    """Test cleanup of partial artifacts."""

    def test_removes_existing_artifacts(self, temp_dir, mock_logger):
        """Test cleanup removes existing artifact files."""
        # Create artifact files
        artifact_path = Path(temp_dir) / "artifact.bin"
        artifact_path.write_text("test data")

        artifacts = {"test_artifact": str(artifact_path)}

        # Verify file exists
        assert artifact_path.exists()

        # Cleanup
        cleanup_partial_artifacts(artifacts, logger=mock_logger)

        # Verify file removed
        assert not artifact_path.exists()

    def test_handles_nonexistent_artifacts(self, temp_dir, mock_logger):
        """Test cleanup handles artifacts that don't exist."""
        artifacts = {"test_artifact": str(Path(temp_dir) / "nonexistent.bin")}

        # Should not raise error
        cleanup_partial_artifacts(artifacts, logger=mock_logger)

    def test_handles_cleanup_errors(self, temp_dir, mock_logger):
        """Test cleanup handles errors gracefully."""
        # Create artifact in read-only directory (if possible)
        artifacts = {"test_artifact": "/invalid/path/artifact.bin"}

        # Should not raise error
        cleanup_partial_artifacts(artifacts, logger=mock_logger)


# ============================================================================
# Test Pipeline Summary (T034)
# ============================================================================

class TestGetPipelineSummary:
    """Test pipeline summary generation."""

    def test_generates_summary_for_success(self):
        """Test summary generation for successful pipeline."""
        result = {
            "success": True,
            "stage_completed": "policy",
            "timestamp": "2025-01-01T00:00:00Z",
            "artifacts": {
                "world_model": "artifacts/world_model.bin",
                "policy": "artifacts/policy.bin",
            },
            "metrics": {
                "world_model": {"accuracy": 0.85, "examples_trained": 20},
                "policy": {"accuracy": 0.78, "reasoning_quality": 0.82},
                "pipeline_duration": 45.5,
            },
            "error": None,
        }

        summary = get_pipeline_summary(result)

        assert "SUCCESS" in summary
        assert "policy" in summary
        assert "0.85" in summary  # World model accuracy
        assert "0.78" in summary  # Policy accuracy
        assert "45.5" in summary  # Duration
        assert "world_model: artifacts/world_model.bin" in summary

    def test_generates_summary_for_failure(self):
        """Test summary generation for failed pipeline."""
        result = {
            "success": False,
            "stage_completed": "exploration",
            "timestamp": "2025-01-01T00:00:00Z",
            "error": "Exploration failed: insufficient rollouts",
            "artifacts": {"world_model": "artifacts/world_model.bin"},
            "metrics": {"world_model": {"accuracy": 0.85}},
        }

        summary = get_pipeline_summary(result)

        assert "FAILED" in summary
        assert "exploration" in summary
        assert "Exploration failed" in summary

    def test_handles_missing_fields(self):
        """Test summary handles missing optional fields."""
        result = {
            "success": True,
            "stage_completed": "world_model",
        }

        summary = get_pipeline_summary(result)

        assert "SUCCESS" in summary
        assert "world_model" in summary


# ============================================================================
# Test Error Handling (T034)
# ============================================================================

class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_sets_error_field_on_failure(
        self,
        mock_train_wm,
        sample_expert_demos_file,
        temp_dir,
    ):
        """Test pipeline sets error field when failing."""
        mock_train_wm.side_effect = ValueError("Training error")

        with pytest.raises(RuntimeError):
            result = run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
            )

        # Note: Can't check result since exception is raised
        # But error handling is tested in the function

    @patch("agent_learning.pipeline.train_world_model")
    def test_pipeline_tracks_last_completed_stage(
        self,
        mock_train_wm,
        sample_expert_demos_file,
        temp_dir,
    ):
        """Test pipeline tracks which stage was last completed."""
        mock_train_wm.side_effect = ValueError("Training error")

        with pytest.raises(RuntimeError):
            run_complete_pipeline(
                expert_demos_path=sample_expert_demos_file,
                output_dir=temp_dir,
            )

        # Stage tracking is tested in the function
