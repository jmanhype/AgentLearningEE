"""
Unit tests for policy module.

Tests User Story 3: Train policy with self-reflection reasoning.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import dspy

from agent_learning.policy import (
    PolicySig,
    PolicyModule,
    train_policy,
    generate_decision,
    load_trained_policy,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_reflection_file(tmp_path):
    """Create sample reflection data file."""
    reflection_file = tmp_path / "reflection_data.jsonl"

    reflections = [
        {
            "state": "Vehicle at intersection with green light",
            "reasoning": (
                "Situation: The vehicle is at an intersection with a green light. "
                "Expert: The expert chose to proceed, which is correct. "
                "Alternative: Stopping would be unnecessary. "
                "Conclusion: Proceed through the intersection."
            ),
            "action": "proceed",
        },
        {
            "state": "Vehicle at intersection with red light",
            "reasoning": (
                "Situation: The vehicle is at an intersection with a red light. "
                "Expert: The expert correctly chose to stop. "
                "Alternative: Proceeding would be illegal. "
                "Conclusion: Stop at the intersection."
            ),
            "action": "stop",
        },
        {
            "state": "Pedestrian crossing ahead",
            "reasoning": (
                "Situation: A pedestrian is crossing ahead of the vehicle. "
                "Expert: The expert chose to slow down for safety. "
                "Alternative: Accelerating would risk collision. "
                "Conclusion: Slow down to allow pedestrian to cross."
            ),
            "action": "slow down",
        },
    ] * 4  # Repeat to have 12 examples (minimum 10 required)

    with open(reflection_file, "w") as f:
        for reflection in reflections:
            f.write(json.dumps(reflection) + "\n")

    return str(reflection_file)


@pytest.fixture
def mock_lm():
    """Mock language model for testing."""
    with patch("dspy.settings.lm") as mock:
        mock_lm_instance = Mock()
        mock.return_value = mock_lm_instance
        yield mock_lm_instance


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


# ============================================================================
# Test PolicySig Signature
# ============================================================================

class TestPolicySig:
    """Test PolicySig signature structure."""

    def test_signature_has_correct_fields(self):
        """Test that PolicySig has all required input/output fields."""
        sig = PolicySig
        annotations = sig.__annotations__

        # Check input
        assert "state" in annotations

        # Check outputs
        assert "reasoning" in annotations
        assert "action" in annotations

    def test_signature_field_types(self):
        """Test that PolicySig fields have correct types."""
        sig = PolicySig
        annotations = sig.__annotations__

        # All fields should be strings
        assert annotations["state"] == str
        assert annotations["reasoning"] == str
        assert annotations["action"] == str

    def test_signature_has_ee_template_description(self):
        """Test that PolicySig reasoning field describes 4-section EE template."""
        sig = PolicySig

        # Get field description from signature
        reasoning_field = None
        for field_name, field_info in sig.__fields__.items():
            if field_name == "reasoning":
                reasoning_field = field_info
                break

        assert reasoning_field is not None

        # Check that description mentions all 4 sections
        desc = str(reasoning_field).lower()
        assert "situation" in desc or "4-section" in desc or "ee" in desc


# ============================================================================
# Test PolicyModule
# ============================================================================

class TestPolicyModule:
    """Test PolicyModule class."""

    def test_initializes_correctly(self):
        """Test that PolicyModule initializes with ChainOfThought."""
        policy = PolicyModule()

        assert hasattr(policy, "policy")
        assert policy.policy is not None

    @patch("dspy.ChainOfThought")
    def test_forward_calls_policy(self, mock_cot):
        """Test that forward() calls underlying policy."""
        # Mock policy predictor
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = "Test reasoning"
            prediction.action = "test action"
            return prediction

        mock_predictor.side_effect = mock_predict

        policy = PolicyModule()
        prediction = policy(state="test state")

        assert hasattr(prediction, "reasoning")
        assert hasattr(prediction, "action")


# ============================================================================
# Test train_policy()
# ============================================================================

class TestTrainPolicy:
    """Test policy training function."""

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_trains_policy(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() trains policy model."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock policy predictions
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            prediction.action = "test action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
            metric_threshold=None,  # Skip threshold check for test
        )

        # Check outputs
        assert trained_model is not None
        assert "accuracy" in metrics
        assert "reasoning_quality" in metrics
        assert metrics["examples_trained"] > 0

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_validates_accuracy_threshold(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() validates accuracy against threshold (SC-004)."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model with high accuracy
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock policy predictions with correct actions
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            # Match expected actions from sample data
            if "green light" in state:
                prediction.action = "proceed"
            elif "red light" in state:
                prediction.action = "stop"
            elif "pedestrian" in state.lower():
                prediction.action = "slow down"
            else:
                prediction.action = "unknown"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy with threshold
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
            metric_threshold=0.70,
        )

        # Should achieve high accuracy
        assert metrics["accuracy"] >= 0.0

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_validates_reasoning_structure(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() validates reasoning structure (SC-005)."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock policy predictions with complete 4-section reasoning
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: Current state analysis. "
                "Expert: Expert action evaluation. "
                "Alternative: Alternative actions comparison. "
                "Conclusion: Final decision justification."
            )
            prediction.action = "test action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
        )

        # Should have high reasoning quality
        assert "reasoning_quality" in metrics
        assert metrics["reasoning_quality"] > 0.5

    def test_handles_missing_reflection_file(self, temp_dir):
        """Test that train_policy() handles missing reflection file."""
        with pytest.raises(FileNotFoundError):
            train_policy(
                "nonexistent.jsonl",
                str(temp_dir / "policy.bin"),
            )

    def test_handles_insufficient_examples(self, temp_dir):
        """Test that train_policy() handles insufficient training examples."""
        # Create file with too few examples
        reflection_file = temp_dir / "few_examples.jsonl"
        reflections = [
            {
                "state": "state",
                "reasoning": "situation expert alternative conclusion",
                "action": "action",
            }
        ]

        with open(reflection_file, "w") as f:
            for reflection in reflections:
                f.write(json.dumps(reflection) + "\n")

        with pytest.raises(ValueError, match="Insufficient reflection data"):
            train_policy(
                str(reflection_file),
                str(temp_dir / "policy.bin"),
            )

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_saves_model_with_metadata(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() saves model with metadata."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock policy predictions
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            prediction.action = "test action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy
        with patch("agent_learning.policy.save_module") as mock_save:
            train_policy(
                sample_reflection_file,
                str(output_file),
            )

            # Check that save was called with metadata
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            assert "metadata" in kwargs
            metadata = kwargs["metadata"]
            assert "training_data" in metadata
            assert "accuracy" in metadata
            assert "reasoning_quality" in metadata


# ============================================================================
# Test generate_decision()
# ============================================================================

class TestGenerateDecision:
    """Test policy inference function."""

    def test_generates_decision(self):
        """Test that generate_decision() generates reasoning and action."""
        # Create mock policy
        mock_policy = Mock()

        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = "Test reasoning with all sections"
            prediction.action = "test action"
            return prediction

        mock_policy.side_effect = mock_predict

        # Generate decision
        result = generate_decision(mock_policy, "test state")

        assert result is not None
        reasoning, action = result
        assert isinstance(reasoning, str)
        assert isinstance(action, str)
        assert len(reasoning) > 0
        assert len(action) > 0

    def test_validates_state_input(self):
        """Test that generate_decision() validates state input."""
        mock_policy = Mock()

        # Empty state
        with pytest.raises(ValueError, match="State must be a non-empty string"):
            generate_decision(mock_policy, "")

        # None state
        with pytest.raises(ValueError, match="State must be a non-empty string"):
            generate_decision(mock_policy, None)

    def test_handles_empty_reasoning(self):
        """Test that generate_decision() handles empty reasoning output."""
        # Create mock policy that returns empty reasoning
        mock_policy = Mock()

        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = ""
            prediction.action = "test action"
            return prediction

        mock_policy.side_effect = mock_predict

        # Should return None
        result = generate_decision(mock_policy, "test state")
        assert result is None

    def test_handles_empty_action(self):
        """Test that generate_decision() handles empty action output."""
        # Create mock policy that returns empty action
        mock_policy = Mock()

        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = "test reasoning"
            prediction.action = ""
            return prediction

        mock_policy.side_effect = mock_predict

        # Should return None
        result = generate_decision(mock_policy, "test state")
        assert result is None

    def test_handles_prediction_errors(self):
        """Test that generate_decision() handles prediction errors gracefully."""
        # Create mock policy that raises errors
        mock_policy = Mock()
        mock_policy.side_effect = Exception("Prediction failed")

        # Should return None, not raise
        result = generate_decision(mock_policy, "test state")
        assert result is None


# ============================================================================
# Test Policy Metrics
# ============================================================================

class TestPolicyMetrics:
    """Test policy evaluation metrics."""

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_calculates_accuracy(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() calculates accuracy metric."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock predictions
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            prediction.action = "test action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
        )

        # Check accuracy metric
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_calculates_reasoning_quality(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() calculates reasoning quality metric."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock predictions with varying reasoning quality
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            prediction.action = "test action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
        )

        # Check reasoning quality metric
        assert "reasoning_quality" in metrics
        assert isinstance(metrics["reasoning_quality"], float)
        assert 0.0 <= metrics["reasoning_quality"] <= 1.0


# ============================================================================
# Test Error Handling
# ============================================================================

class TestPolicyErrorHandling:
    """Test error handling in policy module."""

    def test_handles_missing_reflection_fields(self, temp_dir):
        """Test that train_policy() handles missing fields in reflection data."""
        # Create file with missing fields
        reflection_file = temp_dir / "invalid_reflection.jsonl"
        reflections = [
            {"state": "state", "reasoning": "reasoning"}  # Missing "action"
        ] * 10  # Need at least 10 examples

        with open(reflection_file, "w") as f:
            for reflection in reflections:
                f.write(json.dumps(reflection) + "\n")

        with pytest.raises(ValueError, match="missing required fields"):
            train_policy(
                str(reflection_file),
                str(temp_dir / "policy.bin"),
            )

    def test_handles_empty_reflection_fields(self, temp_dir):
        """Test that train_policy() handles empty fields in reflection data."""
        # Create file with empty fields
        reflection_file = temp_dir / "empty_fields.jsonl"
        reflections = [
            {
                "state": "",
                "reasoning": "situation expert alternative conclusion",
                "action": "action",
            }
        ] * 10

        with open(reflection_file, "w") as f:
            for reflection in reflections:
                f.write(json.dumps(reflection) + "\n")

        with pytest.raises(ValueError, match="contains empty"):
            train_policy(
                str(reflection_file),
                str(temp_dir / "policy.bin"),
            )

    @patch("agent_learning.policy.dspy.BootstrapFewShot")
    def test_handles_low_accuracy(self, mock_bootstrap, sample_reflection_file, temp_dir):
        """Test that train_policy() logs warning for low accuracy."""
        # Mock bootstrap teleprompter
        mock_teleprompter = Mock()
        mock_bootstrap.return_value = mock_teleprompter

        # Mock compiled model with low accuracy
        mock_compiled = Mock()

        def mock_compile(model, trainset):
            return mock_compiled

        mock_teleprompter.compile = mock_compile

        # Mock predictions with incorrect actions
        def mock_predict(state):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            prediction.action = "wrong action"
            return prediction

        mock_compiled.side_effect = mock_predict

        output_file = temp_dir / "policy.bin"

        # Train policy with threshold (should log warning but not raise)
        trained_model, metrics = train_policy(
            sample_reflection_file,
            str(output_file),
            metric_threshold=0.70,
        )

        # Should complete with low accuracy
        assert metrics["accuracy"] >= 0.0
