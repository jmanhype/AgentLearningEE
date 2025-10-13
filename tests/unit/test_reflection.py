"""
Unit tests for reflection module.

Tests User Story 3: Generate self-reflection training data with EE-style reasoning.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import dspy

from agent_learning.reflection import (
    ReflectionSig,
    generate_reflection_data,
    validate_reasoning_structure,
    validate_reflection_data,
    check_reasoning_quality,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_rollouts_file(tmp_path):
    """Create sample exploratory rollouts file."""
    rollouts_file = tmp_path / "exploratory_rollouts.jsonl"

    rollouts = [
        {
            "state": "Vehicle at intersection with green light",
            "action": "proceed",
            "next_state": "Vehicle crossing intersection",
            "expert_action": "proceed",
            "expert_next_state": "Vehicle crossing intersection",
            "source_demo_id": 0,
        },
        {
            "state": "Vehicle at intersection with red light",
            "action": "proceed",
            "next_state": "Vehicle entering intersection illegally",
            "expert_action": "stop",
            "expert_next_state": "Vehicle stopped at intersection",
            "source_demo_id": 1,
        },
        {
            "state": "Pedestrian crossing ahead",
            "action": "accelerate",
            "next_state": "Risk of collision with pedestrian",
            "expert_action": "slow down",
            "expert_next_state": "Vehicle slowing, pedestrian crosses safely",
            "source_demo_id": 2,
        },
    ]

    with open(rollouts_file, "w") as f:
        for rollout in rollouts:
            f.write(json.dumps(rollout) + "\n")

    return str(rollouts_file)


@pytest.fixture
def sample_reflection_file(tmp_path):
    """Create sample reflection data file."""
    reflection_file = tmp_path / "reflection_data.jsonl"

    reflections = [
        {
            "state": "Vehicle at intersection with green light",
            "reasoning": (
                "Situation: The vehicle is at an intersection with a green light, indicating it is safe to proceed. "
                "Expert Action: The expert chose to proceed through the intersection, which is the correct action. "
                "Alternative: An alternative would be to stop, but this would be unnecessary and disrupt traffic flow. "
                "Conclusion: Proceeding is the optimal action in this situation."
            ),
            "action": "proceed",
            "source_rollout_id": 0,
        },
        {
            "state": "Vehicle at intersection with red light",
            "reasoning": (
                "Situation: The vehicle is at an intersection with a red light, requiring a stop. "
                "Expert Action: The expert correctly chose to stop at the red light to comply with traffic laws. "
                "Alternative: Proceeding through the red light would be illegal and dangerous. "
                "Conclusion: Stopping is the only safe and legal action."
            ),
            "action": "stop",
            "source_rollout_id": 1,
        },
    ]

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
# Test ReflectionSig Signature
# ============================================================================

class TestReflectionSig:
    """Test ReflectionSig signature structure."""

    def test_signature_has_correct_fields(self):
        """Test that ReflectionSig has all required input/output fields."""
        sig = ReflectionSig
        annotations = sig.__annotations__

        # Check inputs
        assert "state" in annotations
        assert "expert_action" in annotations
        assert "expert_next_state" in annotations
        assert "alternative_action" in annotations
        assert "alternative_next_state" in annotations

        # Check output
        assert "reasoning" in annotations

    def test_signature_field_types(self):
        """Test that ReflectionSig fields have correct types."""
        sig = ReflectionSig
        annotations = sig.__annotations__

        # All fields should be strings
        assert annotations["state"] == str
        assert annotations["expert_action"] == str
        assert annotations["expert_next_state"] == str
        assert annotations["alternative_action"] == str
        assert annotations["alternative_next_state"] == str
        assert annotations["reasoning"] == str


# ============================================================================
# Test generate_reflection_data()
# ============================================================================

class TestGenerateReflectionData:
    """Test reflection data generation from exploratory rollouts."""

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_generates_reflection_data(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that generate_reflection_data() creates reflection examples."""
        # Mock reflection generator
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            prediction.reasoning = (
                f"Situation: {kwargs['state']}. "
                f"Expert action: {kwargs['expert_action']}. "
                f"Alternative: {kwargs['alternative_action']}. "
                f"Conclusion: Expert action is correct."
            )
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Generate reflection data
        num_reflections, metrics = generate_reflection_data(
            sample_rollouts_file,
            str(output_file),
        )

        # Check output
        assert num_reflections > 0
        assert output_file.exists()

        # Load and validate
        with open(output_file, "r") as f:
            reflections = [json.loads(line) for line in f]

        assert len(reflections) == num_reflections
        assert all("state" in r for r in reflections)
        assert all("reasoning" in r for r in reflections)
        assert all("action" in r for r in reflections)

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_validates_reasoning_structure(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that generate_reflection_data() validates reasoning structure (SC-005)."""
        # Mock reflection generator with valid 4-section reasoning
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            # Include all 4 required sections
            prediction.reasoning = (
                "Situation: Current state analysis. "
                "Expert action evaluation: Why expert chose this. "
                "Alternative actions: Comparing other options. "
                "Conclusion: Final decision and justification."
            )
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Generate reflection data
        num_reflections, metrics = generate_reflection_data(
            sample_rollouts_file,
            str(output_file),
        )

        # All reflections should pass validation
        assert num_reflections > 0
        assert metrics["success_rate"] > 0.0

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_handles_invalid_reasoning_structure(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that generate_reflection_data() skips invalid reasoning."""
        # Mock reflection generator with incomplete reasoning
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            # Missing required sections
            prediction.reasoning = "This is incomplete reasoning."
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Generate reflection data
        num_reflections, metrics = generate_reflection_data(
            sample_rollouts_file,
            str(output_file),
        )

        # Should skip all due to invalid structure
        assert num_reflections == 0
        assert metrics["failed_generations"] > 0

    def test_handles_missing_rollouts_file(self, temp_dir):
        """Test that generate_reflection_data() handles missing input file."""
        with pytest.raises(FileNotFoundError):
            generate_reflection_data(
                "nonexistent.jsonl",
                str(temp_dir / "output.jsonl"),
            )

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_respects_max_reflections_limit(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that generate_reflection_data() respects max_reflections parameter."""
        # Mock reflection generator
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Limit to 2 reflections
        num_reflections, metrics = generate_reflection_data(
            sample_rollouts_file,
            str(output_file),
            max_reflections=2,
        )

        # Should generate at most 2
        assert num_reflections <= 2

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_uses_expert_action_as_target(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that reflection data uses expert_action as the target action."""
        # Mock reflection generator
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Generate reflection data
        generate_reflection_data(sample_rollouts_file, str(output_file))

        # Load and check actions
        with open(output_file, "r") as f:
            reflections = [json.loads(line) for line in f]

        # All actions should match expert_action from rollouts
        for reflection in reflections:
            assert "action" in reflection
            assert "expert_action" in reflection
            assert reflection["action"] == reflection["expert_action"]


# ============================================================================
# Test validate_reasoning_structure()
# ============================================================================

class TestValidateReasoningStructure:
    """Test reasoning structure validation (SC-005)."""

    def test_validates_complete_reasoning(self):
        """Test that valid 4-section reasoning passes validation."""
        reasoning = (
            "Situation: Current state is analyzed here. "
            "Expert action: The expert chose this action. "
            "Alternative actions: Other options were considered. "
            "Conclusion: Final decision is made here."
        )

        assert validate_reasoning_structure(reasoning) is True

    def test_rejects_incomplete_reasoning(self):
        """Test that reasoning missing sections fails validation."""
        # Missing "situation" and "alternative"
        reasoning = "Expert action: The expert chose this. Conclusion: Final decision."

        assert validate_reasoning_structure(reasoning) is False

    def test_rejects_empty_reasoning(self):
        """Test that empty reasoning fails validation."""
        assert validate_reasoning_structure("") is False
        assert validate_reasoning_structure(None) is False

    def test_case_insensitive_validation(self):
        """Test that validation is case-insensitive."""
        reasoning = (
            "SITUATION: Analysis. EXPERT: Evaluation. "
            "ALTERNATIVE: Comparison. CONCLUSION: Decision."
        )

        assert validate_reasoning_structure(reasoning) is True

    def test_accepts_variations_in_section_names(self):
        """Test that variations in section naming are accepted."""
        reasoning = (
            "The situation is analyzed. "
            "The expert's action is evaluated. "
            "Alternative approaches are considered. "
            "In conclusion, the decision is made."
        )

        assert validate_reasoning_structure(reasoning) is True


# ============================================================================
# Test validate_reflection_data()
# ============================================================================

class TestValidateReflectionData:
    """Test reflection data validation."""

    def test_validates_correct_data(self, sample_reflection_file):
        """Test that valid reflection data passes validation."""
        is_valid, report = validate_reflection_data(sample_reflection_file)

        assert is_valid is True
        assert report["total_items"] == 2
        assert report["valid_items"] == 2
        assert report["valid_ratio"] >= 0.9

    def test_detects_missing_file(self):
        """Test that validation detects missing file."""
        is_valid, report = validate_reflection_data("nonexistent.jsonl")

        assert is_valid is False
        assert "error" in report

    def test_detects_missing_fields(self, temp_dir):
        """Test that validation detects missing required fields."""
        reflection_file = temp_dir / "invalid_reflection.jsonl"

        # Missing "action" field
        invalid_reflection = {
            "state": "some state",
            "reasoning": "situation expert alternative conclusion",
        }

        with open(reflection_file, "w") as f:
            f.write(json.dumps(invalid_reflection) + "\n")

        is_valid, report = validate_reflection_data(str(reflection_file))

        assert is_valid is False
        assert report["invalid_items"] > 0

    def test_detects_invalid_structure(self, temp_dir):
        """Test that validation detects invalid reasoning structure."""
        reflection_file = temp_dir / "invalid_structure.jsonl"

        # Missing required sections
        invalid_reflection = {
            "state": "some state",
            "reasoning": "incomplete reasoning",
            "action": "some action",
        }

        with open(reflection_file, "w") as f:
            f.write(json.dumps(invalid_reflection) + "\n")

        is_valid, report = validate_reflection_data(str(reflection_file))

        assert is_valid is False
        assert report["structure_quality"] < 0.75

    def test_detects_empty_fields(self, temp_dir):
        """Test that validation detects empty fields."""
        reflection_file = temp_dir / "empty_fields.jsonl"

        invalid_reflection = {
            "state": "",
            "reasoning": "situation expert alternative conclusion",
            "action": "some action",
        }

        with open(reflection_file, "w") as f:
            f.write(json.dumps(invalid_reflection) + "\n")

        is_valid, report = validate_reflection_data(str(reflection_file))

        assert is_valid is False
        assert report["invalid_items"] > 0

    def test_detects_duplicate_states(self, temp_dir):
        """Test that validation detects duplicate states."""
        reflection_file = temp_dir / "duplicates.jsonl"

        reflections = [
            {
                "state": "same state",
                "reasoning": "situation expert alternative conclusion",
                "action": "action 1",
            },
            {
                "state": "same state",
                "reasoning": "situation expert alternative conclusion",
                "action": "action 2",
            },
        ]

        with open(reflection_file, "w") as f:
            for reflection in reflections:
                f.write(json.dumps(reflection) + "\n")

        is_valid, report = validate_reflection_data(str(reflection_file))

        assert report["duplicate_count"] > 0


# ============================================================================
# Test check_reasoning_quality()
# ============================================================================

class TestCheckReasoningQuality:
    """Test reasoning quality metrics."""

    def test_calculates_structure_completeness(self, sample_reflection_file):
        """Test that quality check calculates structure completeness."""
        metrics = check_reasoning_quality(sample_reflection_file)

        assert "structure_completeness" in metrics
        assert 0.0 <= metrics["structure_completeness"] <= 1.0

    def test_calculates_avg_reasoning_length(self, sample_reflection_file):
        """Test that quality check calculates average reasoning length."""
        metrics = check_reasoning_quality(sample_reflection_file)

        assert "avg_reasoning_length" in metrics
        assert metrics["avg_reasoning_length"] > 0

    def test_calculates_section_balance(self, sample_reflection_file):
        """Test that quality check calculates section balance."""
        metrics = check_reasoning_quality(sample_reflection_file)

        assert "section_balance" in metrics
        assert 0.0 <= metrics["section_balance"] <= 1.0

    def test_handles_empty_file(self, temp_dir):
        """Test that quality check handles empty file."""
        empty_file = temp_dir / "empty.jsonl"
        empty_file.write_text("")

        metrics = check_reasoning_quality(str(empty_file))

        assert metrics["structure_completeness"] == 0.0
        assert metrics["avg_reasoning_length"] == 0.0
        assert metrics["section_balance"] == 0.0


# ============================================================================
# Test Error Handling
# ============================================================================

class TestReflectionErrorHandling:
    """Test error handling in reflection module."""

    def test_handles_insufficient_rollouts(self, temp_dir):
        """Test that generate_reflection_data() handles insufficient rollouts."""
        # Create file with too few rollouts
        rollouts_file = temp_dir / "few_rollouts.jsonl"
        rollouts = [
            {
                "state": "state",
                "action": "action",
                "next_state": "next",
                "expert_action": "expert",
                "expert_next_state": "expert_next",
            }
        ]

        with open(rollouts_file, "w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + "\n")

        with pytest.raises(ValueError, match="Insufficient exploratory rollouts"):
            generate_reflection_data(
                str(rollouts_file),
                str(temp_dir / "output.jsonl"),
            )

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_handles_generation_errors(self, mock_cot, sample_rollouts_file, temp_dir):
        """Test that generate_reflection_data() handles generation errors gracefully."""
        # Mock reflection generator that raises errors
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor
        mock_predictor.side_effect = Exception("Generation failed")

        output_file = temp_dir / "reflection_data.jsonl"

        # Should not raise, but log errors
        num_reflections, metrics = generate_reflection_data(
            sample_rollouts_file,
            str(output_file),
        )

        # All generations should fail
        assert num_reflections == 0
        assert metrics["failed_generations"] > 0
        assert metrics["success_rate"] == 0.0

    @patch("agent_learning.reflection.dspy.ChainOfThought")
    def test_skips_rollouts_with_missing_fields(self, mock_cot, temp_dir):
        """Test that generate_reflection_data() skips rollouts with missing fields."""
        # Create rollouts with missing fields
        rollouts_file = temp_dir / "incomplete_rollouts.jsonl"
        rollouts = [
            {"state": "state1", "action": "action1"},  # Missing fields
            {"state": "state2", "action": "action2"},  # Missing fields
        ] * 5  # Need at least 10 to pass minimum check

        with open(rollouts_file, "w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout) + "\n")

        # Mock reflection generator
        mock_predictor = Mock()
        mock_cot.return_value = mock_predictor

        def mock_predict(**kwargs):
            prediction = Mock()
            prediction.reasoning = (
                "Situation: analysis. Expert: evaluation. "
                "Alternative: comparison. Conclusion: decision."
            )
            return prediction

        mock_predictor.side_effect = mock_predict

        output_file = temp_dir / "reflection_data.jsonl"

        # Should skip all rollouts due to missing fields
        num_reflections, metrics = generate_reflection_data(
            str(rollouts_file),
            str(output_file),
        )

        assert num_reflections == 0
        assert metrics["failed_generations"] > 0
