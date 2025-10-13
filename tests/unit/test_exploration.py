"""
Unit tests for Agent Learning via Early Experience exploration module.

Tests exploratory rollout generation, alternative action creation, and validation.
Covers User Story 2 (Phase 4) functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import dspy

from agent_learning.exploration import (
    AlternativeActionSig,
    generate_alternative_actions,
    generate_exploratory_rollouts,
    validate_exploratory_data,
    check_alternative_coverage,
    calculate_expansion_ratio,
)
from agent_learning.world_model import WorldModelModule, predict_next_state
from agent_learning.utils import (
    save_jsonl,
    load_jsonl,
    setup_logger,
    MetricsTracker,
)
from tests.fixtures.deterministic_seeds import set_seed, SAMPLE_EXPERT_DEMOS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_demos_file(temp_dir):
    """Create temporary JSONL file with sample expert demos."""
    demos_path = temp_dir / "expert_demos.jsonl"
    save_jsonl(SAMPLE_EXPERT_DEMOS, demos_path)
    return demos_path


@pytest.fixture
def mock_world_model():
    """Create mock world model that returns predictable next states."""
    mock_model = Mock(spec=WorldModelModule)

    def mock_predict(state, action):
        """Return mock prediction with action appended to state."""
        prediction = Mock()
        prediction.next_state = f"{state} after {action}"
        return prediction

    mock_model.side_effect = mock_predict
    return mock_model


@pytest.fixture
def mock_lm():
    """Configure mock language model for testing."""
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=150)
    dspy.settings.configure(lm=lm)
    return lm


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    set_seed(42)


# ============================================================================
# Test AlternativeActionSig Signature (T018)
# ============================================================================

class TestAlternativeActionSig:
    """Test AlternativeActionSig signature structure and validation."""

    def test_signature_has_correct_fields(self):
        """Test that AlternativeActionSig has required input/output fields."""
        sig = AlternativeActionSig
        annotations = sig.__annotations__

        # Check input fields
        assert "state" in annotations, "Signature should have 'state' input field"
        assert "expert_action" in annotations, "Signature should have 'expert_action' input field"

        # Check output field
        assert "alternatives" in annotations, "Signature should have 'alternatives' output field"

    def test_signature_field_descriptions(self):
        """Test that AlternativeActionSig fields have proper descriptions."""
        # Access field metadata through __fields__
        fields = AlternativeActionSig.__fields__

        assert "state" in fields
        assert "expert_action" in fields
        assert "alternatives" in fields

        # Verify output field has formatting instructions
        alternatives_desc = fields["alternatives"].json_schema_extra.get("desc", "")
        assert "newlines" in alternatives_desc.lower() or "line" in alternatives_desc.lower()


# ============================================================================
# Test generate_alternative_actions() Function (T018)
# ============================================================================

class TestGenerateAlternativeActions:
    """Test alternative action generation with diversity enforcement."""

    def test_generates_alternatives_for_valid_input(self, mock_lm):
        """Test that generate_alternative_actions() produces alternative actions."""
        logger = setup_logger("test_exploration")

        state = "Vehicle approaching intersection with red light"
        expert_action = "stop"

        with patch("dspy.Predict") as mock_predict:
            # Mock DSPy prediction
            mock_prediction = Mock()
            mock_prediction.alternatives = "slow down\nwait\nturn right"
            mock_predict.return_value = Mock(return_value=mock_prediction)

            alternatives = generate_alternative_actions(
                state=state,
                expert_action=expert_action,
                num_alternatives=3,
                ensure_diversity=False,
                logger=logger,
            )

            # Should return list of alternatives
            assert isinstance(alternatives, list)
            assert len(alternatives) > 0
            assert all(isinstance(alt, str) for alt in alternatives)

    def test_enforces_diversity_requirement(self, mock_lm):
        """Test that generate_alternative_actions() enforces 50%+ diversity (SC-003)."""
        logger = setup_logger("test_exploration")

        state = "Vehicle approaching intersection with red light"
        expert_action = "stop"

        with patch("dspy.Predict") as mock_predict:
            # Mock DSPy returning all duplicates of expert action
            mock_prediction = Mock()
            mock_prediction.alternatives = "stop\nstop\nstop"
            mock_predict.return_value = Mock(return_value=mock_prediction)

            alternatives = generate_alternative_actions(
                state=state,
                expert_action=expert_action,
                num_alternatives=3,
                ensure_diversity=True,
                logger=logger,
            )

            # Should have forced diversity
            assert len(alternatives) >= 2

            # Count how many differ from expert
            num_different = sum(1 for alt in alternatives if alt.lower() != expert_action.lower())
            diversity_ratio = num_different / len(alternatives)

            # Should meet 50%+ diversity requirement
            assert diversity_ratio >= 0.5, f"Diversity ratio {diversity_ratio:.2%} below 50% threshold"

    def test_handles_empty_state(self, mock_lm):
        """Test that generate_alternative_actions() rejects empty state."""
        logger = setup_logger("test_exploration")

        with pytest.raises(ValueError, match="State must be a non-empty string"):
            generate_alternative_actions(
                state="",
                expert_action="stop",
                logger=logger,
            )

    def test_handles_empty_expert_action(self, mock_lm):
        """Test that generate_alternative_actions() rejects empty expert action."""
        logger = setup_logger("test_exploration")

        with pytest.raises(ValueError, match="Expert action must be a non-empty string"):
            generate_alternative_actions(
                state="Vehicle approaching intersection",
                expert_action="",
                logger=logger,
            )

    def test_handles_invalid_state_type(self, mock_lm):
        """Test that generate_alternative_actions() rejects non-string state."""
        logger = setup_logger("test_exploration")

        with pytest.raises(ValueError, match="State must be a non-empty string"):
            generate_alternative_actions(
                state=123,  # Invalid type
                expert_action="stop",
                logger=logger,
            )

    def test_returns_requested_number_of_alternatives(self, mock_lm):
        """Test that generate_alternative_actions() returns approximately requested number."""
        logger = setup_logger("test_exploration")

        state = "Vehicle approaching intersection with red light"
        expert_action = "stop"

        with patch("dspy.Predict") as mock_predict:
            # Mock DSPy returning 4 alternatives
            mock_prediction = Mock()
            mock_prediction.alternatives = "slow down\nwait\nturn right\nproceed with caution"
            mock_predict.return_value = Mock(return_value=mock_prediction)

            alternatives = generate_alternative_actions(
                state=state,
                expert_action=expert_action,
                num_alternatives=4,
                ensure_diversity=False,
                logger=logger,
            )

            # Should return close to requested number (within reasonable range)
            assert 2 <= len(alternatives) <= 6

    def test_parses_newline_separated_alternatives(self, mock_lm):
        """Test that generate_alternative_actions() correctly parses newline-separated alternatives."""
        logger = setup_logger("test_exploration")

        state = "Vehicle approaching intersection"
        expert_action = "stop"

        with patch("dspy.Predict") as mock_predict:
            # Mock DSPy returning newline-separated alternatives
            mock_prediction = Mock()
            mock_prediction.alternatives = "slow down\nwait and observe\nturn right on red"
            mock_predict.return_value = Mock(return_value=mock_prediction)

            alternatives = generate_alternative_actions(
                state=state,
                expert_action=expert_action,
                ensure_diversity=False,
                logger=logger,
            )

            # Should parse into separate alternatives
            assert len(alternatives) == 3
            assert "slow down" in alternatives
            assert "wait and observe" in alternatives
            assert "turn right on red" in alternatives


# ============================================================================
# Test generate_exploratory_rollouts() Function (T019)
# ============================================================================

class TestGenerateExploratoryRollouts:
    """Test exploratory rollout generation and data expansion."""

    def test_generates_rollouts_from_expert_demos(self, sample_demos_file, temp_dir, mock_lm):
        """Test that generate_exploratory_rollouts() creates rollouts from expert demos."""
        logger = setup_logger("test_exploration")
        output_path = temp_dir / "exploratory_rollouts.jsonl"

        # Create mock world model
        mock_world_model = Mock(spec=WorldModelModule)

        def mock_predict(state, action):
            prediction = Mock()
            prediction.next_state = f"{state} after {action}"
            return prediction

        mock_world_model.side_effect = mock_predict

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            with patch("agent_learning.exploration.predict_next_state") as mock_predict_state:
                # Mock alternative generation
                mock_gen_alts.return_value = ["alternative_1", "alternative_2"]

                # Mock world model prediction
                mock_predict_state.side_effect = lambda wm, s, a, l: f"{s} after {a}"

                num_rollouts, metrics = generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    num_alternatives_per_demo=2,
                    logger=logger,
                )

                # Should generate rollouts
                assert num_rollouts > 0
                assert isinstance(metrics, dict)

                # Output file should exist
                assert output_path.exists()

                # Load and validate rollouts
                rollouts = load_jsonl(output_path)
                assert len(rollouts) == num_rollouts

    def test_achieves_target_expansion_ratio(self, sample_demos_file, temp_dir, mock_lm):
        """Test that generate_exploratory_rollouts() achieves 3x expansion ratio (SC-002)."""
        logger = setup_logger("test_exploration")
        output_path = temp_dir / "exploratory_rollouts.jsonl"

        # Create mock world model
        mock_world_model = Mock(spec=WorldModelModule)
        mock_world_model.side_effect = lambda s, a: Mock(next_state=f"{s} after {a}")

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            with patch("agent_learning.exploration.predict_next_state") as mock_predict_state:
                # Mock alternative generation (2 alternatives per demo)
                mock_gen_alts.return_value = ["alternative_1", "alternative_2"]
                mock_predict_state.side_effect = lambda wm, s, a, l: f"{s} after {a}"

                num_rollouts, metrics = generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    num_alternatives_per_demo=2,
                    target_expansion_ratio=3.0,
                    logger=logger,
                )

                # Check expansion ratio in metrics
                assert "expansion_ratio" in metrics
                expansion_ratio = metrics["expansion_ratio"]

                # Should be close to 3.0x (original + 2x alternatives = 3x total)
                assert expansion_ratio >= 2.0, f"Expansion ratio {expansion_ratio:.2f}x below minimum"

    def test_validates_alternative_coverage(self, sample_demos_file, temp_dir, mock_lm):
        """Test that generate_exploratory_rollouts() validates 50%+ alternative coverage (SC-003)."""
        logger = setup_logger("test_exploration")
        output_path = temp_dir / "exploratory_rollouts.jsonl"

        mock_world_model = Mock(spec=WorldModelModule)
        mock_world_model.side_effect = lambda s, a: Mock(next_state=f"{s} after {a}")

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            with patch("agent_learning.exploration.predict_next_state") as mock_predict_state:
                # Mock alternatives that differ from expert
                mock_gen_alts.return_value = ["different_action_1", "different_action_2"]
                mock_predict_state.side_effect = lambda wm, s, a, l: f"{s} after {a}"

                num_rollouts, metrics = generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    num_alternatives_per_demo=2,
                    logger=logger,
                )

                # Check alternative coverage in metrics
                assert "alternative_coverage" in metrics
                coverage = metrics["alternative_coverage"]

                # Should meet 50%+ coverage requirement
                assert coverage >= 0.5, f"Alternative coverage {coverage:.2%} below 50% threshold"

    def test_creates_rollouts_with_required_fields(self, sample_demos_file, temp_dir, mock_lm):
        """Test that generated rollouts contain all required fields."""
        logger = setup_logger("test_exploration")
        output_path = temp_dir / "exploratory_rollouts.jsonl"

        mock_world_model = Mock(spec=WorldModelModule)
        mock_world_model.side_effect = lambda s, a: Mock(next_state=f"{s} after {a}")

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            with patch("agent_learning.exploration.predict_next_state") as mock_predict_state:
                mock_gen_alts.return_value = ["alternative_1"]
                mock_predict_state.side_effect = lambda wm, s, a, l: f"{s} after {a}"

                generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    num_alternatives_per_demo=1,
                    logger=logger,
                )

                # Load rollouts
                rollouts = load_jsonl(output_path)

                # Check required fields
                required_fields = ["state", "action", "next_state", "source_demo_id",
                                 "expert_action", "expert_next_state"]

                for rollout in rollouts:
                    for field in required_fields:
                        assert field in rollout, f"Rollout missing required field: {field}"
                        assert rollout[field] is not None, f"Field {field} is None"

    def test_handles_insufficient_expert_demos(self, temp_dir, mock_lm):
        """Test that generate_exploratory_rollouts() rejects too few expert demos."""
        logger = setup_logger("test_exploration")

        # Create file with insufficient demos
        insufficient_demos = SAMPLE_EXPERT_DEMOS[:5]  # Less than 10
        demos_path = temp_dir / "insufficient_demos.jsonl"
        save_jsonl(insufficient_demos, demos_path)

        output_path = temp_dir / "exploratory_rollouts.jsonl"

        mock_world_model = Mock(spec=WorldModelModule)

        with pytest.raises(ValueError, match="at least 10"):
            generate_exploratory_rollouts(
                expert_demos_path=str(demos_path),
                world_model=mock_world_model,
                output_path=str(output_path),
                logger=logger,
            )

    def test_tracks_metrics_correctly(self, sample_demos_file, temp_dir, mock_lm):
        """Test that generate_exploratory_rollouts() tracks all required metrics."""
        logger = setup_logger("test_exploration")
        metrics_tracker = MetricsTracker()
        output_path = temp_dir / "exploratory_rollouts.jsonl"

        mock_world_model = Mock(spec=WorldModelModule)
        mock_world_model.side_effect = lambda s, a: Mock(next_state=f"{s} after {a}")

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            with patch("agent_learning.exploration.predict_next_state") as mock_predict_state:
                mock_gen_alts.return_value = ["alternative_1", "alternative_2"]
                mock_predict_state.side_effect = lambda wm, s, a, l: f"{s} after {a}"

                num_rollouts, metrics = generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    num_alternatives_per_demo=2,
                    logger=logger,
                    metrics_tracker=metrics_tracker,
                )

                # Check all expected metrics are present
                expected_metrics = [
                    "num_rollouts",
                    "expansion_ratio",
                    "alternative_coverage",
                    "duration",
                ]

                for metric_name in expected_metrics:
                    assert metric_name in metrics, f"Missing metric: {metric_name}"


# ============================================================================
# Test Validation Functions (T020)
# ============================================================================

class TestValidateExploratoryData:
    """Test exploratory rollout data validation."""

    def test_validates_correct_data(self, sample_demos_file, temp_dir):
        """Test that validate_exploratory_data() accepts correct data."""
        logger = setup_logger("test_exploration")

        # Create valid exploratory rollouts
        expert_demos = load_jsonl(sample_demos_file)
        exploratory_rollouts = [
            {
                "state": "test state 1",
                "action": "test action 1",
                "next_state": "test next state 1",
                "source_demo_id": 0,
                "expert_action": "expert action 1",
                "expert_next_state": "expert next state 1",
            },
            {
                "state": "test state 2",
                "action": "test action 2",  # Different from expert
                "next_state": "test next state 2",
                "source_demo_id": 1,
                "expert_action": "expert action 2",
                "expert_next_state": "expert next state 2",
            },
        ]

        result = validate_exploratory_data(
            exploratory_rollouts=exploratory_rollouts,
            expert_demos=expert_demos,
            min_alternative_coverage=0.5,
            min_expansion_ratio=1.0,
            logger=logger,
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_detects_missing_fields(self, sample_demos_file):
        """Test that validate_exploratory_data() detects missing required fields."""
        logger = setup_logger("test_exploration")

        expert_demos = load_jsonl(sample_demos_file)

        # Create rollout with missing field
        invalid_rollouts = [
            {
                "state": "test state",
                "action": "test action",
                # Missing: next_state
                "source_demo_id": 0,
                "expert_action": "expert action",
                "expert_next_state": "expert next state",
            },
        ]

        result = validate_exploratory_data(
            exploratory_rollouts=invalid_rollouts,
            expert_demos=expert_demos,
            logger=logger,
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("missing" in error.lower() for error in result["errors"])

    def test_detects_low_alternative_coverage(self, sample_demos_file):
        """Test that validate_exploratory_data() detects low alternative coverage (<50%)."""
        logger = setup_logger("test_exploration")

        expert_demos = load_jsonl(sample_demos_file)

        # Create rollouts where all alternatives match expert action
        low_coverage_rollouts = [
            {
                "state": "test state",
                "action": "same action",  # Same as expert
                "next_state": "test next state",
                "source_demo_id": 0,
                "expert_action": "same action",
                "expert_next_state": "expert next state",
            },
        ]

        result = validate_exploratory_data(
            exploratory_rollouts=low_coverage_rollouts,
            expert_demos=expert_demos,
            min_alternative_coverage=0.5,
            logger=logger,
        )

        # Should have error or warning about low coverage
        assert result["valid"] is False or len(result["warnings"]) > 0
        assert "alternative_coverage" in result

    def test_detects_low_expansion_ratio(self, sample_demos_file):
        """Test that validate_exploratory_data() detects low expansion ratio."""
        logger = setup_logger("test_exploration")

        expert_demos = load_jsonl(sample_demos_file)

        # Create only 1 rollout for many expert demos (very low expansion)
        low_expansion_rollouts = [
            {
                "state": "test state",
                "action": "test action",
                "next_state": "test next state",
                "source_demo_id": 0,
                "expert_action": "expert action",
                "expert_next_state": "expert next state",
            },
        ]

        result = validate_exploratory_data(
            exploratory_rollouts=low_expansion_rollouts,
            expert_demos=expert_demos,
            min_expansion_ratio=2.0,
            logger=logger,
        )

        # Should have warning about low expansion
        assert len(result["warnings"]) > 0 or result["valid"] is False
        assert "expansion_ratio" in result


class TestCheckAlternativeCoverage:
    """Test alternative coverage calculation."""

    def test_calculates_correct_coverage(self):
        """Test that check_alternative_coverage() calculates coverage correctly."""
        exploratory_rollouts = [
            {
                "action": "different_action_1",
                "expert_action": "expert_action",
            },
            {
                "action": "different_action_2",
                "expert_action": "expert_action",
            },
            {
                "action": "expert_action",  # Same as expert
                "expert_action": "expert_action",
            },
        ]

        coverage = check_alternative_coverage(exploratory_rollouts, threshold=0.5)

        # 2 out of 3 differ = 66.7% coverage
        assert coverage > 0.6
        assert coverage < 0.7

    def test_handles_empty_rollouts(self):
        """Test that check_alternative_coverage() handles empty rollouts list."""
        coverage = check_alternative_coverage([], threshold=0.5)

        # Should return 0.0 for empty list
        assert coverage == 0.0

    def test_handles_all_same_actions(self):
        """Test coverage when all actions match expert."""
        rollouts = [
            {"action": "same", "expert_action": "same"},
            {"action": "same", "expert_action": "same"},
        ]

        coverage = check_alternative_coverage(rollouts, threshold=0.5)

        # Should return 0.0 when all actions match
        assert coverage == 0.0

    def test_handles_all_different_actions(self):
        """Test coverage when all actions differ from expert."""
        rollouts = [
            {"action": "different_1", "expert_action": "expert"},
            {"action": "different_2", "expert_action": "expert"},
        ]

        coverage = check_alternative_coverage(rollouts, threshold=0.5)

        # Should return 1.0 when all actions differ
        assert coverage == 1.0


class TestCalculateExpansionRatio:
    """Test expansion ratio calculation."""

    def test_calculates_correct_expansion_ratio(self):
        """Test that calculate_expansion_ratio() calculates ratio correctly."""
        num_expert_demos = 10
        num_exploratory_rollouts = 20

        expansion_ratio = calculate_expansion_ratio(num_expert_demos, num_exploratory_rollouts)

        # (10 + 20) / 10 = 3.0x
        assert expansion_ratio == 3.0

    def test_handles_zero_expansion(self):
        """Test expansion ratio with no exploratory rollouts."""
        num_expert_demos = 10
        num_exploratory_rollouts = 0

        expansion_ratio = calculate_expansion_ratio(num_expert_demos, num_exploratory_rollouts)

        # (10 + 0) / 10 = 1.0x (no expansion)
        assert expansion_ratio == 1.0

    def test_handles_equal_amounts(self):
        """Test expansion ratio with equal expert demos and rollouts."""
        num_expert_demos = 10
        num_exploratory_rollouts = 10

        expansion_ratio = calculate_expansion_ratio(num_expert_demos, num_exploratory_rollouts)

        # (10 + 10) / 10 = 2.0x
        assert expansion_ratio == 2.0

    def test_handles_large_expansion(self):
        """Test expansion ratio with large expansion."""
        num_expert_demos = 10
        num_exploratory_rollouts = 50

        expansion_ratio = calculate_expansion_ratio(num_expert_demos, num_exploratory_rollouts)

        # (10 + 50) / 10 = 6.0x
        assert expansion_ratio == 6.0


# ============================================================================
# Test Error Handling and Edge Cases
# ============================================================================

class TestExplorationErrorHandling:
    """Test error handling in exploration module."""

    def test_handles_missing_expert_demos_file(self, temp_dir):
        """Test handling of missing expert demos file."""
        logger = setup_logger("test_exploration")

        nonexistent_path = temp_dir / "nonexistent.jsonl"
        output_path = temp_dir / "output.jsonl"

        mock_world_model = Mock(spec=WorldModelModule)

        with pytest.raises(FileNotFoundError):
            generate_exploratory_rollouts(
                expert_demos_path=str(nonexistent_path),
                world_model=mock_world_model,
                output_path=str(output_path),
                logger=logger,
            )

    def test_handles_malformed_expert_demos(self, temp_dir):
        """Test handling of malformed expert demos."""
        logger = setup_logger("test_exploration")

        # Create file with malformed data
        malformed_demos = [
            {"state": "test", "action": "test"},  # Missing next_state
        ]
        demos_path = temp_dir / "malformed_demos.jsonl"
        save_jsonl(malformed_demos, demos_path)

        output_path = temp_dir / "output.jsonl"
        mock_world_model = Mock(spec=WorldModelModule)

        with pytest.raises(ValueError, match="missing required fields"):
            generate_exploratory_rollouts(
                expert_demos_path=str(demos_path),
                world_model=mock_world_model,
                output_path=str(output_path),
                logger=logger,
            )

    def test_handles_world_model_prediction_failure(self, sample_demos_file, temp_dir):
        """Test handling when world model prediction fails."""
        logger = setup_logger("test_exploration")
        output_path = temp_dir / "output.jsonl"

        # Mock world model that raises exception
        mock_world_model = Mock(spec=WorldModelModule)
        mock_world_model.side_effect = Exception("Model prediction failed")

        with patch("agent_learning.exploration.generate_alternative_actions") as mock_gen_alts:
            mock_gen_alts.return_value = ["alternative_1"]

            # Should handle gracefully and log error
            with pytest.raises(Exception):
                generate_exploratory_rollouts(
                    expert_demos_path=str(sample_demos_file),
                    world_model=mock_world_model,
                    output_path=str(output_path),
                    logger=logger,
                )
