"""
Tests for playbook context injection.

Validates:
- Playbook augmentation
- Graceful no-op behavior
- Token budget enforcement
- Header formatting
- Batch processing
- Validation utilities
"""

import pytest
from ee_ace_bridge import InMemoryAceClient
from ee_ace_bridge.injector import (
    augment_state_with_playbook,
    augment_states_batch,
    estimate_token_overhead,
    validate_injection,
    PLAYBOOK_HEADER,
    STATE_HEADER,
    INSTRUCTIONS,
)


class TestAugmentStateWithPlaybook:
    """Test suite for playbook augmentation function."""

    def test_graceful_noop_when_playbook_empty(self):
        """Test that empty playbook returns original state unchanged."""
        ace = InMemoryAceClient()
        state = "Current state description"

        augmented = augment_state_with_playbook(state, ace)

        assert augmented == state

    def test_prepends_playbook_when_present(self):
        """Test that playbook is prepended to state."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability first",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        state = "Hotel booking: 2 nights in NYC"
        augmented = augment_state_with_playbook(state, ace)

        # Should contain both playbook and original state
        assert "Check availability first" in augmented
        assert state in augmented
        # Playbook should come before state
        assert augmented.index("Check availability") < augmented.index(state)

    def test_includes_playbook_header(self):
        """Test that playbook header is included."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        state = "Test state"
        augmented = augment_state_with_playbook(state, ace)

        assert PLAYBOOK_HEADER in augmented

    def test_includes_state_header(self):
        """Test that state header is included."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        state = "Test state"
        augmented = augment_state_with_playbook(state, ace)

        assert STATE_HEADER in augmented

    def test_includes_instructions(self):
        """Test that usage instructions are included."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        state = "Test state"
        augmented = augment_state_with_playbook(state, ace)

        assert INSTRUCTIONS in augmented

    def test_respects_section_filter(self):
        """Test that section filtering is applied."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Availability rule",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.ingest_insight({
            "task": "payment",
            "insight_kind": "rule",
            "insight_text": "Payment rule",
            "tags": ["payment"],
            "created_at": "2025-01-15T10:01:00Z"
        })

        state = "Test state"
        augmented = augment_state_with_playbook(
            state,
            ace,
            sections=["Payment/Validation"]
        )

        # Should only include payment section
        assert "Payment rule" in augmented
        assert "Availability rule" not in augmented

    def test_respects_token_budget(self):
        """Test that token budget limits playbook size."""
        ace = InMemoryAceClient()

        # Add many insights
        for i in range(100):
            ace.ingest_insight({
                "task": "booking",
                "insight_kind": "rule",
                "insight_text": f"Rule {i}: Long rule text with many words",
                "tags": ["general"],
                "created_at": "2025-01-15T10:00:00Z"
            })

        state = "Test state"
        augmented = augment_state_with_playbook(
            state,
            ace,
            token_budget=200  # Small budget
        )

        # Augmented text should be reasonable size
        # (~200 tokens * 4 chars + headers + state)
        assert len(augmented) < 1500

    def test_preserves_state_integrity(self):
        """Test that original state appears exactly as provided."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        state = "Complex state with\nmultiple lines\nand special chars: @#$"
        augmented = augment_state_with_playbook(state, ace)

        # State should appear exactly at the end
        assert augmented.endswith(state)


class TestAugmentStatesBatch:
    """Test suite for batch augmentation."""

    def test_augments_multiple_states(self):
        """Test batch augmentation of multiple states."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        states = ["State 1", "State 2", "State 3"]
        augmented = augment_states_batch(states, ace)

        assert len(augmented) == 3
        # All should contain playbook
        assert all("Test rule" in state for state in augmented)
        # Each should preserve original state
        for i, aug_state in enumerate(augmented):
            assert states[i] in aug_state

    def test_graceful_noop_for_batch(self):
        """Test batch no-op when playbook empty."""
        ace = InMemoryAceClient()
        states = ["State 1", "State 2", "State 3"]

        augmented = augment_states_batch(states, ace)

        # Should return original states unchanged
        assert augmented == states

    def test_preserves_order(self):
        """Test that batch preserves state order."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        states = [f"State {i}" for i in range(10)]
        augmented = augment_states_batch(states, ace)

        # Extract original states from augmented (should be at end)
        for i, aug_state in enumerate(augmented):
            assert aug_state.endswith(states[i])

    def test_uses_same_playbook_for_all(self):
        """Test that same playbook is applied to all states."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Common rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        states = ["State 1", "State 2"]
        augmented = augment_states_batch(states, ace)

        # Extract playbook parts (everything before STATE_HEADER)
        playbook_parts = [
            aug.split(STATE_HEADER)[0]
            for aug in augmented
        ]

        # All playbook parts should be identical
        assert playbook_parts[0] == playbook_parts[1]


class TestEstimateTokenOverhead:
    """Test suite for token estimation."""

    def test_returns_reasonable_estimate(self):
        """Test that token estimate is reasonable."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Short rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        overhead = estimate_token_overhead(ace)

        # Should be a positive number
        assert overhead > 0
        # Rough sanity check (~4 chars per token)
        assert overhead < 1000

    def test_scales_with_playbook_size(self):
        """Test that estimate scales with playbook size."""
        ace = InMemoryAceClient()

        # Small playbook
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Small",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        small_overhead = estimate_token_overhead(ace)

        # Add many more insights
        for i in range(50):
            ace.ingest_insight({
                "task": "booking",
                "insight_kind": "rule",
                "insight_text": f"Rule {i} with more text",
                "tags": ["general"],
                "created_at": "2025-01-15T10:00:00Z"
            })

        large_overhead = estimate_token_overhead(ace)

        assert large_overhead > small_overhead

    def test_respects_section_filter(self):
        """Test that section filter affects estimate."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Availability rule",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.ingest_insight({
            "task": "payment",
            "insight_kind": "rule",
            "insight_text": "Payment rule",
            "tags": ["payment"],
            "created_at": "2025-01-15T10:01:00Z"
        })

        # Estimate with all sections
        full_overhead = estimate_token_overhead(ace)

        # Estimate with filtered sections
        filtered_overhead = estimate_token_overhead(
            ace,
            sections=["Payment/Validation"]
        )

        assert filtered_overhead < full_overhead


class TestValidateInjection:
    """Test suite for injection validation."""

    def test_validates_successful_injection(self):
        """Test that valid injection passes validation."""
        ace = InMemoryAceClient()
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        original = "Test state"
        augmented = augment_state_with_playbook(original, ace)

        is_valid = validate_injection(original, augmented)

        assert is_valid

    def test_validates_noop_injection(self):
        """Test that no-op injection is valid."""
        ace = InMemoryAceClient()
        state = "Test state"

        augmented = augment_state_with_playbook(state, ace)

        is_valid = validate_injection(state, augmented)

        assert is_valid

    def test_detects_missing_original_state(self):
        """Test detection when original state is missing."""
        original = "Test state"
        augmented = "Different content without original"

        is_valid = validate_injection(original, augmented)

        assert not is_valid

    def test_detects_missing_headers(self):
        """Test detection when required headers are missing."""
        original = "Test state"
        # Manually construct invalid augmentation
        augmented = f"Some playbook content\n{original}"

        is_valid = validate_injection(original, augmented)

        # Should fail because headers are missing
        assert not is_valid

    def test_detects_suspicious_length(self):
        """Test detection when augmented is shorter than original."""
        original = "Test state"
        augmented = "Short"  # Shorter than original, should fail

        is_valid = validate_injection(original, augmented)

        assert not is_valid


class TestHeaderConstants:
    """Test suite for header constant definitions."""

    def test_playbook_header_is_markdown(self):
        """Test that playbook header is markdown format."""
        assert PLAYBOOK_HEADER.startswith("###")

    def test_state_header_is_markdown(self):
        """Test that state header is markdown format."""
        assert STATE_HEADER.startswith("###")

    def test_instructions_not_empty(self):
        """Test that instructions are not empty."""
        assert len(INSTRUCTIONS) > 0

    def test_instructions_mention_playbook(self):
        """Test that instructions reference the playbook."""
        assert "playbook" in INSTRUCTIONS.lower()
