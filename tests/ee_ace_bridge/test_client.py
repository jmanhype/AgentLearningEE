"""
Tests for AceClient protocol and InMemoryAceClient stub.

Validates:
- Protocol compliance
- Insight ingestion
- Playbook rendering
- Section routing
- Token budget enforcement
"""

import pytest
from ee_ace_bridge import InMemoryAceClient


class TestInMemoryAceClient:
    """Test suite for InMemoryAceClient stub implementation."""

    def test_ingest_insight_returns_id(self):
        """Test that ingest_insight returns unique identifier."""
        ace = InMemoryAceClient()

        insight = {
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability first",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        }

        insight_id = ace.ingest_insight(insight)

        assert isinstance(insight_id, str)
        assert insight_id.startswith("mem-")

    def test_ingest_insight_increments_counter(self):
        """Test that multiple ingestions get unique IDs."""
        ace = InMemoryAceClient()

        insight1 = {
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "First rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        }

        insight2 = {
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Second rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:01:00Z"
        }

        id1 = ace.ingest_insight(insight1)
        id2 = ace.ingest_insight(insight2)

        assert id1 != id2

    def test_render_playbook_empty_when_no_insights(self):
        """Test that empty playbook renders as empty string."""
        ace = InMemoryAceClient()

        playbook = ace.render_playbook()

        assert playbook == ""

    def test_render_playbook_formats_insights(self):
        """Test playbook renders with proper markdown formatting."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability first",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()

        assert "## Decision Heuristics/Availability" in playbook
        assert "- Check availability first" in playbook

    def test_render_playbook_respects_token_budget(self):
        """Test that playbook rendering respects token budget."""
        ace = InMemoryAceClient()

        # Add many insights
        for i in range(100):
            ace.ingest_insight({
                "task": "booking",
                "insight_kind": "rule",
                "insight_text": f"Rule number {i}: This is a long rule with many words",
                "tags": ["general"],
                "created_at": "2025-01-15T10:00:00Z"
            })

        # Render with small token budget
        playbook = ace.render_playbook(token_budget=100)

        # Should be clipped (~400 chars for 100 tokens)
        assert len(playbook) <= 400

    def test_render_playbook_filters_by_section(self):
        """Test that playbook can filter by section names."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.ingest_insight({
            "task": "payment",
            "insight_kind": "rule",
            "insight_text": "Validate credit card",
            "tags": ["payment"],
            "created_at": "2025-01-15T10:01:00Z"
        })

        # Filter to only payment section
        playbook = ace.render_playbook(sections=["Payment/Validation"])

        assert "Validate credit card" in playbook
        assert "Check availability" not in playbook

    def test_section_routing_availability(self):
        """Test that availability tags route to correct section."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()
        assert "Decision Heuristics/Availability" in playbook

    def test_section_routing_payment(self):
        """Test that payment tags route to correct section."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Validate payment",
            "tags": ["payment"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()
        assert "Payment/Validation" in playbook

    def test_section_routing_error_avoidance(self):
        """Test that error-avoidance tags route to correct section."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Avoid double booking",
            "tags": ["error-avoidance"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()
        assert "Error Prevention" in playbook

    def test_section_routing_safety(self):
        """Test that safety tags route to correct section."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Ensure secure connection",
            "tags": ["safety"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()
        assert "Safety Rules" in playbook

    def test_section_routing_fallback_to_task(self):
        """Test that unknown tags fallback to task-based section."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Some rule",
            "tags": ["unknown-tag"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        playbook = ace.render_playbook()
        assert "General/Booking" in playbook

    def test_health_returns_status(self):
        """Test that health check returns proper status."""
        ace = InMemoryAceClient()

        health = ace.health()

        assert health["status"] == "ok"
        assert "sections" in health
        assert "insights" in health

    def test_health_reflects_state(self):
        """Test that health reflects current state."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        health = ace.health()

        assert int(health["sections"]) >= 1
        assert int(health["insights"]) >= 1

    def test_version_returns_string(self):
        """Test that version returns stub identifier."""
        ace = InMemoryAceClient()

        version = ace.version()

        assert isinstance(version, str)
        assert "stub" in version.lower()

    def test_clear_removes_all_insights(self):
        """Test that clear() removes all stored insights."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.clear()

        assert ace.get_section_count() == 0
        assert ace.get_insight_count() == 0
        assert ace.render_playbook() == ""

    def test_get_section_count(self):
        """Test that get_section_count returns accurate count."""
        ace = InMemoryAceClient()

        # Add insights to different sections
        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Check availability",
            "tags": ["availability"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.ingest_insight({
            "task": "payment",
            "insight_kind": "rule",
            "insight_text": "Validate payment",
            "tags": ["payment"],
            "created_at": "2025-01-15T10:01:00Z"
        })

        assert ace.get_section_count() >= 2

    def test_get_insight_count(self):
        """Test that get_insight_count returns accurate total."""
        ace = InMemoryAceClient()

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Rule 1",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        })

        ace.ingest_insight({
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Rule 2",
            "tags": ["general"],
            "created_at": "2025-01-15T10:01:00Z"
        })

        assert ace.get_insight_count() == 2

    def test_preserves_provided_id(self):
        """Test that client preserves insight ID if provided."""
        ace = InMemoryAceClient()

        custom_id = "custom-id-123"
        insight = {
            "id": custom_id,
            "task": "booking",
            "insight_kind": "rule",
            "insight_text": "Test rule",
            "tags": ["general"],
            "created_at": "2025-01-15T10:00:00Z"
        }

        returned_id = ace.ingest_insight(insight)

        assert returned_id == custom_id
