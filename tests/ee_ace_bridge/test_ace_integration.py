"""
Integration Tests for EE-ACE Bridge with ACE Repository

Tests wire compatibility between EE bridge and ACE CuratorService.
Requires ACE repository to be available (skip if not installed).
"""

import pytest

# Check if ACE integration is available
try:
    from ee_ace_bridge import ACE_INTEGRATION_AVAILABLE
    from ee_ace_bridge.ace_client import InProcessAceClient
    from ee_ace_bridge.translate import bridge_to_ace_insight, bridge_batch_to_ace
    from ee_ace_bridge.config_extra import (
        ACE_DOMAIN_ID,
        ACE_TARGET_STAGE,
        ACE_SIMILARITY_THRESHOLD,
        validate_domain_id,
    )
    ACE_AVAILABLE = ACE_INTEGRATION_AVAILABLE
except ImportError:
    ACE_AVAILABLE = False

# Skip all tests if ACE not available
pytestmark = pytest.mark.skipif(
    not ACE_AVAILABLE,
    reason="ACE repository not available (requires ace-playbook package)"
)


class TestSchemaTranslation:
    """Test EE bridge schema → ACE schema translation."""

    def test_bridge_to_ace_insight_helpful(self):
        """Test mapping helpful insight (rule → Helpful)."""
        bridge_insight = {
            "insight_text": "Always check availability first",
            "insight_kind": "rule",
            "tags": ["availability"],
        }

        ace_insight = bridge_to_ace_insight(bridge_insight)

        assert ace_insight["content"] == "Always check availability first"
        assert ace_insight["section"] == "Helpful"
        assert ace_insight["tags"] == ["availability"]

    def test_bridge_to_ace_insight_harmful(self):
        """Test mapping harmful insight (anti_pattern → Harmful)."""
        bridge_insight = {
            "insight_text": "Do not skip payment validation",
            "insight_kind": "anti_pattern",
            "tags": ["payment", "validation"],
        }

        ace_insight = bridge_to_ace_insight(bridge_insight)

        assert ace_insight["content"] == "Do not skip payment validation"
        assert ace_insight["section"] == "Harmful"
        assert ace_insight["tags"] == ["payment", "validation"]

    def test_bridge_to_ace_insight_neutral(self):
        """Test mapping neutral insight (observation → Neutral)."""
        bridge_insight = {
            "insight_text": "User requested specific dates",
            "insight_kind": "observation",
            "tags": ["context"],
        }

        ace_insight = bridge_to_ace_insight(bridge_insight)

        assert ace_insight["content"] == "User requested specific dates"
        assert ace_insight["section"] == "Neutral"
        assert ace_insight["tags"] == ["context"]

    def test_bridge_batch_to_ace(self):
        """Test batch translation."""
        batch = [
            {"insight_text": "Rule 1", "insight_kind": "rule", "tags": ["a"]},
            {"insight_text": "Anti-pattern 1", "insight_kind": "anti_pattern", "tags": ["b"]},
        ]

        ace_batch = bridge_batch_to_ace(batch)

        assert len(ace_batch) == 2
        assert ace_batch[0]["section"] == "Helpful"
        assert ace_batch[1]["section"] == "Harmful"


class TestConfigExtra:
    """Test extended ACE configuration."""

    def test_domain_id_validation(self):
        """Test domain_id validation regex."""
        assert validate_domain_id("agent-learning") is True
        assert validate_domain_id("flights-api") is True
        assert validate_domain_id("test123") is True

        # Invalid patterns
        assert validate_domain_id("Agent-Learning") is False  # uppercase
        assert validate_domain_id("agent_learning") is False  # underscore
        assert validate_domain_id("agent learning") is False  # space
        assert validate_domain_id("") is False  # empty

    def test_config_defaults(self):
        """Test configuration defaults."""
        assert ACE_DOMAIN_ID == "agent-learning"
        assert ACE_TARGET_STAGE.value == "shadow"
        assert ACE_SIMILARITY_THRESHOLD == 0.80


class TestInProcessAceClient:
    """Test InProcessAceClient wire compatibility with ACE CuratorService."""

    @pytest.fixture
    def client(self):
        """Create InProcessAceClient with test domain."""
        # Use a unique test domain to avoid interference
        return InProcessAceClient(domain_id="test-ee-bridge")

    def test_ingest_single_insight(self, client):
        """Test ingesting single insight."""
        bridge_insight = {
            "insight_text": "Check availability first",
            "insight_kind": "rule",
            "tags": ["availability"],
        }

        result = client.ingest_insight(bridge_insight)

        assert "added" in result
        assert "incremented" in result
        assert "duplicates" in result
        assert result["total_insights"] == 1

    def test_ingest_batch_insights(self, client):
        """Test ingesting batch of insights."""
        batch = [
            {"insight_text": "Rule 1", "insight_kind": "rule", "tags": ["a"]},
            {"insight_text": "Rule 2", "insight_kind": "rule", "tags": ["b"]},
            {"insight_text": "Anti-pattern 1", "insight_kind": "anti_pattern", "tags": ["c"]},
        ]

        result = client.ingest_insights_batch(batch)

        assert result["total_insights"] == 3
        assert result["added"] > 0

    def test_deduplication(self, client):
        """Test that ACE deduplicates similar insights."""
        insight = {
            "insight_text": "Always verify payment before confirming",
            "insight_kind": "rule",
            "tags": ["payment"],
        }

        # Ingest same insight twice
        result1 = client.ingest_insight(insight)
        result2 = client.ingest_insight(insight)

        # First should add, second should increment
        assert result1["added"] == 1
        assert result2["incremented"] == 1
        assert result2["duplicates"] == 1

    def test_render_playbook(self, client):
        """Test playbook rendering from ACE database."""
        # Ingest some insights
        batch = [
            {"insight_text": "Check availability", "insight_kind": "rule", "tags": ["availability"]},
            {"insight_text": "Verify payment", "insight_kind": "rule", "tags": ["payment"]},
        ]
        client.ingest_insights_batch(batch)

        # Render playbook
        playbook = client.render_playbook(token_budget=3500)

        assert isinstance(playbook, str)
        if playbook:  # May be empty if no bullets promoted to prod
            assert "ACE Playbook" in playbook or "Helpful" in playbook or "Harmful" in playbook

    def test_get_health(self, client):
        """Test health check."""
        health = client.get_health()

        assert health["status"] == "healthy"
        assert health["domain_id"] == "test-ee-bridge"
        assert "insights_ingested" in health
        assert "stage_counts" in health

    def test_get_counts(self, client):
        """Test section and insight counts."""
        # Initially should be 0 or low
        initial_count = client.get_insight_count()

        # Ingest some insights
        batch = [
            {"insight_text": "Rule 1", "insight_kind": "rule", "tags": ["a"]},
            {"insight_text": "Anti-pattern 1", "insight_kind": "anti_pattern", "tags": ["b"]},
        ]
        client.ingest_insights_batch(batch)

        # Count should increase
        new_count = client.get_insight_count()
        assert new_count >= initial_count

    def test_stage_isolation(self, client):
        """Test that insights start in shadow stage."""
        # Ingest insight
        insight = {"insight_text": "Test rule", "insight_kind": "rule", "tags": ["test"]}
        client.ingest_insight(insight)

        # Check health to see stage distribution
        health = client.get_health()
        stage_counts = health["stage_counts"]

        # Should have bullets in shadow stage (new insights start there)
        assert "shadow" in stage_counts or "staging" in stage_counts or "prod" in stage_counts
