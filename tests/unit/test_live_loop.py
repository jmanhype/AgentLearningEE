"""
Unit tests for Live Exploration Loop.

Tests the continuous learning loop orchestrator.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from agent_learning.live_loop import (
    LiveExplorationLoop,
    LiveLoopConfig,
    Episode,
    LiveLoopMetrics,
)


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, num_episodes=10):
        self.num_episodes = num_episodes
        self.episode_count = 0

    def reset(self):
        """Return mock state."""
        return f"state_{self.episode_count}"

    def step(self, action):
        """Return mock next state."""
        next_state = f"next_state_{self.episode_count}"
        self.episode_count += 1
        done = self.episode_count >= self.num_episodes
        return next_state, done


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_policy():
    """Create temporary mock policy file."""
    temp_dir = Path(tempfile.mkdtemp())
    policy_path = temp_dir / "policy.pkl"

    # Create minimal policy file
    import pickle

    with open(policy_path, "wb") as f:
        pickle.dump({"mock": "policy"}, f)

    yield policy_path

    shutil.rmtree(temp_dir)


class TestLiveLoopConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LiveLoopConfig()

        assert config.episode_batch_size == 10
        assert config.max_episodes is None
        assert config.reflection_interval == 10
        assert config.min_episodes_for_reflection == 5
        assert config.ace_enabled is True
        assert config.save_episodes is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LiveLoopConfig(
            episode_batch_size=20,
            max_episodes=100,
            ace_enabled=False,
        )

        assert config.episode_batch_size == 20
        assert config.max_episodes == 100
        assert config.ace_enabled is False


class TestEpisode:
    """Test Episode dataclass."""

    def test_episode_creation(self):
        """Test creating episode."""
        episode = Episode(
            state="test state",
            action="test action",
            next_state="test next state",
        )

        assert episode.state == "test state"
        assert episode.action == "test action"
        assert episode.next_state == "test next state"
        assert episode.timestamp > 0


class TestLiveLoopMetrics:
    """Test metrics tracking."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = LiveLoopMetrics()

        assert metrics.total_episodes == 0
        assert metrics.total_reflections == 0
        assert metrics.total_ace_updates == 0
        assert metrics.runtime_seconds() >= 0

    def test_episodes_per_minute(self):
        """Test throughput calculation."""
        metrics = LiveLoopMetrics()
        metrics.total_episodes = 60
        metrics.loop_start_time -= 60  # Simulate 1 minute runtime

        assert metrics.episodes_per_minute() == pytest.approx(60.0, rel=0.1)


class TestLiveExplorationLoop:
    """Test live exploration loop orchestrator."""

    @patch("agent_learning.policy.load_trained_policy")
    def test_initialization(self, mock_load_policy, temp_output_dir, temp_policy):
        """Test loop initialization."""
        environment = MockEnvironment()
        config = LiveLoopConfig(output_dir=temp_output_dir)

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        assert loop.environment == environment
        assert loop.policy_path == temp_policy
        assert loop.config == config
        assert len(loop.episode_buffer) == 0
        assert loop.metrics.total_episodes == 0

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    def test_episode_collection(
        self, mock_generate, mock_load_policy, temp_output_dir, temp_policy
    ):
        """Test single episode collection."""
        # Mock policy
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy
        mock_generate.return_value = ("reasoning", "test_action")

        environment = MockEnvironment()
        config = LiveLoopConfig(output_dir=temp_output_dir, ace_enabled=False)

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        episode = loop._collect_episode()

        assert episode is not None
        assert episode.state == "state_0"
        assert episode.action == "test_action"
        assert episode.next_state == "next_state_0"
        mock_generate.assert_called_once()

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    def test_run_with_max_episodes(
        self, mock_generate, mock_load_policy, temp_output_dir, temp_policy
    ):
        """Test running loop with episode limit."""
        # Mock policy
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy
        mock_generate.return_value = ("reasoning", "test_action")

        environment = MockEnvironment()
        config = LiveLoopConfig(
            output_dir=temp_output_dir,
            max_episodes=5,
            reflection_interval=100,  # Disable reflections for this test
            ace_enabled=False,
        )

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        metrics = loop.run()

        assert metrics.total_episodes == 5
        assert len(loop.episode_buffer) == 5

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    @patch("dspy.ChainOfThought")
    def test_reflection_trigger(
        self,
        mock_cot_cls,
        mock_generate,
        mock_load_policy,
        temp_output_dir,
        temp_policy,
    ):
        """Test reflection generation trigger."""
        # Mock policy
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy
        mock_generate.return_value = ("reasoning", "test_action")

        # Mock ChainOfThought reflection
        mock_cot = Mock()
        mock_result = Mock()
        mock_result.reasoning = "test reasoning"
        mock_cot.return_value = mock_result
        mock_cot_cls.return_value = mock_cot

        environment = MockEnvironment()
        config = LiveLoopConfig(
            output_dir=temp_output_dir,
            max_episodes=10,
            reflection_interval=10,  # Reflect after 10 episodes
            min_episodes_for_reflection=5,
            ace_enabled=False,
        )

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        metrics = loop.run()

        assert metrics.total_episodes == 10
        assert metrics.total_reflections > 0  # Should have generated reflections

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    def test_health_check(
        self, mock_generate, mock_load_policy, temp_output_dir, temp_policy
    ):
        """Test health check functionality."""
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy

        environment = MockEnvironment()
        config = LiveLoopConfig(output_dir=temp_output_dir, ace_enabled=False)

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        health = loop._health_check()

        assert health["status"] == "healthy"
        assert "metrics" in health
        assert "buffer_size" in health
        assert health["running"] is False  # Not started yet

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    def test_graceful_stop(
        self, mock_generate, mock_load_policy, temp_output_dir, temp_policy
    ):
        """Test graceful shutdown."""
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy
        mock_generate.return_value = ("reasoning", "test_action")

        environment = MockEnvironment()
        config = LiveLoopConfig(
            output_dir=temp_output_dir,
            max_episodes=100,  # High limit
            ace_enabled=False,
        )

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        # Stop after 1 episode
        def stop_after_one(*args, **kwargs):
            if loop.metrics.total_episodes >= 1:
                loop.stop()
            return "reasoning", "test_action"

        mock_generate.side_effect = stop_after_one

        metrics = loop.run()

        # Should stop early
        assert metrics.total_episodes < 100
        assert loop._should_stop is True


class TestACEIntegration:
    """Test ACE integration in live loop."""

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.generate_decision")
    @patch("agent_learning.policy.get_ace_client")
    def test_ace_client_loading(
        self, mock_get_ace, mock_generate, mock_load_policy, temp_output_dir, temp_policy
    ):
        """Test ACE client loading when enabled."""
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy

        mock_ace_client = Mock()
        mock_get_ace.return_value = mock_ace_client

        environment = MockEnvironment()
        config = LiveLoopConfig(output_dir=temp_output_dir, ace_enabled=True)

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        client = loop._get_ace_client()

        assert client == mock_ace_client
        mock_get_ace.assert_called_once()

    @patch("agent_learning.policy.load_trained_policy")
    @patch("agent_learning.policy.get_ace_client")
    def test_ace_disabled(self, mock_get_ace, mock_load_policy, temp_output_dir, temp_policy):
        """Test that ACE is not loaded when disabled."""
        mock_policy = Mock()
        mock_load_policy.return_value = mock_policy

        environment = MockEnvironment()
        config = LiveLoopConfig(output_dir=temp_output_dir, ace_enabled=False)

        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=temp_policy,
            config=config,
        )

        client = loop._get_ace_client()

        assert client is None
        mock_get_ace.assert_not_called()
