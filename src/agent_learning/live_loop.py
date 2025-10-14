"""
Live Exploration Loop - Continuous Learning System

Implements online learning loop combining:
1. Policy execution with playbook context
2. Experience collection (episodes)
3. Reflection generation on experiences
4. ACE playbook updates
5. Continuous improvement cycle

Architecture:
    Environment → Policy (+ Playbook) → Action
         ↓
    Experience Buffer → Reflection → ACE Insights → Playbook
         ↑________________________________________________|
                        (Continuous Loop)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol
from collections import deque
import logging

from agent_learning.utils import save_jsonl, setup_logger
from agent_learning.policy import PolicyModule


# Environment Protocol (user-provided)
class Environment(Protocol):
    """Protocol for environment that generates experiences."""

    def reset(self) -> str:
        """Reset environment and return initial state."""
        ...

    def step(self, action: str) -> tuple[str, bool]:
        """
        Execute action in environment.

        Returns:
            next_state: Resulting state after action
            done: Whether episode is complete
        """
        ...


@dataclass
class Episode:
    """Single experience tuple."""

    state: str
    action: str
    next_state: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiveLoopConfig:
    """Configuration for live exploration loop."""

    # Episode collection
    episode_batch_size: int = 10  # Reflect after N episodes
    max_episodes: Optional[int] = None  # None = run indefinitely
    episode_timeout: float = 30.0  # Max seconds per episode

    # Reflection triggers
    reflection_interval: int = 10  # Reflect every N episodes
    min_episodes_for_reflection: int = 5  # Min episodes needed

    # ACE integration
    ace_enabled: bool = True  # Enable ACE playbook updates
    ace_update_interval: int = 10  # Update ACE every N reflections

    # Storage
    output_dir: Path = Path("live_loop_artifacts/")
    save_episodes: bool = True
    save_reflections: bool = True

    # Monitoring
    log_level: int = logging.INFO
    health_check_interval: int = 50  # Check health every N episodes


@dataclass
class LiveLoopMetrics:
    """Metrics tracked during live loop."""

    total_episodes: int = 0
    total_reflections: int = 0
    total_ace_updates: int = 0
    loop_start_time: float = field(default_factory=time.time)
    last_reflection_time: Optional[float] = None
    last_ace_update_time: Optional[float] = None

    def runtime_seconds(self) -> float:
        """Get total runtime in seconds."""
        return time.time() - self.loop_start_time

    def episodes_per_minute(self) -> float:
        """Calculate throughput."""
        runtime_minutes = self.runtime_seconds() / 60.0
        if runtime_minutes == 0:
            return 0.0
        return self.total_episodes / runtime_minutes


class LiveExplorationLoop:
    """
    Continuous learning loop with online reflection and ACE updates.

    Example:
        >>> from agent_learning.live_loop import LiveExplorationLoop, LiveLoopConfig
        >>>
        >>> # Define environment
        >>> class MyEnvironment:
        ...     def reset(self):
        ...         return "initial state"
        ...     def step(self, action):
        ...         return "next state", False
        >>>
        >>> # Configure loop
        >>> config = LiveLoopConfig(
        ...     episode_batch_size=10,
        ...     max_episodes=100,
        ...     ace_enabled=True,
        ... )
        >>>
        >>> # Run loop
        >>> loop = LiveExplorationLoop(
        ...     environment=MyEnvironment(),
        ...     policy_path="artifacts/policy.pkl",
        ...     config=config,
        ... )
        >>> metrics = loop.run()
    """

    def __init__(
        self,
        environment: Environment,
        policy_path: str | Path,
        config: Optional[LiveLoopConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize live exploration loop.

        Args:
            environment: Environment implementing reset() and step()
            policy_path: Path to trained policy
            config: Loop configuration
            logger: Optional logger instance
        """
        self.environment = environment
        self.policy_path = Path(policy_path)
        self.config = config or LiveLoopConfig()
        self.logger = logger or setup_logger("live_loop", level=self.config.log_level)

        # Episode buffer
        self.episode_buffer: deque[Episode] = deque(
            maxlen=self.config.episode_batch_size * 2
        )

        # Metrics
        self.metrics = LiveLoopMetrics()

        # Components (lazy-loaded)
        self._policy_module: Optional[PolicyModule] = None
        self._ace_client: Optional[Any] = None

        # Control flags
        self._running = False
        self._should_stop = False

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Live loop initialized with policy: {self.policy_path}")

    def _load_policy(self) -> PolicyModule:
        """Load policy module (cached)."""
        if self._policy_module is None:
            from agent_learning.policy import load_trained_policy

            self._policy_module = load_trained_policy(str(self.policy_path))
            self.logger.info(f"Policy loaded from {self.policy_path}")
        return self._policy_module

    def _get_ace_client(self) -> Optional[Any]:
        """Get ACE client if enabled (cached)."""
        if not self.config.ace_enabled:
            return None

        if self._ace_client is None:
            try:
                from agent_learning.policy import get_ace_client

                self._ace_client = get_ace_client()
                if self._ace_client:
                    self.logger.info("ACE client loaded successfully")
                else:
                    self.logger.warning("ACE client unavailable")
            except ImportError:
                self.logger.warning("ACE integration not available")
                self._ace_client = None

        return self._ace_client

    def _generate_action(self, state: str) -> tuple[str, str]:
        """
        Generate action for given state using policy.

        Returns:
            action: Action to take
            reasoning: Policy's reasoning (with playbook context if ACE enabled)
        """
        policy = self._load_policy()

        # Get playbook context if ACE enabled
        playbook_context = ""
        ace_client = self._get_ace_client()
        if ace_client:
            try:
                from ee_ace_bridge import augment_state_with_playbook

                playbook_context = augment_state_with_playbook(
                    client=ace_client,
                    state=state,
                    token_budget=3500,
                )
            except Exception as e:
                self.logger.warning(f"Playbook augmentation failed: {e}")

        # Generate decision
        from agent_learning.policy import generate_decision

        reasoning, action = generate_decision(policy, state, playbook_context)

        return action, reasoning

    def _collect_episode(self) -> Optional[Episode]:
        """
        Collect single episode from environment.

        Returns:
            Episode if successful, None if timeout/error
        """
        try:
            # Reset environment
            state = self.environment.reset()

            # Generate action
            action, reasoning = self._generate_action(state)

            # Execute in environment
            start_time = time.time()
            next_state, done = self.environment.step(action)

            # Check timeout
            if time.time() - start_time > self.config.episode_timeout:
                self.logger.warning(f"Episode timeout: {state[:50]}...")
                return None

            episode = Episode(
                state=state,
                action=action,
                next_state=next_state,
                timestamp=start_time,
            )

            self.logger.debug(f"Episode collected: state={state[:50]}... action={action}")

            return episode

        except Exception as e:
            self.logger.error(f"Episode collection failed: {e}", exc_info=True)
            return None

    def _trigger_reflection(self) -> List[Dict]:
        """
        Generate reflections on buffered episodes.

        Returns:
            List of reflection dicts
        """
        if len(self.episode_buffer) < self.config.min_episodes_for_reflection:
            self.logger.debug(
                f"Insufficient episodes for reflection: "
                f"{len(self.episode_buffer)}/{self.config.min_episodes_for_reflection}"
            )
            return []

        self.logger.info(f"Triggering reflection on {len(self.episode_buffer)} episodes")

        # Convert episodes to rollout format
        rollouts = [
            {
                "state": ep.state,
                "action": ep.action,
                "next_state": ep.next_state,
                "expert_action": ep.action,  # In live loop, policy action is "expert"
            }
            for ep in self.episode_buffer
        ]

        # Generate reflections using DSPy module
        reflections = []

        # Import ReflectionSig for generating reflections
        from agent_learning.reflection import ReflectionSig
        import dspy

        reflection_module = dspy.ChainOfThought(ReflectionSig)

        for rollout in rollouts:
            try:
                # Generate reflection for this rollout
                result = reflection_module(
                    state=rollout["state"],
                    expert_action=rollout["expert_action"],
                    alternative_actions=rollout.get("alternative_actions", ""),
                    expert_outcome=rollout["next_state"],
                    alternative_outcomes=rollout.get("alternative_outcomes", ""),
                )

                reflection = {
                    "state": rollout["state"],
                    "reasoning": result.reasoning,
                    "action": rollout["expert_action"],
                }
                reflections.append(reflection)
            except Exception as e:
                self.logger.warning(f"Reflection generation failed: {e}")

        self.metrics.total_reflections += len(reflections)
        self.metrics.last_reflection_time = time.time()

        # Save reflections if configured
        if self.config.save_reflections and reflections:
            reflection_path = (
                self.config.output_dir / f"reflections_{int(time.time())}.jsonl"
            )
            save_jsonl(reflections, str(reflection_path))
            self.logger.info(f"Reflections saved to {reflection_path}")

        return reflections

    def _update_ace_playbook(self, reflections: List[Dict]) -> None:
        """
        Update ACE playbook with new reflections.

        Args:
            reflections: List of reflection dicts to ingest
        """
        if not reflections:
            return

        ace_client = self._get_ace_client()
        if not ace_client:
            self.logger.debug("ACE client unavailable, skipping update")
            return

        self.logger.info(f"Updating ACE playbook with {len(reflections)} reflections")

        try:
            # Convert reflections to bridge insights
            from ee_ace_bridge import reflection_to_insight

            insights = [reflection_to_insight(r) for r in reflections]

            # Ingest batch
            result = ace_client.ingest_insights_batch(insights)

            self.metrics.total_ace_updates += 1
            self.metrics.last_ace_update_time = time.time()

            self.logger.info(
                f"ACE playbook updated - added: {result['added']}, "
                f"incremented: {result['incremented']}, duplicates: {result['duplicates']}"
            )

        except Exception as e:
            self.logger.error(f"ACE update failed: {e}", exc_info=True)

    def _health_check(self) -> Dict[str, Any]:
        """
        Perform health check on loop components.

        Returns:
            Health status dict
        """
        health = {
            "status": "healthy",
            "metrics": {
                "total_episodes": self.metrics.total_episodes,
                "total_reflections": self.metrics.total_reflections,
                "total_ace_updates": self.metrics.total_ace_updates,
                "runtime_seconds": self.metrics.runtime_seconds(),
                "episodes_per_minute": self.metrics.episodes_per_minute(),
            },
            "buffer_size": len(self.episode_buffer),
            "running": self._running,
        }

        # Check ACE health if enabled
        ace_client = self._get_ace_client()
        if ace_client:
            try:
                ace_health = ace_client.get_health()
                health["ace"] = ace_health
            except Exception as e:
                health["ace"] = {"status": "error", "error": str(e)}
                health["status"] = "degraded"

        return health

    def run(self) -> LiveLoopMetrics:
        """
        Run live exploration loop.

        Returns:
            Final metrics
        """
        self._running = True
        self._should_stop = False

        self.logger.info(f"Live loop starting - max_episodes={self.config.max_episodes}")

        try:
            while not self._should_stop:
                # Check episode limit
                if (
                    self.config.max_episodes
                    and self.metrics.total_episodes >= self.config.max_episodes
                ):
                    self.logger.info(f"Max episodes reached: {self.config.max_episodes}")
                    break

                # Collect episode
                episode = self._collect_episode()
                if episode:
                    self.episode_buffer.append(episode)
                    self.metrics.total_episodes += 1

                    # Save episode if configured
                    if self.config.save_episodes:
                        episode_path = self.config.output_dir / "episodes.jsonl"
                        save_jsonl(
                            [
                                {
                                    "state": episode.state,
                                    "action": episode.action,
                                    "next_state": episode.next_state,
                                }
                            ],
                            str(episode_path),
                        )

                # Check reflection trigger
                if self.metrics.total_episodes % self.config.reflection_interval == 0:
                    reflections = self._trigger_reflection()

                    # Check ACE update trigger
                    if (
                        reflections
                        and self.metrics.total_reflections
                        % self.config.ace_update_interval
                        == 0
                    ):
                        self._update_ace_playbook(reflections)

                # Health check
                if self.metrics.total_episodes % self.config.health_check_interval == 0:
                    health = self._health_check()
                    self.logger.info(f"Health check: {health['status']}")

        except KeyboardInterrupt:
            self.logger.info("Loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Loop error: {e}", exc_info=True)
        finally:
            self._running = False

            # Final reflection on remaining episodes
            if len(self.episode_buffer) >= self.config.min_episodes_for_reflection:
                self.logger.info("Final reflection pass")
                reflections = self._trigger_reflection()
                if reflections:
                    self._update_ace_playbook(reflections)

            # Final health check
            final_health = self._health_check()
            self.logger.info(f"Live loop stopped - status: {final_health['status']}")

        return self.metrics

    def stop(self) -> None:
        """Request graceful shutdown of loop."""
        self.logger.info("Stop requested")
        self._should_stop = True
