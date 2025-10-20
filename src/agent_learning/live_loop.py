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
from typing import Any, Dict, List, Optional, Protocol, Tuple
from collections import deque
import json
import logging

import dspy

from agent_learning.utils import save_jsonl, setup_logger
from agent_learning.policy import PolicyModule
from guardrails import get_guardrail
from guardrails.base import NumericGuardrail


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
    task_id: Optional[str] = None
    domain: Optional[str] = None
    ground_truth: Optional[str] = None
    guardrail_passed: Optional[bool] = None
    guardrail_corrected_action: Optional[str] = None


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
    default_guardrail_domain: Optional[str] = None
    apply_guardrails: bool = True

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
    total_guardrail_checks: int = 0
    guardrail_passes: int = 0
    guardrail_failures: int = 0
    guardrail_auto_corrections: int = 0
    loop_start_time: float = field(default_factory=time.time)
    last_reflection_time: Optional[float] = None
    last_ace_update_time: Optional[float] = None
    reflections_since_last_update: int = 0

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
        self._pending_reflections: List[Dict[str, Any]] = []

        # Control flags
        self._running = False
        self._should_stop = False

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Artifact paths scoped per run
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._episode_log_path = (
            self.config.output_dir / f"episodes_{timestamp}.jsonl"
        )
        self._run_id = timestamp
        self.episode_log_path = self._episode_log_path
        self.run_id = self._run_id

        self.logger.info(f"Live loop initialized with policy: {self.policy_path}")

    def _load_policy(self) -> PolicyModule:
        """Load policy module (cached)."""
        if self._policy_module is None:
            from agent_learning.policy import load_trained_policy

            self._policy_module = load_trained_policy(str(self.policy_path))
            self.logger.info(f"Policy loaded from {self.policy_path}")
        return self._policy_module

    def _parse_environment_state(self, raw_state: Any) -> Tuple[str, Dict[str, Any]]:
        """Normalize environment reset() output into state text and metadata."""

        metadata: Dict[str, Any] = {}

        if isinstance(raw_state, tuple) and len(raw_state) == 2 and isinstance(raw_state[1], dict):
            state_text = str(raw_state[0])
            metadata = dict(raw_state[1])
        elif isinstance(raw_state, dict):
            state_text = str(raw_state.get("state") or raw_state.get("description") or "")
            metadata = {k: v for k, v in raw_state.items() if k not in {"state", "description"}}
        else:
            state_text = str(raw_state)

        return state_text, metadata

    def _augment_with_guardrail(self, state: str, guardrail: NumericGuardrail) -> str:
        """Append guardrail instructions to state prompt."""

        return (
            f"{state}\n\nGuardrail: {guardrail.instructions} "
            "Return only the final value exactly as specified."
        )

    def _apply_guardrails(
        self,
        action: str,
        guardrail: Optional[NumericGuardrail],
        ground_truth: Optional[str],
        task_id: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply guardrail auto-correction and evaluation."""

        if not guardrail or not self.config.apply_guardrails:
            return action, {"evaluated": False}

        evaluation_answer = action.strip()
        auto_corrected: Optional[str] = None

        if guardrail.auto_correct:
            canonical = guardrail.canonical_answer()
            if canonical and canonical != evaluation_answer:
                auto_corrected = canonical
                evaluation_answer = canonical
                self.metrics.guardrail_auto_corrections += 1
                self.logger.info(
                    "guardrail_auto_corrected",
                    extra={
                        "task_id": task_id,
                        "before": action,
                        "after": canonical,
                    },
                )

        passed: Optional[bool] = None
        self.metrics.total_guardrail_checks += 1

        if ground_truth:
            passed = guardrail.validate(evaluation_answer, ground_truth)
            if passed:
                self.metrics.guardrail_passes += 1
            else:
                self.metrics.guardrail_failures += 1
                self.logger.warning(
                    "guardrail_violation",
                    extra={
                        "task_id": task_id,
                        "answer": evaluation_answer,
                        "ground_truth": ground_truth,
                    },
                )

        return evaluation_answer, {
            "evaluated": True,
            "passed": passed,
            "auto_corrected": auto_corrected,
            "evaluation_answer": evaluation_answer,
        }

    def _persist_episode(self, episode: Episode) -> None:
        """Append a single episode record to the run artifact."""

        if not self.config.save_episodes:
            return

        record = {
            "state": episode.state,
            "action": episode.action,
            "next_state": episode.next_state,
            "timestamp": episode.timestamp,
            "task_id": episode.task_id,
            "domain": episode.domain,
            "ground_truth": episode.ground_truth,
            "guardrail_passed": episode.guardrail_passed,
            "guardrail_corrected_action": episode.guardrail_corrected_action,
        }

        with open(self._episode_log_path, "a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")

    def _dequeue_episode_batch(self, force: bool = False) -> List[Episode]:
        """Retrieve the next batch of episodes for reflection."""

        available = len(self.episode_buffer)
        if available == 0:
            return []

        min_required = max(1, self.config.min_episodes_for_reflection)
        if not force and available < min_required:
            return []

        batch_size = self.config.episode_batch_size
        if batch_size <= 0 or (force and available < batch_size):
            batch_size = available

        batch_size = min(batch_size, available)

        return [self.episode_buffer.popleft() for _ in range(batch_size)]

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

        # Generate decision (ACE integration handled internally now)
        from agent_learning.policy import generate_decision

        result = generate_decision(policy, state, self.logger)

        # Handle failure case
        if result is None:
            raise RuntimeError(f"Policy failed to generate decision for state: {state[:100]}...")

        reasoning, action = result
        return action, reasoning

    def _collect_episode(self) -> List[Episode]:
        """
        Collect one complete environment rollout as one or more episodes.

        Returns:
            List of Episode objects (empty if collection failed)
        """
        episodes: List[Episode] = []

        try:
            # Reset environment
            raw_state = self.environment.reset()
            state, metadata = self._parse_environment_state(raw_state)

            task_id = metadata.get("task_id")
            domain = metadata.get("domain") or self.config.default_guardrail_domain
            ground_truth = metadata.get("ground_truth")

            guardrail: Optional[NumericGuardrail] = None
            if task_id:
                guardrail = get_guardrail(task_id, domain=domain)
                if guardrail and hasattr(guardrail, "reset"):
                    guardrail.reset()

            episode_start = time.time()
            current_state = state

            while True:
                prompt_state = current_state
                if guardrail and self.config.apply_guardrails:
                    prompt_state = self._augment_with_guardrail(current_state, guardrail)

                # Generate action
                action, reasoning = self._generate_action(prompt_state)

                corrected_action = action
                guardrail_result = {"evaluated": False}
                if guardrail and self.config.apply_guardrails:
                    corrected_action, guardrail_result = self._apply_guardrails(
                        action,
                        guardrail,
                        ground_truth,
                        task_id,
                    )

                step_start = time.time()
                next_state, done = self.environment.step(corrected_action)

                if time.time() - episode_start > self.config.episode_timeout:
                    self.logger.warning(
                        "Episode timeout",
                        extra={
                            "task_id": task_id,
                            "elapsed": time.time() - episode_start,
                            "timeout": self.config.episode_timeout,
                        },
                    )
                    break

                episode = Episode(
                    state=current_state,
                    action=corrected_action,
                    next_state=str(next_state),
                    timestamp=step_start,
                    task_id=task_id,
                    domain=domain,
                    ground_truth=ground_truth,
                    guardrail_passed=guardrail_result.get("passed"),
                    guardrail_corrected_action=guardrail_result.get("auto_corrected"),
                )
                episodes.append(episode)

                self.logger.debug(
                    "Episode step collected",
                    extra={
                        "state_preview": current_state[:50],
                        "action": corrected_action,
                        "task_id": task_id,
                        "guardrail_passed": guardrail_result.get("passed"),
                        "done": done,
                    },
                )

                if done:
                    break

                current_state = str(next_state)

            return episodes

        except Exception as e:
            self.logger.error(f"Episode collection failed: {e}", exc_info=True)
            return []

    def _trigger_reflection(
        self, episodes: List[Episode], force: bool = False
    ) -> List[Dict]:
        """
        Generate reflections on buffered episodes.

        Returns:
            List of reflection dicts
        """
        if dspy.settings.lm is None:
            raise RuntimeError(
                "Reflections requested but no LM is configured. Configure DSPy with dspy.configure(lm=...)"
            )

        if (not force) and len(episodes) < self.config.min_episodes_for_reflection:
            self.logger.debug(
                f"Insufficient episodes for reflection: "
                f"{len(episodes)}/{self.config.min_episodes_for_reflection}"
            )
            return []

        self.logger.info(f"Triggering reflection on {len(episodes)} episodes")

        # Convert episodes to rollout format
        rollouts = [
            {
                "state": ep.state,
                "expert_action": ep.action,
                "expert_next_state": ep.next_state,
                "alternative_action": "",
                "alternative_next_state": "",
            }
            for ep in episodes
        ]

        # Generate reflections using DSPy module
        reflections = []

        # Import ReflectionSig for generating reflections
        from agent_learning.reflection import ReflectionSig

        reflection_module = dspy.ChainOfThought(ReflectionSig)

        for rollout in rollouts:
            try:
                # Generate reflection for this rollout
                result = reflection_module(
                    state=rollout["state"],
                    expert_action=rollout["expert_action"],
                    expert_next_state=rollout["expert_next_state"],
                    alternative_action=rollout["alternative_action"],
                    alternative_next_state=rollout["alternative_next_state"],
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

    def _record_reflections(
        self, reflections: List[Dict], force_flush: bool = False
    ) -> None:
        """Track reflections and trigger ACE ingestion when thresholds are met."""

        if not reflections:
            if force_flush:
                self._flush_pending_reflections(force=True)
            return

        self._pending_reflections.extend(reflections)
        self.metrics.reflections_since_last_update += len(reflections)

        if not self.config.ace_enabled:
            # Nothing to ingest; drop accumulated reflections to avoid growth
            self._pending_reflections.clear()
            self.metrics.reflections_since_last_update = 0
            return

        if force_flush:
            self._flush_pending_reflections(force=True)
            return

        interval = self.config.ace_update_interval
        if interval <= 0 or self.metrics.reflections_since_last_update >= interval:
            self._flush_pending_reflections()

    def _flush_pending_reflections(self, force: bool = False) -> None:
        """Ingest accumulated reflections into ACE if configured."""

        if not self._pending_reflections:
            if force:
                self.metrics.reflections_since_last_update = 0
            return

        payload = list(self._pending_reflections)
        self._pending_reflections.clear()
        self.metrics.reflections_since_last_update = 0

        if not self.config.ace_enabled and not force:
            return

        self._update_ace_playbook(payload)

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
                "guardrail_checks": self.metrics.total_guardrail_checks,
                "guardrail_passes": self.metrics.guardrail_passes,
                "guardrail_failures": self.metrics.guardrail_failures,
                "guardrail_auto_corrections": self.metrics.guardrail_auto_corrections,
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

                # Collect episode(s)
                new_episodes = self._collect_episode()
                if new_episodes:
                    if self.config.max_episodes is not None:
                        remaining = self.config.max_episodes - self.metrics.total_episodes
                        if remaining <= 0:
                            break
                        if len(new_episodes) > remaining:
                            self.logger.info(
                                "Truncating episode batch to respect max_episodes",
                                extra={
                                    "requested": len(new_episodes),
                                    "allowed": remaining,
                                },
                            )
                            new_episodes = new_episodes[:remaining]

                    for episode in new_episodes:
                        self.episode_buffer.append(episode)
                        self.metrics.total_episodes += 1
                        self._persist_episode(episode)

                    while True:
                        should_reflect_by_batch = (
                            self.config.episode_batch_size > 0
                            and len(self.episode_buffer)
                            >= self.config.episode_batch_size
                        )
                        should_reflect_by_interval = (
                            self.config.reflection_interval > 0
                            and self.metrics.total_episodes
                            % self.config.reflection_interval
                            == 0
                            and len(self.episode_buffer)
                            >= self.config.min_episodes_for_reflection
                        )

                        if not (should_reflect_by_batch or should_reflect_by_interval):
                            break

                        episodes_for_reflection = self._dequeue_episode_batch()
                        if not episodes_for_reflection:
                            break

                        reflections = self._trigger_reflection(episodes_for_reflection)
                        self._record_reflections(reflections)

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
            if self.episode_buffer:
                self.logger.info("Final reflection pass")
            while self.episode_buffer:
                episodes_for_reflection = self._dequeue_episode_batch(force=True)
                if not episodes_for_reflection:
                    break
                reflections = self._trigger_reflection(
                    episodes_for_reflection, force=True
                )
                self._record_reflections(reflections, force_flush=True)

            # Flush any reflections collected but not yet ingested
            self._flush_pending_reflections(force=True)

            # Final health check
            final_health = self._health_check()
            self.logger.info(f"Live loop stopped - status: {final_health['status']}")

        return self.metrics

    def stop(self) -> None:
        """Request graceful shutdown of loop."""
        self.logger.info("Stop requested")
        self._should_stop = True
