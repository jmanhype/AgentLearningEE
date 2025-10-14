#!/usr/bin/env python3
"""
Live Exploration Loop Demo

Demonstrates continuous learning with:
1. Simple driving simulation environment
2. Policy execution with playbook context (if ACE enabled)
3. Online reflection generation
4. ACE playbook updates
5. Continuous improvement loop

This shows the full Early Experience + ACE integration:
- Agent generates experiences from environment
- Reflects on experiences to extract insights
- Insights flow into ACE playbook
- Playbook context improves future decisions
"""

import os
import random
import time
from pathlib import Path
from typing import Optional

# Configure ACE if available
os.environ.setdefault("ACE_ENABLED", "1")
os.environ.setdefault("ACE_DOMAIN_ID", "live-loop-demo")
os.environ.setdefault("ACE_TARGET_STAGE", "shadow")

import dspy
from agent_learning.live_loop import LiveExplorationLoop, LiveLoopConfig
from agent_learning.utils import setup_logger

# Configure DSPy LM for inference
# Requires OPENAI_API_KEY and OPENAI_API_BASE environment variables
if not os.getenv('OPENAI_API_KEY'):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    print()
    print("Please set your OpenRouter or OpenAI API key:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("  export OPENAI_API_BASE='https://openrouter.ai/api/v1'  # For OpenRouter")
    print()
    exit(1)

api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
lm = dspy.LM(
    'openai/gpt-4o-mini',
    api_base=api_base,
    api_key=os.environ['OPENAI_API_KEY']
)
dspy.configure(lm=lm)


class DrivingSimulator:
    """
    Simple driving simulation for demonstration.

    Generates realistic driving scenarios and evaluates actions.
    """

    # Scenario templates
    SCENARIOS = [
        {
            "state": "Vehicle approaching intersection with red light",
            "good_actions": ["stop", "brake"],
            "bad_actions": ["proceed", "accelerate"],
        },
        {
            "state": "Pedestrian crossing at crosswalk",
            "good_actions": ["yield", "stop"],
            "bad_actions": ["proceed", "honk"],
        },
        {
            "state": "Vehicle in left lane with obstacle ahead",
            "good_actions": ["change lane", "merge right"],
            "bad_actions": ["maintain speed", "accelerate"],
        },
        {
            "state": "Speed limit 35mph, current speed 45mph",
            "good_actions": ["decelerate", "reduce speed"],
            "bad_actions": ["maintain speed", "accelerate"],
        },
        {
            "state": "Vehicle behind attempting to pass on highway",
            "good_actions": ["maintain lane", "signal and move right"],
            "bad_actions": ["brake suddenly", "swerve left"],
        },
        {
            "state": "Approaching stop sign at empty intersection",
            "good_actions": ["stop", "full stop"],
            "bad_actions": ["roll through", "proceed without stopping"],
        },
        {
            "state": "Rain starting, road becoming slippery",
            "good_actions": ["reduce speed", "increase following distance"],
            "bad_actions": ["maintain speed", "brake hard"],
        },
        {
            "state": "Emergency vehicle approaching from behind with sirens",
            "good_actions": ["pull over", "move to right lane"],
            "bad_actions": ["maintain speed", "ignore"],
        },
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize simulator."""
        self.random = random.Random(seed)
        self.current_scenario = None
        self.episode_count = 0

    def reset(self) -> str:
        """Reset and return new scenario state."""
        self.current_scenario = self.random.choice(self.SCENARIOS)
        return self.current_scenario["state"]

    def step(self, action: str) -> tuple[str, bool]:
        """
        Execute action and return outcome.

        Returns:
            next_state: Description of outcome
            done: Always True (episodic environment)
        """
        if not self.current_scenario:
            raise RuntimeError("Must call reset() before step()")

        action_lower = action.lower().strip()

        # Evaluate action
        if any(
            good in action_lower for good in self.current_scenario["good_actions"]
        ):
            # Good action
            outcomes = [
                "Safe maneuver completed",
                "Situation handled appropriately",
                "No incidents occurred",
                "Compliant with traffic rules",
            ]
            outcome = self.random.choice(outcomes)
        elif any(
            bad in action_lower for bad in self.current_scenario["bad_actions"]
        ):
            # Bad action
            outcomes = [
                "Potential safety violation",
                "Risky maneuver attempted",
                "Traffic violation occurred",
                "Near-miss incident",
            ]
            outcome = self.random.choice(outcomes)
        else:
            # Neutral/unclear action
            outcome = "Action taken; outcome unclear"

        next_state = f"{outcome} after action: {action}"

        self.episode_count += 1
        return next_state, True  # Episodes are single-step


def run_live_loop_demo():
    """Run live exploration loop demonstration."""
    print("=" * 70)
    print("Live Exploration Loop Demo - Continuous Learning")
    print("=" * 70)
    print()

    # Check if we have a trained policy
    policy_path = Path("artifacts/policy.pkl")
    if not policy_path.exists():
        print("⚠️  No trained policy found at artifacts/policy.pkl")
        print()
        print("Please train a policy first:")
        print("  1. Generate expert demos: python -m agent_learning.utils")
        print("  2. Run pipeline: python -m agent_learning.pipeline")
        print()
        return

    # Check ACE status
    ace_enabled = os.getenv("ACE_ENABLED", "0") == "1"
    print(f"ACE Integration: {'✓ Enabled' if ace_enabled else '✗ Disabled'}")
    if ace_enabled:
        try:
            from ee_ace_bridge import ACE_INTEGRATION_AVAILABLE

            if ACE_INTEGRATION_AVAILABLE:
                print("  ACE CuratorService: ✓ Available")
            else:
                print("  ACE CuratorService: ✗ Not Available (using in-memory stub)")
        except ImportError:
            print("  ACE Bridge: ✗ Not Available")
    print()

    # Configure live loop
    config = LiveLoopConfig(
        episode_batch_size=10,
        max_episodes=50,  # Demo runs 50 episodes
        reflection_interval=10,  # Reflect every 10 episodes
        min_episodes_for_reflection=5,
        ace_enabled=ace_enabled,
        ace_update_interval=10,  # Update ACE every 10 reflections
        output_dir=Path("live_loop_artifacts/"),
        save_episodes=True,
        save_reflections=True,
    )

    print("Configuration:")
    print(f"  Max Episodes: {config.max_episodes}")
    print(f"  Reflection Interval: {config.reflection_interval} episodes")
    print(f"  ACE Update Interval: {config.ace_update_interval} reflections")
    print(f"  Output Dir: {config.output_dir}")
    print()

    # Create environment
    environment = DrivingSimulator(seed=42)

    # Create logger
    logger = setup_logger("live_loop_demo")

    print("=" * 70)
    print("Starting Live Loop...")
    print("=" * 70)
    print()

    # Run loop
    try:
        loop = LiveExplorationLoop(
            environment=environment,
            policy_path=policy_path,
            config=config,
            logger=logger,
        )

        start_time = time.time()
        metrics = loop.run()
        runtime = time.time() - start_time

        print()
        print("=" * 70)
        print("✅ Live Loop Completed!")
        print("=" * 70)
        print()

        print("Metrics:")
        print(f"  Total Episodes: {metrics.total_episodes}")
        print(f"  Total Reflections: {metrics.total_reflections}")
        print(f"  Total ACE Updates: {metrics.total_ace_updates}")
        print(f"  Runtime: {runtime:.1f}s")
        print(f"  Throughput: {metrics.episodes_per_minute():.1f} episodes/min")
        print()

        if ace_enabled and metrics.total_ace_updates > 0:
            print("ACE Playbook Status:")
            try:
                from ee_ace_bridge.ace_client import InProcessAceClient

                client = InProcessAceClient(domain_id="live-loop-demo")
                health = client.get_health()

                print(f"  Status: {health['status']}")
                print(f"  Insights Ingested: {health['insights_ingested']}")
                print(f"  Stage Counts: {health['stage_counts']}")
                print(f"  Total Insights: {client.get_insight_count()}")
                print()

                # Show sample of playbook
                playbook = client.render_playbook(token_budget=1000)
                if playbook:
                    print("Sample of Accumulated Knowledge:")
                    print("-" * 70)
                    print(playbook[:400])
                    if len(playbook) > 400:
                        print("  ... (truncated)")
                    print()

            except Exception as e:
                print(f"  Error getting ACE status: {e}")

        print("Artifacts:")
        print(f"  Episodes: {config.output_dir / 'episodes.jsonl'}")
        if metrics.total_reflections > 0:
            print(f"  Reflections: {config.output_dir / 'reflections_*.jsonl'}")
        print()

        print("Next Steps:")
        if ace_enabled:
            print("  1. Inspect playbook to see accumulated insights")
            print("  2. Promote insights from shadow → staging → prod")
            print("  3. Run another loop to see improved decisions with playbook context")
        else:
            print("  1. Enable ACE: export ACE_ENABLED=1")
            print("  2. Re-run to see continuous playbook improvement")
        print()

    except KeyboardInterrupt:
        print()
        print("Loop interrupted by user (Ctrl+C)")
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    run_live_loop_demo()
