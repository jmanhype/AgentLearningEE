"""Utility to run guardrail-driven live loops for SWE-bench and MagicBrush.

This script wraps the small JSONL samples under ``data/swe_bench_samples`` and
``data/magicbrush_samples`` with simple environments so we can exercise the
``LiveExplorationLoop`` without requiring an LLM policy. Actions are replayed
from the dataset and validated by the deterministic guardrails we added.

Usage (SWE-bench demo)::

    python examples/live_loop_swe_magic.py --domain swe-bench --episodes 5

Usage (MagicBrush demo)::

    python examples/live_loop_swe_magic.py --domain magicbrush --episodes 5

Set ``--ace`` if you have ACE configured and want to log corrections into the
playbook. The script defaults to guardrail-driven actions so it runs fully
offline (no API keys required).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List

import guardrails.magicbrush  # Registers guardrails as side effects.
import guardrails.swe_bench
import dspy
import os
from agent_learning.live_loop import LiveExplorationLoop, LiveLoopConfig


def _configure_lm_from_env() -> None:
    if dspy.settings.lm is not None:
        return

    key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )

    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            key = (
                os.getenv("OPENROUTER_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            )

    if not key:
        return

    if os.getenv("OPENROUTER_API_KEY"):
        dspy.configure(
            lm=dspy.LM(
                model=os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen-2.5-7b-instruct"),
                api_key=os.environ["OPENROUTER_API_KEY"],
                api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            )
        )
    elif os.getenv("OPENAI_API_KEY"):
        dspy.configure(
            lm=dspy.LM(
                model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
                api_key=os.environ["OPENAI_API_KEY"],
                api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            )
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        dspy.configure(
            lm=dspy.LM(
                model=os.getenv("ANTHROPIC_MODEL", "anthropic/claude-3-haiku-20240307"),
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        )


def _load_records(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[record["task_id"]] = record
    return records


@dataclass
class SweBenchEnvironment:
    dataset_path: Path

    def __post_init__(self) -> None:
        self.records = _load_records(self.dataset_path)
        self.keys: List[str] = list(self.records.keys())
        self.index = 0
        self.current = None

    def reset(self):
        record = self.records[self.keys[self.index % len(self.keys)]]
        self.current = record
        state_payload = {
            "task_id": record["task_id"],
            "issue": record["state"]["issue"],
            "tests": record["state"].get("failing_tests", []),
        }
        metadata = {
            "task_id": record["task_id"],
            "domain": "swe-bench",
            "ground_truth": record.get("ground_truth", "pass"),
        }
        return json.dumps(state_payload, indent=2), metadata

    def step(self, action: str):
        from guardrails import get_guardrail

        record = self.current
        guardrail = get_guardrail(record["task_id"], domain="swe-bench")
        if hasattr(guardrail, "reset"):
            guardrail.reset()
        guardrail.validate(action, record.get("ground_truth", "pass"))
        info = {
            "task_id": record["task_id"],
            "canonical": guardrail.canonical_answer(),
        }
        self.index += 1
        return json.dumps(info, indent=2), True


@dataclass
class MagicBrushEnvironment:
    dataset_path: Path

    def __post_init__(self) -> None:
        self.records = _load_records(self.dataset_path)
        self.keys: List[str] = list(self.records.keys())
        self.index = 0
        self.current = None

    def reset(self):
        record = self.records[self.keys[self.index % len(self.keys)]]
        self.current = record
        state_payload = {
            "task_id": record["task_id"],
            "instruction": record["state"]["instruction"],
            "size": record["state"].get("metadata", {}),
        }
        metadata = {
            "task_id": record["task_id"],
            "domain": "magicbrush",
            "ground_truth": record.get("ground_truth", "pass"),
        }
        return json.dumps(state_payload, indent=2), metadata

    def step(self, action: str):
        from guardrails import get_guardrail

        record = self.current
        guardrail = get_guardrail(record["task_id"], domain="magicbrush")
        if hasattr(guardrail, "reset"):
            guardrail.reset()
        guardrail.validate(action, record.get("ground_truth", "pass"))
        info = {
            "task_id": record["task_id"],
            "canonical": guardrail.canonical_answer(),
            "metrics": record["next_state"].get("metrics", {}),
        }
        self.index += 1
        return json.dumps(info, indent=2), True


class GuardrailPolicy:
    """Stub policy that replays recorded actions for the current task."""

    def __init__(self, get_task_id: Callable[[], str], records: Dict[str, dict]) -> None:
        self.get_task_id = get_task_id
        self.records = records

    def forward(self, state: str):  # dspy.Module interface
        task_id = self.get_task_id()
        record = self.records[task_id]
        action = record["action"].get("patch") or record["action"].get("edit_prompt") or ""
        return SimpleNamespace(
            reasoning=f"Replaying recorded action for {task_id}",
            action=action,
        )

    __call__ = forward


def run_guardrail_loop(domain: str, dataset_path: Path, episodes: int, ace: bool) -> None:
    _configure_lm_from_env()
    if domain == "swe-bench":
        env = SweBenchEnvironment(dataset_path)
    elif domain == "magicbrush":
        env = MagicBrushEnvironment(dataset_path)
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    config = LiveLoopConfig(
        max_episodes=episodes,
        reflection_interval=5,
        ace_enabled=ace,
        default_guardrail_domain=domain,
        apply_guardrails=True,
        log_level=20,
    )

    loop = LiveExplorationLoop(env, policy_path="artifacts/policy.pkl", config=config)

    # Replace policy with guardrail-driven stub.
    loop._policy_module = GuardrailPolicy(lambda: env.current["task_id"], env.records)

    metrics = loop.run()
    print(f"Loop finished for domain={domain}: episodes={metrics.total_episodes}, guardrail_passes={metrics.guardrail_passes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guardrail-driven live loop demos")
    parser.add_argument("--domain", choices=["swe-bench", "magicbrush"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ace", action="store_true", help="Enable ACE integration if configured")
    args = parser.parse_args()

    dataset = {
        "swe-bench": Path("data/swe_bench_samples/swe_bench_50.jsonl"),
        "magicbrush": Path("data/magicbrush_samples/magicbrush_50.jsonl"),
    }[args.domain]

    run_guardrail_loop(args.domain, dataset, args.episodes, ace=args.ace)


if __name__ == "__main__":
    main()
