"""Run lightweight experience replay training from live loop reflections."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import dspy

from agent_learning.utils import load_jsonl, save_jsonl, setup_logger
from agent_learning.policy import train_policy


def configure_lm_from_env() -> bool:
    if dspy.settings.lm is not None:
        return True

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
        return False

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

    return dspy.settings.lm is not None


def collect_reflections(reflections_dir: Path) -> List[Path]:
    if not reflections_dir.exists():
        return []
    return sorted(reflections_dir.glob("reflections_*.jsonl"))


def aggregate_reflections(reflection_files: List[Path]) -> List[Dict]:
    reflections: List[Dict] = []
    for file_path in reflection_files:
        reflections.extend(load_jsonl(str(file_path)))
    return reflections


def write_aggregated_reflections(output_path: Path, reflections: List[Dict]) -> None:
    if not reflections:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(reflections, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy from live loop reflections")
    parser.add_argument(
        "--reflections-dir",
        type=Path,
        default=Path("live_loop_artifacts"),
        help="Directory containing reflections_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/replay"),
        help="Directory to store replay artifacts",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=Path("artifacts/policy_replay.pkl"),
        help="Path to save the retrained policy",
    )
    parser.add_argument(
        "--min-reflections",
        type=int,
        default=10,
        help="Minimum reflections required to trigger replay training",
    )
    args = parser.parse_args()

    logger = setup_logger("experience_replay")

    reflection_files = collect_reflections(args.reflections_dir)
    if not reflection_files:
        logger.info("No reflection files found; skipping experience replay.")
        return

    reflections = aggregate_reflections(reflection_files)
    if len(reflections) < args.min_reflections:
        logger.info(
            "Insufficient reflections for replay",
            extra={"available": len(reflections), "required": args.min_reflections},
        )
        return

    aggregated_path = args.output_dir / "reflection_data.jsonl"
    write_aggregated_reflections(aggregated_path, reflections)

    if not configure_lm_from_env():
        logger.warning("No language model configured; skipping replay training.")
        return

    logger.info(
        "Running policy training on replay reflections",
        extra={"reflections": len(reflections), "output": str(args.policy_output)},
    )

    _, policy_metrics = train_policy(
        reflection_data_path=str(aggregated_path),
        output_path=str(args.policy_output),
        metric_threshold=None,
        logger=logger,
    )

    metrics_path = args.output_dir / "policy_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(policy_metrics, handle, indent=2, sort_keys=True)

    latest_policy_path = Path("artifacts/policy.pkl")
    if latest_policy_path.exists():
        backup_path = latest_policy_path.with_name("policy_prev.pkl")
        shutil.copy2(latest_policy_path, backup_path)
        logger.info(
            "Backed up previous policy",
            extra={"backup_path": str(backup_path)},
        )

    shutil.copy2(args.policy_output, latest_policy_path)
    logger.info(
        "Promoted replay-trained policy to artifacts/policy.pkl",
        extra={"policy_path": str(args.policy_output)},
    )


if __name__ == "__main__":
    main()
