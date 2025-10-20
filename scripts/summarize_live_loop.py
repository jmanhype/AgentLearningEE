"""Summarize live loop artifacts for monitoring dashboards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Any


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize_episodes(artifacts_dir: Path) -> Dict[str, dict]:
    summary: Dict[str, dict] = defaultdict(
        lambda: {
            "episodes": 0,
            "guardrail_passes": 0,
            "guardrail_failures": 0,
            "guardrail_unknown": 0,
            "auto_corrections": 0,
        }
    )

    for path in artifacts_dir.glob("episodes*.jsonl"):
        for record in read_jsonl(path):
            domain = record.get("domain", "unknown")
            domain_summary = summary[domain]
            domain_summary["episodes"] += 1
            guardrail_passed = record.get("guardrail_passed")
            if guardrail_passed is True:
                domain_summary["guardrail_passes"] += 1
            elif guardrail_passed is False:
                domain_summary["guardrail_failures"] += 1
            else:
                domain_summary["guardrail_unknown"] += 1
            if record.get("guardrail_corrected_action"):
                domain_summary["auto_corrections"] += 1

    return summary


def load_metrics(artifacts_dir: Path) -> Dict[str, Any]:
    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize live loop artifacts")
    parser.add_argument(
        "artifacts",
        type=Path,
        default=Path("live_loop_artifacts"),
        help="Directory containing live loop JSONL artifacts",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")

    episode_summary = summarize_episodes(artifacts_dir)
    metrics = load_metrics(artifacts_dir)

    if metrics:
        total_reflections = metrics.get("total_reflections", 0)
    else:
        # Fallback: count reflection files if metrics absent
        total_reflections = 0
        for path in artifacts_dir.glob("reflections_*.jsonl"):
            total_reflections += sum(1 for _ in read_jsonl(path))

    output = {
        "episodes": episode_summary,
        "total_reflections": total_reflections,
    }

    if metrics:
        output["metrics"] = metrics

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
