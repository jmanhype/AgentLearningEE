"""Append live loop metrics to a rolling history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_history(history_path: Path, record: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update metrics history from live loop run")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("live_loop_artifacts/metrics.json"),
        help="Metrics JSON file produced by the live loop",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/metrics_history.jsonl"),
        help="History file to append (JSONL)",
    )
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    domain = metrics.get("domain", "unknown")

    history_record = {
        "domain": domain,
        "timestamp": metrics.get("timestamp"),
        "total_reflections": metrics.get("total_reflections"),
        "total_ace_updates": metrics.get("total_ace_updates"),
        "guardrail_passes": metrics.get("guardrail_passes"),
        "guardrail_failures": metrics.get("guardrail_failures"),
        "ace_update_history": metrics.get("ace_update_history", []),
    }

    append_history(args.history, history_record)


if __name__ == "__main__":
    main()
