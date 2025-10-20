"""Validate live loop metrics to ensure ACE ingestion occurred when expected."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate live loop metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("live_loop_artifacts/metrics.json"),
        help="Path to metrics JSON file produced by the live loop",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)

    ace_enabled = os.getenv("ACE_ENABLED", "0") == "1"
    total_reflections = metrics.get("total_reflections", 0)
    total_ace_updates = metrics.get("total_ace_updates", 0)
    domain = metrics.get("domain")

    history = metrics.get("ace_update_history", []) or []
    added = sum(int(entry.get("added", 0) or 0) for entry in history)
    incremented = sum(int(entry.get("incremented", 0) or 0) for entry in history)
    duplicates = sum(int(entry.get("duplicates", 0) or 0) for entry in history)

    if ace_enabled and total_reflections > 0 and (added + incremented) == 0:
        raise SystemExit(
            "ACE validation failed: reflections were generated but no ACE playbook updates occurred."
        )

    if ace_enabled and (added + incremented) > 0 and duplicates > (added + incremented) * 2:
        raise SystemExit(
            f"ACE validation failed: duplicates ({duplicates}) exceed allowed threshold relative to added/incremented ({added + incremented})."
        )

    print(
        json.dumps(
            {
                "ace_enabled": ace_enabled,
                "total_reflections": total_reflections,
                "total_ace_updates": total_ace_updates,
                "ace_updates_added": added,
                "ace_updates_incremented": incremented,
                "ace_updates_duplicates": duplicates,
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
