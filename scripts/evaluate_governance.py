from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(metrics: Dict) -> List[Dict]:
    alerts: List[Dict] = []

    # Guardrail failure ratio
    guardrail_failures = metrics.get("guardrail_failures", 0)
    guardrail_passes = metrics.get("guardrail_passes", 0)
    if guardrail_failures > guardrail_passes:
        alerts.append(
            {
                "severity": "high",
                "message": f"Guardrail failures ({guardrail_failures}) exceed passes ({guardrail_passes})",
            }
        )

    # Negative feedback volume
    negative_history = metrics.get("ace_negative_feedback_history") or []
    if negative_history:
        last = negative_history[-1]
        tasks = last.get("tasks", [])
        if len(tasks) >= 5:
            alerts.append(
                {
                    "severity": "medium",
                    "message": f"High guardrail penalties recorded ({len(tasks)} tasks)",
                }
            )

    # Prod demotions
    prod_history = metrics.get("ace_prod_promotion_history") or []
    if prod_history:
        last = prod_history[-1]
        demoted = last.get("demoted", [])
        if demoted:
            alerts.append(
                {
                    "severity": "high",
                    "message": f"Prod demotions detected for {len(demoted)} bullets",
                }
            )

    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate governance SLOs and emit alerts")
    parser.add_argument("--metrics", type=Path, default=Path("live_loop_artifacts/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/governance_alerts.json"))
    args = parser.parse_args()

    metrics = load_metrics(args.metrics)
    alerts = evaluate(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(alerts, handle, indent=2)

    if any(alert.get("severity") == "high" for alert in alerts):
        raise SystemExit("governance_alerts_detected")

    print(json.dumps({"status": "ok", "alerts": alerts}, indent=2))


if __name__ == "__main__":
    main()
