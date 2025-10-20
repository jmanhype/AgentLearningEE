"""Export live loop and promotion metrics in Prometheus format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def emit_metric(name: str, value: float, labels: Dict[str, str]) -> str:
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    if label_str:
        return f"{name}{{{label_str}}} {value}"
    return f"{name} {value}"


def summarize_ace_history(history: list[Dict[str, Any]]) -> Dict[str, int]:
    totals = {"added": 0, "incremented": 0, "duplicates": 0, "total_insights": 0}
    for record in history:
        for key in totals:
            totals[key] += int(record.get(key, 0) or 0)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Prometheus metrics for live loop runs")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("live_loop_artifacts/metrics.json"),
        help="Path to metrics JSON produced by the live loop",
    )
    parser.add_argument(
        "--promotion-report",
        type=Path,
        default=Path("artifacts/replay/promotion_report.json"),
        help="Path to promotion report JSON produced by replay training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("live_loop_artifacts/metrics.prom"),
        help="Output path for Prometheus metrics text",
    )
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    promotion = load_json(args.promotion_report)

    domain = metrics.get("domain") or promotion.get("domain_id") or "unknown"
    labels = {"domain": domain}

    lines: list[str] = []

    if metrics:
        lines.append(emit_metric("ace_total_reflections", metrics.get("total_reflections", 0), labels))
        lines.append(emit_metric("ace_total_guardrail_checks", metrics.get("total_guardrail_checks", 0), labels))
        lines.append(emit_metric("ace_total_auto_corrections", metrics.get("guardrail_auto_corrections", 0), labels))
        lines.append(emit_metric("ace_total_updates", metrics.get("total_ace_updates", 0), labels))

        history = metrics.get("ace_update_history", []) or []
        if history:
            totals = summarize_ace_history(history)
            for key, value in totals.items():
                metric_name = f"ace_insights_{key}"
                lines.append(emit_metric(metric_name, value, labels))

    if promotion:
        lines.append(emit_metric("ace_promoted_bullets", len(promotion.get("promoted", [])), labels))
        lines.append(emit_metric("ace_quarantined_bullets", len(promotion.get("quarantined", [])), labels))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
