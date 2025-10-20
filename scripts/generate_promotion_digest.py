"""Generate markdown digest for ACE promotions and quarantines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from ace.utils.database import get_session
from ace.repositories.playbook_repository import PlaybookRepository

METRICS_DEFAULT = Path("live_loop_artifacts/metrics.json")

CHECKLIST_ITEMS = [
    "Review promoted bullets and confirm accuracy",
    "Investigate newly demoted or penalized items",
    "Acknowledge benchmark failures and assign follow-up",
    "Update prod rollback notes if demotions occurred",
]


def reviewer_checklist() -> List[str]:
    lines = ["## Reviewer Checklist", ""]
    for item in CHECKLIST_ITEMS:
        lines.append(f"- [ ] {item}")
    lines.append("")
    return lines


def load_promotion_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_feedback(metrics: Dict[str, Any]) -> List[str]:
    if not metrics:
        return []

    lines: List[str] = []

    benchmark_history = metrics.get("ace_benchmark_feedback_history") or []
    if benchmark_history:
        last = benchmark_history[-1]
        successes = last.get("tasks_succeeded", [])
        failures = last.get("tasks_failed", [])
        lines.append("## Benchmark Feedback")
        lines.append(f"- Successes: {len(successes)}")
        if successes:
            lines.append("  - Tasks: " + ", ".join(successes[:10]))
            if len(successes) > 10:
                lines.append(f"  - (+{len(successes) - 10} more)")
        lines.append(f"- Failures: {len(failures)}")
        if failures:
            lines.append("  - Tasks: " + ", ".join(failures[:10]))
            if len(failures) > 10:
                lines.append(f"  - (+{len(failures) - 10} more)")
        lines.append("")

    negative_history = metrics.get("ace_negative_feedback_history") or []
    if negative_history:
        last = negative_history[-1]
        tasks = last.get("tasks", [])
        lines.append("## Guardrail Penalties")
        lines.append(f"- Penalized tasks: {len(tasks)}")
        if tasks:
            lines.append("  - Tasks: " + ", ".join(tasks[:10]))
            if len(tasks) > 10:
                lines.append(f"  - (+{len(tasks) - 10} more)")
        lines.append("")

    prod_history = metrics.get("ace_prod_promotion_history") or []
    if prod_history:
        last = prod_history[-1]
        promoted = last.get("promoted", [])
        demoted = last.get("demoted", [])
        lines.append("## Prod Promotion Summary")
        lines.append(f"- Promoted to prod: {len(promoted)}")
        if promoted:
            lines.append("  - Bullets: " + ", ".join(promoted[:10]))
            if len(promoted) > 10:
                lines.append(f"  - (+{len(promoted) - 10} more)")
        lines.append(f"- Demoted from prod: {len(demoted)}")
        if demoted:
            lines.append("  - Bullets: " + ", ".join(demoted[:10]))
            if len(demoted) > 10:
                lines.append(f"  - (+{len(demoted) - 10} more)")
        lines.append("")

    return lines


def format_bullet(bullet) -> str:
    preview = bullet.content.strip().splitlines()[0]
    if len(preview) > 200:
        preview = preview[:197] + "..."
    return (
        f"- **ID:** `{bullet.id}`\n"
        f"  - Stage: `{bullet.stage.value}`\n"
        f"  - Helpful: {bullet.helpful_count}, Harmful: {bullet.harmful_count}\n"
        f"  - Section: {bullet.section}\n"
        f"  - Preview: {preview}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ACE promotion digest")
    parser.add_argument(
        "--promotion-report",
        type=Path,
        default=Path("artifacts/replay/promotion_report.json"),
        help="Path to promotion report JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/replay/promotion_review.md"),
        help="Path to write markdown digest",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_DEFAULT,
        help="Path to live loop metrics JSON",
    )
    args = parser.parse_args()

    report = load_promotion_report(args.promotion_report)
    if not report:
        domain_id = os.getenv("ACE_DOMAIN_ID", "unknown")
        promoted_ids: list[str] = []
        quarantined_ids: list[str] = []
    else:
        domain_id = report.get("domain_id", "default")
        promoted_ids = report.get("promoted", [])
        quarantined_ids = report.get("quarantined", [])

    metrics_data: Dict[str, Any] = {}
    if args.metrics.exists():
        with args.metrics.open("r", encoding="utf-8") as handle:
            try:
                metrics_data = json.load(handle)
            except json.JSONDecodeError:
                metrics_data = {}

    lines = [f"# ACE Promotion Digest ({domain_id})\n"]
    lines.extend(reviewer_checklist())

    if promoted_ids or quarantined_ids:
        with get_session() as session:
            repo = PlaybookRepository(session)

            if promoted_ids:
                lines.append("## Promoted Bullets\n")
                for bullet_id in promoted_ids:
                    bullet = repo.get_by_id(bullet_id, domain_id)
                    if bullet:
                        lines.append(format_bullet(bullet))
                    else:
                        lines.append(f"- `{bullet_id}` (details unavailable)")
                lines.append("")

            if quarantined_ids:
                lines.append("## Quarantined Bullets\n")
                for bullet_id in quarantined_ids:
                    bullet = repo.get_by_id(bullet_id, domain_id)
                    if bullet:
                        lines.append(format_bullet(bullet))
                    else:
                        lines.append(f"- `{bullet_id}` (details unavailable)")
                lines.append("")

    feedback = summarize_feedback(metrics_data)
    if feedback:
        lines.extend(feedback)

    if len(lines) == 1:
        lines.append("No promotions or quarantines recorded in this run.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
