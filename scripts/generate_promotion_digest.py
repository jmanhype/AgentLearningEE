"""Generate markdown digest for ACE promotions and quarantines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

from ace.utils.database import get_session
from ace.repositories.playbook_repository import PlaybookRepository


def load_promotion_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    lines = [f"# ACE Promotion Digest ({domain_id})\n"]

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

    if len(lines) == 1:
        lines.append("No promotions or quarantines recorded in this run.\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
