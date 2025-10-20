"""Perform guardrail integrity sweep on ACE playbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ace.utils.database import get_session
from ace.repositories.playbook_repository import PlaybookRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guardrail integrity sweep")
    parser.add_argument(
        "--domain",
        default="default",
        help="ACE domain identifier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/replay/guardrail_sweep.json"),
        help="Path for sweep report",
    )
    args = parser.parse_args()

    violations = []
    with get_session() as session:
        repo = PlaybookRepository(session)
        bullets = repo.get_by_domain(args.domain)
        for bullet in bullets:
            harmful = bullet.harmful_count
            helpful = bullet.helpful_count
            if harmful > helpful:
                violations.append(
                    {
                        "id": bullet.id,
                        "stage": bullet.stage.value,
                        "helpful": helpful,
                        "harmful": harmful,
                        "section": bullet.section,
                    }
                )

    report = {
        "domain": args.domain,
        "violations": violations,
        "status": "fail" if violations else "pass",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if violations:
        raise SystemExit(
            f"Guardrail sweep failed: {len(violations)} bullets have harmful_count > helpful_count"
        )


if __name__ == "__main__":
    main()
