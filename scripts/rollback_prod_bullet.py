from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ace.models.playbook import PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session


def _load_ids(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback prod playbook bullets to staging or shadow")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--bullet-id", action="append", dest="bullet_ids", default=[])
    parser.add_argument("--bullet-file", type=Path, help="Path to file containing bullet IDs (one per line)")
    parser.add_argument(
        "--target-stage",
        choices=["staging", "shadow", "quarantine"],
        default="staging",
        help="Stage to move bullet to",
    )
    args = parser.parse_args()

    bullet_ids = list(args.bullet_ids)
    if args.bullet_file and args.bullet_file.exists():
        bullet_ids.extend(list(_load_ids(args.bullet_file)))

    if not bullet_ids:
        raise SystemExit("No bullet IDs provided")

    stage_map = {
        "staging": PlaybookStage.STAGING,
        "shadow": PlaybookStage.SHADOW,
        "quarantine": PlaybookStage.QUARANTINED,
    }
    target_stage = stage_map[args.target_stage]

    with get_session() as session:
        repo = PlaybookRepository(session)
        updated = []
        for bullet_id in bullet_ids:
            bullet = repo.get_by_id(bullet_id, args.domain)
            if not bullet:
                continue
            bullet.stage = target_stage
            if target_stage == PlaybookStage.SHADOW:
                bullet.helpful_count = 0
            repo.update(bullet)
            updated.append(bullet_id)
        if updated:
            session.commit()

    print(f"Rolled back {len(updated)} bullets to {args.target_stage}")


if __name__ == "__main__":
    main()
