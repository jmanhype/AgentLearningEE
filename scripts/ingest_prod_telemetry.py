from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from ace.models.playbook import PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_metrics(path: Path, domain: str) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {"domain": domain}
    return {"domain": domain}


def _persist_metrics(path: Path, metrics: Dict, entry: Dict) -> None:
    history = metrics.setdefault("ace_prod_telemetry_history", [])
    history.append(entry)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def _apply_feedback(
    *,
    domain_id: str,
    task_ids: Sequence[str],
    helpful_increment: int = 0,
    harmful_increment: int = 0,
) -> List[str]:
    if not task_ids:
        return []

    tags_to_match = {f"task:{task_id}" for task_id in task_ids}
    updated_ids: List[str] = []

    with get_session() as session:
        repo = PlaybookRepository(session)
        bullets = repo.get_by_domain(domain_id)

        for bullet in bullets:
            if bullet.stage == PlaybookStage.QUARANTINED:
                continue

            tags = bullet.tags or []
            if not any(tag in tags_to_match for tag in tags):
                continue

            if helpful_increment:
                bullet.helpful_count += helpful_increment
            if harmful_increment:
                bullet.harmful_count += harmful_increment

            repo.update(bullet)
            updated_ids.append(bullet.id)

        if updated_ids:
            session.commit()

    return updated_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply production telemetry feedback to ACE")
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/prod_telemetry.jsonl"))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--metrics-path", type=Path, default=Path("live_loop_artifacts/metrics.json"))
    parser.add_argument("--success-increment", type=int, default=1)
    parser.add_argument("--failure-increment", type=int, default=1)
    args = parser.parse_args()

    records = _load_jsonl(args.telemetry)
    if not records:
        print(json.dumps({"status": "skipped", "reason": "no telemetry"}))
        return

    successes: List[str] = []
    failures: List[str] = []
    for record in records:
        task_id = record.get("task_id")
        if not task_id:
            continue
        status = (record.get("status") or "").lower()
        if status in {"success", "succeeded", "pass", "passed"}:
            successes.append(task_id)
        elif status in {"failure", "failed", "error"}:
            failures.append(task_id)

    success_ids = _apply_feedback(
        domain_id=args.domain,
        task_ids=successes,
        helpful_increment=args.success_increment,
        harmful_increment=0,
    )
    failure_ids = _apply_feedback(
        domain_id=args.domain,
        task_ids=failures,
        helpful_increment=0,
        harmful_increment=args.failure_increment,
    )

    entry = {
        "timestamp": time.time(),
        "tasks_succeeded": successes,
        "tasks_failed": failures,
        "updated_success_bullets": success_ids,
        "updated_failure_bullets": failure_ids,
        "success_increment": args.success_increment,
        "failure_increment": args.failure_increment,
    }

    metrics = _load_metrics(args.metrics_path, args.domain)
    _persist_metrics(args.metrics_path, metrics, entry)

    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
