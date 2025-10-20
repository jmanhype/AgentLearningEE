"""Apply benchmark outcome feedback to ACE playbooks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ace.models.playbook import PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session


def _load_results(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark results not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_outcomes(evaluations: Sequence[Dict]) -> Tuple[List[str], List[str]]:
    successes: List[str] = []
    failures: List[str] = []

    for record in evaluations:
        task_id = record.get("task_id")
        if not task_id:
            continue
        if record.get("correct"):
            successes.append(task_id)
        else:
            failures.append(task_id)

    return successes, failures


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


def _update_metrics(
    metrics_path: Path,
    *,
    domain_id: str,
    successes: Sequence[str],
    failures: Sequence[str],
    helpful_increment: int,
    harmful_increment: int,
    updated_success_ids: Sequence[str],
    updated_failure_ids: Sequence[str],
) -> None:
    metrics: Dict = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        metrics["domain"] = domain_id

    entry = {
        "timestamp": time.time(),
        "tasks_succeeded": list(successes),
        "tasks_failed": list(failures),
        "helpful_increment": helpful_increment,
        "harmful_increment": harmful_increment,
        "updated_success_bullets": list(updated_success_ids),
        "updated_failure_bullets": list(updated_failure_ids),
    }

    metrics.setdefault("ace_benchmark_feedback_history", []).append(entry)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply benchmark outcomes to ACE")
    parser.add_argument("--results", type=Path, required=True, help="Path to benchmark results JSON")
    parser.add_argument("--domain", type=str, required=True, help="ACE domain identifier")
    parser.add_argument("--helpful-increment", type=int, default=1, help="Helpful increment for successful tasks")
    parser.add_argument("--harmful-increment", type=int, default=1, help="Harmful increment for failed tasks")
    parser.add_argument("--metrics-path", type=Path, default=Path("live_loop_artifacts/metrics.json"), help="Path to live loop metrics JSON")
    args = parser.parse_args()

    results = _load_results(args.results)
    evaluations = results.get("evaluations", [])

    if not evaluations:
        print("No evaluations in results; skipping benchmark feedback")
        return

    successes, failures = _split_outcomes(evaluations)

    success_ids = _apply_feedback(
        domain_id=args.domain,
        task_ids=successes,
        helpful_increment=args.helpful_increment,
        harmful_increment=0,
    )
    failure_ids = _apply_feedback(
        domain_id=args.domain,
        task_ids=failures,
        helpful_increment=0,
        harmful_increment=args.harmful_increment,
    )

    if success_ids or failure_ids:
        _update_metrics(
            args.metrics_path,
            domain_id=args.domain,
            successes=successes,
            failures=failures,
            helpful_increment=args.helpful_increment,
            harmful_increment=args.harmful_increment,
            updated_success_ids=success_ids,
            updated_failure_ids=failure_ids,
        )

    print(
        json.dumps(
            {
                "domain": args.domain,
                "success_tasks": successes,
                "failure_tasks": failures,
                "updated_success_bullets": success_ids,
                "updated_failure_bullets": failure_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
