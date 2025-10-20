from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from ace.models.playbook import PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
from ace.utils.database import get_session


def _load_metrics(path: Path, domain: str) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        metrics = {"domain": domain}
    return metrics


def _persist_metrics(path: Path, metrics: Dict, entry: Dict) -> None:
    history = metrics.setdefault("ace_prod_promotion_history", [])
    history.append(entry)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def promote_prod(
    domain: str,
    *,
    staging_helpful_threshold: int,
    prod_helpful_threshold: int,
    harmful_cap: int,
) -> Dict:
    promoted_ids: List[str] = []
    demoted_ids: List[str] = []

    with get_session() as session:
        repo = PlaybookRepository(session)
        bullets = repo.get_by_domain(domain)

        for bullet in bullets:
            if bullet.stage == PlaybookStage.QUARANTINED:
                continue

            # Promote shadow/staging → staging/prod
            if bullet.stage == PlaybookStage.SHADOW:
                if bullet.helpful_count >= staging_helpful_threshold and bullet.harmful_count <= harmful_cap:
                    bullet.stage = PlaybookStage.STAGING
                    repo.update(bullet)
                    promoted_ids.append(bullet.id)

            elif bullet.stage == PlaybookStage.STAGING:
                if bullet.helpful_count >= prod_helpful_threshold and bullet.harmful_count <= harmful_cap:
                    bullet.stage = PlaybookStage.PROD
                    repo.update(bullet)
                    promoted_ids.append(bullet.id)
                elif bullet.harmful_count > harmful_cap:
                    bullet.stage = PlaybookStage.SHADOW
                    repo.update(bullet)
                    demoted_ids.append(bullet.id)

            elif bullet.stage == PlaybookStage.PROD:
                if bullet.harmful_count > harmful_cap:
                    bullet.stage = PlaybookStage.STAGING
                    repo.update(bullet)
                    demoted_ids.append(bullet.id)

        if promoted_ids or demoted_ids:
            session.commit()

    return {
        "promoted": promoted_ids,
        "demoted": demoted_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote staging bullets to prod based on thresholds")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--metrics-path", type=Path, default=Path("live_loop_artifacts/metrics.json"))
    parser.add_argument("--staging-helpful-threshold", type=int, default=3)
    parser.add_argument("--prod-helpful-threshold", type=int, default=5)
    parser.add_argument("--harmful-cap", type=int, default=1)
    args = parser.parse_args()

    result = promote_prod(
        args.domain,
        staging_helpful_threshold=args.staging_helpful_threshold,
        prod_helpful_threshold=args.prod_helpful_threshold,
        harmful_cap=args.harmful_cap,
    )

    metrics = _load_metrics(args.metrics_path, args.domain)
    entry = {
        "timestamp": time.time(),
        "promoted": result["promoted"],
        "demoted": result["demoted"],
        "staging_threshold": args.staging_helpful_threshold,
        "prod_threshold": args.prod_helpful_threshold,
        "harmful_cap": args.harmful_cap,
    }
    _persist_metrics(args.metrics_path, metrics, entry)

    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
