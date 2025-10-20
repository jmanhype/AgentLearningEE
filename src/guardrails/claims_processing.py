"""Guardrails for insurance claims processing domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from guardrails import register_domain


class ClaimsDecisionGuardrail:
    """Guardrail that enforces binary approve/deny decisions."""

    auto_correct = True
    format = "string"
    decimals = None

    def __init__(self, *, task_id: str, expected: str, instructions: str) -> None:
        self.task_id = task_id
        self.expected = self._normalize(expected)
        self.instructions = instructions

    def canonical_answer(self) -> str:
        return self.expected

    def validate(self, answer: str, ground_truth: str) -> bool:
        normalized = self._normalize(answer)
        return normalized == self.expected

    def reset(self) -> None:
        return

    def _normalize(self, answer: str) -> str:
        normalized = (answer or "").strip().lower()
        if normalized in {"approve", "approved", "yes"}:
            return "approve"
        if normalized in {"deny", "denied", "no"}:
            return "deny"
        return normalized


def _load_guardrails() -> Dict[str, ClaimsDecisionGuardrail]:
    domain_guardrails: Dict[str, ClaimsDecisionGuardrail] = {}
    root = Path(__file__).resolve().parents[2]
    dataset_path = root / "data" / "claims_samples" / "claims_20.jsonl"
    if not dataset_path.exists():
        return domain_guardrails

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record["task_id"]
            guardrail_meta = record.get("guardrail", {})
            expected = record.get("ground_truth", guardrail_meta.get("value", "deny"))
            instructions = guardrail_meta.get(
                "instructions",
                "Return either 'approve' or 'deny'.",
            )

            domain_guardrails[task_id] = ClaimsDecisionGuardrail(
                task_id=task_id,
                expected=expected,
                instructions=instructions,
            )

    return domain_guardrails


DOMAIN_GUARDRAILS = _load_guardrails()


def get_guardrail(task_id: str) -> Optional[ClaimsDecisionGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain("claims-processing", DOMAIN_GUARDRAILS)


__all__ = ["ClaimsDecisionGuardrail", "get_guardrail", "DOMAIN_GUARDRAILS"]
