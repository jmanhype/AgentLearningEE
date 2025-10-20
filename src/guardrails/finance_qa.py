"""Guardrails for finance QA benchmark."""

from __future__ import annotations

from typing import Dict, Optional

from guardrails import register_domain


class FinanceQAGuardrail:
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
        if normalized in {"pass", "passed", "yes"}:
            return "pass"
        if normalized in {"fail", "failed", "no"}:
            return "fail"
        return normalized


FINANCE_DATA = [
    {
        "task_id": "finance-case-001",
        "ground_truth": "pass",
        "instructions": "Return 'pass' if growth >= 0.05 else 'fail'.",
    },
    {
        "task_id": "finance-case-002",
        "ground_truth": "fail",
        "instructions": "Return 'pass' if growth >= 0.05 else 'fail'.",
    },
    {
        "task_id": "finance-case-003",
        "ground_truth": "pass",
        "instructions": "Return 'pass' if margin >= 0.15 else 'fail'.",
    },
    {
        "task_id": "finance-case-004",
        "ground_truth": "fail",
        "instructions": "Return 'pass' if margin >= 0.15 else 'fail'.",
    },
    {
        "task_id": "finance-case-005",
        "ground_truth": "fail",
        "instructions": "Return 'pass' if cash flow >= 0 else 'fail'.",
    },
]


def _load_guardrails() -> Dict[str, FinanceQAGuardrail]:
    guardrails: Dict[str, FinanceQAGuardrail] = {}
    for entry in FINANCE_DATA:
        guardrails[entry["task_id"]] = FinanceQAGuardrail(
            task_id=entry["task_id"],
            expected=entry["ground_truth"],
            instructions=entry["instructions"],
        )
    return guardrails


DOMAIN_GUARDRAILS = _load_guardrails()


def get_guardrail(task_id: str) -> Optional[FinanceQAGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain("finance-qa", DOMAIN_GUARDRAILS)


__all__ = ["FinanceQAGuardrail", "get_guardrail", "DOMAIN_GUARDRAILS"]
