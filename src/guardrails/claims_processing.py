"""Guardrails for insurance claims processing domain."""

from __future__ import annotations

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


CLAIMS_DATA = [
    {
        "task_id": "claims-case-001",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Return 'approve' or 'deny'. Approve when driver not at fault and estimate below $2000.",
        },
    },
    {
        "task_id": "claims-case-002",
        "ground_truth": "deny",
        "guardrail": {
            "instructions": "Return 'approve' or 'deny'. Deny basic policies when estimate exceeds $3000.",
        },
    },
    {
        "task_id": "claims-case-003",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Return 'approve' or 'deny'. Approve minor damage under $1000.",
        },
    },
    {
        "task_id": "claims-case-004",
        "ground_truth": "deny",
        "guardrail": {
            "instructions": "Return 'approve' or 'deny'. Deny when driver at fault due to policy violation.",
        },
    },
    {
        "task_id": "claims-case-005",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Return 'approve' or 'deny'. Approve theft claims for premium tier regardless of amount.",
        },
    },
    {
        "task_id": "claims-case-006",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Approve basic policy vandalism claims when estimate is below $1,000.",
        },
    },
    {
        "task_id": "claims-case-007",
        "ground_truth": "deny",
        "guardrail": {
            "instructions": "Deny hit-and-run claims if documentation is missing regardless of tier.",
        },
    },
    {
        "task_id": "claims-case-008",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Premium tier covers all weather incidents; approve by default.",
        },
    },
    {
        "task_id": "claims-case-009",
        "ground_truth": "deny",
        "guardrail": {
            "instructions": "Deny mechanical wear claims for basic policies.",
        },
    },
    {
        "task_id": "claims-case-010",
        "ground_truth": "approve",
        "guardrail": {
            "instructions": "Approve minor glass repairs under $500 for standard tier.",
        },
    },
]


def _load_guardrails() -> Dict[str, ClaimsDecisionGuardrail]:
    domain_guardrails: Dict[str, ClaimsDecisionGuardrail] = {}
    for entry in CLAIMS_DATA:
        guardrail_meta = entry.get("guardrail", {})
        instructions = guardrail_meta.get(
            "instructions",
            "Return either 'approve' or 'deny'.",
        )
        expected = entry.get("ground_truth", "deny")

        domain_guardrails[entry["task_id"]] = ClaimsDecisionGuardrail(
            task_id=entry["task_id"],
            expected=expected,
            instructions=instructions,
        )

    return domain_guardrails


DOMAIN_GUARDRAILS = _load_guardrails()


def get_guardrail(task_id: str) -> Optional[ClaimsDecisionGuardrail]:
    return DOMAIN_GUARDRAILS.get(task_id)


register_domain("claims-processing", DOMAIN_GUARDRAILS)


__all__ = ["ClaimsDecisionGuardrail", "get_guardrail", "DOMAIN_GUARDRAILS"]
