"""Domain guardrail registry for AgentLearningEE."""

from __future__ import annotations

from typing import Dict, Optional

from .base import NumericGuardrail


_REGISTRY: Dict[str, Dict[str, NumericGuardrail]] = {}


def register_domain(domain: str, guardrails: Dict[str, NumericGuardrail]) -> None:
    """Register guardrails for a domain."""

    _REGISTRY[domain] = guardrails


def all_domains() -> Dict[str, Dict[str, NumericGuardrail]]:
    """Return mapping of domain → guardrails."""

    return dict(_REGISTRY)


def get_guardrail(task_id: str, domain: Optional[str] = None) -> Optional[NumericGuardrail]:
    """Fetch guardrail for a task, optionally scoped by domain."""

    if domain:
        domain_guardrails = _REGISTRY.get(domain)
        if domain_guardrails:
            return domain_guardrails.get(task_id)
        return None

    for domain_guardrails in _REGISTRY.values():
        if task_id in domain_guardrails:
            return domain_guardrails[task_id]
    return None


# Import default domains
from . import finance  # noqa: E402,F401
from . import claims_processing  # noqa: E402,F401


__all__ = ["NumericGuardrail", "register_domain", "all_domains", "get_guardrail"]
