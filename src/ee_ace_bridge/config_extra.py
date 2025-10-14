"""
Extended ACE Configuration

Additional configuration parameters for wire compatibility with ACE Curator.
Supplements config.py with domain isolation, stage management, and similarity tuning.
"""

import os
from enum import Enum
from typing import Optional


class Stage(str, Enum):
    """ACE playbook deployment stages."""

    SHADOW = "shadow"  # Insights logged, not used in retrieval
    STAGING = "staging"  # Used by 5% of traffic
    PROD = "prod"  # Used by all production traffic


# Domain Isolation
ACE_DOMAIN_ID = os.getenv("ACE_DOMAIN_ID", "agent-learning").strip()
"""
Domain namespace for multi-tenant isolation (e.g., 'flights', 'hotels', 'agent-learning').
Must match pattern: ^[a-z0-9-]+$ (lowercase letters, digits, hyphens only).
"""

# Stage Management
ACE_TARGET_STAGE_STR = os.getenv("ACE_TARGET_STAGE", "shadow").lower()
ACE_TARGET_STAGE = Stage(ACE_TARGET_STAGE_STR) if ACE_TARGET_STAGE_STR in [s.value for s in Stage] else Stage.SHADOW
"""
Target stage for new insights (shadow/staging/prod).
Default: shadow (safe for gradual rollout).
"""

# Semantic Similarity
ACE_SIMILARITY_THRESHOLD = float(os.getenv("ACE_SIMILARITY_THRESHOLD", "0.80"))
"""
Cosine similarity threshold for deduplication (0.0-1.0).
Default: 0.80 (ACE recommended value).
- 0.75-0.78: Aggressive dedup (may merge distinct patterns)
- 0.80-0.82: Balanced (ACE default)
- 0.85-0.90: Conservative (preserves more variations)
"""


def validate_domain_id(domain_id: str) -> bool:
    """
    Validate domain_id against ACE pattern.

    Args:
        domain_id: Domain namespace to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    pattern = r"^[a-z0-9-]+$"
    return bool(re.match(pattern, domain_id))


def get_config_summary() -> dict:
    """
    Get summary of extended ACE configuration.

    Returns:
        Dict with all ACE configuration parameters
    """
    return {
        "domain_id": ACE_DOMAIN_ID,
        "target_stage": ACE_TARGET_STAGE.value,
        "similarity_threshold": ACE_SIMILARITY_THRESHOLD,
        "domain_id_valid": validate_domain_id(ACE_DOMAIN_ID),
    }


def validate_extended_config() -> list[str]:
    """
    Validate extended ACE configuration and return warnings.

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []

    # Validate domain_id pattern
    if not validate_domain_id(ACE_DOMAIN_ID):
        warnings.append(
            f"ACE_DOMAIN_ID '{ACE_DOMAIN_ID}' does not match required pattern ^[a-z0-9-]+$"
        )

    # Validate similarity threshold range
    if not 0.0 <= ACE_SIMILARITY_THRESHOLD <= 1.0:
        warnings.append(
            f"ACE_SIMILARITY_THRESHOLD {ACE_SIMILARITY_THRESHOLD} must be between 0.0 and 1.0"
        )
    elif ACE_SIMILARITY_THRESHOLD < 0.75:
        warnings.append(
            f"ACE_SIMILARITY_THRESHOLD {ACE_SIMILARITY_THRESHOLD} is very low (< 0.75) - may over-merge distinct patterns"
        )
    elif ACE_SIMILARITY_THRESHOLD > 0.90:
        warnings.append(
            f"ACE_SIMILARITY_THRESHOLD {ACE_SIMILARITY_THRESHOLD} is very high (> 0.90) - may create many near-duplicates"
        )

    return warnings
