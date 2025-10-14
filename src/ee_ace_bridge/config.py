"""
Configuration Module - Feature flags and settings for ACE bridge integration.

This module provides environment-based configuration for gradual rollout of ACE
playbook features. All settings default to safe, non-intrusive values.

Feature Flags:
    ACE_ENABLED: Master toggle for all ACE functionality (default: False)
    ACE_SECTIONS: Optional comma-separated list of playbook sections to include
    ACE_TOKEN_BUDGET: Maximum tokens for rendered playbook context (default: 3500)
    ACE_ENDPOINT: HTTP endpoint for future HttpAceClient (default: None)

Environment Variables:
    Set these via environment or .env file:
    - ACE_ENABLED=1              # Enable ACE bridge
    - ACE_SECTIONS=Payment,Auth  # Filter to specific sections
    - ACE_TOKEN_BUDGET=5000      # Increase token budget
    - ACE_ENDPOINT=http://...    # Use HTTP client instead of stub

Usage:
    from ee_ace_bridge import config

    if config.ACE_ENABLED:
        # Use ACE features
        pass
"""

import os
from typing import Optional, List


# ============================================================================
# Feature Flags
# ============================================================================

ACE_ENABLED: bool = os.getenv("ACE_ENABLED", "0") == "1"
"""
Master toggle for ACE bridge functionality.

When False (default), all ACE features are disabled and the system behaves
exactly as before. When True, enables playbook injection during inference.

Set via environment: ACE_ENABLED=1
"""


ACE_SECTIONS: Optional[List[str]] = (
    [s.strip() for s in os.getenv("ACE_SECTIONS", "").split(",") if s.strip()]
    if os.getenv("ACE_SECTIONS")
    else None
)
"""
Optional filter for playbook sections to include.

If None (default), all sections are included. If specified, only listed
sections will be rendered in playbook context.

Set via environment: ACE_SECTIONS=Payment,Auth,Validation
"""


ACE_TOKEN_BUDGET: int = int(os.getenv("ACE_TOKEN_BUDGET", "3500"))
"""
Maximum tokens allocated for playbook context in policy prompts.

Default 3500 provides ~14KB of context (4 chars ≈ 1 token).
Increase if policy has larger context window.

Set via environment: ACE_TOKEN_BUDGET=5000
"""


ACE_ENDPOINT: Optional[str] = os.getenv("ACE_ENDPOINT")
"""
HTTP endpoint for ACE playbook service (future HttpAceClient).

When None (default), uses InMemoryAceClient stub. When set, bridge
should use HttpAceClient to communicate with real ACE backend.

Set via environment: ACE_ENDPOINT=http://ace-service:8080
"""


# ============================================================================
# Validation
# ============================================================================

def validate_config() -> List[str]:
    """
    Validate configuration values for common issues.

    Returns:
        List of validation warnings (empty if all valid)

    Example:
        >>> warnings = validate_config()
        >>> if warnings:
        ...     print("Config warnings:", warnings)
    """
    warnings = []

    # Check token budget is reasonable
    if ACE_TOKEN_BUDGET < 500:
        warnings.append(
            f"ACE_TOKEN_BUDGET ({ACE_TOKEN_BUDGET}) is very low, may truncate playbooks"
        )
    elif ACE_TOKEN_BUDGET > 10000:
        warnings.append(
            f"ACE_TOKEN_BUDGET ({ACE_TOKEN_BUDGET}) is very high, may exceed context window"
        )

    # Check endpoint format if provided
    if ACE_ENDPOINT:
        if not ACE_ENDPOINT.startswith(("http://", "https://")):
            warnings.append(
                f"ACE_ENDPOINT should start with http:// or https://, got: {ACE_ENDPOINT}"
            )

    # Warn if sections specified but ACE disabled
    if ACE_SECTIONS and not ACE_ENABLED:
        warnings.append(
            "ACE_SECTIONS specified but ACE_ENABLED=False, sections will be ignored"
        )

    # Warn if endpoint specified but ACE disabled
    if ACE_ENDPOINT and not ACE_ENABLED:
        warnings.append(
            "ACE_ENDPOINT specified but ACE_ENABLED=False, endpoint will be ignored"
        )

    return warnings


def get_config_summary() -> dict:
    """
    Get dictionary summary of current configuration.

    Returns:
        Dictionary with all config values

    Example:
        >>> import json
        >>> print(json.dumps(get_config_summary(), indent=2))
    """
    return {
        "ACE_ENABLED": ACE_ENABLED,
        "ACE_SECTIONS": ACE_SECTIONS,
        "ACE_TOKEN_BUDGET": ACE_TOKEN_BUDGET,
        "ACE_ENDPOINT": ACE_ENDPOINT,
        "warnings": validate_config(),
    }
