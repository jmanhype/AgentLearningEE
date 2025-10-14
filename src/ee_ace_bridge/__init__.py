"""
EE-ACE Bridge Package - Protocol-based adapter for ACE playbook integration.

This package provides a clean interface between the EE (Early Experience) agent learning
system and the ACE (Adaptive Code Evolution) playbook system. It enables contract-first
integration with an in-memory stub for immediate testing, allowing seamless swap to
a real ACE backend without code changes.

Key Components:
    - client: AceClient protocol and InMemoryAceClient stub
    - mapper: Deterministic ReflectionEvent → ACE.Insight mapping
    - injector: Playbook context injection for policy prompts
    - config: Feature flags for gradual rollout

Usage:
    from ee_ace_bridge import InMemoryAceClient, augment_state_with_playbook
    from ee_ace_bridge import reflection_to_insight, config

    # Initialize client
    ace = InMemoryAceClient()

    # Training time: seed with reflections
    for reflection in reflection_data:
        insight = reflection_to_insight(reflection)
        ace.ingest_insight(insight)

    # Inference time: augment state with playbook
    if config.ACE_ENABLED:
        state = augment_state_with_playbook(state, ace)
"""

from .client import AceClient, InMemoryAceClient
from .mapper import reflection_to_insight
from .injector import augment_state_with_playbook
from .config import (
    ACE_ENABLED,
    ACE_SECTIONS,
    ACE_TOKEN_BUDGET,
    ACE_ENDPOINT,
)

__all__ = [
    "AceClient",
    "InMemoryAceClient",
    "reflection_to_insight",
    "augment_state_with_playbook",
    "ACE_ENABLED",
    "ACE_SECTIONS",
    "ACE_TOKEN_BUDGET",
    "ACE_ENDPOINT",
]

__version__ = "0.1.0"
