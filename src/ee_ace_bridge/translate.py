"""
Schema Translation: EE Bridge → ACE Curator

Maps EE bridge insight schema to ACE's expected format for wire compatibility.
"""

from __future__ import annotations
from typing import Dict, List

# Section mapping: EE insight_kind → ACE section
_HELPFUL = {"rule", "pattern", "checklist"}
_HARMFUL = {"anti_pattern"}
_NEUTRAL = {"observation", "context"}


def bridge_to_ace_insight(bridge_insight: Dict) -> Dict:
    """
    Map EE bridge insight schema → ACE Curator insight schema.

    EE Bridge Format:
        {
            "insight_kind": "rule|pattern|anti_pattern|checklist|observation",
            "insight_text": "Check availability first",
            "tags": ["availability"],
            ...
        }

    ACE Format:
        {
            "content": "Check availability first",
            "section": "Helpful|Harmful|Neutral",
            "tags": ["availability"]
        }

    Args:
        bridge_insight: Insight in EE bridge format

    Returns:
        Insight in ACE Curator format
    """
    kind = (bridge_insight.get("insight_kind") or "").lower()

    # Map insight_kind → section
    if kind in _HARMFUL:
        section = "Harmful"
    elif kind in _HELPFUL:
        section = "Helpful"
    else:
        section = "Neutral"

    # Extract content (truncate to 280 chars per ACE convention)
    content = bridge_insight.get("insight_text", "").strip()[:280]

    return {
        "content": content,
        "section": section,
        "tags": bridge_insight.get("tags") or [],
    }


def bridge_batch_to_ace(batch: List[Dict]) -> List[Dict]:
    """
    Convert a batch of EE bridge insights to ACE format.

    Args:
        batch: List of insights in EE bridge format

    Returns:
        List of insights in ACE Curator format
    """
    return [bridge_to_ace_insight(x) for x in batch]


def ace_to_bridge_section(section: str) -> str:
    """
    Reverse mapping: ACE section → EE bridge insight_kind.

    Useful for reading back from ACE and displaying in EE format.

    Args:
        section: ACE section (Helpful/Harmful/Neutral)

    Returns:
        EE bridge insight_kind
    """
    section_lower = section.lower()

    if section_lower == "harmful":
        return "anti_pattern"
    elif section_lower == "helpful":
        return "rule"
    else:
        return "observation"
