"""
Client Module - Protocol abstraction and in-memory stub for ACE playbook system.

This module defines the AceClient protocol that all ACE implementations must satisfy,
along with an InMemoryAceClient stub for immediate testing and development.

The protocol ensures EE remains decoupled from ACE implementation details. When the
real ACE backend is ready, implement HttpAceClient with the same protocol and swap
via configuration - no code changes required.

Protocol Methods:
    - ingest_insight(insight): Store an insight in the playbook
    - render_playbook(sections, token_budget): Render playbook as text
    - health(): Check system health
    - version(): Get implementation version

Usage:
    from ee_ace_bridge import InMemoryAceClient

    # Initialize stub client
    ace = InMemoryAceClient()

    # Ingest insights
    insight_id = ace.ingest_insight({
        "task": "booking",
        "insight_kind": "rule",
        "insight_text": "Always check availability before confirming",
        "tags": ["availability"],
        ...
    })

    # Render playbook
    playbook = ace.render_playbook(sections=["Decision Heuristics"], token_budget=3500)
"""

from typing import Protocol, Dict, List, Optional, Any
import itertools


# ============================================================================
# Protocol Definition
# ============================================================================

class AceClient(Protocol):
    """
    Protocol defining the interface all ACE client implementations must satisfy.

    This protocol enforces a clean boundary between EE and ACE systems. Any
    implementation (InMemoryAceClient, HttpAceClient, MockAceClient) must
    provide these methods with matching signatures.

    The protocol enables dependency injection and testing without coupling
    to specific implementations.
    """

    def ingest_insight(self, insight: Dict[str, Any]) -> str:
        """
        Store an insight in the ACE playbook system.

        Args:
            insight: Insight dictionary matching ACE.Insight.v0.json schema

        Returns:
            Unique identifier for the ingested insight

        Example:
            >>> insight_id = client.ingest_insight({
            ...     "task": "booking",
            ...     "insight_kind": "rule",
            ...     "insight_text": "Check availability first",
            ...     "tags": ["availability"],
            ...     "created_at": "2025-01-15T10:30:00Z"
            ... })
        """
        ...

    def render_playbook(
        self,
        *,
        sections: Optional[List[str]] = None,
        token_budget: int = 3500
    ) -> str:
        """
        Render playbook sections as formatted text for policy context.

        Args:
            sections: Optional list of section names to include (None = all)
            token_budget: Maximum tokens to allocate (~4 chars ≈ 1 token)

        Returns:
            Formatted playbook text with sections and insights

        Example:
            >>> playbook = client.render_playbook(
            ...     sections=["Payment", "Validation"],
            ...     token_budget=2000
            ... )
            >>> print(playbook)
            ## Payment
            - Rule: Validate credit card format before submission
            ## Validation
            - Rule: Check required fields are non-empty
        """
        ...

    def health(self) -> Dict[str, str]:
        """
        Check health status of ACE system.

        Returns:
            Dictionary with status and diagnostic info

        Example:
            >>> health = client.health()
            >>> print(health["status"])
            ok
        """
        ...

    def version(self) -> str:
        """
        Get version identifier of ACE client implementation.

        Returns:
            Version string (e.g., "ace-stub-0.1", "ace-http-1.2.3")

        Example:
            >>> print(client.version())
            ace-stub-0.1
        """
        ...


# ============================================================================
# In-Memory Stub Implementation
# ============================================================================

class InMemoryAceClient:
    """
    In-memory stub implementation of AceClient for immediate testing.

    This stub provides deterministic, local-only ACE functionality without
    external dependencies. It stores insights in memory organized by sections,
    and renders them with simple formatting.

    The stub is intentionally simple - all sophisticated curation, deduplication,
    and ranking stays in the future ACE backend. The bridge only does basic
    deterministic mapping and rendering.

    Thread Safety:
        Not thread-safe. Use separate instances per thread if needed.

    Persistence:
        Memory-only, lost on process restart. For persistence, use HttpAceClient
        with ACE backend.

    Attributes:
        _sections: Dict mapping section names to lists of insight texts
        _ids: Counter for generating unique insight IDs

    Example:
        >>> ace = InMemoryAceClient()
        >>> ace.ingest_insight({
        ...     "task": "booking",
        ...     "insight_text": "Check availability before confirming",
        ...     "tags": ["availability"],
        ...     "insight_kind": "rule",
        ...     "created_at": "2025-01-15T10:30:00Z"
        ... })
        'mem-1'
        >>> print(ace.render_playbook())
        ## Decision Heuristics/Availability
        - Rule: Check availability before confirming
    """

    _ids = itertools.count(1)  # Class-level counter for unique IDs

    def __init__(self):
        """Initialize empty in-memory playbook."""
        self._sections: Dict[str, List[str]] = {}

    def ingest_insight(self, insight: Dict[str, Any]) -> str:
        """
        Store insight in appropriate section based on tags.

        The section is chosen deterministically using _choose_section() which
        examines insight tags and task. All insights for a section are stored
        as a flat list of insight_text strings.

        Args:
            insight: Insight dictionary (should match ACE.Insight.v0.json)

        Returns:
            Unique identifier (format: "mem-{counter}")
        """
        section = self._choose_section(insight)
        self._sections.setdefault(section, []).append(insight["insight_text"])

        # Return unique ID (use provided ID or generate new one)
        return insight.get("id") or f"mem-{next(self._ids)}"

    def render_playbook(
        self,
        *,
        sections: Optional[List[str]] = None,
        token_budget: int = 3500
    ) -> str:
        """
        Render playbook as formatted markdown text.

        Format:
            ## Section Name
            - Insight 1
            - Insight 2
            ## Another Section
            - Insight 3

        The output is clipped to fit within token_budget using rough estimate
        of 4 characters ≈ 1 token.

        Args:
            sections: Section names to include (None = all sections)
            token_budget: Maximum tokens (~4 chars ≈ 1 token)

        Returns:
            Formatted playbook text (may be empty if no insights)
        """
        # Select sections (all or filtered)
        selected = sections or list(self._sections.keys())

        # Build formatted output
        lines: List[str] = []
        for section_name in selected:
            entries = self._sections.get(section_name, [])
            if not entries:
                continue

            lines.append(f"## {section_name}")
            for entry in entries:
                lines.append(f"- {entry}")

        text = "\n".join(lines)

        # Clip to token budget (rough: 4 chars ≈ 1 token)
        max_chars = token_budget * 4
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0]  # Clip at line boundary

        return text

    def health(self) -> Dict[str, str]:
        """
        Return health status of in-memory stub.

        Returns:
            Dict with "status" and "sections" count
        """
        return {
            "status": "ok",
            "sections": str(len(self._sections)),
            "insights": str(sum(len(v) for v in self._sections.values())),
        }

    def version(self) -> str:
        """
        Return version identifier for stub implementation.

        Returns:
            Version string "ace-stub-0.1"
        """
        return "ace-stub-0.1"

    def _choose_section(self, insight: Dict[str, Any]) -> str:
        """
        Deterministically choose section name for insight based on tags.

        Section Selection Logic:
        1. Check for specific tags that map to known sections
        2. Fall back to task-based section name
        3. Use "General" as last resort

        This is intentionally simple and deterministic. Sophisticated section
        management and hierarchy belongs in the ACE backend.

        Args:
            insight: Insight dictionary with tags and task

        Returns:
            Section name (e.g., "Decision Heuristics/Availability")
        """
        tags = insight.get("tags") or []

        # Known tag → section mappings
        if "availability" in tags:
            return "Decision Heuristics/Availability"
        if "payment" in tags or "validation" in tags:
            return "Payment/Validation"
        if "error-avoidance" in tags:
            return "Error Prevention"
        if "safety" in tags:
            return "Safety Rules"

        # Fall back to task-based section
        task = insight.get("task", "general")
        return f"General/{task.capitalize()}"

    def clear(self):
        """
        Clear all insights from memory.

        Useful for testing and resetting state.

        Example:
            >>> ace.clear()
            >>> assert len(ace._sections) == 0
        """
        self._sections.clear()

    def get_section_count(self) -> int:
        """
        Get number of sections currently in playbook.

        Returns:
            Number of sections with at least one insight
        """
        return len(self._sections)

    def get_insight_count(self) -> int:
        """
        Get total number of insights across all sections.

        Returns:
            Total insight count
        """
        return sum(len(entries) for entries in self._sections.values())
