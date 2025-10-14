"""
Injector Module - Playbook context injection for policy prompts.

This module provides functions to augment policy input states with rendered
playbook context. The injection is designed to be:
- Non-intrusive: gracefully no-ops if playbook is empty
- Token-aware: respects budget constraints
- Clear: uses explicit headers to separate playbook from state

The injector is the final step in the ACE bridge pipeline, called at inference
time to provide the policy with relevant learned rules and patterns.

Key Functions:
    - augment_state_with_playbook(): Main injection function
    - _format_playbook_context(): Format playbook with clear structure

Usage:
    from ee_ace_bridge import augment_state_with_playbook, InMemoryAceClient

    ace = InMemoryAceClient()
    # ... populate ace with insights ...

    # At inference time
    original_state = "Current booking state: hotel search for 2 nights"

    augmented_state = augment_state_with_playbook(
        state=original_state,
        ace=ace,
        sections=["Payment", "Validation"],
        token_budget=3500
    )

    # augmented_state now includes playbook context before state
"""

from typing import Optional, List
from .client import AceClient


# ============================================================================
# Playbook Headers and Instructions
# ============================================================================

PLAYBOOK_HEADER = "### Playbook (Ground Rules from Prior Experience)"
"""Header that introduces the playbook section in augmented prompts."""

STATE_HEADER = "### Current State"
"""Header that introduces the actual state after playbook."""

INSTRUCTIONS = (
    "Follow the Playbook rules when deciding the next action. "
    "If rules conflict, prefer safety and task completion."
)
"""Instructions for how policy should use playbook context."""


# ============================================================================
# Main Injection Function
# ============================================================================

def augment_state_with_playbook(
    state: str,
    ace: AceClient,
    *,
    sections: Optional[List[str]] = None,
    token_budget: int = 3500
) -> str:
    """
    Augment state string with rendered playbook context.

    This function prepends the playbook to the state with clear structural
    headers, making it easy for the policy to distinguish between learned
    rules and current state.

    Format (when playbook exists):
        ### Playbook (Ground Rules from Prior Experience)
        ## Section Name
        - Rule: ...
        - Rule: ...

        ### Current State
        Follow the Playbook rules when deciding...

        {original state}

    If playbook is empty, returns original state unchanged (graceful no-op).

    Args:
        state: Original state description string
        ace: AceClient instance (InMemoryAceClient or HttpAceClient)
        sections: Optional list of section names to include (None = all)
        token_budget: Maximum tokens for playbook rendering (~4 chars ≈ 1 token)

    Returns:
        Augmented state string with playbook prepended, or original state if
        playbook is empty

    Example:
        >>> ace = InMemoryAceClient()
        >>> ace.ingest_insight({
        ...     "insight_text": "Always check availability first",
        ...     "tags": ["availability"],
        ...     "insight_kind": "rule",
        ...     "task": "booking",
        ...     "created_at": "2025-01-15T10:00:00Z"
        ... })
        'mem-1'
        >>> state = "Hotel booking: 2 nights in NYC"
        >>> augmented = augment_state_with_playbook(state, ace, token_budget=1000)
        >>> print(augmented)
        ### Playbook (Ground Rules from Prior Experience)
        ## Decision Heuristics/Availability
        - Always check availability first
        <BLANKLINE>
        ### Current State
        Follow the Playbook rules when deciding...
        <BLANKLINE>
        Hotel booking: 2 nights in NYC
    """
    # Render playbook from ACE client
    playbook = ace.render_playbook(sections=sections, token_budget=token_budget)

    # Graceful no-op if playbook is empty
    if not playbook or not playbook.strip():
        return state

    # Format complete context with headers
    return _format_playbook_context(playbook, state)


def _format_playbook_context(playbook: str, state: str) -> str:
    """
    Format playbook and state with clear structural headers.

    Creates a well-structured prompt that clearly separates:
    1. Playbook header and content (rules from experience)
    2. State header and instructions (how to use rules)
    3. Original state (current situation)

    Args:
        playbook: Rendered playbook text (assumed non-empty)
        state: Original state string

    Returns:
        Formatted string with clear sections

    Example:
        >>> playbook = "## Payment\\n- Rule: Validate credit card"
        >>> state = "Processing payment"
        >>> print(_format_playbook_context(playbook, state))
        ### Playbook (Ground Rules from Prior Experience)
        ## Payment
        - Rule: Validate credit card
        <BLANKLINE>
        ### Current State
        Follow the Playbook rules...
        <BLANKLINE>
        Processing payment
    """
    return (
        f"{PLAYBOOK_HEADER}\n"
        f"{playbook}\n"
        f"\n"
        f"{STATE_HEADER}\n"
        f"{INSTRUCTIONS}\n"
        f"\n"
        f"{state}"
    )


# ============================================================================
# Batch Processing
# ============================================================================

def augment_states_batch(
    states: List[str],
    ace: AceClient,
    *,
    sections: Optional[List[str]] = None,
    token_budget: int = 3500
) -> List[str]:
    """
    Augment multiple states with same playbook (for batch inference).

    Useful when running batch predictions - renders playbook once and
    applies to all states.

    Args:
        states: List of state description strings
        ace: AceClient instance
        sections: Optional section filter
        token_budget: Token budget for playbook

    Returns:
        List of augmented state strings (same order as input)

    Example:
        >>> states = ["State 1", "State 2", "State 3"]
        >>> augmented = augment_states_batch(states, ace)
        >>> len(augmented)
        3
    """
    # Render playbook once
    playbook = ace.render_playbook(sections=sections, token_budget=token_budget)

    # Graceful no-op if playbook empty
    if not playbook or not playbook.strip():
        return states

    # Apply same playbook to all states
    return [_format_playbook_context(playbook, state) for state in states]


# ============================================================================
# Diagnostic Utilities
# ============================================================================

def estimate_token_overhead(ace: AceClient, sections: Optional[List[str]] = None) -> int:
    """
    Estimate token overhead added by playbook injection.

    Useful for capacity planning and checking if playbook will fit in
    policy context window.

    Estimation: ~4 characters ≈ 1 token (OpenAI estimate)

    Args:
        ace: AceClient instance
        sections: Optional section filter

    Returns:
        Estimated token count for full playbook

    Example:
        >>> overhead = estimate_token_overhead(ace, sections=["Payment"])
        >>> print(f"Playbook adds ~{overhead} tokens")
        Playbook adds ~250 tokens
    """
    # Render with large budget to get full size
    playbook = ace.render_playbook(sections=sections, token_budget=100000)

    # Add overhead for headers and instructions
    headers = PLAYBOOK_HEADER + "\n\n" + STATE_HEADER + "\n" + INSTRUCTIONS + "\n\n"
    total_text = headers + playbook

    # Estimate: 4 chars ≈ 1 token
    return len(total_text) // 4


def validate_injection(
    original_state: str,
    augmented_state: str
) -> bool:
    """
    Validate that injection preserved original state.

    Checks:
    1. Original state appears in augmented state
    2. Augmented state is longer than original (unless no-op)
    3. Headers are present (if not no-op)

    Args:
        original_state: Original state before injection
        augmented_state: State after augment_state_with_playbook()

    Returns:
        True if injection looks valid, False if suspicious

    Example:
        >>> original = "Book hotel"
        >>> augmented = augment_state_with_playbook(original, ace)
        >>> assert validate_injection(original, augmented)
    """
    # Check original state is preserved
    if original_state not in augmented_state:
        return False

    # If no-op (playbook empty), should be identical
    if original_state == augmented_state:
        return True

    # If playbook added, check headers present
    if STATE_HEADER not in augmented_state:
        return False
    if PLAYBOOK_HEADER not in augmented_state:
        return False

    # Check augmented is longer
    if len(augmented_state) <= len(original_state):
        return False

    return True
