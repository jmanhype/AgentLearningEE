"""
Mapper Module - Deterministic transformation of ReflectionEvent to ACE.Insight.

This module provides NO-LLM, deterministic mapping from EE reflection data to
ACE insights. The mapping uses simple heuristics and pattern matching to extract
actionable rules from structured reasoning.

The mapper is intentionally conservative and rule-based. Rich semantic analysis,
deduplication, and curation belong in the ACE backend, not in this bridge layer.

Key Functions:
    - reflection_to_insight(): Main mapping function
    - _extract_lesson(): Extract actionable lesson from rationale
    - _derive_tags(): Derive categorization tags from content
    - _normalize(): Normalize text for consistent formatting

Usage:
    from ee_ace_bridge import reflection_to_insight

    reflection = {
        "state": "...",
        "reasoning": "...",
        "action": "check_availability",
        "expert_action": "check_availability",
        "alternative_action": "book_immediately",
        "source_rollout_id": 42,
        "confidence": 0.9
    }

    insight = reflection_to_insight(reflection)
    print(insight["insight_text"])  # "Rule: Check availability before booking."
"""

from datetime import datetime, timezone
from typing import Dict, List
import re


# ============================================================================
# Main Mapping Function
# ============================================================================

def reflection_to_insight(refl: Dict) -> Dict:
    """
    Transform reflection data into ACE insight format.

    This is a deterministic, no-LLM mapping that extracts actionable insights
    from EE reflection data. The mapping is conservative - it's better to
    create generic insights than to hallucinate specifics.

    Mapping Strategy:
    1. Extract lesson from reasoning/rationale field
    2. Derive tags for section routing
    3. Preserve confidence from reflection
    4. Set novelty to 1.0 (true novelty assessment is ACE's job)
    5. Include source reference for traceability

    Args:
        refl: Reflection dictionary with keys:
            - state: Current environment state (optional, for context)
            - reasoning: Structured EE-style reasoning (required)
            - action: Chosen action (required for task inference)
            - expert_action: Expert's chosen action (optional)
            - alternative_action: Alternative considered (optional)
            - source_rollout_id: Rollout identifier (optional)
            - confidence: Confidence score 0-1 (optional, default 0.5)

    Returns:
        Dictionary matching ACE.Insight.v0.json schema with keys:
            - task: Task domain (inferred from action)
            - insight_kind: Always "rule" (patterns are ACE's job)
            - insight_text: Actionable lesson text
            - tags: List of categorization tags
            - evidence_ref: Reference to source reflection
            - quality: {confidence, novelty}
            - created_at: ISO 8601 timestamp

    Example:
        >>> reflection = {
        ...     "reasoning": "The expert checked availability first, therefore avoiding errors.",
        ...     "action": "check_availability",
        ...     "confidence": 0.85
        ... }
        >>> insight = reflection_to_insight(reflection)
        >>> print(insight["insight_text"])
        Rule: Therefore check availability first to avoid errors.
        >>> print(insight["tags"])
        ['availability', 'error-avoidance']
    """
    # Extract rationale (check multiple possible field names)
    rationale = refl.get("reasoning") or refl.get("rationale", "")

    # Extract lesson from rationale
    lesson = _extract_lesson(rationale)

    # Derive tags for section routing
    tags = _derive_tags(lesson, refl)

    task_id = refl.get("task_id")
    if task_id:
        tags.append(f"task:{task_id}")

    domain = refl.get("domain")
    if domain:
        tags.append(f"domain:{domain}")

    # Infer task from action
    action = refl.get("action") or refl.get("expert_action", "")
    task = _infer_task(action)

    # Build insight with schema compliance
    insight = {
        "task": task,
        "insight_kind": "rule",  # Conservative choice; ACE can refine to pattern/anti_pattern
        "insight_text": lesson,
        "tags": tags,
        "evidence_ref": _build_evidence_ref(refl),
        "quality": {
            "confidence": float(refl.get("confidence", 0.5)),
            "novelty": 1.0  # Leave true novelty assessment to ACE Curator
        },
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    return insight


# ============================================================================
# Lesson Extraction Heuristics
# ============================================================================

def _extract_lesson(rationale: str) -> str:
    """
    Extract actionable lesson from rationale text using heuristics.

    Strategy:
    1. Look for sentences with causal/prescriptive keywords:
       - "therefore", "so", "thus" (causal conclusion)
       - "should", "must", "always", "never" (prescriptive)
       - "to avoid", "to prevent" (preventive)
    2. If found, extract that sentence as the lesson
    3. Otherwise, take first sentence (bounded to 240 chars)
    4. Normalize and ensure imperative voice

    This is intentionally conservative. We want clear, simple rules.
    Complex pattern extraction belongs in ACE.

    Args:
        rationale: Structured reasoning text from reflection

    Returns:
        Normalized lesson text starting with "Rule:" prefix

    Example:
        >>> text = "The expert chose X. This is good because Y. Therefore always do X first."
        >>> lesson = _extract_lesson(text)
        >>> print(lesson)
        Rule: Therefore always do X first.
    """
    if not rationale or not isinstance(rationale, str):
        return "Rule: Follow established best practices."

    # Try to find sentence with prescriptive/causal keywords
    for cue in [r"\btherefore\b", r"\bso\b", r"\bthus\b", r"\bto avoid\b",
                r"\bshould\b", r"\bmust\b", r"\balways\b", r"\bnever\b",
                r"\bto prevent\b", r"\bensure\b"]:
        match = re.search(rf"([^.!?]*{cue}[^.!?]*)[.!?]", rationale, flags=re.IGNORECASE)
        if match:
            sentence = match.group(1).strip()
            if len(sentence) > 10:  # Skip trivial matches
                return _normalize(sentence)

    # Fallback: take first sentence (bounded)
    first_sentence = re.split(r'[.!?]', rationale)[0].strip()
    if len(first_sentence) > 240:
        first_sentence = first_sentence[:237] + "..."

    return _normalize(first_sentence)


def _normalize(text: str) -> str:
    """
    Normalize lesson text for consistent formatting.

    Normalization:
    1. Strip whitespace
    2. Ensure ends with period
    3. Add "Rule:" prefix if not already prescriptive

    Args:
        text: Raw lesson text

    Returns:
        Normalized text with "Rule:" prefix

    Example:
        >>> _normalize("check availability first")
        'Rule: Check availability first.'
        >>> _normalize("Always validate input.")
        'Rule: Always validate input.'
    """
    text = text.strip()

    # Ensure ends with period
    if not text.endswith((".", "!", "?")):
        text += "."

    # Add "Rule:" prefix if not already imperative/prescriptive
    if not any(text.lower().startswith(prefix) for prefix in
               ["rule:", "always", "never", "before", "when", "if", "ensure"]):
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        text = "Rule: " + text

    return text


# ============================================================================
# Tag Derivation
# ============================================================================

def _derive_tags(lesson: str, refl: Dict) -> List[str]:
    """
    Derive categorization tags from lesson content and reflection context.

    Tags are used by InMemoryAceClient._choose_section() to route insights
    to appropriate playbook sections. This is simple keyword matching -
    sophisticated taxonomy belongs in ACE.

    Tag Derivation Logic:
    - "sold out", "availability", "check" → "availability"
    - "credit card", "payment", "format", "validate" → "payment", "validation"
    - "error", "fail", "avoid", "prevent" → "error-avoidance"
    - "safe", "safety", "secure" → "safety"

    Args:
        lesson: Normalized lesson text
        refl: Reflection dictionary (for additional context)

    Returns:
        List of tag strings (deduplicated, preserving order)

    Example:
        >>> tags = _derive_tags("Rule: Check availability to avoid sold out errors.", {})
        >>> print(sorted(tags))
        ['availability', 'error-avoidance']
    """
    tags = []

    # Combine lesson with rationale for richer context
    text = (lesson + " " + refl.get("reasoning", "") + " " + refl.get("rationale", "")).lower()

    # Keyword-based tag mapping
    if any(kw in text for kw in ["sold out", "availability", "available"]):
        tags.append("availability")

    if any(kw in text for kw in ["credit card", "payment", "charge"]):
        tags.append("payment")

    if any(kw in text for kw in ["format", "validate", "validation", "check"]):
        tags.append("validation")

    if any(kw in text for kw in ["error", "fail", "avoid", "prevent", "mistake"]):
        tags.append("error-avoidance")

    if any(kw in text for kw in ["safe", "safety", "secure", "security"]):
        tags.append("safety")

    if any(kw in text for kw in ["performance", "slow", "fast", "optimize"]):
        tags.append("performance")

    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    # If no tags derived, add generic tag
    if not unique_tags:
        unique_tags.append("general")

    return unique_tags


# ============================================================================
# Helper Functions
# ============================================================================

def _infer_task(action: str) -> str:
    """
    Infer task domain from action string.

    Simple heuristic: extract first word or use action as-is.
    Sophisticated task taxonomy belongs in ACE.

    Args:
        action: Action string (e.g., "check_availability", "book_hotel")

    Returns:
        Task domain string (e.g., "check", "book", "general")

    Example:
        >>> _infer_task("check_availability")
        'check'
        >>> _infer_task("booking")
        'booking'
    """
    if not action:
        return "general"

    # Split on underscore or camelCase
    parts = re.split(r'[_\s]|(?<=[a-z])(?=[A-Z])', action)
    if parts:
        return parts[0].lower()

    return action.lower() or "general"


def _build_evidence_ref(refl: Dict) -> str:
    """
    Build evidence reference string for traceability.

    Format: "rollout:{id}" or "reflection" if no ID available.

    Args:
        refl: Reflection dictionary

    Returns:
        Evidence reference string

    Example:
        >>> _build_evidence_ref({"source_rollout_id": 42})
        'rollout:42'
        >>> _build_evidence_ref({})
        'reflection'
    """
    rollout_id = refl.get("source_rollout_id")
    if rollout_id is not None:
        return f"rollout:{rollout_id}"

    return "reflection"


# ============================================================================
# Batch Processing
# ============================================================================

def reflections_to_insights(reflections: List[Dict]) -> List[Dict]:
    """
    Batch convert multiple reflections to insights.

    Convenience function for processing reflection datasets in bulk.

    Args:
        reflections: List of reflection dictionaries

    Returns:
        List of insight dictionaries (same order as input)

    Example:
        >>> reflections = [
        ...     {"reasoning": "...", "action": "check"},
        ...     {"reasoning": "...", "action": "book"}
        ... ]
        >>> insights = reflections_to_insights(reflections)
        >>> print(len(insights))
        2
    """
    return [reflection_to_insight(refl) for refl in reflections]
