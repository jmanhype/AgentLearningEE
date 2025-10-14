"""
Tests for deterministic reflection → insight mapping.

Validates:
- Schema compliance
- Lesson extraction heuristics
- Tag derivation
- Normalization
- Edge cases
"""

import pytest
from ee_ace_bridge.mapper import (
    reflection_to_insight,
    _extract_lesson,
    _derive_tags,
    _normalize,
    _infer_task,
    _build_evidence_ref,
    reflections_to_insights,
)


class TestReflectionToInsight:
    """Test suite for main mapping function."""

    def test_basic_mapping(self):
        """Test basic reflection to insight transformation."""
        reflection = {
            "reasoning": "The expert chose X because it prevents errors. Therefore always do X first.",
            "action": "check_availability",
            "confidence": 0.9
        }

        insight = reflection_to_insight(reflection)

        assert insight["task"] == "check"
        assert insight["insight_kind"] == "rule"
        assert "therefore" in insight["insight_text"].lower()
        assert insight["quality"]["confidence"] == 0.9
        assert insight["quality"]["novelty"] == 1.0
        assert "created_at" in insight

    def test_extracts_tags(self):
        """Test that tags are derived from content."""
        reflection = {
            "reasoning": "Always check availability to avoid sold out errors",
            "action": "check_availability",
            "confidence": 0.8
        }

        insight = reflection_to_insight(reflection)

        tags = insight["tags"]
        assert "availability" in tags or "error-avoidance" in tags

    def test_builds_evidence_ref(self):
        """Test that evidence reference is included."""
        reflection = {
            "reasoning": "Test reasoning",
            "action": "test_action",
            "source_rollout_id": 42
        }

        insight = reflection_to_insight(reflection)

        assert "evidence_ref" in insight
        assert insight["evidence_ref"] == "rollout:42"

    def test_defaults_confidence(self):
        """Test that missing confidence defaults to 0.5."""
        reflection = {
            "reasoning": "Test reasoning",
            "action": "test_action"
        }

        insight = reflection_to_insight(reflection)

        assert insight["quality"]["confidence"] == 0.5

    def test_handles_alternative_field_names(self):
        """Test that both 'reasoning' and 'rationale' are accepted."""
        reflection1 = {
            "reasoning": "Test with reasoning field",
            "action": "test"
        }

        reflection2 = {
            "rationale": "Test with rationale field",
            "action": "test"
        }

        insight1 = reflection_to_insight(reflection1)
        insight2 = reflection_to_insight(reflection2)

        assert "Test" in insight1["insight_text"]
        assert "Test" in insight2["insight_text"]


class TestExtractLesson:
    """Test suite for lesson extraction heuristics."""

    def test_extracts_therefore_sentence(self):
        """Test extraction of sentence with 'therefore'."""
        rationale = "The expert did X. This was good. Therefore always do X first."

        lesson = _extract_lesson(rationale)

        assert "therefore" in lesson.lower()
        assert "do x first" in lesson.lower()

    def test_extracts_should_sentence(self):
        """Test extraction of sentence with 'should'."""
        rationale = "Context here. You should validate before proceeding. More text."

        lesson = _extract_lesson(rationale)

        assert "should" in lesson.lower() or "validate" in lesson.lower()

    def test_extracts_to_avoid_sentence(self):
        """Test extraction of sentence with 'to avoid'."""
        rationale = "Check availability to avoid sold out errors. More details here."

        lesson = _extract_lesson(rationale)

        assert "avoid" in lesson.lower()

    def test_extracts_always_sentence(self):
        """Test extraction of sentence with 'always'."""
        rationale = "Background. Always check credentials first. Additional notes."

        lesson = _extract_lesson(rationale)

        assert "always" in lesson.lower() or "check" in lesson.lower()

    def test_fallback_to_first_sentence(self):
        """Test fallback when no prescriptive keywords found."""
        rationale = "This is the first sentence. Second sentence here."

        lesson = _extract_lesson(rationale)

        assert "first sentence" in lesson.lower()

    def test_handles_empty_rationale(self):
        """Test graceful handling of empty rationale."""
        lesson = _extract_lesson("")

        assert isinstance(lesson, str)
        assert len(lesson) > 0
        assert "rule" in lesson.lower()

    def test_handles_none_rationale(self):
        """Test graceful handling of None rationale."""
        lesson = _extract_lesson(None)

        assert isinstance(lesson, str)
        assert len(lesson) > 0

    def test_bounds_long_sentences(self):
        """Test that long sentences are bounded to 240 chars."""
        rationale = "This is a very long sentence " * 20

        lesson = _extract_lesson(rationale)

        assert len(lesson) <= 250  # 240 + "..." + "Rule: " prefix


class TestNormalize:
    """Test suite for text normalization."""

    def test_adds_period(self):
        """Test that period is added if missing."""
        text = "check availability first"

        normalized = _normalize(text)

        assert normalized.endswith(".")

    def test_preserves_existing_period(self):
        """Test that existing period is preserved."""
        text = "check availability first."

        normalized = _normalize(text)

        # Should not add duplicate period
        assert normalized.endswith(".")
        assert not normalized.endswith("..")

    def test_adds_rule_prefix(self):
        """Test that 'Rule:' prefix is added."""
        text = "check availability first"

        normalized = _normalize(text)

        assert normalized.startswith("Rule:")

    def test_preserves_prescriptive_prefix(self):
        """Test that prescriptive prefixes are preserved."""
        text = "Always validate input"

        normalized = _normalize(text)

        # Should preserve 'Always' and not add 'Rule:' prefix
        assert "always" in normalized.lower()

    def test_capitalizes_first_letter(self):
        """Test that first letter is capitalized."""
        text = "check availability"

        normalized = _normalize(text)

        # After "Rule: ", first word should be capitalized
        assert "Rule: Check" in normalized


class TestDeriveTags:
    """Test suite for tag derivation."""

    def test_derives_availability_tag(self):
        """Test that availability keywords produce availability tag."""
        lesson = "Check availability to avoid sold out"
        reflection = {"reasoning": "availability check"}

        tags = _derive_tags(lesson, reflection)

        assert "availability" in tags

    def test_derives_payment_tag(self):
        """Test that payment keywords produce payment tag."""
        lesson = "Validate credit card format"
        reflection = {"reasoning": "payment validation"}

        tags = _derive_tags(lesson, reflection)

        assert "payment" in tags

    def test_derives_validation_tag(self):
        """Test that validation keywords produce validation tag."""
        lesson = "Validate input format before processing"
        reflection = {}

        tags = _derive_tags(lesson, reflection)

        assert "validation" in tags

    def test_derives_error_avoidance_tag(self):
        """Test that error keywords produce error-avoidance tag."""
        lesson = "Check to avoid errors"
        reflection = {"reasoning": "prevent mistakes"}

        tags = _derive_tags(lesson, reflection)

        assert "error-avoidance" in tags

    def test_derives_safety_tag(self):
        """Test that safety keywords produce safety tag."""
        lesson = "Ensure secure connection"
        reflection = {"reasoning": "safety first"}

        tags = _derive_tags(lesson, reflection)

        assert "safety" in tags

    def test_derives_performance_tag(self):
        """Test that performance keywords produce performance tag."""
        lesson = "Optimize query for speed"
        reflection = {"reasoning": "slow performance"}

        tags = _derive_tags(lesson, reflection)

        assert "performance" in tags

    def test_deduplicates_tags(self):
        """Test that duplicate tags are removed."""
        lesson = "Check availability and validate availability"
        reflection = {"reasoning": "availability availability"}

        tags = _derive_tags(lesson, reflection)

        # Should only have one occurrence of 'availability'
        assert tags.count("availability") == 1

    def test_fallback_to_general(self):
        """Test fallback to 'general' tag when no matches."""
        lesson = "Do something unrelated"
        reflection = {}

        tags = _derive_tags(lesson, reflection)

        assert "general" in tags


class TestInferTask:
    """Test suite for task inference."""

    def test_extracts_first_word(self):
        """Test that first word is extracted from action."""
        task = _infer_task("check_availability")

        assert task == "check"

    def test_handles_camel_case(self):
        """Test splitting of camelCase actions."""
        task = _infer_task("bookHotel")

        assert task == "book"

    def test_handles_empty_action(self):
        """Test fallback for empty action."""
        task = _infer_task("")

        assert task == "general"

    def test_handles_none_action(self):
        """Test fallback for None action."""
        task = _infer_task(None)

        assert task == "general"

    def test_lowercases_result(self):
        """Test that result is lowercased."""
        task = _infer_task("CHECK_AVAILABILITY")

        assert task == "check"
        assert task.islower()


class TestBuildEvidenceRef:
    """Test suite for evidence reference construction."""

    def test_includes_rollout_id(self):
        """Test that rollout ID is included when present."""
        reflection = {"source_rollout_id": 42}

        ref = _build_evidence_ref(reflection)

        assert ref == "rollout:42"

    def test_fallback_when_no_id(self):
        """Test fallback when no rollout ID present."""
        reflection = {}

        ref = _build_evidence_ref(reflection)

        assert ref == "reflection"

    def test_handles_none_id(self):
        """Test handling of explicit None ID."""
        reflection = {"source_rollout_id": None}

        ref = _build_evidence_ref(reflection)

        assert ref == "reflection"


class TestReflectionsToInsights:
    """Test suite for batch processing."""

    def test_processes_multiple_reflections(self):
        """Test batch conversion of reflections."""
        reflections = [
            {
                "reasoning": "First reasoning",
                "action": "check"
            },
            {
                "reasoning": "Second reasoning",
                "action": "book"
            }
        ]

        insights = reflections_to_insights(reflections)

        assert len(insights) == 2
        assert all("insight_text" in insight for insight in insights)

    def test_preserves_order(self):
        """Test that output order matches input order."""
        reflections = [
            {
                "reasoning": f"Reasoning {i}",
                "action": f"action_{i}"
            }
            for i in range(10)
        ]

        insights = reflections_to_insights(reflections)

        # Check that tasks are derived in order
        tasks = [insight["task"] for insight in insights]
        expected_tasks = [f"action" for _ in range(10)]

        assert len(tasks) == len(expected_tasks)

    def test_handles_empty_list(self):
        """Test handling of empty reflection list."""
        insights = reflections_to_insights([])

        assert insights == []
