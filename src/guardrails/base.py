"""Shared guardrail primitives for domain validation.

Offers utilities for building deterministic calculators, canonical formatting,
and exact-match validation with structured logging. Domain modules extend
``NumericGuardrail`` to declare instructions and optional calculators.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
import logging
from typing import Callable, Optional
import re


getcontext().prec = 28


logger = logging.getLogger(__name__)


DecimalCalculator = Callable[[], Decimal]


def _quantizer(decimals: int) -> Decimal:
    """
    Create a Decimal quantizer for rounding to specified decimal places.

    Args:
        decimals: Number of decimal places

    Returns:
        Decimal quantizer (e.g., 0.01 for 2 decimals)
    """
    return Decimal(1).scaleb(-decimals)


def _format_value(value: Decimal, decimals: Optional[int]) -> str:
    """
    Format a Decimal value to string with optional decimal places.

    Args:
        value: The Decimal value to format
        decimals: Number of decimal places (None for automatic normalization)

    Returns:
        Formatted string representation of the value
    """
    if decimals is not None:
        quantizer = _quantizer(decimals)
        value = value.quantize(quantizer)
        return format(value, f".{decimals}f")

    normalized = value.normalize()
    return format(normalized, "f")


def _extract_final_token(text: str) -> Optional[str]:
    """
    Extract the last numeric token from text (supports decimals and percentages).

    Args:
        text: Input text to search for numeric values

    Returns:
        Last numeric token found, or None if no matches
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?%?", text)
    if not matches:
        return None
    return matches[-1]


def _to_decimal(value: str, value_format: str) -> Decimal:
    """
    Convert a string value to Decimal, handling format-specific preprocessing.

    Args:
        value: String value to convert
        value_format: Format type ("number", "percent", or "string")

    Returns:
        Decimal representation of the value

    Raises:
        InvalidOperation: If value cannot be converted to Decimal
    """
    cleaned = value.strip()
    if value_format == "percent":
        cleaned = cleaned.rstrip("%")
    return Decimal(cleaned)


@dataclass(frozen=True)
class NumericGuardrail:
    """Numeric guardrail with formatter and deterministic calculator."""

    instructions: str
    calculator: Optional[DecimalCalculator] = None
    format: str = "number"
    auto_correct: bool = False
    decimals: Optional[int] = None

    def validate(self, answer: str, ground_truth: str) -> bool:
        """Enforce exact-match scoring with optional extraction fallback."""

        normalized_answer = answer.strip()
        normalized_gt = ground_truth.strip()

        if normalized_answer == normalized_gt:
            self._log_formula_check(normalized_gt)
            return True

        extracted = _extract_final_token(normalized_answer)
        if extracted == normalized_gt:
            self._log_formula_check(normalized_gt)
            return True

        logger.warning(
            "guardrail_mismatch",
            extra={
                "expected": normalized_gt,
                "provided": normalized_answer,
                "extracted_token": extracted,
            },
        )
        self._log_formula_check(normalized_gt)
        return False

    def canonical_answer(self) -> Optional[str]:
        if not self.calculator:
            return None

        value = self.calculator()
        if isinstance(value, Decimal):
            text = _format_value(value, self.decimals)
            if self.format == "percent":
                return f"{text}%"
            return text
        return str(value)

    def parse_numeric(self, answer: str) -> Optional[Decimal]:
        if not answer:
            return None

        token = _extract_final_token(answer)
        if not token:
            return None

        cleaned = token.rstrip("%")
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None

    def _log_formula_check(self, ground_truth: str) -> None:
        if not self.calculator:
            return

        calculated = self.calculator()
        if not isinstance(calculated, Decimal):
            return

        try:
            gt_decimal = _to_decimal(ground_truth, self.format)
        except Exception:
            logger.warning(
                "guardrail_ground_truth_parse_error",
                extra={"ground_truth": ground_truth, "format": self.format},
            )
            return

        if abs(gt_decimal - calculated) > Decimal("0.01"):
            logger.warning(
                "guardrail_formula_deviation",
                extra={
                    "ground_truth": ground_truth,
                    "calculator_value": str(calculated),
                    "format": self.format,
                },
            )


def constant_guardrail(
    instructions: str,
    value: str,
    *,
    format: str = "number",
    decimals: Optional[int] = None,
) -> NumericGuardrail:
    """
    Create a guardrail that always returns a canonical value.

    Args:
        instructions: Guardrail validation instructions
        value: The canonical value to return
        format: Output format ("number" or "percent")
        decimals: Number of decimal places for formatting

    Returns:
        NumericGuardrail configured with the constant value

    Raises:
        ValueError: If value is empty or format is invalid
    """
    if not instructions:
        raise ValueError("instructions must be a non-empty string")
    if not value:
        raise ValueError("value must be a non-empty string")
    if format not in {"number", "percent", "string"}:
        raise ValueError(f"format must be 'number', 'percent', or 'string', got '{format}'")

    def _calculator() -> Decimal:
        text = value.rstrip("%")
        if format in {"number", "percent"}:
            try:
                return Decimal(text)
            except (InvalidOperation, ValueError) as e:
                raise ValueError(f"Invalid numeric value '{text}': {e}")
        return Decimal(0)  # Fallback for string format

    return NumericGuardrail(
        instructions=instructions,
        calculator=_calculator,
        format=format,
        auto_correct=True,
        decimals=decimals,
    )


__all__ = ["NumericGuardrail", "DecimalCalculator", "constant_guardrail"]
