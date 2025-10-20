"""Finance guardrails for benchmark validation."""

from __future__ import annotations

from decimal import Decimal, getcontext
import math
from typing import Dict, Optional

from .base import NumericGuardrail
from . import register_domain


getcontext().prec = 28


def _round_two(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _compound_interest(principal: Decimal, rate: Decimal, periods_per_year: int, years: int) -> Decimal:
    amount = principal * (Decimal(1) + rate / periods_per_year) ** (periods_per_year * years)
    return _round_two(amount - principal)


def _annualized_return(initial: Decimal, final: Decimal, years: Decimal) -> Decimal:
    growth = final / initial
    annualized = growth ** (Decimal(1) / years) - Decimal(1)
    return (annualized * Decimal(100)).quantize(Decimal("0.01"))


def _net_present_value(cashflows: tuple[Decimal, ...], discount_rate: Decimal) -> Decimal:
    total = Decimal(0)
    for idx, cashflow in enumerate(cashflows):
        total += cashflow / (Decimal(1) + discount_rate) ** idx
    return _round_two(total)


def _irr_newton(cashflows: tuple[Decimal, ...], initial_guess: Decimal = Decimal("0.1")) -> Decimal:
    rate = initial_guess
    for _ in range(100):
        npv = Decimal(0)
        derivative = Decimal(0)
        for period, cashflow in enumerate(cashflows):
            denominator = (Decimal(1) + rate) ** period
            npv += cashflow / denominator
            if period > 0:
                derivative -= (period * cashflow) / ((Decimal(1) + rate) ** (period + 1))
        if abs(npv) < Decimal("1e-8"):
            break
        if derivative == 0:
            break
        rate -= npv / derivative
    return (rate * Decimal(100)).quantize(Decimal("0.01"))


def _future_value_annuity(payment: Decimal, rate: Decimal, periods: int) -> Decimal:
    amount = payment * (((Decimal(1) + rate) ** periods - Decimal(1)) / rate)
    return _round_two(amount)


def _wacc(weights_equity: Decimal, cost_equity: Decimal, weights_debt: Decimal, cost_debt: Decimal) -> Decimal:
    value = weights_equity * cost_equity + weights_debt * cost_debt
    return (value * Decimal(100)).quantize(Decimal("0.01"))


def _cagr(initial: Decimal, final: Decimal, years: Decimal) -> Decimal:
    ratio = float(final / initial)
    exponent = float(Decimal(1) / years)
    growth = Decimal(math.pow(ratio, exponent) - 1)
    return (growth * Decimal(100)).quantize(Decimal("0.01"))


def _loan_payment(principal: Decimal, annual_rate: Decimal, years: Decimal, periods_per_year: int) -> Decimal:
    rate_per_period = annual_rate / Decimal(periods_per_year)
    total_periods = periods_per_year * int(years)
    numerator = principal * rate_per_period
    denominator = Decimal(1) - (Decimal(1) + rate_per_period) ** (-total_periods)
    payment = numerator / denominator
    return _round_two(payment)


def _percentage(value: Decimal) -> Decimal:
    return (value * Decimal(100)).quantize(Decimal("0.01"))


def _net_income(revenue: Decimal, expenses: Decimal) -> Decimal:
    return revenue - expenses


def _simple_interest(principal: Decimal, rate: Decimal, years: Decimal) -> Decimal:
    return principal * rate * years


def _break_even_units(fixed_costs: Decimal, unit_price: Decimal, unit_cost: Decimal) -> Decimal:
    return (fixed_costs / (unit_price - unit_cost)).quantize(Decimal("1"))


def _roi(return_amount: Decimal, cost: Decimal) -> Decimal:
    return ((return_amount - cost) / cost * Decimal(100)).quantize(Decimal("0.01"))


def _net_profit_margin(net_income: Decimal, revenue: Decimal) -> Decimal:
    return (net_income / revenue * Decimal(100)).quantize(Decimal("0.01"))


def _payback_period(initial_investment: Decimal, annual_inflow: Decimal) -> Decimal:
    return (initial_investment / annual_inflow).quantize(Decimal("1"))


def _ebitda(revenue: Decimal, cogs: Decimal, operating_expenses: Decimal) -> Decimal:
    return revenue - cogs - operating_expenses


def _working_capital(current_assets: Decimal, current_liabilities: Decimal) -> Decimal:
    return current_assets - current_liabilities


def _inventory_turnover(cogs: Decimal, average_inventory: Decimal) -> Decimal:
    return (cogs / average_inventory).quantize(Decimal("1"))


def _debt_to_equity(total_liabilities: Decimal, equity: Decimal) -> Decimal:
    return (total_liabilities / equity).quantize(Decimal("0.1"))


def _price_to_earnings(price: Decimal, eps: Decimal) -> Decimal:
    return (price / eps).quantize(Decimal("1"))


def _operating_leverage(contribution_margin: Decimal, operating_income: Decimal) -> Decimal:
    return (contribution_margin / operating_income).quantize(Decimal("1"))


def _retention_ratio(earnings: Decimal, dividends: Decimal) -> Decimal:
    return ((earnings - dividends) / earnings * Decimal(100)).quantize(Decimal("0.01"))


FINANCE_GUARDRAILS: Dict[str, NumericGuardrail] = {
    "fin-001": NumericGuardrail(
        instructions=(
            "Compute net income as revenue minus expenses using revenue 1200 and expenses 450. "
            "Return only the dollar amount with no additional text."
        ),
        calculator=lambda: _net_income(Decimal("1200"), Decimal("450")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-002": NumericGuardrail(
        instructions=(
            "Apply the CAGR formula ((final / initial) ** (1 / years) - 1) * 100 with initial value 100, final value 169, and years = 2. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."
        ),
        calculator=lambda: _cagr(Decimal("100"), Decimal("169"), Decimal("2")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-003": NumericGuardrail(
        instructions=(
            "Use simple interest P * r * t with principal 600, annual rate 5%, and duration 4 years. "
            "Return only the interest earned as a whole number."
        ),
        calculator=lambda: _simple_interest(Decimal("600"), Decimal("0.05"), Decimal("4")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-004": NumericGuardrail(
        instructions=(
            "Calculate break-even units as fixed_costs / (unit_price - unit_cost) with fixed costs 1500, price 50, and cost 20. "
            "Return only the whole number of units."
        ),
        calculator=lambda: _break_even_units(Decimal("1500"), Decimal("50"), Decimal("20")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-005": NumericGuardrail(
        instructions=(
            "Calculate straight-line depreciation using (cost - salvage) / useful_life with cost 1200, salvage 200, and useful life 4 years. "
            "Return only the annual depreciation as a whole number."
        ),
        calculator=lambda: (Decimal("1200") - Decimal("200")) / Decimal("4"),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-006": NumericGuardrail(
        instructions=(
            "Compute return on investment ((return - cost) / cost) * 100 with cost 800 and return 1040. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."
        ),
        calculator=lambda: _roi(Decimal("1040"), Decimal("800")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-007": NumericGuardrail(
        instructions=(
            "Calculate net profit margin (net income / revenue) * 100 with revenue 5000 and net income 750. "
            "Return only the percentage with a trailing % sign."
        ),
        calculator=lambda: _net_profit_margin(Decimal("750"), Decimal("5000")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-008": NumericGuardrail(
        instructions=(
            "Determine payback period as initial investment / annual cash inflow with investment 2000 and inflow 500. "
            "Return only the number of years as a whole number."
        ),
        calculator=lambda: _payback_period(Decimal("2000"), Decimal("500")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-009": NumericGuardrail(
        instructions=(
            "Use the compound interest formula A = P(1 + r/n)^(n*t). Compute the interest earned (A - P) and return only that value rounded to two decimals."
        ),
        calculator=lambda: _compound_interest(Decimal("400"), Decimal("0.06"), 4, 3),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-010": NumericGuardrail(
        instructions=(
            "Calculate EBITDA as revenue - COGS - operating expenses with revenue 8000, COGS 3000, and operating expenses 2500. "
            "Return only the dollar amount."
        ),
        calculator=lambda: _ebitda(Decimal("8000"), Decimal("3000"), Decimal("2500")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-011": NumericGuardrail(
        instructions=(
            "Compute working capital as current assets minus current liabilities with assets 4200 and liabilities 1600. "
            "Return only the dollar amount."
        ),
        calculator=lambda: _working_capital(Decimal("4200"), Decimal("1600")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-012": NumericGuardrail(
        instructions=(
            "Compute gross margin percentage as ((revenue - COGS) / revenue) * 100 with revenue 9000 and COGS 6300. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."
        ),
        calculator=lambda: _percentage((Decimal("9000") - Decimal("6300")) / Decimal("9000")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-013": NumericGuardrail(
        instructions=(
            "Calculate inventory turnover as COGS / average inventory with COGS 12000 and average inventory 2000. "
            "Return only the whole number."
        ),
        calculator=lambda: _inventory_turnover(Decimal("12000"), Decimal("2000")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-014": NumericGuardrail(
        instructions=(
            "Convert 18 months to 1.5 years and compute the annualized return ((final / initial) ** (1 / years) - 1) * 100. "
            "Carry full precision, round once at the end, and return only the percentage with two decimals and a trailing % sign."
        ),
        calculator=lambda: _annualized_return(Decimal("2000"), Decimal("2600"), Decimal("1.5")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-015": NumericGuardrail(
        instructions=(
            "Compute debt-to-equity ratio as total liabilities / shareholders' equity with liabilities 3500 and equity 2500. "
            "Return only the ratio rounded to one decimal place."
        ),
        calculator=lambda: _debt_to_equity(Decimal("3500"), Decimal("2500")),
        format="number",
        auto_correct=True,
        decimals=1,
    ),
    "fin-016": NumericGuardrail(
        instructions=(
            "Determine price-to-earnings ratio as share price / earnings per share with price 45 and EPS 3. "
            "Return only the whole number."
        ),
        calculator=lambda: _price_to_earnings(Decimal("45"), Decimal("3")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-017": NumericGuardrail(
        instructions=(
            "Calculate dividend yield as (dividend per share / share price) * 100 using dividend per share 2 and share price 40. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."
        ),
        calculator=lambda: _percentage(Decimal("2") / Decimal("40")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-018": NumericGuardrail(
        instructions=(
            "Discount each cash flow by (1 + 0.10)^period with period starting at 0. Sum the discounted values and report the net present value rounded to two decimals."
        ),
        calculator=lambda: _net_present_value(
            (Decimal("-5000"), Decimal("2000"), Decimal("2500"), Decimal("3000")), Decimal("0.10")
        ),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-019": NumericGuardrail(
        instructions=(
            "Solve for the internal rate of return that makes the net present value of [-3000, 1200, 1500, 1800] equal to zero. "
            "Return the IRR as a percentage with two decimals and a trailing % sign."
        ),
        calculator=lambda: _irr_newton((Decimal("-3000"), Decimal("1200"), Decimal("1500"), Decimal("1800"))),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-020": NumericGuardrail(
        instructions=(
            "Compute the fixed monthly payment for a $10,000 loan at 5% annual interest over 5 years with monthly compounding. "
            "Return only the payment rounded to two decimals (no currency symbol)."
        ),
        calculator=lambda: _loan_payment(Decimal("10000"), Decimal("0.05"), Decimal("5"), 12),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-021": NumericGuardrail(
        instructions=(
            "Compute the contribution margin ratio as ((sales price - variable cost) / sales price) * 100 using sales price 70 and variable cost 28. "
            "Return only the percentage as an integer with a trailing % sign."
        ),
        calculator=lambda: _percentage((Decimal("70") - Decimal("28")) / Decimal("70")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-022": NumericGuardrail(
        instructions=(
            "Determine operating leverage as contribution margin / operating income with values 4500 and 1500. "
            "Return only the whole number."
        ),
        calculator=lambda: _operating_leverage(Decimal("4500"), Decimal("1500")),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-023": NumericGuardrail(
        instructions=(
            "Compute the effective annual rate as ((1 + nominal_rate/12) ** 12 - 1) * 100 with nominal rate 0.09. "
            "Return only the percentage rounded to two decimals with a trailing % sign."
        ),
        calculator=lambda: ((Decimal(1) + Decimal("0.09") / Decimal(12)) ** 12 - Decimal(1)) * Decimal(100),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-024": NumericGuardrail(
        instructions=(
            "Use the ordinary annuity future value formula FV = payment * (((1 + r)^n - 1) / r) with payment 400, r = 0.04, and n = 8. "
            "Return only the future value rounded to two decimals."
        ),
        calculator=lambda: _future_value_annuity(Decimal("400"), Decimal("0.04"), 8),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-025": NumericGuardrail(
        instructions=(
            "Compute the weighted average cost of capital as 0.60 * 0.10 + 0.40 * 0.06 (no tax adjustment). "
            "Return the result as a percentage with two decimals and a trailing % sign."
        ),
        calculator=lambda: _wacc(Decimal("0.60"), Decimal("0.10"), Decimal("0.40"), Decimal("0.06")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-026": NumericGuardrail(
        instructions=(
            "Compute the retention ratio as (earnings - dividends) / earnings * 100 with earnings 1200 and dividends 300. "
            "Return only the percentage with a trailing % sign."
        ),
        calculator=lambda: _retention_ratio(Decimal("1200"), Decimal("300")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
}


def get_guardrail(task_id: str) -> Optional[NumericGuardrail]:
    return FINANCE_GUARDRAILS.get(task_id)


register_domain("finance", FINANCE_GUARDRAILS)


__all__ = ["FINANCE_GUARDRAILS", "get_guardrail"]
