"""Deterministic legal calculators for formula-like labor-law tasks."""

from __future__ import annotations

import re


def extract_months_and_salary(query: str) -> tuple[int | None, float | None]:
    months: int | None = None
    salary: float | None = None

    month_match = re.search(r"(\d+)\s*个?月", query)
    if month_match:
        months = int(month_match.group(1))

    salary_pattern = re.compile(r"(月薪|工资|每月)?\s*(\d+(?:\.\d+)?)\s*(k|K|千|元|块)?")
    for salary_match in salary_pattern.finditer(query):
        prefix = salary_match.group(1) or ""
        value = float(salary_match.group(2))
        unit = salary_match.group(3) or ""
        if unit in {"k", "K", "千"}:
            value *= 1000
        if prefix or unit or value >= 100:
            salary = value
            break

    return months, salary


def estimate_double_wage_gap(query: str) -> str | None:
    """Estimate unpaid written-contract double-wage difference.

    Simplified rule for MVP demonstration:
    - start after the first month of employment
    - cap at 11 months
    - return only an informational estimate, not a legal conclusion
    """

    months, salary = extract_months_and_salary(query)
    if months is None or salary is None or months <= 1:
        return None

    compensable_months = min(months - 1, 11)
    amount = compensable_months * salary
    return (
        f"按简化规则估算，若工作 {months} 个月、月工资约 {salary:.0f} 元，"
        f"未签书面劳动合同二倍工资差额的可计算月份约为 {compensable_months} 个月，"
        f"金额约 {amount:.0f} 元。实际金额仍需结合入职日期、工资口径、仲裁时效和当地裁审口径确认。"
    )
