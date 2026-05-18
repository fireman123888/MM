"""Query understanding for the labor-law MVP."""

from __future__ import annotations

import re

from .calculators import extract_months_and_salary
from .models import QueryFacts


def classify_query(query: str) -> QueryFacts:
    text = query.lower()
    confirmed: list[str] = []
    inferred: list[str] = []
    missing: list[str] = []
    risk_flags: list[str] = []

    if "未签" in query or "没签" in query or "劳动合同" in query:
        inferred.append("可能涉及未签书面劳动合同责任")
    if "辞退" in query or "解除" in query or "开除" in query:
        inferred.append("可能涉及解除劳动合同或经济补偿")
    if "社保" in query or "社会保险" in query:
        inferred.append("可能涉及社会保险缴纳争议")
    if "试用期" in query:
        inferred.append("可能涉及试用期劳动合同规则")

    month_match = re.search(r"(\d+)\s*个?月", query)
    if month_match:
        confirmed.append(f"工作时长约 {month_match.group(1)} 个月")
    _, salary = extract_months_and_salary(query)
    if salary is not None:
        confirmed.append("用户提供了工资或金额信息")
    else:
        missing.append("月工资标准")

    if not re.search(r"\d{4}[-年]", query) and not re.search(r"\d+\s*月\s*\d+\s*日", query):
        missing.append("入职日期或争议发生时间")
    if not any(region in query for region in ["北京", "上海", "广州", "深圳", "杭州", "成都", "江苏", "浙江"]):
        missing.append("劳动合同履行地或所在城市")
    if "离职" not in query and "在职" not in query and ("辞退" in query or "解除" in query):
        missing.append("是否已经离职以及离职原因")

    if any(term in text for term in ["刑事", "伪造", "作假", "规避监管"]):
        risk_flags.append("high_risk_or_unsafe_request")
    if missing:
        risk_flags.append("missing_material_facts")

    intent = "labor_consultation"
    if "第" in query and "条" in query:
        intent = "article_lookup"
    elif "赔" in query or "补偿" in query or "工资" in query:
        intent = "compensation_consultation"

    return QueryFacts(
        domain="labor_law",
        intent=intent,
        confirmed=confirmed,
        missing=dedupe(missing),
        inferred=dedupe(inferred),
        risk_flags=dedupe(risk_flags),
    )


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
