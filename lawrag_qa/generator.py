"""Answer generation for the legal RAG MVP."""

from __future__ import annotations

import json

from .calculators import estimate_double_wage_gap
from .llm_client import ChatMessage, LLMError, OpenAICompatibleClient
from .models import AnswerCitation, EvidencePack, StructuredAnswer


DISCLAIMER = "本回答仅基于检索材料和你提供的信息作一般法律信息参考，不构成正式法律意见；具体处理建议咨询专业律师或当地劳动仲裁机构。"


class AnswerGenerator:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None, use_llm: bool = True):
        self.llm_client = llm_client
        self.use_llm = use_llm

    def generate(self, pack: EvidencePack) -> StructuredAnswer:
        if self.use_llm and self.llm_client is not None:
            try:
                return self._generate_with_llm(pack)
            except LLMError:
                return self._generate_rule_based(pack, degraded=True)
        return self._generate_rule_based(pack, degraded=pack.degraded)

    def _generate_with_llm(self, pack: EvidencePack) -> StructuredAnswer:
        raw = self.llm_client.chat(self._build_messages(pack))
        try:
            parsed = self._parse_json_response(raw)
        except LLMError:
            return self._generate_from_unstructured_llm_text(raw, pack)
        citations = self._citations_from_pack(pack)
        return StructuredAnswer(
            conclusion=parsed.get("conclusion", "").strip() or "依据检索材料，可以给出一般性分析，但仍需补充事实。",
            legal_basis=listify(parsed.get("legal_basis")),
            analysis=parsed.get("analysis", "").strip() or "请结合引用材料和补充事实进一步判断。",
            missing_facts=listify(parsed.get("missing_facts")) or pack.facts.missing,
            risk_tips=listify(parsed.get("risk_tips")) or self._default_risks(pack),
            citations=citations,
            disclaimer=parsed.get("disclaimer", "").strip() or DISCLAIMER,
            confidence=parsed.get("confidence", "medium"),
            degraded=pack.degraded,
        )

    def _generate_from_unstructured_llm_text(self, raw: str, pack: EvidencePack) -> StructuredAnswer:
        fallback = self._generate_rule_based(pack, degraded=pack.degraded)
        text = raw.strip()
        if not text:
            return fallback
        return StructuredAnswer(
            conclusion=fallback.conclusion,
            legal_basis=fallback.legal_basis,
            analysis=f"{text}\n\n{fallback.analysis}",
            missing_facts=fallback.missing_facts,
            risk_tips=fallback.risk_tips,
            citations=fallback.citations,
            disclaimer=fallback.disclaimer,
            confidence=fallback.confidence,
            degraded=pack.degraded,
        )

    def _build_messages(self, pack: EvidencePack) -> list[ChatMessage]:
        contexts = "\n\n".join(
            (
                f"[{item.source_id}]\n"
                f"标题：{item.chunk.title}\n"
                f"条号：{item.chunk.article_no}\n"
                f"效力：{item.chunk.status}\n"
                f"正文：{item.chunk.text}"
            )
            for item in pack.evidence
        )
        user = f"""用户问题：
{pack.query}

已识别事实：
确认事实：{pack.facts.confirmed}
缺失事实：{pack.facts.missing}
系统推断：{pack.facts.inferred}
风险标记：{pack.facts.risk_flags}

检索材料：
{contexts}

请只基于检索材料回答，输出 JSON，不要输出 Markdown。字段：
conclusion, legal_basis(list), analysis, missing_facts(list), risk_tips(list), confidence, disclaimer。
每个关键法律依据必须带 [S1] 这种引用编号。"""
        system = """你是法律信息检索与问答辅助系统。只能基于提供材料回答，不得编造法条、案例、案号或来源。事实不足时必须说明限制并提出需要补充的信息。回答不构成正式法律意见。"""
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def _parse_json_response(self, raw: str) -> dict[str, object]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM did not return JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM JSON response is not an object")
        return parsed

    def _generate_rule_based(self, pack: EvidencePack, degraded: bool = False) -> StructuredAnswer:
        citations = self._citations_from_pack(pack)
        basis = [
            f"{citation.doc_title}{citation.article_no}：{truncate(citation.text, 120)} [{citation.source_id}]"
            for citation in citations[:3]
        ]
        calculation = estimate_double_wage_gap(pack.query)
        conclusion = "根据已检索到的劳动法律材料，可以先作一般性判断："
        if any("未签书面劳动合同" in item for item in pack.facts.inferred):
            conclusion += "如果确实存在劳动关系且用人单位超过一个月未订立书面劳动合同，通常可能涉及二倍工资差额责任。"
        elif any("解除" in item or "经济补偿" in item for item in pack.facts.inferred):
            conclusion += "解除或终止劳动合同是否需要补偿，需要结合解除原因、工作年限和证据判断。"
        else:
            conclusion += "请结合下列法律依据和事实补充进一步判断。"

        analysis_parts = []
        if calculation:
            analysis_parts.append(calculation)
        if pack.facts.confirmed:
            analysis_parts.append("已识别事实：" + "；".join(pack.facts.confirmed) + "。")
        if pack.facts.missing:
            analysis_parts.append("由于仍缺少关键信息，当前只能给出一般规则，不能直接替代具体个案判断。")
        if not analysis_parts:
            analysis_parts.append("检索材料可作为初步分析依据，但仍需结合合同、工资流水、考勤记录、沟通记录等证据。")

        return StructuredAnswer(
            conclusion=conclusion,
            legal_basis=basis,
            analysis="".join(analysis_parts),
            missing_facts=pack.facts.missing,
            risk_tips=self._default_risks(pack),
            citations=citations,
            disclaimer=DISCLAIMER,
            confidence="low" if pack.facts.missing else "medium",
            degraded=degraded,
        )

    def _citations_from_pack(self, pack: EvidencePack) -> list[AnswerCitation]:
        return [
            AnswerCitation(
                source_id=item.source_id,
                doc_title=item.chunk.title,
                article_no=item.chunk.article_no,
                text=item.chunk.text,
                source_url=item.chunk.source_url,
            )
            for item in pack.evidence
        ]

    def _default_risks(self, pack: EvidencePack) -> list[str]:
        risks = ["劳动争议通常存在仲裁时效和举证要求，建议及时保存合同、工资流水、考勤、聊天记录等证据。"]
        if pack.facts.missing:
            risks.append("当前事实不足，金额、时效和责任判断可能随补充事实变化。")
        if "high_risk_or_unsafe_request" in pack.facts.risk_flags:
            risks.append("问题包含高风险或不适当请求，系统只能提供合法合规的一般信息。")
        return risks


def truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def listify(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
