"""Data models shared by ingestion, retrieval, generation, and API layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LegalDocument:
    doc_id: str
    title: str
    text: str
    doc_type: str = "statute"
    jurisdiction: str = "CN"
    status: str = "effective"
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    article_no: str = ""
    section_path: str = ""
    doc_type: str = "statute"
    jurisdiction: str = "CN"
    status: str = "effective"
    source_url: str = ""
    tenant_id: str = "public"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedEvidence:
    source_id: str
    chunk: LegalChunk
    score: float
    rank_reason: str = ""
    supports: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryFacts:
    domain: str
    intent: str
    confirmed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    inferred: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidencePack:
    query: str
    facts: QueryFacts
    evidence: list[RetrievedEvidence]
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerCitation:
    source_id: str
    doc_title: str
    article_no: str
    text: str
    source_url: str = ""


@dataclass(frozen=True)
class StructuredAnswer:
    conclusion: str
    legal_basis: list[str]
    analysis: str
    missing_facts: list[str]
    risk_tips: list[str]
    citations: list[AnswerCitation]
    disclaimer: str
    confidence: str = "medium"
    degraded: bool = False

    def to_markdown(self) -> str:
        basis = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(self.legal_basis))
        missing = "\n".join(f"- {item}" for item in self.missing_facts) or "- 暂无"
        risks = "\n".join(f"- {item}" for item in self.risk_tips) or "- 暂无"
        return (
            "【初步结论】\n"
            f"{self.conclusion}\n\n"
            "【法律依据】\n"
            f"{basis or '暂无可引用依据'}\n\n"
            "【适用分析】\n"
            f"{self.analysis}\n\n"
            "【仍需补充的信息】\n"
            f"{missing}\n\n"
            "【风险提示】\n"
            f"{risks}\n\n"
            "【免责声明】\n"
            f"{self.disclaimer}"
        )
