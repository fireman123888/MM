"""Citation and answer verification helpers."""

from __future__ import annotations

import re
from dataclasses import asdict

from .models import CitationSpan, ClaimCheck, EvidencePack, StructuredAnswer
from .text_utils import tokenize


CITATION_RE = re.compile(r"\[(S\d+)\]")
MIN_SUPPORT_SCORE = 0.08


def verify_answer(answer: StructuredAnswer, pack: EvidencePack) -> dict[str, object]:
    available = {item.source_id for item in pack.evidence}
    used = {citation.source_id for citation in answer.citations}
    missing = sorted(used - available)

    evidence_text = {item.source_id: item.chunk.text for item in pack.evidence}
    claim_checks = check_claims(answer, evidence_text, available)
    unsupported = [claim.text for claim in claim_checks if not claim.supported]
    citation_spans = [span for claim in claim_checks for span in claim.support_spans]

    risk_reasons: list[str] = []
    if missing:
        risk_reasons.append("missing_citation_sources")
    if unsupported:
        risk_reasons.append("unsupported_claims")
    if pack.degraded:
        risk_reasons.append("degraded_answer")
    if "high_risk_or_unsafe_request" in pack.facts.risk_flags:
        risk_reasons.append("high_risk_request")

    ok = not missing and not unsupported and not pack.degraded

    return {
        "ok": ok,
        "needs_review": not ok or bool(risk_reasons),
        "risk_reasons": risk_reasons,
        "missing_citations": missing,
        "unsupported_basis": unsupported,
        "used_citations": sorted(used),
        "claim_checks": [claim_to_dict(claim) for claim in claim_checks],
        "citation_spans": [asdict(span) for span in citation_spans],
    }


def check_claims(answer: StructuredAnswer, evidence_text: dict[str, str], available_sources: set[str]) -> list[ClaimCheck]:
    claims = extract_claims(answer)
    checks: list[ClaimCheck] = []
    for index, claim_text in enumerate(claims, start=1):
        claim_id = f"C{index}"
        cited_ids = [source_id for source_id in CITATION_RE.findall(claim_text) if source_id in available_sources]
        if not cited_ids and available_sources and claim_text.strip():
            checks.append(
                ClaimCheck(
                    claim_id=claim_id,
                    text=claim_text,
                    cited_source_ids=[],
                    supported=False,
                    support_score=0.0,
                    issue="missing_claim_citation",
                )
            )
            continue

        spans: list[CitationSpan] = []
        for source_id in cited_ids:
            span = best_support_span(claim_id, claim_text, source_id, evidence_text.get(source_id, ""))
            if span is not None:
                spans.append(span)

        best_score = max((span.support_score for span in spans), default=0.0)
        supported = best_score >= MIN_SUPPORT_SCORE if cited_ids else not available_sources
        checks.append(
            ClaimCheck(
                claim_id=claim_id,
                text=claim_text,
                cited_source_ids=cited_ids,
                supported=supported,
                support_score=round(best_score, 4),
                support_spans=spans,
                issue="" if supported else "citation_does_not_support_claim",
            )
        )
    return checks


def extract_claims(answer: StructuredAnswer) -> list[str]:
    raw_claims: list[str] = []
    raw_claims.extend(answer.legal_basis)
    raw_claims.extend(sentence for sentence in split_sentences(answer.conclusion) if not is_generic_claim(sentence))
    raw_claims.extend(sentence for sentence in split_sentences(answer.analysis) if CITATION_RE.search(sentence))
    return [normalize_claim(claim) for claim in raw_claims if normalize_claim(claim)]


def best_support_span(claim_id: str, claim: str, source_id: str, evidence: str) -> CitationSpan | None:
    if not evidence:
        return None
    best: CitationSpan | None = None
    for start, end, sentence in iter_sentence_spans(evidence):
        score = support_score(claim, sentence)
        if best is None or score > best.support_score:
            best = CitationSpan(
                source_id=source_id,
                claim_id=claim_id,
                start_char=start,
                end_char=end,
                text=sentence,
                support_score=round(score, 4),
            )
    return best


def support_score(claim: str, evidence_sentence: str) -> float:
    claim_tokens = meaningful_tokens(claim)
    evidence_tokens = meaningful_tokens(evidence_sentence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return len(overlap) / max(len(claim_tokens), 1)


def meaningful_tokens(text: str) -> set[str]:
    stop_tokens = {"s1", "s2", "s3", "s4", "s5", "s6", "第", "条", "的", "了", "和", "或", "及", "与"}
    return {token for token in tokenize(strip_citations(text)) if len(token) >= 2 and token not in stop_tokens}


def iter_sentence_spans(text: str):
    start = 0
    for match in re.finditer(r"[。；;!?！？\n]", text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            yield start, end, sentence
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            yield start, len(text), sentence


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for _, _, sentence in iter_sentence_spans(text):
        if CITATION_RE.fullmatch(sentence.strip()) and sentences:
            sentences[-1] = f"{sentences[-1]} {sentence.strip()}"
        else:
            sentences.append(sentence)
    return sentences


def normalize_claim(claim: str) -> str:
    return re.sub(r"\s+", " ", claim).strip()


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def is_generic_claim(text: str) -> bool:
    stripped = strip_citations(text)
    generic_markers = ["请结合", "一般性判断", "进一步判断", "仅供参考", "不构成正式法律意见"]
    return any(marker in stripped for marker in generic_markers)


def claim_to_dict(claim: ClaimCheck) -> dict[str, object]:
    data = asdict(claim)
    data["support_spans"] = [asdict(span) for span in claim.support_spans]
    return data
