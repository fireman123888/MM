"""Citation and answer verification helpers."""

from __future__ import annotations

import re

from .models import EvidencePack, StructuredAnswer
from .text_utils import tokenize


def verify_answer(answer: StructuredAnswer, pack: EvidencePack) -> dict[str, object]:
    available = {item.source_id for item in pack.evidence}
    used = {citation.source_id for citation in answer.citations}
    missing = sorted(used - available)

    unsupported: list[str] = []
    evidence_text = {item.source_id: item.chunk.text for item in pack.evidence}
    for basis in answer.legal_basis:
        cited_ids = set(re.findall(r"\[(S\d+)\]", basis))
        if not cited_ids:
            unsupported.append(basis)
            continue
        basis_tokens = set(tokenize(basis))
        if not any(len(basis_tokens & set(tokenize(evidence_text.get(source_id, "")))) >= 2 for source_id in cited_ids):
            unsupported.append(basis)

    return {
        "ok": not missing and not unsupported,
        "missing_citations": missing,
        "unsupported_basis": unsupported,
        "used_citations": sorted(used),
    }
