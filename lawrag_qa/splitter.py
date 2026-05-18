"""Legal document splitters."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .models import LegalChunk, LegalDocument
from .text_utils import normalize_text


ARTICLE_PATTERN = re.compile(r"(第[一二三四五六七八九十百千万零〇0-9]+条)")
SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万零〇0-9]+[章节编]\s*(.+)?$")


def stable_id(*parts: str, prefix: str = "chunk") -> str:
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def split_statute(document: LegalDocument) -> list[LegalChunk]:
    """Split a statute-like document into article-level chunks."""

    text = normalize_text(document.text)
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return [
            LegalChunk(
                chunk_id=stable_id(document.doc_id, document.title, prefix="chunk"),
                doc_id=document.doc_id,
                title=document.title,
                text=text,
                doc_type=document.doc_type,
                jurisdiction=document.jurisdiction,
                status=document.status,
                source_url=document.source_url,
                tenant_id=document.metadata.get("tenant_id", "public"),
                metadata=dict(document.metadata),
            )
        ]

    chunks: list[LegalChunk] = []
    section_path = ""
    preamble = text[: matches[0].start()]
    for line in preamble.splitlines():
        section_match = SECTION_PATTERN.match(line.strip())
        if section_match:
            section_path = line.strip()

    for index, match in enumerate(matches):
        article_no = match.group(1)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk_text = normalize_text(text[start:end])
        if not chunk_text:
            continue
        chunk_id = stable_id(document.doc_id, article_no, chunk_text, prefix="law")
        chunks.append(
            LegalChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                title=document.title,
                article_no=article_no,
                section_path=section_path,
                text=chunk_text,
                doc_type=document.doc_type,
                jurisdiction=document.jurisdiction,
                status=document.status,
                source_url=document.source_url,
                tenant_id=document.metadata.get("tenant_id", "public"),
                metadata=dict(document.metadata),
            )
        )
    return chunks


def split_documents(documents: Iterable[LegalDocument]) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    for document in documents:
        if document.doc_type == "statute":
            chunks.extend(split_statute(document))
        else:
            chunks.extend(split_statute(document))
    return chunks
