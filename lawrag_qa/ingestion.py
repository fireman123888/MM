"""Document ingestion helpers for local files and API payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import LegalDocument
from .text_utils import normalize_text


SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".jsonl"}


def document_from_dict(payload: dict[str, object], default_doc_type: str = "statute") -> LegalDocument:
    """Create a LegalDocument from an API or JSON payload."""

    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("document text is required")

    title = str(payload.get("title") or "").strip() or "未命名法律文档"
    doc_id = str(payload.get("doc_id") or payload.get("id") or slug_id(title)).strip()
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return LegalDocument(
        doc_id=doc_id,
        title=title,
        text=normalize_text(text),
        doc_type=str(payload.get("doc_type") or default_doc_type),
        jurisdiction=str(payload.get("jurisdiction") or "CN"),
        status=str(payload.get("status") or "effective"),
        source_url=str(payload.get("source_url") or ""),
        metadata={str(key): value for key, value in metadata.items()},
    )


def load_documents(path: str | Path) -> list[LegalDocument]:
    """Load LegalDocument objects from a file or directory.

    Supported formats:
    - .txt/.md: first non-empty line becomes title when it starts with '#'
    - .json: one object or an array of objects
    - .jsonl: one document object per line
    """

    root = Path(path)
    if root.is_dir():
        docs: list[LegalDocument] = []
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.extend(load_documents(file_path))
        return docs
    if not root.exists():
        raise FileNotFoundError(str(root))

    suffix = root.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [_document_from_text_file(root)]
    if suffix == ".json":
        data = json.loads(root.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [document_from_dict(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [document_from_dict(data)]
        raise ValueError(f"unsupported JSON document shape: {root}")
    if suffix == ".jsonl":
        return [
            document_from_dict(json.loads(line))
            for line in root.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError(f"unsupported file type: {root.suffix}")


def save_documents_jsonl(documents: Iterable[LegalDocument], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for doc in documents:
        lines.append(
            json.dumps(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "doc_type": doc.doc_type,
                    "jurisdiction": doc.jurisdiction,
                    "status": doc.status,
                    "source_url": doc.source_url,
                    "metadata": doc.metadata,
                },
                ensure_ascii=False,
            )
        )
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _document_from_text_file(path: Path) -> LegalDocument:
    text = normalize_text(path.read_text(encoding="utf-8"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = path.stem
    if lines and lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip() or title
        text = normalize_text("\n".join(lines[1:]))

    return LegalDocument(
        doc_id=slug_id(path.stem),
        title=title,
        text=text,
        source_url=str(path),
        metadata={"tenant_id": "public", "source_file": str(path)},
    )


def slug_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "document"
