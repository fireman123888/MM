"""End-to-end legal RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict
from threading import RLock
from uuid import uuid4

from .config import Settings, load_settings
from .generator import AnswerGenerator
from .ingestion import load_documents, save_documents_jsonl
from .llm_client import OpenAICompatibleClient
from .models import EvidencePack, LegalDocument, QueryFacts, StructuredAnswer
from .query import classify_query
from .retrieval import InMemoryHybridRetriever
from .review import ReviewItem, ReviewStore
from .sessions import SessionState, SessionStore, merge_query_facts
from .splitter import split_documents
from .storage import SQLiteStore
from .verifier import verify_answer


class LawRagPipeline:
    def __init__(self, documents: list[LegalDocument], settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.documents = list(documents)
        self.storage = SQLiteStore(self.settings.db_path) if self.settings.db_path else None
        self.sessions = SessionStore(persistence=self.storage)
        self.reviews = ReviewStore(persistence=self.storage)
        self._lock = RLock()
        self._rebuild_index()
        llm_client = OpenAICompatibleClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )
        self.generator = AnswerGenerator(llm_client=llm_client, use_llm=self.settings.use_llm)

    def _rebuild_index(self) -> None:
        self.chunks = split_documents(self.documents)
        self.retriever = InMemoryHybridRetriever(self.chunks)

    def add_documents(self, documents: list[LegalDocument]) -> dict[str, int]:
        with self._lock:
            before_documents = len(self.documents)
            before_chunks = len(self.chunks)
            self.documents.extend(documents)
            self._rebuild_index()
            return {
                "documents_added": len(self.documents) - before_documents,
                "chunks_added": len(self.chunks) - before_chunks,
                "total_documents": len(self.documents),
                "total_chunks": len(self.chunks),
            }

    def add_documents_from_path(self, path: str) -> dict[str, int]:
        return self.add_documents(load_documents(path))

    def persist_documents(self, path: str | None = None) -> None:
        save_documents_jsonl(self.documents, path or self.settings.corpus_path)

    def build_evidence_pack(self, query: str, tenant_id: str = "public", session_id: str | None = None) -> EvidencePack:
        current_facts = classify_query(query)
        session = self.sessions.get_or_create(session_id) if session_id else None
        facts = merge_query_facts(current_facts, session)
        retrieval_query = self._retrieval_query(query, facts)
        with self._lock:
            evidence = self.retriever.search(
                query=retrieval_query,
                top_k=self.settings.retrieval_top_k,
                filters={"tenant_id": tenant_id, "status": "effective"},
            )
        notes = []
        degraded = False
        if not evidence:
            degraded = True
            notes.append("未检索到可用依据，回答应降级为依据不足。")
        if session_id:
            self.sessions.update(session_id, query, current_facts)
        return EvidencePack(query=query, facts=facts, evidence=evidence, degraded=degraded, notes=notes)

    def ask(
        self,
        query: str,
        tenant_id: str = "public",
        session_id: str | None = None,
    ) -> tuple[StructuredAnswer, EvidencePack, dict[str, object], SessionState | None, ReviewItem | None]:
        pack = self.build_evidence_pack(query, tenant_id=tenant_id, session_id=session_id)
        answer = self.generator.generate(pack)
        verification = verify_answer(answer, pack)
        session = self.sessions.get(session_id) if session_id else None
        answer_markdown = answer.to_markdown()
        review = None
        if verification.get("needs_review"):
            review = self.reviews.enqueue(
                query=query,
                answer_markdown=answer_markdown,
                verification=verification,
                session_id=session_id,
            )
        self._record_qa_log(query, answer_markdown, answer, pack, verification, session_id=session_id)
        return answer, pack, verification, session, review

    def list_qa_logs(self, limit: int = 50) -> list[dict[str, object]]:
        if not self.storage:
            return []
        return self.storage.list_qa_logs(limit=limit)

    def _record_qa_log(
        self,
        query: str,
        answer_markdown: str,
        answer: StructuredAnswer,
        pack: EvidencePack,
        verification: dict[str, object],
        session_id: str | None = None,
    ) -> None:
        if not self.storage:
            return
        self.storage.insert_qa_log(
            qa_id=f"qa_{uuid4().hex[:12]}",
            query=query,
            answer_markdown=answer_markdown,
            facts=asdict(pack.facts),
            citations=[asdict(citation) for citation in answer.citations],
            verification=verification,
            session_id=session_id,
        )

    def _retrieval_query(self, query: str, facts: QueryFacts) -> str:
        context = " ".join([*facts.confirmed, *facts.inferred])
        return f"{query} {context}".strip()

    @classmethod
    def from_sample(cls, settings: Settings | None = None) -> "LawRagPipeline":
        from .sample_data import sample_documents

        return cls(sample_documents(), settings=settings)

    @classmethod
    def from_sample_and_corpus(cls, settings: Settings | None = None) -> "LawRagPipeline":
        from pathlib import Path

        from .sample_data import sample_documents

        resolved_settings = settings or load_settings()
        documents = sample_documents()
        corpus_path = Path(resolved_settings.corpus_path)
        if corpus_path.exists():
            documents.extend(load_documents(corpus_path))
        return cls(documents, settings=resolved_settings)
