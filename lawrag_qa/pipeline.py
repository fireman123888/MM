"""End-to-end legal RAG pipeline."""

from __future__ import annotations

from .config import Settings, load_settings
from .generator import AnswerGenerator
from .llm_client import OpenAICompatibleClient
from .models import EvidencePack, LegalDocument, StructuredAnswer
from .query import classify_query
from .retrieval import InMemoryHybridRetriever
from .splitter import split_documents
from .verifier import verify_answer


class LawRagPipeline:
    def __init__(self, documents: list[LegalDocument], settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.chunks = split_documents(documents)
        self.retriever = InMemoryHybridRetriever(self.chunks)
        llm_client = OpenAICompatibleClient(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )
        self.generator = AnswerGenerator(llm_client=llm_client, use_llm=self.settings.use_llm)

    def build_evidence_pack(self, query: str, tenant_id: str = "public") -> EvidencePack:
        facts = classify_query(query)
        evidence = self.retriever.search(
            query=query,
            top_k=self.settings.retrieval_top_k,
            filters={"tenant_id": tenant_id, "status": "effective"},
        )
        notes = []
        degraded = False
        if not evidence:
            degraded = True
            notes.append("未检索到可用依据，回答应降级为依据不足。")
        return EvidencePack(query=query, facts=facts, evidence=evidence, degraded=degraded, notes=notes)

    def ask(self, query: str, tenant_id: str = "public") -> tuple[StructuredAnswer, EvidencePack, dict[str, object]]:
        pack = self.build_evidence_pack(query, tenant_id=tenant_id)
        answer = self.generator.generate(pack)
        verification = verify_answer(answer, pack)
        return answer, pack, verification

    @classmethod
    def from_sample(cls, settings: Settings | None = None) -> "LawRagPipeline":
        from .sample_data import sample_documents

        return cls(sample_documents(), settings=settings)
