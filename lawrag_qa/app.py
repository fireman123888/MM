"""FastAPI app for the legal RAG MVP."""

from .pipeline import LawRagPipeline


def create_app(pipeline: LawRagPipeline | None = None):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install fastapi and uvicorn to run the API.") from exc

    app = FastAPI(title="LawRAG-QA MVP", version="0.1.0")
    rag = pipeline or LawRagPipeline.from_sample()

    class QARequest(BaseModel):
        query: str = Field(..., min_length=1)
        tenant_id: str = "public"

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=1)
        top_k: int = 6
        tenant_id: str = "public"

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(rag.chunks)}

    @app.post("/api/v1/search")
    def search(request: SearchRequest):
        evidence = rag.retriever.search(
            request.query,
            top_k=request.top_k,
            filters={"tenant_id": request.tenant_id, "status": "effective"},
        )
        return {
            "items": [
                {
                    "source_id": item.source_id,
                    "score": item.score,
                    "rank_reason": item.rank_reason,
                    "doc_title": item.chunk.title,
                    "article_no": item.chunk.article_no,
                    "text": item.chunk.text,
                    "source_url": item.chunk.source_url,
                }
                for item in evidence
            ]
        }

    @app.post("/api/v1/qa")
    def qa(request: QARequest):
        answer, pack, verification = rag.ask(request.query, tenant_id=request.tenant_id)
        return {
            "status": "ok" if verification["ok"] else "needs_review",
            "answer": answer.to_markdown(),
            "structured_answer": {
                "conclusion": answer.conclusion,
                "legal_basis": answer.legal_basis,
                "analysis": answer.analysis,
                "missing_facts": answer.missing_facts,
                "risk_tips": answer.risk_tips,
                "confidence": answer.confidence,
                "degraded": answer.degraded,
                "disclaimer": answer.disclaimer,
            },
            "citations": [
                {
                    "source_id": citation.source_id,
                    "doc_title": citation.doc_title,
                    "article_no": citation.article_no,
                    "text": citation.text,
                    "source_url": citation.source_url,
                }
                for citation in answer.citations
            ],
            "facts": {
                "domain": pack.facts.domain,
                "intent": pack.facts.intent,
                "confirmed": pack.facts.confirmed,
                "missing": pack.facts.missing,
                "inferred": pack.facts.inferred,
                "risk_flags": pack.facts.risk_flags,
            },
            "verification": verification,
        }

    return app


app = create_app()
