"""FastAPI app for the legal RAG MVP."""

from pathlib import Path

from .ingestion import document_from_dict
from .models import LegalDocument
from .pipeline import LawRagPipeline


def create_app(pipeline: LawRagPipeline | None = None):
    try:
        from fastapi import FastAPI, Query
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install fastapi and uvicorn to run the API.") from exc

    app = FastAPI(title="LawRAG-QA MVP", version="0.1.0")
    rag = pipeline or LawRagPipeline.from_sample_and_corpus()

    class QARequest(BaseModel):
        query: str = Field(..., min_length=1)
        tenant_id: str = "public"
        session_id: str | None = None

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=1)
        top_k: int = 6
        tenant_id: str = "public"

    class DocumentPayload(BaseModel):
        doc_id: str | None = None
        title: str
        text: str
        doc_type: str = "statute"
        jurisdiction: str = "CN"
        status: str = "effective"
        source_url: str = ""
        metadata: dict[str, object] = Field(default_factory=dict)

    class ImportRequest(BaseModel):
        documents: list[DocumentPayload]
        persist: bool = True

    class FeedbackRequest(BaseModel):
        query: str = Field(..., min_length=1)
        issue_type: str = "general"
        comment: str = ""
        rating: int | None = None
        session_id: str | None = None

    class ResolveReviewRequest(BaseModel):
        note: str = ""

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = Path(__file__).parent / "static" / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(rag.documents), "chunks": len(rag.chunks)}

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
        answer, pack, verification, session, review = rag.ask(
            request.query,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
        )
        return {
            "status": "ok" if verification["ok"] else "needs_review",
            "session": serialize_session(session),
            "review": serialize_review(review),
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

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str):
        session = rag.sessions.get(session_id)
        if session is None:
            return {"status": "not_found", "session": None}
        return {"status": "ok", "session": serialize_session(session)}

    @app.post("/api/v1/documents/import")
    def import_documents(request: ImportRequest):
        documents: list[LegalDocument] = [
            document_from_dict(model_to_dict(payload)) for payload in request.documents
        ]
        summary = rag.add_documents(documents)
        if request.persist:
            rag.persist_documents()
        return {"status": "ok", **summary}

    @app.post("/api/v1/feedback")
    def feedback(request: FeedbackRequest):
        item = rag.reviews.add_feedback(
            query=request.query,
            issue_type=request.issue_type,
            comment=request.comment,
            rating=request.rating,
            session_id=request.session_id,
        )
        return {"status": "ok", "feedback": item.__dict__}

    @app.get("/api/v1/feedback")
    def list_feedback():
        return {"status": "ok", "items": rag.reviews.list_feedback()}

    @app.get("/api/v1/qa-logs")
    def qa_logs(limit: int = Query(50, ge=1, le=200)):
        return {"status": "ok", "items": rag.list_qa_logs(limit=limit)}

    @app.get("/api/v1/review-queue")
    def review_queue(status: str | None = None):
        return {"status": "ok", "items": rag.reviews.list_reviews(status=status)}

    @app.post("/api/v1/review-queue/{review_id}/resolve")
    def resolve_review(review_id: str, request: ResolveReviewRequest):
        item = rag.reviews.resolve(review_id, note=request.note)
        if item is None:
            return {"status": "not_found", "review": None}
        return {"status": "ok", "review": item}

    return app


app = create_app()


def serialize_session(session):
    if session is None:
        return None
    return {
        "session_id": session.session_id,
        "confirmed_facts": session.confirmed_facts,
        "inferred_topics": session.inferred_topics,
        "missing_facts": session.missing_facts,
        "turn_count": len(session.turns),
        "turns": session.turns[-10:],
        "updated_at": session.updated_at,
    }


def serialize_review(review):
    if review is None:
        return None
    return {
        "review_id": review.review_id,
        "status": review.status,
        "risk_reasons": review.risk_reasons,
        "created_at": review.created_at,
    }


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
