"""Feedback and review queue primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Protocol
from uuid import uuid4

from .sessions import now_iso


@dataclass
class ReviewItem:
    review_id: str
    query: str
    status: str
    risk_reasons: list[str]
    verification: dict[str, object]
    answer_markdown: str
    session_id: str | None = None
    created_at: str = field(default_factory=now_iso)
    resolved_at: str | None = None
    resolution_note: str = ""


@dataclass
class FeedbackItem:
    feedback_id: str
    query: str
    rating: int | None
    issue_type: str
    comment: str
    session_id: str | None = None
    created_at: str = field(default_factory=now_iso)


class ReviewPersistence(Protocol):
    def load_reviews(self) -> dict[str, ReviewItem]:
        ...

    def upsert_review(self, review: ReviewItem) -> None:
        ...

    def load_feedback(self) -> dict[str, FeedbackItem]:
        ...

    def upsert_feedback(self, feedback: FeedbackItem) -> None:
        ...


class ReviewStore:
    def __init__(self, persistence: ReviewPersistence | None = None):
        self._persistence = persistence
        self._reviews: dict[str, ReviewItem] = persistence.load_reviews() if persistence else {}
        self._feedback: dict[str, FeedbackItem] = persistence.load_feedback() if persistence else {}
        self._lock = RLock()

    def enqueue(
        self,
        query: str,
        answer_markdown: str,
        verification: dict[str, object],
        session_id: str | None = None,
    ) -> ReviewItem:
        with self._lock:
            review = ReviewItem(
                review_id=f"rev_{uuid4().hex[:12]}",
                query=query,
                status="open",
                risk_reasons=[str(item) for item in verification.get("risk_reasons", [])],
                verification=verification,
                answer_markdown=answer_markdown,
                session_id=session_id,
            )
            self._reviews[review.review_id] = review
            if self._persistence:
                self._persistence.upsert_review(review)
            return review

    def list_reviews(self, status: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            reviews = list(self._reviews.values())
            if status:
                reviews = [item for item in reviews if item.status == status]
            return [asdict(item) for item in sorted(reviews, key=lambda item: item.created_at, reverse=True)]

    def resolve(self, review_id: str, note: str = "") -> dict[str, object] | None:
        with self._lock:
            review = self._reviews.get(review_id)
            if review is None:
                return None
            review.status = "resolved"
            review.resolved_at = now_iso()
            review.resolution_note = note
            if self._persistence:
                self._persistence.upsert_review(review)
            return asdict(review)

    def add_feedback(
        self,
        query: str,
        issue_type: str,
        comment: str = "",
        rating: int | None = None,
        session_id: str | None = None,
    ) -> FeedbackItem:
        with self._lock:
            feedback = FeedbackItem(
                feedback_id=f"fb_{uuid4().hex[:12]}",
                query=query,
                rating=rating,
                issue_type=issue_type,
                comment=comment,
                session_id=session_id,
            )
            self._feedback[feedback.feedback_id] = feedback
            if self._persistence:
                self._persistence.upsert_feedback(feedback)
            return feedback

    def list_feedback(self) -> list[dict[str, object]]:
        with self._lock:
            return [asdict(item) for item in sorted(self._feedback.values(), key=lambda item: item.created_at, reverse=True)]
