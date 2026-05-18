"""SQLite persistence for sessions, feedback, reviews, and QA logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from .review import FeedbackItem, ReviewItem
from .sessions import SessionState, now_iso


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    confirmed_facts TEXT NOT NULL,
                    inferred_topics TEXT NOT NULL,
                    missing_facts TEXT NOT NULL,
                    turns TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS qa_logs (
                    qa_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    answer_markdown TEXT NOT NULL,
                    facts TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    verification TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_reasons TEXT NOT NULL,
                    verification TEXT NOT NULL,
                    answer_markdown TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_note TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    rating INTEGER,
                    issue_type TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_session(self, session: SessionState) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, confirmed_facts, inferred_topics, missing_facts, turns, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    confirmed_facts=excluded.confirmed_facts,
                    inferred_topics=excluded.inferred_topics,
                    missing_facts=excluded.missing_facts,
                    turns=excluded.turns,
                    updated_at=excluded.updated_at
                """,
                (
                    session.session_id,
                    dumps(session.confirmed_facts),
                    dumps(session.inferred_topics),
                    dumps(session.missing_facts),
                    dumps(session.turns),
                    session.updated_at,
                ),
            )

    def load_sessions(self) -> dict[str, SessionState]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions").fetchall()
        result: dict[str, SessionState] = {}
        for row in rows:
            result[row["session_id"]] = SessionState(
                session_id=row["session_id"],
                confirmed_facts=loads(row["confirmed_facts"], []),
                inferred_topics=loads(row["inferred_topics"], []),
                missing_facts=loads(row["missing_facts"], []),
                turns=loads(row["turns"], []),
                updated_at=row["updated_at"],
            )
        return result

    def insert_qa_log(
        self,
        qa_id: str,
        query: str,
        answer_markdown: str,
        facts: dict[str, Any],
        citations: list[dict[str, Any]],
        verification: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_logs(qa_id, session_id, query, answer_markdown, facts, citations, verification, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    qa_id,
                    session_id,
                    query,
                    answer_markdown,
                    dumps(facts),
                    dumps(citations),
                    dumps(verification),
                    now_iso(),
                ),
            )

    def list_qa_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row_to_dict(row, json_fields={"facts", "citations", "verification"}) for row in rows]

    def upsert_review(self, review: ReviewItem) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reviews(review_id, query, status, risk_reasons, verification, answer_markdown, session_id, created_at, resolved_at, resolution_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    status=excluded.status,
                    resolved_at=excluded.resolved_at,
                    resolution_note=excluded.resolution_note
                """,
                (
                    review.review_id,
                    review.query,
                    review.status,
                    dumps(review.risk_reasons),
                    dumps(review.verification),
                    review.answer_markdown,
                    review.session_id,
                    review.created_at,
                    review.resolved_at,
                    review.resolution_note,
                ),
            )

    def load_reviews(self) -> dict[str, ReviewItem]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM reviews").fetchall()
        return {
            row["review_id"]: ReviewItem(
                review_id=row["review_id"],
                query=row["query"],
                status=row["status"],
                risk_reasons=loads(row["risk_reasons"], []),
                verification=loads(row["verification"], {}),
                answer_markdown=row["answer_markdown"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                resolved_at=row["resolved_at"],
                resolution_note=row["resolution_note"],
            )
            for row in rows
        }

    def upsert_feedback(self, feedback: FeedbackItem) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback(feedback_id, query, rating, issue_type, comment, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback.feedback_id,
                    feedback.query,
                    feedback.rating,
                    feedback.issue_type,
                    feedback.comment,
                    feedback.session_id,
                    feedback.created_at,
                ),
            )

    def load_feedback(self) -> dict[str, FeedbackItem]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM feedback").fetchall()
        return {
            row["feedback_id"]: FeedbackItem(
                feedback_id=row["feedback_id"],
                query=row["query"],
                rating=row["rating"],
                issue_type=row["issue_type"],
                comment=row["comment"],
                session_id=row["session_id"],
                created_at=row["created_at"],
            )
            for row in rows
        }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def row_to_dict(row: sqlite3.Row, json_fields: set[str]) -> dict[str, Any]:
    data = dict(row)
    for field in json_fields:
        data[field] = loads(data.get(field, ""), json_default(field))
    return data


def json_default(field: str) -> Any:
    if field in {"facts", "verification"}:
        return {}
    return []
