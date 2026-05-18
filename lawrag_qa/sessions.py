"""In-memory session state for multi-turn legal consultations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Protocol
from uuid import uuid4

from .models import QueryFacts


@dataclass
class SessionState:
    session_id: str
    confirmed_facts: list[str] = field(default_factory=list)
    inferred_topics: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    turns: list[dict[str, object]] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: now_iso())


class SessionPersistence(Protocol):
    def load_sessions(self) -> dict[str, SessionState]:
        ...

    def upsert_session(self, session: SessionState) -> None:
        ...


class SessionStore:
    def __init__(self, persistence: SessionPersistence | None = None):
        self._persistence = persistence
        self._sessions: dict[str, SessionState] = persistence.load_sessions() if persistence else {}
        self._lock = RLock()

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        with self._lock:
            if not session_id:
                session_id = uuid4().hex
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session_id: str, query: str, facts: QueryFacts) -> SessionState:
        with self._lock:
            state = self.get_or_create(session_id)
            state.confirmed_facts = merge_unique(state.confirmed_facts, facts.confirmed)
            state.inferred_topics = merge_unique(state.inferred_topics, facts.inferred)
            state.missing_facts = [
                item for item in merge_unique(state.missing_facts, facts.missing) if item not in state.confirmed_facts
            ]
            state.turns.append(
                {
                    "query": query,
                    "intent": facts.intent,
                    "confirmed": facts.confirmed,
                    "missing": facts.missing,
                    "inferred": facts.inferred,
                    "at": now_iso(),
                }
            )
            state.updated_at = now_iso()
            if self._persistence:
                self._persistence.upsert_session(state)
            return state


def merge_query_facts(current: QueryFacts, session: SessionState | None) -> QueryFacts:
    if session is None:
        return current
    confirmed = merge_unique(session.confirmed_facts, current.confirmed)
    inferred = merge_unique(session.inferred_topics, current.inferred)
    missing = [item for item in merge_unique(current.missing, session.missing_facts) if item not in confirmed]
    return QueryFacts(
        domain=current.domain,
        intent=current.intent,
        confirmed=confirmed,
        missing=missing,
        inferred=inferred,
        risk_flags=current.risk_flags,
    )


def merge_unique(first: list[str], second: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in [*first, *second]:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
