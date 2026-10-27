"""Conversation sessions and context preparation (application logic, no model calls).

The goal is conversational continuity — enough recent context for the router and
agent to interpret follow-ups — not long-term memory retrieval.
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

RECENT_TURNS = 6
EARLIER_REQUESTS = 5
ROUTER_TURN_CHARS = 400
AGENT_TURN_CHARS = 2000
EARLIER_REQUEST_CHARS = 160

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent: str | None = None
    status: str | None = None
    task_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Session(BaseModel):
    id: str
    turns: list[Turn] = Field(default_factory=list)
    # Reused when present; Phase 1 does not generate summaries.
    summary: str | None = None


class SessionStore:
    """One JSON file per session under ``<data_dir>/sessions``."""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir) / "sessions"
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID.match(session_id):
            raise ValueError("session_id must be 1-64 chars of letters, digits, '-' or '_'")
        return self._dir / f"{session_id}.json"

    def load(self, session_id: str | None) -> Session:
        if not session_id:
            return Session(id=uuid.uuid4().hex)
        path = self._path(session_id)
        if path.exists():
            return Session.model_validate_json(path.read_text())
        return Session(id=session_id)

    def append(self, session_id: str, *turns: Turn) -> None:
        with self._lock:
            session = self.load(session_id)
            session.turns.extend(turns)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path(session_id).write_text(session.model_dump_json(indent=2))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class ConversationContext(BaseModel):
    session_id: str
    message: str
    recent: list[Turn] = Field(default_factory=list)
    earlier_requests: list[str] = Field(default_factory=list)
    summary: str | None = None
    last_agent: str | None = None
    active_task_id: str | None = None

    def router_state(self, request: str) -> dict:
        """Compact state for the decision model."""
        state: dict = {"current_request": request}
        if request != self.message:
            state["original_message"] = self.message
        if self.recent:
            state["recent_conversation"] = [
                f"{t.role}{f' ({t.agent})' if t.agent else ''}: {_clip(t.content, ROUTER_TURN_CHARS)}"
                for t in self.recent
            ]
        if self.summary:
            state["conversation_summary"] = self.summary
        if self.earlier_requests:
            state["earlier_requests"] = self.earlier_requests
        if self.last_agent:
            state["previous_turn_handled_by"] = self.last_agent
        return state

    def history_messages(self) -> list[BaseMessage]:
        """Recent turns as chat messages for the top-level executing agent."""
        out: list[BaseMessage] = []
        for t in self.recent:
            content = _clip(t.content, AGENT_TURN_CHARS)
            out.append(HumanMessage(content=content) if t.role == "user" else AIMessage(content=content))
        return out

    def background(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Conversation summary: {self.summary}")
        if self.earlier_requests:
            parts.append(
                "Earlier requests in this conversation:\n"
                + "\n".join(f"- {r}" for r in self.earlier_requests)
            )
        return "\n\n".join(parts)


def prepare_context(session: Session, message: str) -> ConversationContext:
    recent = session.turns[-RECENT_TURNS:]
    older = session.turns[:-RECENT_TURNS] if len(session.turns) > RECENT_TURNS else []
    earlier = [
        _clip(t.content, EARLIER_REQUEST_CHARS) for t in older if t.role == "user"
    ][-EARLIER_REQUESTS:]
    last_assistant = next((t for t in reversed(session.turns) if t.role == "assistant"), None)
    return ConversationContext(
        session_id=session.id,
        message=message,
        recent=recent,
        earlier_requests=[] if session.summary else earlier,
        summary=session.summary,
        last_agent=last_assistant.agent if last_assistant else None,
        active_task_id=last_assistant.task_id if last_assistant else None,
    )
