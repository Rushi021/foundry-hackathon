"""Long-term memory (basic): Private and Shared stores behind an access-controlled gateway.

Private Memory is shared exclusively between the user and the Co-Founder agent. The
gateway enforces this in code; prompts are never the access control.

Phase 1 scope: file-backed JSON stores, keyword retrieval, user-initiated writes only.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MemoryScope = Literal["private", "shared"]
SCOPES: tuple[MemoryScope, ...] = ("private", "shared")

# Hard boundary independent of registry configuration: a misconfigured agent spec
# cannot grant Private Memory to anyone else.
PRIVATE_MEMORY_AGENTS = frozenset({"cofounder"})

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from",
        "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "of", "on", "or", "our", "so", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "was", "we", "what", "when", "where", "which", "who",
        "why", "will", "with", "you", "your", "about", "should", "would", "could", "can",
    }
)


class MemoryAccessDenied(PermissionError):
    pass


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    scope: MemoryScope
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class MemoryHit(BaseModel):
    item: MemoryItem
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class MemoryStore:
    """One JSON file per scope under ``<data_dir>/memory``."""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir) / "memory"
        self._lock = threading.Lock()

    def _path(self, scope: MemoryScope) -> Path:
        if scope not in SCOPES:
            raise ValueError(f"Unknown memory scope {scope!r}")
        return self._dir / f"{scope}.json"

    def list(self, scope: MemoryScope) -> list[MemoryItem]:
        path = self._path(scope)
        if not path.exists():
            return []
        return [MemoryItem.model_validate(row) for row in json.loads(path.read_text())]

    def add(self, scope: MemoryScope, content: str) -> MemoryItem:
        item = MemoryItem(scope=scope, content=content.strip())
        with self._lock:
            items = self.list(scope)
            items.append(item)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path(scope).write_text(
                json.dumps([i.model_dump() for i in items], indent=2)
            )
        return item

    def search(self, scope: MemoryScope, query: str, k: int = 4) -> list[MemoryHit]:
        """Keyword-overlap retrieval; good enough until embeddings land."""
        q = set(_tokens(query))
        if not q:
            return []
        hits: list[MemoryHit] = []
        for item in self.list(scope):
            words = _tokens(item.content)
            overlap = len(q & set(words))
            if overlap:
                hits.append(MemoryHit(item=item, score=overlap / math.sqrt(len(words))))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


class MemoryGateway:
    """The only path agents have to long-term memory."""

    def __init__(self, store: MemoryStore, scopes_by_agent: dict[str, frozenset[str]]) -> None:
        self.store = store
        self._scopes_by_agent = scopes_by_agent

    def may_access(self, agent_id: str, scope: str) -> bool:
        if scope == "private" and agent_id not in PRIVATE_MEMORY_AGENTS:
            return False
        return scope in self._scopes_by_agent.get(agent_id, frozenset())

    def authorized_scopes(self, agent_id: str) -> list[MemoryScope]:
        return [s for s in SCOPES if self.may_access(agent_id, s)]

    def search(self, agent_id: str, scope: str, query: str, k: int = 4) -> list[MemoryHit]:
        if not self.may_access(agent_id, scope):
            raise MemoryAccessDenied(f"Agent '{agent_id}' is not authorized to read {scope} memory.")
        return self.store.search(scope, query, k=k)  # type: ignore[arg-type]


_SHINGLE = 6


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def contains_private_content(text: str, private_texts: list[str]) -> bool:
    """True if ``text`` reproduces any retrieved private memory (verbatim or 6-word runs)."""
    if not private_texts:
        return False
    words = re.findall(r"[a-z0-9]+", text.lower())
    joined = " ".join(words)
    text_shingles = _shingles(words)
    for private in private_texts:
        p_words = re.findall(r"[a-z0-9]+", private.lower())
        if not p_words:
            continue
        if len(p_words) < _SHINGLE:
            if " ".join(p_words) in joined:
                return True
        elif text_shingles & _shingles(p_words):
            return True
    return False
