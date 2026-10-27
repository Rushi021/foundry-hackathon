"""FastAPI entrypoint — primary interface for web / other clients."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from foundry.config import get_settings
from foundry.graph import run_workflow
from foundry.memory import MemoryItem, MemoryStore

app = FastAPI(
    title="Foundry",
    description="AI company workflows — routed agents with shared and private memory",
    version="0.2.0",
)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, description="User message to route and handle")
    session_id: str | None = Field(
        default=None,
        description="Conversation id; omit to start a new conversation (returned in response)",
    )


class RunResponse(BaseModel):
    run_id: str
    session_id: str
    status: str
    route: str | None
    route_rationale: str | None
    route_confidence: float | None = None
    decision: dict[str, Any] | None = None
    response: str
    task: dict[str, Any]
    metrics: dict[str, Any]
    latency_ms: int


class MemoryCreate(BaseModel):
    scope: Literal["private", "shared"]
    content: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/workflows/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    try:
        # verbose=True prints each step to the API server terminal for verification
        result = run_workflow(request.prompt, session_id=request.session_id, verbose=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunResponse(**result)


# Memory endpoints are the user's own access (the user owns both scopes). Agents only
# reach memory through the runtime's access-controlled gateway.
@app.post("/v1/memory", response_model=MemoryItem)
def add_memory(body: MemoryCreate) -> MemoryItem:
    return MemoryStore(get_settings().foundry_data_dir).add(body.scope, body.content)


@app.get("/v1/memory/{scope}", response_model=list[MemoryItem])
def list_memory(scope: Literal["private", "shared"]) -> list[MemoryItem]:
    return MemoryStore(get_settings().foundry_data_dir).list(scope)
