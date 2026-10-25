"""FastAPI entrypoint — primary interface for web / other clients."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from foundry.graph import run_workflow

app = FastAPI(
    title="Foundry",
    description="AI consultancy workflows — founder agent POC",
    version="0.1.0",
)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, description="User prompt to route and handle")
    user_context: str | None = Field(
        default=None,
        description="Optional context hook for future persistent brain (unused in v0 logic)",
    )


class RunResponse(BaseModel):
    run_id: str
    route: str | None
    route_rationale: str | None
    route_confidence: float | None = None
    response: str
    latency_ms: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/workflows/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    try:
        # verbose=True prints each step to the API server terminal for verification
        result = run_workflow(
            request.prompt,
            user_context=request.user_context,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface as 500 for POC
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunResponse(**result)
