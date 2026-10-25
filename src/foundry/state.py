"""Shared LangGraph state for Foundry workflows."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

AgentRoute = Literal[
    "founder",
    "lawyer",
    "sales",
    "gtm",
    "engineer",
    "assistant",
    "unsupported",
]

FUTURE_AGENTS = frozenset({"lawyer", "sales", "gtm", "engineer", "assistant"})


class RouteDecision(BaseModel):
    route: AgentRoute
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_prompt: str
    # Hook for future persistent company brain — unused in v0
    user_context: str | None
    route: AgentRoute
    route_rationale: str
    route_confidence: float
    run_id: str
    final_response: str
    metadata: dict[str, Any]
