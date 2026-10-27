"""Shared LangGraph state for Foundry workflows."""

from __future__ import annotations

from typing import TypedDict

from foundry.context import ConversationContext
from foundry.routing.models import RoutingDecision
from foundry.runtime import TaskResult


class WorkflowState(TypedDict, total=False):
    run_id: str
    session_id: str | None
    # The user's message as received.
    user_message: str
    # The request being executed; a standalone rewrite after context resolution.
    request: str
    context: ConversationContext
    decision: RoutingDecision
    route_pass: int
    # Agent id, or a control step: "resolve", "reroute", "clarify".
    destination: str
    route_reason: str
    result: TaskResult
    final_response: str
    status: str
