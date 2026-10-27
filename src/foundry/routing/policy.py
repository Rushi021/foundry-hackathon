"""Application policy applied on top of the router's recommendation."""

from __future__ import annotations

from pydantic import BaseModel

from foundry.config import Settings
from foundry.routing.models import CLARIFY, RESOLVE, RoutingDecision

FALLBACK_DESTINATION = "direct"


class RoutePlan(BaseModel):
    destination: str
    preload_memory: bool
    reason: str


def plan_route(
    decision: RoutingDecision,
    *,
    route_pass: int,
    routable: set[str],
    settings: Settings,
    has_history: bool = True,
) -> RoutePlan:
    """Turn a decision into the next step.

    Uncertainty first goes to context resolution (once). After that the workflow never
    loops: it falls back to the Direct Assistant, which can ask or escalate itself.
    """
    preload = decision.memory_probability >= settings.router_memory_threshold
    known = decision.destination in routable
    confident = known and decision.confidence >= settings.router_min_confidence
    wants_clarification = (
        decision.destination == CLARIFY
        or decision.clarification_probability >= settings.router_clarify_threshold
    )
    # Resolution needs either missing information to ask about or history to resolve from.
    resolvable = wants_clarification or has_history

    if route_pass == 0 and (wants_clarification or not confident) and resolvable:
        reason = "clarification likely needed" if wants_clarification else "low routing confidence"
        return RoutePlan(destination=RESOLVE, preload_memory=preload, reason=reason)
    if confident:
        return RoutePlan(destination=decision.destination, preload_memory=preload, reason="router")
    return RoutePlan(
        destination=FALLBACK_DESTINATION,
        preload_memory=preload,
        reason="still uncertain after context resolution",
    )
