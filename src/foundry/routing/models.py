"""Router contracts, independent of the decision provider."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

CLARIFY = "clarify"
RESOLVE = "resolve"
NO_TOOL = "none"

Complexity = Literal["conversational", "single_agent", "multi_step", "multi_agent"]

COMPLEXITY_OPTIONS: dict[str, str] = {
    "conversational": "A straightforward question or conversational request.",
    "single_agent": "One specialized task a single expert can complete.",
    "multi_step": "An operational workflow with several dependent steps.",
    "multi_agent": "Needs several different experts or teams coordinated together.",
}

CLARIFY_OPTION = (
    "The request cannot be interpreted: it depends on something not established in the "
    "conversation (e.g. an unknown person, file or project)."
)


class RoutingDecision(BaseModel):
    """Structured routing output. A recommendation — never an authorization."""

    destination: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    complexity: Complexity = "conversational"
    memory_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_hint: str = NO_TOOL
    provider: str
    rationale: str | None = None
    usage: dict[str, Any] | None = None


class DecisionProvider(Protocol):
    name: str

    def decide(
        self,
        state: dict[str, Any],
        *,
        destinations: dict[str, str],
        tools: dict[str, str],
    ) -> RoutingDecision:
        """Classify the request. ``destinations``/``tools`` map option id → description."""
