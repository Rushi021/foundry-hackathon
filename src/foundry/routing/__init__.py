"""Decision routing: Jev by default, LLM fallback only when Jev is unavailable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foundry.config import Settings
from foundry.routing.jev import JevDecisionProvider
from foundry.routing.llm_fallback import LLMDecisionProvider
from foundry.routing.models import CLARIFY, RESOLVE, DecisionProvider, RoutingDecision
from foundry.routing.policy import RoutePlan, plan_route

__all__ = [
    "CLARIFY",
    "RESOLVE",
    "DecisionProvider",
    "RoutePlan",
    "RoutingDecision",
    "get_decision_providers",
    "plan_route",
]


def get_decision_providers(
    settings: Settings,
    model_factory: Callable[..., Any],
) -> tuple[DecisionProvider | None, DecisionProvider, str | None]:
    """Return ``(primary, fallback, reason_primary_unavailable)``.

    ``primary`` is Jev; ``fallback`` is the LLM router used if Jev is unconfigured or errors.
    """
    mode = (settings.router_provider or "auto").strip().lower()
    llm = LLMDecisionProvider(settings.categorize_llm, model_factory)
    if mode == "llm":
        return None, llm, "ROUTER_PROVIDER=llm"
    if mode not in {"auto", "jev"}:
        raise ValueError(f"Unknown ROUTER_PROVIDER={mode!r}. Use auto | jev | llm.")
    if not settings.jev_configured:
        if mode == "jev":
            raise ValueError("ROUTER_PROVIDER=jev but TYPESAFE_API_KEY is not set.")
        return None, llm, "TYPESAFE_API_KEY not set"
    jev = JevDecisionProvider(
        settings.typesafe_api_key,  # type: ignore[arg-type]
        model=settings.jev_model,
        base_url=settings.jev_base_url,
        timeout_s=settings.jev_timeout_s,
    )
    return jev, llm, None
