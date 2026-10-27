"""LLM router — used only when Jev is unconfigured or fails (flagged in metrics)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from foundry.routing.models import (
    CLARIFY,
    CLARIFY_OPTION,
    COMPLEXITY_OPTIONS,
    NO_TOOL,
    RoutingDecision,
)
from foundry.text import message_text, usage_of


def _system_prompt(destinations: dict[str, str], tools: dict[str, str]) -> str:
    dest_lines = "\n".join(f"- {k}: {v}" for k, v in {**destinations, CLARIFY: CLARIFY_OPTION}.items())
    tool_lines = "\n".join(f"- {k}: {v}" for k, v in {NO_TOOL: "no external tool", **tools}.items())
    cx_lines = "\n".join(f"- {k}: {v}" for k, v in COMPLEXITY_OPTIONS.items())
    return f"""You are a request router. Classify; do not answer the request.
Interpret current_request in the context of recent_conversation.
Prefer the simplest destination that can fully handle the request.

destination options:
{dest_lines}

complexity options:
{cx_lines}

tool options:
{tool_lines}

Respond with ONLY a JSON object (no markdown fences):
{{"destination":"<option>","confidence":<0-1>,"complexity":"<option>","tool":"<option>",
"needs_memory":<0-1 probability the answer depends on the user's stored projects/goals/prior decisions>,
"needs_clarification":<0-1 probability essential info is missing and not inferable>,
"rationale":"<short why>"}}"""


def parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


class LLMDecisionProvider:
    name = "llm"

    def __init__(self, model_spec: str, model_factory: Callable[..., Any]) -> None:
        self._model_spec = model_spec
        self._factory = model_factory

    def decide(
        self,
        state: dict[str, Any],
        *,
        destinations: dict[str, str],
        tools: dict[str, str],
    ) -> RoutingDecision:
        model = self._factory(self._model_spec, purpose="router", temperature=0)
        ai = model.invoke(
            [
                SystemMessage(content=_system_prompt(destinations, tools)),
                HumanMessage(content=json.dumps(state, ensure_ascii=False)),
            ]
        )
        text = message_text(ai)
        try:
            data = parse_json_object(text)
            dest = data["destination"]
            if dest not in destinations and dest != CLARIFY:
                raise ValueError(f"unknown destination {dest!r}")
            tool = data.get("tool", NO_TOOL)
            complexity = data.get("complexity", "conversational")
            return RoutingDecision(
                destination=dest,
                confidence=float(data.get("confidence", 0.0)),
                complexity=complexity if complexity in COMPLEXITY_OPTIONS else "conversational",
                memory_probability=float(data.get("needs_memory", 0.0)),
                clarification_probability=float(data.get("needs_clarification", 0.0)),
                tool_hint=tool if tool in tools else NO_TOOL,
                provider=self.name,
                rationale=data.get("rationale"),
                usage=usage_of(ai),
            )
        except Exception as exc:
            raise ValueError(f"LLM router returned an invalid decision: {text[:400]}") from exc
