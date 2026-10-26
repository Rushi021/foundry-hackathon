"""Chief of Agents — categorize and route the user prompt."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from foundry.config import get_settings
from foundry.llm import get_chat_model
from foundry.progress import emit
from foundry.state import RouteDecision, WorkflowState

CATEGORIZE_SYSTEM = """You are the Chief of Agents for Foundry, an AI consultancy.

Route the user prompt to exactly one specialist:

- founder: co-founder work — ideas, strategy, brainstorming, critique, product direction,
  "should I build X", market/positioning debates, pushback on plans
- lawyer: legal, contracts, compliance, IP
- sales: outbound, sales scripts, deal strategy
- gtm: go-to-market plans, launches, channels, positioning campaigns
- engineer: implementation, architecture, code, technical execution
- assistant: scheduling, reminders, personal admin, light logistics
- unsupported: unclear, off-topic, or cannot map confidently

Prefer "founder" when the user wants an opinionated partner on their company/idea.

Respond with ONLY a JSON object (no markdown fences) shaped as:
{"route":"<one of the routes above>","rationale":"<short why>","confidence":<0.0-1.0>}
"""


def categorize_node(state: WorkflowState) -> dict:
    settings = get_settings()
    run_id = state.get("run_id")
    # Parse JSON from text — ChatGPT plan usage streams Responses without a
    # LangChain `parsed` field, so with_structured_output fails on that path.
    model = get_chat_model(settings.categorize_llm, temperature=0)

    prompt = state["user_prompt"]
    context = state.get("user_context")
    user_content = prompt
    if context:
        user_content = f"User context (persistent brain):\n{context}\n\nPrompt:\n{prompt}"

    resolved = getattr(model, "model_name", None) or settings.categorize_llm
    emit("categorize.start", run_id=run_id, model=resolved)
    ai: AIMessage = model.invoke(
        [
            SystemMessage(content=CATEGORIZE_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    decision = _parse_route_decision(_message_text(ai))
    emit(
        "categorize.decision",
        run_id=run_id,
        route=decision.route,
        confidence=decision.confidence,
        rationale=decision.rationale,
    )

    return {
        "route": decision.route,
        "route_rationale": decision.rationale,
        "route_confidence": decision.confidence,
        "messages": [HumanMessage(content=prompt)],
    }


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return str(content)


def _parse_route_decision(text: str) -> RouteDecision:
    """Parse RouteDecision JSON from model text (fenced or raw)."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        return RouteDecision.model_validate(json.loads(raw))
    except Exception as exc:
        raise ValueError(
            f"Categorize model did not return valid RouteDecision JSON: {text[:400]}"
        ) from exc
