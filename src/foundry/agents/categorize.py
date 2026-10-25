"""Chief of Agents — categorize and route the user prompt."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from foundry.config import get_settings
from foundry.llm import get_chat_model
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
Return structured output only.
"""


def categorize_node(state: WorkflowState) -> dict:
    settings = get_settings()
    model = get_chat_model(settings.categorize_llm, temperature=0).with_structured_output(
        RouteDecision
    )

    prompt = state["user_prompt"]
    context = state.get("user_context")
    user_content = prompt
    if context:
        user_content = f"User context (persistent brain):\n{context}\n\nPrompt:\n{prompt}"

    decision: RouteDecision = model.invoke(
        [
            SystemMessage(content=CATEGORIZE_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )

    return {
        "route": decision.route,
        "route_rationale": decision.rationale,
        "route_confidence": decision.confidence,
        "messages": [HumanMessage(content=prompt)],
    }
