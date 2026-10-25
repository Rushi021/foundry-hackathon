"""Founder agent — opinionated co-founder, never sugarcoats."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from foundry.config import get_settings
from foundry.llm import get_chat_model
from foundry.state import WorkflowState
from foundry.tools.research import build_research_tool

FOUNDER_SYSTEM = """You are the user's co-founder inside Foundry — not a cheerleader, not a chatbot.

Rules:
- Have your own opinions. Push back, negate, or refine when the idea is weak, crowded, vague, or poorly scoped.
- Never sugarcoat. Empty praise is a failure mode.
- When the idea has legs, brainstorm hard and be concrete about next moves.
- Use the `research` tool when you need market/competitor/factual grounding. Do not invent stats.
- Structure your reply clearly:
  1) Verdict posture: PUSH BACK | REFINE | PURSUE
  2) Why (blunt)
  3) Counters / risks
  4) Concrete next moves
  5) Research notes (only if you used research)

Speak as a peer co-founder who shares the goal of building something that works.
"""

MAX_TOOL_ROUNDS = 4


def founder_node(state: WorkflowState) -> dict:
    settings = get_settings()
    research = build_research_tool()
    tools = [research]
    model = get_chat_model(settings.founder_llm, temperature=0.4).bind_tools(tools)
    tool_node = ToolNode(tools)

    system = FOUNDER_SYSTEM
    context = state.get("user_context")
    if context:
        system += f"\n\nPersistent user/company context:\n{context}"

    messages: list = [SystemMessage(content=system), HumanMessage(content=state["user_prompt"])]

    for _ in range(MAX_TOOL_ROUNDS):
        ai: AIMessage = model.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            return {"final_response": _message_text(ai), "messages": messages[1:]}

        tool_result = tool_node.invoke({"messages": [ai]})
        tool_messages = tool_result["messages"]
        messages.extend(tool_messages)

    # Force a closing answer without tools if we hit the cap
    closing = get_chat_model(settings.founder_llm, temperature=0.4).invoke(
        messages
        + [
            HumanMessage(
                content="Stop calling tools. Give your final co-founder response now."
            )
        ]
    )
    messages.append(closing)
    return {"final_response": _message_text(closing), "messages": messages[1:]}


def _message_text(message: AIMessage | ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return str(content)
