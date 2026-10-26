"""Founder agent — opinionated co-founder, never sugarcoats."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from foundry.config import get_settings
from foundry.llm import get_chat_model
from foundry.progress import emit
from foundry.state import WorkflowState
from foundry.tools.research import build_research_tool

FOUNDER_SYSTEM = """You are the user's co-founder inside Foundry — not a cheerleader, not a chatbot.

Rules:
- Have your own opinions. Push back, negate, or refine when the idea is weak, crowded, or poorly scoped.
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
    run_id = state.get("run_id")
    research = build_research_tool(run_id=run_id)
    tools = [research]
    model = get_chat_model(settings.founder_llm, temperature=0.4).bind_tools(tools)
    tool_node = ToolNode(tools)

    system = FOUNDER_SYSTEM
    context = state.get("user_context")
    if context:
        system += f"\n\nPersistent user/company context:\n{context}"

    resolved = getattr(model, "model_name", None) or settings.founder_llm
    emit("founder.start", run_id=run_id, model=resolved)
    messages: list = [SystemMessage(content=system), HumanMessage(content=state["user_prompt"])]

    for round_idx in range(MAX_TOOL_ROUNDS):
        emit("founder.llm", run_id=run_id, round=round_idx + 1)
        ai: AIMessage = model.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            text = _message_text(ai)
            emit("founder.response", run_id=run_id, response=text)
            return {"final_response": text, "messages": messages[1:]}

        for call in ai.tool_calls:
            emit(
                "founder.tool_call",
                run_id=run_id,
                tool=call.get("name"),
                args=call.get("args"),
            )

        tool_result = tool_node.invoke({"messages": [ai]})
        tool_messages = tool_result["messages"]
        for tm in tool_messages:
            if isinstance(tm, ToolMessage):
                emit(
                    "founder.tool_result",
                    run_id=run_id,
                    tool=getattr(tm, "name", None),
                    result=_message_text(tm),
                )
        messages.extend(tool_messages)

    emit("founder.force_close", run_id=run_id, reason="max_tool_rounds")
    closing = get_chat_model(settings.founder_llm, temperature=0.4).invoke(
        messages
        + [
            HumanMessage(
                content="Stop calling tools. Give your final co-founder response now."
            )
        ]
    )
    messages.append(closing)
    text = _message_text(closing)
    emit("founder.response", run_id=run_id, response=text)
    return {"final_response": text, "messages": messages[1:]}


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
