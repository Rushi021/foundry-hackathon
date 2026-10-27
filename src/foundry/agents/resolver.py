"""Context resolution — runs only when routing is uncertain or information seems missing."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from foundry.context import ConversationContext
from foundry.routing.llm_fallback import parse_json_object
from foundry.runtime import RunContext
from foundry.text import message_text

RESOLVER_SYSTEM = """You resolve what the user's latest message refers to, using the recent conversation.

- If the conversation establishes what the message means, rewrite it as one standalone
  request that preserves the user's intent exactly.
- If essential information is genuinely missing, write one short clarifying question.
- Never invent people, files, projects or facts.

Respond with ONLY a JSON object (no markdown fences):
{"resolved": true|false, "standalone_request": "<rewritten request or empty>", "clarifying_question": "<question or empty>"}"""


class Resolution(BaseModel):
    resolved: bool
    standalone_request: str = ""
    clarifying_question: str = ""


def resolve_context(ctx: ConversationContext, request: str, run: RunContext) -> Resolution:
    model = run.model_factory(run.settings.categorize_llm, purpose="resolver", temperature=0)
    ai = run.invoke_model(
        model,
        [
            SystemMessage(content=RESOLVER_SYSTEM),
            HumanMessage(content=json.dumps(ctx.router_state(request), ensure_ascii=False)),
        ],
        "resolver",
    )
    try:
        res = Resolution.model_validate(parse_json_object(message_text(ai)))
    except Exception:  # noqa: BLE001 — an unusable resolution must not guess
        return Resolution(
            resolved=False,
            clarifying_question="Could you add a bit more detail about what you'd like me to do?",
        )
    if res.resolved and not res.standalone_request.strip():
        res.standalone_request = request
    if not res.resolved and not res.clarifying_question.strip():
        res.clarifying_question = "Could you add a bit more detail about what you'd like me to do?"
    return res
