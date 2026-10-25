"""Stub node for agents not yet implemented."""

from __future__ import annotations

from foundry.progress import emit
from foundry.state import FUTURE_AGENTS, WorkflowState

AGENT_LABELS = {
    "lawyer": "Lawyer",
    "sales": "Sales / Marketing",
    "gtm": "Go-to-Market Engineer",
    "engineer": "Software Developer",
    "assistant": "Personal Assistant",
    "unsupported": "a specialist",
}


def unsupported_node(state: WorkflowState) -> dict:
    route = state.get("route", "unsupported")
    rationale = state.get("route_rationale", "")
    run_id = state.get("run_id")
    label = AGENT_LABELS.get(route, route)

    emit("unsupported.start", run_id=run_id, route=route, label=label)

    if route in FUTURE_AGENTS:
        response = (
            f"Routed to **{label}** — this agent is not built yet in Foundry v0.\n\n"
            f"Why this route: {rationale}\n\n"
            "Today only the **Founder (co-founder)** workflow is live. "
            "Reframe as strategy/idea/critique work to use the Founder agent, "
            "or check back as we ship more specialists."
        )
    else:
        response = (
            "Could not map this prompt to a specialist confidently.\n\n"
            f"Rationale: {rationale}\n\n"
            "Try a clearer ask — e.g. co-founder brainstorming, critique of an idea, "
            "or a go/no-go decision."
        )

    emit("unsupported.response", run_id=run_id, response=response)
    return {"final_response": response}
