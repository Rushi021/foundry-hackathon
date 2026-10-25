"""LangGraph orchestration: categorize → founder | unsupported."""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from foundry.agents.categorize import categorize_node
from foundry.agents.founder import founder_node
from foundry.agents.unsupported import unsupported_node
from foundry.persistence.supabase_logger import log_workflow_run
from foundry.state import WorkflowState


def _route_after_categorize(state: WorkflowState) -> str:
    return "founder" if state.get("route") == "founder" else "unsupported"


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("categorize", categorize_node)
    graph.add_node("founder", founder_node)
    graph.add_node("unsupported", unsupported_node)

    graph.add_edge(START, "categorize")
    graph.add_conditional_edges(
        "categorize",
        _route_after_categorize,
        {"founder": "founder", "unsupported": "unsupported"},
    )
    graph.add_edge("founder", END)
    graph.add_edge("unsupported", END)
    return graph.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run_workflow(
    prompt: str,
    *,
    user_context: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the founder workflow end-to-end and optionally log to Supabase."""
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    graph = get_graph()

    result = graph.invoke(
        {
            "user_prompt": prompt,
            "user_context": user_context,
            "run_id": run_id,
            "messages": [],
            "metadata": metadata or {},
        }
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    final = result.get("final_response", "")
    route = result.get("route")
    rationale = result.get("route_rationale")

    log_workflow_run(
        run_id=run_id,
        user_prompt=prompt,
        route=route,
        route_rationale=rationale,
        response=final,
        latency_ms=latency_ms,
        metadata={
            **(metadata or {}),
            "route_confidence": result.get("route_confidence"),
            "has_user_context": bool(user_context),
        },
    )

    return {
        "run_id": run_id,
        "route": route,
        "route_rationale": rationale,
        "route_confidence": result.get("route_confidence"),
        "response": final,
        "latency_ms": latency_ms,
    }
