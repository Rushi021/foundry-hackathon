"""LangGraph orchestration: categorize → founder | unsupported."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from foundry.agents.categorize import categorize_node
from foundry.agents.founder import founder_node
from foundry.agents.unsupported import unsupported_node
from foundry.persistence.supabase_logger import log_workflow_run
from foundry.progress import emit, set_verbose
from foundry.state import WorkflowState

StepCallback = Callable[[dict[str, Any]], None]


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
    verbose: bool = True,
    on_step: StepCallback | None = None,
) -> dict[str, Any]:
    """Run the workflow with real-time per-node streaming to the terminal."""
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    set_verbose(verbose)
    graph = get_graph()

    emit("run.start", run_id=run_id, prompt=prompt)
    if on_step:
        on_step({"step": "run.start", "run_id": run_id, "prompt": prompt})

    inputs: WorkflowState = {
        "user_prompt": prompt,
        "user_context": user_context,
        "run_id": run_id,
        "messages": [],
        "metadata": metadata or {},
    }

    merged: dict[str, Any] = dict(inputs)

    # Prefer updates+custom; fall back if installed LangGraph lacks multi-mode/version.
    stream_kwargs: dict[str, Any] = {"stream_mode": ["updates", "custom"]}
    try:
        stream_iter = graph.stream(inputs, **stream_kwargs, version="v2")
    except TypeError:
        try:
            stream_iter = graph.stream(inputs, **stream_kwargs)
        except TypeError:
            stream_iter = graph.stream(inputs, stream_mode="updates")

    for chunk in stream_iter:
        _consume_stream_chunk(chunk, merged, run_id, on_step)

    latency_ms = int((time.perf_counter() - started) * 1000)
    final = merged.get("final_response", "")
    route = merged.get("route")
    rationale = merged.get("route_rationale")

    emit(
        "run.complete",
        run_id=run_id,
        route=route,
        latency_ms=latency_ms,
        response=final,
    )
    if on_step:
        on_step(
            {
                "step": "run.complete",
                "run_id": run_id,
                "route": route,
                "latency_ms": latency_ms,
                "response": final,
            }
        )

    log_workflow_run(
        run_id=run_id,
        user_prompt=prompt,
        route=route,
        route_rationale=rationale,
        response=final,
        latency_ms=latency_ms,
        metadata={
            **(metadata or {}),
            "route_confidence": merged.get("route_confidence"),
            "has_user_context": bool(user_context),
        },
    )

    return {
        "run_id": run_id,
        "route": route,
        "route_rationale": rationale,
        "route_confidence": merged.get("route_confidence"),
        "response": final,
        "latency_ms": latency_ms,
    }


def _consume_stream_chunk(
    chunk: Any,
    merged: dict[str, Any],
    run_id: str,
    on_step: StepCallback | None,
) -> None:
    # v2: {"type": "updates"|"custom", "data": ...}
    if isinstance(chunk, dict) and "type" in chunk and "data" in chunk:
        kind = chunk["type"]
        data = chunk["data"]
        if kind == "custom" and isinstance(data, dict):
            if on_step:
                on_step({**data, "run_id": data.get("run_id", run_id)})
            return
        if kind == "updates" and isinstance(data, dict):
            _apply_updates(data, merged, run_id, on_step)
            return

    # Multi-mode tuple: (mode, payload)
    if isinstance(chunk, tuple) and len(chunk) == 2:
        mode, payload = chunk
        if mode == "custom" and isinstance(payload, dict):
            if on_step:
                on_step({**payload, "run_id": payload.get("run_id", run_id)})
            return
        if mode == "updates" and isinstance(payload, dict):
            _apply_updates(payload, merged, run_id, on_step)
            return

    # Single-mode updates: {node_name: update}
    if isinstance(chunk, dict):
        _apply_updates(chunk, merged, run_id, on_step)


def _apply_updates(
    updates: dict[str, Any],
    merged: dict[str, Any],
    run_id: str,
    on_step: StepCallback | None,
) -> None:
    for node_name, update in updates.items():
        if node_name.startswith("__"):
            continue
        if isinstance(update, dict):
            merged.update(update)
            emit(
                f"node.{node_name}.done",
                run_id=run_id,
                route=update.get("route", merged.get("route")),
                route_rationale=update.get("route_rationale"),
                route_confidence=update.get("route_confidence"),
                response=update.get("final_response"),
            )
            if on_step:
                on_step(
                    {
                        "step": f"node.{node_name}.done",
                        "run_id": run_id,
                        **{k: v for k, v in update.items() if k != "messages"},
                    }
                )
