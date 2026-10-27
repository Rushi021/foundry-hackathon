"""LangGraph orchestration.

prepare_context → route ─┬─► execute ──► finalize
                         └─► resolve ─┬─► route (once, with a standalone request)
                                      ├─► execute
                                      └─► finalize (clarifying question)

Agents are dispatched through the registry, so adding one does not change this graph.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from foundry.agents.registry import memory_scopes_by_agent, routable_agents
from foundry.agents.resolver import resolve_context
from foundry.config import Settings, get_settings
from foundry.context import SessionStore, Turn, prepare_context
from foundry.memory import MemoryGateway, MemoryStore
from foundry.persistence.supabase_logger import log_workflow_run
from foundry.progress import emit, set_verbose
from foundry.routing import (
    RESOLVE,
    DecisionProvider,
    RoutingDecision,
    get_decision_providers,
    plan_route,
)
from foundry.runtime import (
    ModelFactory,
    RunContext,
    TaskBrief,
    TaskResult,
    default_model_factory,
    run_agent,
)
from foundry.state import WorkflowState
from foundry.tools.registry import TOOLS

StepCallback = Callable[[dict[str, Any]], None]

REROUTE = "reroute"
CLARIFY_STEP = "clarify"


@dataclass
class _Deps:
    run: RunContext
    sessions: SessionStore
    primary: DecisionProvider | None
    fallback: DecisionProvider
    primary_unavailable: str | None


def _deps(config: RunnableConfig) -> _Deps:
    return config["configurable"]["deps"]


def prepare_context_node(state: WorkflowState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    session = deps.sessions.load(state.get("session_id"))
    ctx = prepare_context(session, state["user_message"])
    emit(
        "context.prepared",
        run_id=state["run_id"],
        session_id=session.id,
        recent_turns=len(ctx.recent),
        last_agent=ctx.last_agent,
    )
    return {
        "session_id": session.id,
        "context": ctx,
        "request": state["user_message"],
        "route_pass": 0,
    }


def _decide(deps: _Deps, state_payload: dict[str, Any]) -> RoutingDecision:
    run = deps.run
    destinations = {k: a.route_description for k, a in routable_agents().items()}
    tools = {k: t.description for k, t in TOOLS.items()}
    if deps.primary is not None:
        try:
            decision = deps.primary.decide(state_payload, destinations=destinations, tools=tools)
            run.metrics.record_decision_call(decision.usage)
            run.metrics.router_provider = decision.provider
            return decision
        except Exception as exc:  # noqa: BLE001 — fall back loudly, never silently
            reason = f"{deps.primary.name} error: {exc}"
            emit("router.fallback", run_id=run.run_id, reason=reason)
            run.metrics.router_fallback_reason = reason
    else:
        run.metrics.router_fallback_reason = deps.primary_unavailable
        emit("router.fallback", run_id=run.run_id, reason=deps.primary_unavailable)
    decision = deps.fallback.decide(state_payload, destinations=destinations, tools=tools)
    run.metrics.record_model_call("router", decision.usage)
    run.metrics.router_provider = decision.provider
    return decision


def route_node(state: WorkflowState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    run = deps.run
    ctx = state["context"]
    emit("router.start", run_id=run.run_id, route_pass=state["route_pass"])
    decision = _decide(deps, ctx.router_state(state["request"]))
    plan = plan_route(
        decision,
        route_pass=state["route_pass"],
        routable=set(routable_agents()),
        settings=run.settings,
        has_history=bool(ctx.recent),
    )
    run.preload_memory = plan.preload_memory
    run.tool_hint = decision.tool_hint
    emit(
        "router.decision",
        run_id=run.run_id,
        provider=decision.provider,
        destination=decision.destination,
        confidence=round(decision.confidence, 3),
        complexity=decision.complexity,
        tool_hint=decision.tool_hint,
        memory_p=round(decision.memory_probability, 3),
        clarify_p=round(decision.clarification_probability, 3),
        next=plan.destination,
        reason=plan.reason,
    )
    return {"decision": decision, "destination": plan.destination, "route_reason": plan.reason}


def resolve_node(state: WorkflowState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    run = deps.run
    request = state["request"]
    res = resolve_context(state["context"], request, run)
    emit(
        "resolver.result",
        run_id=run.run_id,
        resolved=res.resolved,
        standalone_request=res.standalone_request or None,
        question=res.clarifying_question or None,
    )
    if not res.resolved:
        result = TaskResult(
            task_id=uuid.uuid4().hex[:12],
            agent="context_resolution",
            status="needs_clarification",
            output=res.clarifying_question,
        )
        return {"destination": CLARIFY_STEP, "result": result}

    standalone = res.standalone_request.strip()
    if " ".join(standalone.lower().split()) != " ".join(request.lower().split()):
        return {"request": standalone, "route_pass": 1, "destination": REROUTE}
    # Nothing to rewrite: re-asking the router would return the same answer.
    plan = plan_route(
        state["decision"],
        route_pass=1,
        routable=set(routable_agents()),
        settings=run.settings,
    )
    return {"route_pass": 1, "destination": plan.destination, "route_reason": plan.reason}


def execute_node(state: WorkflowState, config: RunnableConfig) -> dict:
    run = _deps(config).run
    ctx = state["context"]
    result = run_agent(
        state["destination"],
        TaskBrief(objective=state["request"], requested_by="user"),
        run,
        history=ctx.history_messages(),
        background=ctx.background(),
    )
    return {"result": result}


def finalize_node(state: WorkflowState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    result = state["result"]
    deps.sessions.append(
        state["session_id"],
        Turn(role="user", content=state["user_message"]),
        Turn(
            role="assistant",
            content=result.output,
            agent=result.agent,
            status=result.status,
            task_id=result.task_id,
        ),
    )
    return {"final_response": result.output, "status": result.status}


def _after_route(state: WorkflowState) -> str:
    return "resolve" if state["destination"] == RESOLVE else "execute"


def _after_resolve(state: WorkflowState) -> str:
    dest = state["destination"]
    if dest == REROUTE:
        return "route"
    if dest == CLARIFY_STEP:
        return "finalize"
    return "execute"


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("route", route_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "route")
    graph.add_conditional_edges("route", _after_route, {"resolve": "resolve", "execute": "execute"})
    graph.add_conditional_edges(
        "resolve",
        _after_resolve,
        {"route": "route", "execute": "execute", "finalize": "finalize"},
    )
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", END)
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
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
    on_step: StepCallback | None = None,
    settings: Settings | None = None,
    model_factory: ModelFactory | None = None,
    decision_provider: DecisionProvider | None = None,
    memory_store: MemoryStore | None = None,
    session_store: SessionStore | None = None,
) -> dict[str, Any]:
    """Run one request through the workflow, streaming steps to the terminal."""
    settings = settings or get_settings()
    model_factory = model_factory or default_model_factory
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    set_verbose(verbose)

    if decision_provider is not None:
        primary, fallback, unavailable = decision_provider, decision_provider, None
    else:
        primary, fallback, unavailable = get_decision_providers(settings, model_factory)
    run = RunContext(
        run_id=run_id,
        settings=settings,
        memory=MemoryGateway(
            memory_store or MemoryStore(settings.foundry_data_dir), memory_scopes_by_agent()
        ),
        model_factory=model_factory,
    )
    deps = _Deps(
        run=run,
        sessions=session_store or SessionStore(settings.foundry_data_dir),
        primary=primary,
        fallback=fallback,
        primary_unavailable=unavailable,
    )

    emit("run.start", run_id=run_id, session_id=session_id, prompt=prompt)
    if on_step:
        on_step({"step": "run.start", "run_id": run_id, "prompt": prompt})

    inputs: WorkflowState = {"run_id": run_id, "session_id": session_id, "user_message": prompt}
    merged: dict[str, Any] = dict(inputs)
    config: RunnableConfig = {"configurable": {"deps": deps}}
    for chunk in get_graph().stream(inputs, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name.startswith("__") or not isinstance(update, dict):
                continue
            merged.update(update)
            if on_step:
                on_step({"step": f"node.{node_name}.done", "run_id": run_id})

    latency_ms = int((time.perf_counter() - started) * 1000)
    result: TaskResult = merged["result"]
    decision: RoutingDecision | None = merged.get("decision")
    metrics = run.metrics.as_dict()

    emit(
        "run.complete",
        run_id=run_id,
        agent=result.agent,
        status=result.status,
        latency_ms=latency_ms,
        model_calls=metrics["model_calls"],
        decision_calls=metrics["decision_calls"],
        tool_calls=len(metrics["tool_calls"]),
        delegations=len(metrics["delegations"]),
    )
    if on_step:
        on_step({"step": "run.complete", "run_id": run_id, "status": result.status})

    route_rationale = merged.get("route_reason")
    if decision and decision.rationale:
        route_rationale = f"{route_rationale}: {decision.rationale}"
    log_workflow_run(
        run_id=run_id,
        user_prompt=prompt,
        route=result.agent,
        route_rationale=route_rationale,
        response=result.output,
        latency_ms=latency_ms,
        metadata={
            **(metadata or {}),
            "session_id": merged.get("session_id"),
            "status": result.status,
            "decision": decision.model_dump() if decision else None,
            "metrics": metrics,
        },
    )

    return {
        "run_id": run_id,
        "session_id": merged.get("session_id"),
        "status": result.status,
        "route": result.agent,
        "route_rationale": route_rationale,
        "route_confidence": decision.confidence if decision else None,
        "decision": decision.model_dump(exclude={"usage"}) if decision else None,
        "response": result.output,
        "task": result.model_dump(),
        "metrics": metrics,
        "latency_ms": latency_ms,
    }
