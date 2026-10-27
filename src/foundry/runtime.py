"""Agent runtime: one execution loop for every registered agent.

The application — not the model — enforces tool authorization, memory access,
approval gates, delegation limits, and task status.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from foundry.agents.prompts import STATUS_PROTOCOL
from foundry.agents.registry import AGENTS, AgentSpec
from foundry.config import Settings
from foundry.llm import get_chat_model
from foundry.memory import MemoryAccessDenied, MemoryGateway, MemoryHit, contains_private_content
from foundry.metrics import RunMetrics
from foundry.progress import emit
from foundry.text import message_text, usage_of
from foundry.tools.base import ToolError
from foundry.tools.registry import TOOLS
from foundry.tools.research import ResearchProvider, get_research_provider

TaskStatus = Literal["completed", "partial", "awaiting_approval", "needs_clarification", "failed"]
_SEVERITY: dict[str, int] = {
    "completed": 0,
    "partial": 1,
    "awaiting_approval": 2,
    "needs_clarification": 3,
    "failed": 4,
}
_STATUS_MARKER = re.compile(
    r"\[\[status:(partial|needs_clarification|awaiting_approval|failed)\]\]", re.IGNORECASE
)
_SYSTEM_TOOLS = frozenset({"recall_memory", "delegate", "escalate"})

# Agents nested deeper than this cannot delegate further (Founder → COO → specialist).
MAX_DELEGATION_DEPTH = 3
PRELOAD_MEMORY_K = 4


class TaskBrief(BaseModel):
    objective: str
    context: str = ""
    deliverables: str = ""
    constraints: str = ""
    requested_by: str = "user"

    def render(self) -> str:
        parts = [f"Objective: {self.objective}"]
        for label, value in (
            ("Context", self.context),
            ("Expected deliverables", self.deliverables),
            ("Constraints", self.constraints),
        ):
            if value:
                parts.append(f"{label}: {value}")
        parts.append(f"Requested by: {self.requested_by}")
        return "\n\n".join(parts)


class TaskResult(BaseModel):
    task_id: str
    agent: str
    status: TaskStatus
    output: str
    error: str | None = None
    subtasks: list[TaskResult] = Field(default_factory=list)
    escalated_from: str | None = None
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)


ModelFactory = Callable[..., Any]


def default_model_factory(model_spec: str, *, purpose: str, temperature: float = 0.2) -> Any:
    return get_chat_model(model_spec, temperature=temperature)


@dataclass
class RunContext:
    run_id: str
    settings: Settings
    memory: MemoryGateway
    model_factory: ModelFactory = default_model_factory
    metrics: RunMetrics = field(default_factory=RunMetrics)
    # Router execution hints for the top-level agent.
    tool_hint: str = "none"
    preload_memory: bool = False
    research_cache: dict[str, str] = field(default_factory=dict)
    delegation_cache: dict[tuple[str, str], TaskResult] = field(default_factory=dict)
    # Private memory contents surfaced during this run, used to stop it leaking to
    # agents that are not authorized to see it.
    private_seen: list[str] = field(default_factory=list)
    _research: ResearchProvider | None = None

    def research_provider(self) -> ResearchProvider:
        if self._research is None:
            self._research = get_research_provider(self.settings)
        return self._research

    def invoke_model(self, model: Any, messages: list[BaseMessage], purpose: str) -> Any:
        ai = model.invoke(messages)
        self.metrics.record_model_call(purpose, usage_of(ai))
        return ai

    def note_memory_read(self, agent: str, scope: str, hits: list[MemoryHit], via: str) -> None:
        self.metrics.memory_reads.append(
            {"agent": agent, "scope": scope, "hits": len(hits), "via": via}
        )
        if scope == "private":
            self.private_seen.extend(h.item.content for h in hits)
        emit("memory.read", run_id=self.run_id, agent=agent, scope=scope, hits=len(hits), via=via)


@dataclass
class _Invocation:
    spec: AgentSpec
    chain: tuple[str, ...]
    subtasks: list[TaskResult] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    handoff: tuple[str, str] | None = None


def run_agent(
    agent_id: str,
    brief: TaskBrief,
    run: RunContext,
    *,
    history: list[BaseMessage] | None = None,
    background: str = "",
    parents: tuple[str, ...] = (),
    top_level: bool | None = None,
) -> TaskResult:
    """Execute one task with one agent and return a result with an honest status."""
    spec = AGENTS[agent_id]
    top_level = not parents if top_level is None else top_level
    inv = _Invocation(spec=spec, chain=(*parents, agent_id))
    task_id = uuid.uuid4().hex[:12]
    emit(
        "agent.start",
        run_id=run.run_id,
        agent=agent_id,
        task_id=task_id,
        requested_by=brief.requested_by,
        depth=len(parents),
    )

    try:
        tools = _build_tools(inv, run, top_level=top_level)
        memory_block = _preload_memory(spec, brief, run) if top_level and run.preload_memory else ""
        system = _system_prompt(spec, run, tools, top_level, background, memory_block)
        base_model = run.model_factory(
            getattr(run.settings, spec.llm_setting), purpose=agent_id, temperature=spec.temperature
        )
        model = base_model.bind_tools(list(tools.values())) if tools else base_model
        messages: list[BaseMessage] = [
            SystemMessage(content=system),
            *(history or []),
            HumanMessage(content=brief.objective if top_level else brief.render()),
        ]

        text = ""
        for _ in range(spec.max_rounds):
            ai = run.invoke_model(model, messages, agent_id)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                text = message_text(ai)
                break
            for call in calls:
                messages.append(_execute_tool_call(call, tools, inv, run))
            if inv.handoff:
                break
        else:
            emit("agent.force_close", run_id=run.run_id, agent=agent_id, reason="max_tool_rounds")
            closing = run.invoke_model(
                base_model,
                messages + [HumanMessage(content="Stop calling tools. Give your final response now.")],
                agent_id,
            )
            text = message_text(closing)
    except Exception as exc:  # noqa: BLE001 — failures are results, not crashes
        emit("agent.failed", run_id=run.run_id, agent=agent_id, task_id=task_id, error=str(exc))
        run.metrics.failures.append({"agent": agent_id, "task_id": task_id, "error": str(exc)})
        return TaskResult(
            task_id=task_id,
            agent=agent_id,
            status="failed",
            output=f"{spec.name} could not complete the task: {exc}",
            error=str(exc),
            subtasks=inv.subtasks,
        )

    if inv.handoff:
        target, reason = inv.handoff
        run.metrics.escalations.append({"from": agent_id, "to": target, "reason": reason})
        emit("agent.escalate", run_id=run.run_id, agent=agent_id, to=target, reason=reason)
        result = run_agent(
            target, brief, run, history=history, background=background,
            parents=inv.chain, top_level=True,
        )
        return result.model_copy(update={"escalated_from": agent_id})

    output, declared = _split_status(text)
    status = max(declared, _derived_status(inv), key=_SEVERITY.__getitem__)
    if not output:
        status, output = "failed", f"{spec.name} returned an empty response."
    if status in {"failed", "partial"}:
        run.metrics.failures.append({"agent": agent_id, "task_id": task_id, "status": status})
    emit("agent.result", run_id=run.run_id, agent=agent_id, task_id=task_id, status=status)
    return TaskResult(
        task_id=task_id,
        agent=agent_id,
        status=status,  # type: ignore[arg-type]
        output=output,
        subtasks=inv.subtasks,
        pending_approvals=inv.pending_approvals,
    )


def _split_status(text: str) -> tuple[str, str]:
    markers = _STATUS_MARKER.findall(text)
    declared = markers[-1].lower() if markers else "completed"
    return _STATUS_MARKER.sub("", text).strip(), declared


def _derived_status(inv: _Invocation) -> str:
    """Status from evidence: tools and delegated results, not the agent's say-so."""
    if inv.pending_approvals:
        return "awaiting_approval"
    latest: dict[str, TaskResult] = {}
    for sub in inv.subtasks:
        latest[sub.agent] = sub
    if any(s.status == "awaiting_approval" for s in latest.values()):
        return "awaiting_approval"
    if inv.tool_errors or any(s.status != "completed" for s in latest.values()):
        return "partial"
    return "completed"


def _execute_tool_call(
    call: dict[str, Any], tools: dict[str, StructuredTool], inv: _Invocation, run: RunContext
) -> ToolMessage:
    name = call.get("name") or ""
    args = call.get("args") or {}
    agent = inv.spec.id
    run.metrics.tool_calls.append({"agent": agent, "tool": name})
    emit("agent.tool_call", run_id=run.run_id, agent=agent, tool=name, args=args)
    tool = tools.get(name)
    if tool is None:
        content = f"Tool '{name}' is not authorized for {inv.spec.name}."
        run.metrics.failures.append({"agent": agent, "tool": name, "error": "unauthorized_tool"})
    else:
        try:
            content = str(tool.invoke(args))
        except ToolError as exc:
            content = f"Tool error: {exc}"
            inv.tool_errors.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — surfaced to the agent to retry or report
            content = f"Tool error: {exc}"
            inv.tool_errors.append(f"{name}: {exc}")
    emit("agent.tool_result", run_id=run.run_id, agent=agent, tool=name, result=content)
    return ToolMessage(content=content, tool_call_id=call.get("id") or name, name=name)


def _build_tools(inv: _Invocation, run: RunContext, *, top_level: bool) -> dict[str, StructuredTool]:
    spec = inv.spec
    tools: dict[str, StructuredTool] = {}
    for tool_id in spec.tools:
        tspec = TOOLS.get(tool_id)
        if tspec is None or not tspec.available:
            continue
        tool = tspec.build(run)  # type: ignore[misc]
        tools[tool.name] = _approval_gate(tool, inv, run) if tspec.consequential else tool

    scopes = run.memory.authorized_scopes(spec.id)
    if scopes:
        tools["recall_memory"] = _recall_tool(inv, run, scopes)
    if spec.can_delegate_to and len(inv.chain) < MAX_DELEGATION_DEPTH:
        tools["delegate"] = _delegate_tool(inv, run)
    if spec.can_escalate_to and top_level:
        tools["escalate"] = _escalate_tool(inv)
    return tools


def _approval_gate(tool: StructuredTool, inv: _Invocation, run: RunContext) -> StructuredTool:
    """Consequential actions are recorded for user approval, never executed in Phase 1."""

    def _gated(**kwargs: Any) -> str:
        inv.pending_approvals.append({"tool": tool.name, "args": kwargs, "agent": inv.spec.id})
        emit("approval.required", run_id=run.run_id, agent=inv.spec.id, tool=tool.name)
        return (
            f"'{tool.name}' requires the user's approval. It has NOT been executed. "
            "Tell the user exactly what you propose to do and ask them to approve."
        )

    return StructuredTool.from_function(
        func=_gated, name=tool.name, description=tool.description, args_schema=tool.args_schema
    )


class _RecallArgs(BaseModel):
    query: str = Field(description="What you need to know from long-term memory.")
    scope: str = Field(description="Memory scope to search.")


def _recall_tool(inv: _Invocation, run: RunContext, scopes: list[str]) -> StructuredTool:
    agent = inv.spec.id

    def _recall(query: str, scope: str = scopes[0]) -> str:
        try:
            hits = run.memory.search(agent, scope, query)
        except MemoryAccessDenied as exc:
            run.metrics.memory_denials.append({"agent": agent, "scope": scope, "via": "recall"})
            emit("memory.denied", run_id=run.run_id, agent=agent, scope=scope)
            return f"Access denied: {exc}"
        run.note_memory_read(agent, scope, hits, via="recall")
        if not hits:
            return f"No relevant {scope} memory found."
        return "\n".join(f"- [{scope}] {h.item.content}" for h in hits)

    return StructuredTool.from_function(
        func=_recall,
        name="recall_memory",
        description=(
            "Search long-term memory about the user's projects, goals, preferences and prior "
            f"decisions. Allowed scopes: {', '.join(scopes)}. Use only when the task needs it."
        ),
        args_schema=_RecallArgs,
    )


class _DelegateArgs(BaseModel):
    agent: str = Field(description="Agent id to delegate to.")
    objective: str = Field(description="Clear, self-contained task objective.")
    context: str = Field(default="", description="Only the context the agent needs.")
    deliverables: str = Field(default="", description="What the agent must return.")
    constraints: str = Field(default="", description="Scope limits, deadlines, must-nots.")


def _delegate_tool(inv: _Invocation, run: RunContext) -> StructuredTool:
    caller = inv.spec.id
    targets = inv.spec.can_delegate_to

    def _delegate(
        agent: str, objective: str, context: str = "", deliverables: str = "", constraints: str = ""
    ) -> str:
        if agent not in targets:
            return f"Delegation denied: {caller} may delegate only to {', '.join(targets)}."
        if agent in inv.chain:
            return (
                f"Delegation denied: '{agent}' is already upstream of this task (cycle). "
                "State the open question in your reply instead."
            )
        brief = TaskBrief(
            objective=objective,
            context=context,
            deliverables=deliverables,
            constraints=constraints,
            requested_by=caller,
        )
        if not run.memory.may_access(agent, "private") and contains_private_content(
            brief.render(), run.private_seen
        ):
            run.metrics.memory_denials.append(
                {"agent": agent, "scope": "private", "via": "delegation_brief", "from": caller}
            )
            emit("delegation.blocked", run_id=run.run_id, frm=caller, to=agent, reason="private_memory")
            return (
                f"Delegation blocked: the brief contains Private Memory content, which '{agent}' "
                "is not authorized to see. Rewrite the brief without it."
            )

        key = (agent, " ".join(objective.lower().split()))
        result = run.delegation_cache.get(key)
        if result is not None:
            run.metrics.cache_hits += 1
            emit("delegation.cache_hit", run_id=run.run_id, frm=caller, to=agent)
        else:
            run.metrics.delegations.append({"from": caller, "to": agent})
            emit("delegation.start", run_id=run.run_id, frm=caller, to=agent, objective=objective)
            result = run_agent(agent, brief, run, parents=inv.chain)
            if result.status == "completed":
                run.delegation_cache[key] = result

        if (
            run.memory.may_access(agent, "private")
            and not run.memory.may_access(caller, "private")
            and contains_private_content(result.output, run.private_seen)
        ):
            run.metrics.memory_denials.append(
                {"agent": caller, "scope": "private", "via": "delegation_result", "from": agent}
            )
            result = result.model_copy(
                update={
                    "status": "failed",
                    "output": "Result withheld: it contained Private Memory content.",
                    "error": "private_memory_leak_blocked",
                }
            )

        inv.subtasks.append(result)
        emit("delegation.result", run_id=run.run_id, frm=caller, to=agent, status=result.status)
        body = f"[task {result.task_id} → {agent}] status: {result.status}\n{result.output}"
        return body + (f"\nerror: {result.error}" if result.error else "")

    target_list = "; ".join(f"{t}: {AGENTS[t].name}" for t in targets if t in AGENTS)
    return StructuredTool.from_function(
        func=_delegate,
        name="delegate",
        description=(
            f"Assign a task to another agent and receive its result ({target_list}). "
            "Check the returned status; delegation is not completion."
        ),
        args_schema=_DelegateArgs,
    )


class _EscalateArgs(BaseModel):
    agent: str = Field(description="Agent id to hand the request to.")
    reason: str = Field(description="Why this request needs that agent.")


def _escalate_tool(inv: _Invocation) -> StructuredTool:
    targets = inv.spec.can_escalate_to

    def _escalate(agent: str, reason: str) -> str:
        if agent not in targets:
            return f"Escalation denied: allowed targets are {', '.join(targets)}."
        inv.handoff = (agent, reason)
        return f"Handing the request to {agent}."

    return StructuredTool.from_function(
        func=_escalate,
        name="escalate",
        description=(
            "Hand the whole request to a more suitable agent, who answers the user directly: "
            + "; ".join(f"{t}: {AGENTS[t].name}" for t in targets if t in AGENTS)
        ),
        args_schema=_EscalateArgs,
    )


def _preload_memory(spec: AgentSpec, brief: TaskBrief, run: RunContext) -> str:
    lines: list[str] = []
    for scope in run.memory.authorized_scopes(spec.id):
        hits = run.memory.search(spec.id, scope, brief.objective, k=PRELOAD_MEMORY_K)
        run.note_memory_read(spec.id, scope, hits, via="preload")
        lines.extend(f"- [{scope}] {h.item.content}" for h in hits)
    return "\n".join(lines)


def _system_prompt(
    spec: AgentSpec,
    run: RunContext,
    tools: dict[str, StructuredTool],
    top_level: bool,
    background: str,
    memory_block: str,
) -> str:
    external = [name for name in tools if name not in _SYSTEM_TOOLS]
    lines = [
        spec.persona.strip(),
        "",
        "## Operating context",
        f"- You are `{spec.id}` ({spec.name}).",
        f"- Connected tools: {', '.join(external) or 'none'}.",
    ]
    missing = [TOOLS[t] for t in spec.tools if t in TOOLS and not TOOLS[t].available]
    if missing:
        lines.append(
            "- Not connected yet: "
            + ", ".join(f"{t.id} ({t.description})" for t in missing)
            + ". If the request needs one, say plainly it is unavailable; never pretend it ran."
        )
    if top_level and run.tool_hint not in {"none", ""}:
        if run.tool_hint in tools:
            lines.append(f"- Router hint: this request may need `{run.tool_hint}` (a hint, not a rule).")
        else:
            lines.append(
                f"- Router hint: this request may need `{run.tool_hint}`, which you cannot use. "
                "Report the limitation or escalate."
            )
    if "delegate" in tools:
        lines.append(
            "- You may delegate to: "
            + "; ".join(f"`{t}` ({AGENTS[t].route_description})" for t in spec.can_delegate_to)
            + ". Delegate only what you can't do well yourself; pass only needed context. "
            "Results return to you — integrate them into one answer and report failures honestly."
        )
    if "escalate" in tools:
        lines.append(
            "- If the request is beyond your role you may escalate to: "
            + ", ".join(f"`{t}`" for t in spec.can_escalate_to)
            + ". Escalate rarely."
        )
    if "recall_memory" in tools:
        scopes = ", ".join(run.memory.authorized_scopes(spec.id))
        lines.append(f"- Long-term memory you may read: {scopes}. Use `recall_memory` only if needed.")
    lines += ["", STATUS_PROTOCOL]
    if background:
        lines += ["", "## Conversation background", background]
    if memory_block:
        lines += ["", "## Relevant long-term memory", memory_block]
    return "\n".join(lines)
