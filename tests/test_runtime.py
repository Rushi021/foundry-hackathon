"""Runtime guardrails: authorization, delegation limits, status honesty, efficiency."""

from __future__ import annotations

import pytest
from conftest import Models, ScriptedDecider, call, decision, say
from langchain_core.tools import StructuredTool

from foundry.agents.registry import memory_scopes_by_agent
from foundry.graph import run_workflow
from foundry.memory import MemoryGateway
from foundry.runtime import RunContext, TaskBrief, run_agent
from foundry.tools.base import ToolSpec
from foundry.tools.registry import TOOLS


@pytest.fixture
def ctx(settings, memory_store):
    def _ctx(models, **kw):
        return RunContext(
            run_id="r1",
            settings=settings,
            memory=MemoryGateway(memory_store, memory_scopes_by_agent()),
            model_factory=models,
            **kw,
        )

    return _ctx


def test_direct_assistant_escalates_to_cofounder(ctx):
    models = Models(
        direct=[call("escalate", agent="cofounder", reason="needs strategy")],
        cofounder=[say("PUSH BACK: ...")],
    )
    run = ctx(models)
    result = run_agent("direct", TaskBrief(objective="Should I pivot?"), run)

    assert (result.agent, result.escalated_from, result.status) == ("cofounder", "direct", "completed")
    assert run.metrics.escalations == [{"from": "direct", "to": "cofounder", "reason": "needs strategy"}]
    assert "escalate" not in models["cofounder"].bound


def test_delegation_cycle_is_prevented(ctx):
    models = Models(
        cofounder=[call("delegate", agent="coo", objective="Run launch"), say("done")],
        coo=[
            call("delegate", agent="cofounder", objective="Decide pricing"),
            lambda msgs: say("cycle blocked" if "cycle" in msgs[-1].content else "!"),
        ],
    )
    result = run_agent("cofounder", TaskBrief(objective="Plan launch"), ctx(models))
    assert result.subtasks[0].output == "cycle blocked"


def test_delegation_depth_is_bounded(ctx):
    models = Models(coo=[say("ok")])
    run = ctx(models)
    run_agent("coo", TaskBrief(objective="x"), run, parents=("cofounder", "direct"))
    assert "delegate" not in models["coo"].bound


def test_agent_cannot_delegate_outside_its_allowlist(ctx):
    models = Models(
        coo=[
            call("delegate", agent="direct", objective="x"),
            lambda msgs: say("denied" if "Delegation denied" in msgs[-1].content else "!"),
        ]
    )
    result = run_agent("coo", TaskBrief(objective="x"), ctx(models))
    assert result.output == "denied" and result.subtasks == []


def test_unauthorized_tool_call_is_refused(ctx):
    models = Models(
        marketing=[
            call("web_search", query="x"),
            lambda msgs: say("refused" if "not authorized" in msgs[-1].content else "!"),
        ]
    )
    run = ctx(models)
    result = run_agent("marketing", TaskBrief(objective="x"), run)
    assert result.output == "refused"
    assert "web_search" not in models["marketing"].bound


def test_status_marker_is_parsed_and_stripped(ctx):
    models = Models(direct=[say("Which file do you mean?\n[[status:needs_clarification]]")])
    result = run_agent("direct", TaskBrief(objective="Fix the file"), ctx(models))
    assert result.status == "needs_clarification"
    assert result.output == "Which file do you mean?"


def test_agent_cannot_claim_completion_after_tool_failure(ctx):
    # No research provider configured → web_search fails; the agent's answer is partial.
    models = Models(direct=[call("web_search", query="today's news"), say("Here is the news.")])
    result = run_agent("direct", TaskBrief(objective="news?"), ctx(models))
    assert result.status == "partial"


def test_model_failure_is_a_failed_result_not_a_crash(ctx):
    models = Models(research=[RuntimeError("rate limited")])
    result = run_agent("research", TaskBrief(objective="x"), ctx(models))
    assert result.status == "failed" and "rate limited" in result.error


def test_duplicate_web_search_is_served_from_cache(ctx, research):
    models = Models(
        research=[
            call("web_search", "c1", query="Dental AI market"),
            call("web_search", "c2", query="dental  ai market"),
            say("done"),
        ]
    )
    run = ctx(models)
    run_agent("research", TaskBrief(objective="x"), run)
    assert research.queries == ["Dental AI market"]


def test_private_memory_cannot_leak_through_delegation_brief(ctx, memory_store):
    memory_store.add("private", "I am secretly negotiating an acquisition offer from Acme Corp.")
    models = Models(
        cofounder=[
            call("delegate", agent="research", objective="Research Acme",
                 context="He is secretly negotiating an acquisition offer from Acme Corp."),
            lambda msgs: say("rewrote" if "Delegation blocked" in msgs[-1].content else "!"),
        ]
    )
    run = ctx(models, preload_memory=True)
    result = run_agent("cofounder", TaskBrief(objective="acquisition offer from Acme"), run)

    assert result.output == "rewrote" and result.subtasks == []
    assert run.metrics.memory_denials[0]["via"] == "delegation_brief"


def test_private_memory_cannot_leak_through_cofounder_result_to_coo(ctx, memory_store):
    memory_store.add("private", "My runway ends in June and I will not raise money.")
    models = Models(
        coo=[
            call("delegate", agent="cofounder", objective="Should we cut scope?"),
            lambda msgs: say("withheld" if "Result withheld" in msgs[-1].content else "!"),
        ],
        cofounder=[
            call("recall_memory", query="runway", scope="private"),
            say("Yes — my runway ends in June and I will not raise money."),
        ],
    )
    result = run_agent("coo", TaskBrief(objective="launch"), ctx(models))
    assert result.output == "withheld"
    assert result.subtasks[0].status == "failed"


def test_consequential_tool_requires_approval(ctx, monkeypatch):
    executed = []

    def _send(to: str) -> str:
        executed.append(to)
        return "sent"

    tool = StructuredTool.from_function(func=_send, name="gmail", description="send email")
    monkeypatch.setitem(
        TOOLS,
        "gmail",
        ToolSpec(id="gmail", description="email", build=lambda run: tool, consequential=True),
    )
    models = Models(direct=[call("gmail", to="sam@example.com"), say("Drafted; approve to send.")])
    result = run_agent("direct", TaskBrief(objective="Email Sam"), ctx(models))

    assert executed == []
    assert result.status == "awaiting_approval"
    assert result.pending_approvals == [
        {"tool": "gmail", "args": {"to": "sam@example.com"}, "agent": "direct"}
    ]


def test_jev_failure_falls_back_to_llm_router_visibly(settings, memory_store, session_store,
                                                      monkeypatch):
    def _boom(self, *a, **k):
        raise RuntimeError("jev down")

    monkeypatch.setattr("foundry.routing.jev.JevDecisionProvider.decide", _boom)
    keyed = settings.model_copy(update={"typesafe_api_key": "jv_x"})
    models = Models(
        router=[say('{"destination":"direct","confidence":0.9}')],
        direct=[say("An API is ...")],
    )
    out = run_workflow("What is an API?", verbose=False, settings=keyed, model_factory=models,
                       memory_store=memory_store, session_store=session_store)

    assert out["route"] == "direct"
    assert out["metrics"]["router_provider"] == "llm"
    assert "jev down" in out["metrics"]["router_fallback_reason"]


def test_low_confidence_first_message_falls_back_to_direct_without_resolver(
    settings, memory_store, session_store
):
    models = Models(direct=[say("Sure.")])
    out = run_workflow("hmm", verbose=False, settings=settings, model_factory=models,
                       decision_provider=ScriptedDecider(decision("coo", 0.2)),
                       memory_store=memory_store, session_store=session_store)
    assert out["route"] == "direct"
    assert "resolver" not in out["metrics"]["model_calls_by_purpose"]
