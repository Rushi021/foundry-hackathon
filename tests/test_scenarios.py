"""Phase 1 demonstration scenarios, end to end through the workflow graph."""

from __future__ import annotations

import pytest
from conftest import Models, ScriptedDecider, call, decision, say
from langchain_core.messages import HumanMessage

from foundry.context import Turn
from foundry.graph import run_workflow


@pytest.fixture
def run(settings, memory_store, session_store):
    def _run(prompt, decider, models, session_id=None):
        return run_workflow(
            prompt,
            session_id=session_id,
            verbose=False,
            settings=settings,
            model_factory=models,
            decision_provider=decider,
            memory_store=memory_store,
            session_store=session_store,
        )

    return _run


def test_general_knowledge_question_stays_simple(run, memory_store):
    memory_store.add("shared", "Project Atlas uses a REST API.")
    models = Models(direct=[say("An API is a contract between programs.")])
    out = run("What is an API?", ScriptedDecider(decision("direct")), models)

    assert (out["route"], out["status"]) == ("direct", "completed")
    m = out["metrics"]
    assert m["decision_calls"] == 1 and m["model_calls"] == 1
    assert m["tool_calls"] == [] and m["delegations"] == []
    assert m["memory_reads"] == []  # request needing no memory: no retrieval


def test_current_information_uses_web_search_without_research_agent(run, research):
    models = Models(
        direct=[call("web_search", query="latest OpenAI announcements"), say("Here's the latest...")]
    )
    d = decision("direct", tool_hint="web_search")
    out = run("What are the latest announcements from OpenAI?", ScriptedDecider(d), models)

    assert (out["route"], out["status"]) == ("direct", "completed")
    assert research.queries == ["latest OpenAI announcements"]
    assert out["metrics"]["delegations"] == []
    assert "Router hint: this request may need `web_search`" in models["direct"].system_prompt()


def test_follow_up_question_keeps_conversation_context(run, research):
    models = Models(research=[say("Market overview."), say("Competitors: A, B.")])
    decider = ScriptedDecider(decision("research"), decision("research"))
    first = run("Research the AI receptionist market.", decider, models)
    second = run("What about its competition?", decider, models, session_id=first["session_id"])

    assert second["status"] == "completed"
    state = decider.states[1]
    assert "AI receptionist market" in " ".join(state["recent_conversation"])
    assert state["previous_turn_handled_by"] == "research"
    history = [m.content for m in models["research"].calls[1] if isinstance(m, HumanMessage)]
    assert history[0] == "Research the AI receptionist market."


def test_ambiguous_reference_resolved_from_context(run, session_store):
    session_store.append(
        "s1",
        Turn(role="user", content="I'm talking to John Park about a partnership."),
        Turn(role="assistant", content="Good, prepare an agenda.", agent="direct"),
    )
    models = Models(
        resolver=[
            say('{"resolved": true, "standalone_request": '
                '"Schedule a meeting with John Park tomorrow.", "clarifying_question": ""}')
        ],
        direct=[say("I can't schedule it: Google Calendar isn't connected yet.")],
    )
    decider = ScriptedDecider(
        decision("direct", clarification_probability=0.9),
        decision("direct", tool_hint="google_calendar"),
    )
    out = run("Schedule a meeting with him tomorrow.", decider, models, session_id="s1")

    assert out["route"] == "direct"
    assert decider.states[1]["current_request"] == "Schedule a meeting with John Park tomorrow."
    prompt = models["direct"].system_prompt()
    assert "Not connected yet: gmail" in prompt and "google_calendar" in prompt


def test_ambiguous_request_without_context_asks_for_clarification(run):
    models = Models(
        resolver=[
            say('{"resolved": false, "standalone_request": "", '
                '"clarifying_question": "Who would you like to meet?"}')
        ]
    )
    out = run("Schedule a meeting with him tomorrow.", ScriptedDecider(decision("clarify")), models)

    assert out["status"] == "needs_clarification"
    assert out["route"] == "context_resolution"
    assert out["response"] == "Who would you like to meet?"
    assert out["metrics"]["model_calls_by_purpose"] == {"resolver": 1}


def test_strategic_discussion_goes_to_cofounder(run):
    models = Models(cofounder=[say("Verdict posture: REFINE ...")])
    out = run("Evaluate whether this startup idea is worth pursuing.",
              ScriptedDecider(decision("cofounder")), models)
    assert (out["route"], out["status"]) == ("cofounder", "completed")
    assert out["metrics"]["memory_reads"] == []


def test_personalized_strategy_uses_private_memory(run, memory_store):
    memory_store.add("private", "I want to bootstrap this business idea and avoid VC money.")
    memory_store.add("shared", "Project Atlas business idea: AI receptionist for dentists.")
    models = Models(cofounder=[say("Given you want to bootstrap: PURSUE, narrowly.")])
    d = decision("cofounder", memory_probability=0.9)
    out = run("Considering everything, should I pursue this business idea?",
              ScriptedDecider(d), models)

    prompt = models["cofounder"].system_prompt()
    assert "[private] I want to bootstrap" in prompt
    assert "[shared] Project Atlas" in prompt
    scopes = {r["scope"] for r in out["metrics"]["memory_reads"]}
    assert scopes == {"private", "shared"}


def test_memory_for_non_cofounder_is_shared_only(run, memory_store):
    memory_store.add("private", "My personal savings target is confidential project data.")
    memory_store.add("shared", "Project Atlas launch date is March.")
    models = Models(direct=[say("Atlas launches in March.")])
    out = run("When does project Atlas launch?",
              ScriptedDecider(decision("direct", memory_probability=0.9)), models)

    prompt = models["direct"].system_prompt()
    assert "[shared] Project Atlas launch date" in prompt
    assert "savings" not in prompt
    assert {r["scope"] for r in out["metrics"]["memory_reads"]} == {"shared"}


def test_operational_task_coo_coordinates_specialists(run):
    models = Models(
        coo=[
            call("delegate", "c1", agent="engineering", objective="MVP engineering readiness",
                 deliverables="readiness checklist"),
            call("delegate", "c2", agent="marketing", objective="Initial launch marketing plan"),
            say("Workstreams: engineering (done), marketing (done)."),
        ],
        engineering=[say("Checklist: CI, monitoring, rollback.")],
        marketing=[say("Plan: waitlist, launch post.")],
    )
    out = run("Coordinate the engineering and marketing work required to launch our MVP.",
              ScriptedDecider(decision("coo", complexity="multi_agent")), models)

    assert (out["route"], out["status"]) == ("coo", "completed")
    assert [s["agent"] for s in out["task"]["subtasks"]] == ["engineering", "marketing"]
    assert out["metrics"]["delegations"] == [
        {"from": "coo", "to": "engineering"},
        {"from": "coo", "to": "marketing"},
    ]
    brief = models["engineering"].calls[0][-1].content
    assert "Objective: MVP engineering readiness" in brief and "Requested by: coo" in brief


def test_specialized_research_goes_directly_to_researcher(run, research):
    models = Models(research=[call("web_search", query="receptionist competitors"), say("Report")])
    out = run("Research the competitors of this product.",
              ScriptedDecider(decision("research")), models)
    assert (out["route"], out["status"]) == ("research", "completed")
    assert out["metrics"]["delegations"] == []


def test_delegated_strategic_research_returns_to_cofounder(run, research):
    models = Models(
        cofounder=[
            call("delegate", agent="research", objective="Size the dental AI receptionist market"),
            lambda msgs: say("Based on research: REFINE." if "status: completed" in msgs[-1].content
                             else "unexpected"),
        ],
        research=[call("web_search", query="dental AI receptionist market"), say("TAM ~ $X")],
    )
    out = run("Is the dental AI receptionist market worth entering?",
              ScriptedDecider(decision("cofounder")), models)

    assert out["route"] == "cofounder"
    assert out["response"] == "Based on research: REFINE."
    assert out["task"]["subtasks"][0]["agent"] == "research"


def test_founder_led_execution_coo_reports_back_to_founder(run):
    models = Models(
        cofounder=[
            call("delegate", agent="coo", objective="Prepare an MVP execution plan"),
            say("COO's plan is ready; here is my take."),
        ],
        coo=[call("delegate", agent="engineering", objective="Engineering plan"), say("Plan done")],
        engineering=[say("Eng plan")],
    )
    out = run("Prepare an MVP execution plan based on the strategy we selected.",
              ScriptedDecider(decision("cofounder")), models)

    assert (out["route"], out["status"]) == ("cofounder", "completed")
    coo_task = out["task"]["subtasks"][0]
    assert coo_task["agent"] == "coo"
    assert coo_task["subtasks"][0]["agent"] == "engineering"


def test_unauthorized_private_memory_request_is_denied(run, memory_store, research):
    memory_store.add("private", "Secret: considering selling the company.")
    models = Models(
        research=[
            call("recall_memory", query="company plans", scope="private"),
            lambda msgs: say("Denied as expected." if "Access denied" in msgs[-1].content else "!"),
        ]
    )
    out = run("Research our options.", ScriptedDecider(decision("research")), models)

    assert out["response"] == "Denied as expected."
    assert out["metrics"]["memory_denials"] == [
        {"agent": "research", "scope": "private", "via": "recall"}
    ]
    assert out["metrics"]["memory_reads"] == []


def test_failed_delegated_task_is_reported_not_hidden(run):
    models = Models(
        coo=[
            call("delegate", agent="engineering", objective="Engineering readiness"),
            lambda msgs: say("Engineering failed; launch readiness is incomplete."
                             if "status: failed" in msgs[-1].content else "!"),
        ],
        engineering=[RuntimeError("model provider timeout")],
    )
    out = run("Prepare everything for the MVP launch.", ScriptedDecider(decision("coo")), models)

    assert out["route"] == "coo"
    assert out["status"] == "partial"  # COO cannot claim completion over a failed subtask
    assert out["task"]["subtasks"][0]["status"] == "failed"
    assert any(f.get("agent") == "engineering" for f in out["metrics"]["failures"])


def test_session_turns_are_persisted(run, session_store):
    out = run("What is an API?", ScriptedDecider(decision("direct")), Models(direct=[say("A")]))
    turns = session_store.load(out["session_id"]).turns
    assert [(t.role, t.agent) for t in turns] == [("user", None), ("assistant", "direct")]
