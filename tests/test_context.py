import pytest

from foundry.context import RECENT_TURNS, Session, Turn, prepare_context


def _turns(n: int) -> list[Turn]:
    out = []
    for i in range(n):
        out.append(Turn(role="user", content=f"question {i}"))
        out.append(Turn(role="assistant", content=f"answer {i}", agent="research", task_id=f"t{i}"))
    return out


def test_new_session_has_no_history(session_store):
    session = session_store.load(None)
    ctx = prepare_context(session, "What is an API?")
    assert ctx.recent == [] and ctx.last_agent is None
    assert ctx.router_state("What is an API?") == {"current_request": "What is an API?"}


def test_follow_up_keeps_recent_turns_and_last_agent(session_store):
    session_store.append(
        "s1",
        Turn(role="user", content="Research the AI receptionist market."),
        Turn(role="assistant", content="The market ...", agent="research", task_id="t1"),
    )
    ctx = prepare_context(session_store.load("s1"), "What about its competition?")
    state = ctx.router_state("What about its competition?")
    assert "AI receptionist market" in state["recent_conversation"][0]
    assert state["previous_turn_handled_by"] == "research"
    assert ctx.active_task_id == "t1"
    assert len(ctx.history_messages()) == 2


def test_window_is_bounded_and_older_requests_are_referenced():
    ctx = prepare_context(Session(id="s", turns=_turns(10)), "next")
    assert len(ctx.recent) == RECENT_TURNS
    assert ctx.earlier_requests[-1] == "question 6"
    assert "question 0" not in ctx.background()


def test_existing_summary_replaces_raw_older_requests():
    ctx = prepare_context(Session(id="s", turns=_turns(10), summary="We discussed dental AI."), "x")
    assert ctx.earlier_requests == []
    assert ctx.router_state("x")["conversation_summary"] == "We discussed dental AI."


def test_session_ids_are_validated(session_store):
    with pytest.raises(ValueError):
        session_store.load("../../etc/passwd")
