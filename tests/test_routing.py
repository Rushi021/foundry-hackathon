import json

import httpx
import pytest
from conftest import Models, decision, say

from foundry.agents.registry import routable_agents
from foundry.routing import RESOLVE, get_decision_providers, plan_route
from foundry.routing.jev import JevDecisionProvider, JevError, build_questions
from foundry.routing.llm_fallback import LLMDecisionProvider

DESTS = {k: a.route_description for k, a in routable_agents().items()}
TOOLS = {"web_search": "current info", "gmail": "email"}


def _plan(d, settings, route_pass=0, has_history=True):
    return plan_route(
        d, route_pass=route_pass, routable=set(DESTS), settings=settings, has_history=has_history
    )


def test_confident_decision_goes_to_destination(settings):
    plan = _plan(decision("research", 0.9), settings)
    assert plan.destination == "research"
    assert plan.preload_memory is False


def test_memory_probability_drives_preload(settings):
    assert _plan(decision("cofounder", memory_probability=0.8), settings).preload_memory


def test_uncertain_or_missing_info_goes_to_context_resolution(settings):
    assert _plan(decision("direct", 0.3), settings).destination == RESOLVE
    assert _plan(decision("clarify"), settings).destination == RESOLVE
    assert _plan(decision("direct", clarification_probability=0.9), settings).destination == RESOLVE


def test_low_confidence_without_history_skips_resolution(settings):
    plan = _plan(decision("coo", 0.3), settings, has_history=False)
    assert plan.destination == "direct"


def test_second_pass_never_loops(settings):
    assert _plan(decision("clarify"), settings, route_pass=1).destination == "direct"
    assert _plan(decision("coo", 0.2), settings, route_pass=1).destination == "direct"
    assert _plan(decision("coo", 0.9), settings, route_pass=1).destination == "coo"


def test_jev_questions_cover_all_routing_decisions():
    q = build_questions(DESTS, TOOLS)
    assert set(q) == {"destination", "complexity", "tool", "needs_memory", "needs_clarification"}
    assert set(q["destination"]["criteria"]) == {*DESTS, "clarify"}
    assert q["needs_memory"]["type"] == "noul"
    assert "none" in q["tool"]["criteria"]


def _jev_answers(choice="cofounder", probs=None):
    probs = probs or {choice: 0.9}
    return {
        "model": "jev-1.14.0",
        "answers": {
            "destination": {"type": "choice", "choice": choice, "probabilities": probs,
                            "confidence": 0.7},
            "complexity": {"type": "choice", "choice": "single_agent"},
            "tool": {"type": "choice", "choice": "web_search"},
            "needs_memory": {"type": "noul", "noul": 0.81},
            "needs_clarification": {"type": "noul", "noul": 0.05},
        },
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }


def test_jev_provider_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_jev_answers())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    jev = JevDecisionProvider("jv_test", client=client)
    d = jev.decide({"current_request": "Should I pursue this?"}, destinations=DESTS, tools=TOOLS)

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer jv_test"
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["state"] == {"current_request": "Should I pursue this?"}
    assert d.destination == "cofounder"
    assert d.confidence == 0.9  # probability of the chosen option
    assert d.tool_hint == "web_search"
    assert d.memory_probability == 0.81
    assert d.usage["input_tokens"] == 300


def test_jev_http_error_raises():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(JevError):
        JevDecisionProvider("k", client=client).decide({}, destinations=DESTS, tools=TOOLS)


def test_jev_unknown_destination_rejected():
    payload = _jev_answers(choice="lawyer")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(JevError):
        JevDecisionProvider("k", client=client).decide({}, destinations=DESTS, tools=TOOLS)


def test_llm_fallback_parses_fenced_json():
    reply = say(
        '```json\n{"destination":"direct","confidence":0.8,"complexity":"conversational",'
        '"tool":"none","needs_memory":0.1,"needs_clarification":0.0,"rationale":"simple"}\n```'
    )
    models = Models(router=[reply])
    d = LLMDecisionProvider("openai:x", models).decide({}, destinations=DESTS, tools=TOOLS)
    assert (d.destination, d.provider, d.rationale) == ("direct", "llm", "simple")


def test_llm_fallback_rejects_unknown_destination():
    models = Models(router=[say('{"destination":"lawyer","confidence":0.9}')])
    with pytest.raises(ValueError):
        LLMDecisionProvider("openai:x", models).decide({}, destinations=DESTS, tools=TOOLS)


def test_provider_selection(settings):
    primary, fallback, reason = get_decision_providers(settings, Models())
    assert primary is None and fallback.name == "llm" and "TYPESAFE_API_KEY" in reason

    keyed = settings.model_copy(update={"typesafe_api_key": "jv_x"})
    primary, _, reason = get_decision_providers(keyed, Models())
    assert primary.name == "jev" and reason is None

    forced = settings.model_copy(update={"router_provider": "jev"})
    with pytest.raises(ValueError):
        get_decision_providers(forced, Models())
