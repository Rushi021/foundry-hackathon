from foundry.agents.categorize import _parse_route_decision
from foundry.state import RouteDecision


def test_route_decision_model():
    d = RouteDecision(route="founder", rationale="strategy ask", confidence=0.9)
    assert d.route == "founder"
    assert 0 <= d.confidence <= 1


def test_parse_route_decision_from_streamed_text():
    text = (
        '{"route":"founder","rationale":"strategy ask","confidence":0.99}'
    )
    d = _parse_route_decision(text)
    assert d.route == "founder"
    assert d.confidence == 0.99


def test_parse_route_decision_from_fenced_json():
    text = 'Here you go:\n```json\n{"route":"lawyer","rationale":"contract","confidence":0.8}\n```'
    d = _parse_route_decision(text)
    assert d.route == "lawyer"
