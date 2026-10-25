from foundry.state import RouteDecision


def test_route_decision_model():
    d = RouteDecision(route="founder", rationale="strategy ask", confidence=0.9)
    assert d.route == "founder"
    assert 0 <= d.confidence <= 1
