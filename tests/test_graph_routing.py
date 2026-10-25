from foundry.agents.unsupported import unsupported_node
from foundry.graph import _route_after_categorize, build_graph


def test_route_helper_founder():
    assert _route_after_categorize({"route": "founder"}) == "founder"


def test_route_helper_other():
    assert _route_after_categorize({"route": "lawyer"}) == "unsupported"
    assert _route_after_categorize({"route": "unsupported"}) == "unsupported"


def test_unsupported_stub_names_future_agent():
    out = unsupported_node(
        {
            "route": "lawyer",
            "route_rationale": "legal contract question",
            "user_prompt": "Review this NDA",
        }
    )
    assert "Lawyer" in out["final_response"]
    assert "not built yet" in out["final_response"].lower()


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None
