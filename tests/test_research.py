from foundry.tools.research import (
    UnavailableResearchProvider,
    build_research_tool,
    get_research_provider,
)
from foundry.config import Settings


def test_unavailable_provider_soft_fails():
    provider = UnavailableResearchProvider()
    answer = provider.research("TAM for AI copilots")
    assert "unavailable" in answer.lower()
    assert "TAM for AI copilots" in answer


def test_get_research_provider_without_key():
    settings = Settings(
        perplexity_api_key=None,
        research_provider="perplexity",
    )
    # Force unset even if .env has a key
    object.__setattr__(settings, "perplexity_api_key", None)
    provider = get_research_provider(settings)
    assert provider.name == "unavailable"


def test_research_tool_invokes_provider():
    class Fake:
        name = "fake"

        def research(self, query: str) -> str:
            return f"answer:{query}"

    tool = build_research_tool(Fake())
    assert tool.invoke({"query": "competitors"}) == "answer:competitors"
