"""Answer-only research providers (Perplexity-like).

The agent receives a researched *answer*, not a DIY search/browse workflow.
Swap providers later when users connect apps for different tasks.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from langchain_core.tools import StructuredTool
from openai import OpenAI
from pydantic import BaseModel, Field

from foundry.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ResearchQuery(BaseModel):
    query: str = Field(description="Focused research question needing a grounded answer.")


@runtime_checkable
class ResearchProvider(Protocol):
    name: str

    def research(self, query: str) -> str:
        """Return a researched answer string for the query."""


class PerplexityResearchProvider:
    """Default research: Perplexity Sonar — answer only."""

    name = "perplexity"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "sonar",
        base_url: str = "https://api.perplexity.ai",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def research(self, query: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Answer concisely with facts. "
                        "Cite key sources inline when relevant. Do not pad with filler."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        content = completion.choices[0].message.content
        return content or "No research answer returned."


class UnavailableResearchProvider:
    """Soft-fail when research is not configured."""

    name = "unavailable"

    def research(self, query: str) -> str:
        return (
            "Research is unavailable: PERPLEXITY_API_KEY is not set. "
            f"Cannot ground the query: {query!r}. Proceed with clear uncertainty "
            "and do not invent market facts."
        )


def get_research_provider(settings: Settings | None = None) -> ResearchProvider:
    settings = settings or get_settings()
    provider = (settings.research_provider or "perplexity").lower()

    if provider == "perplexity":
        if not settings.perplexity_configured:
            logger.warning("PERPLEXITY_API_KEY missing — research will soft-fail.")
            return UnavailableResearchProvider()
        return PerplexityResearchProvider(
            api_key=settings.perplexity_api_key,  # type: ignore[arg-type]
            model=settings.perplexity_model,
            base_url=settings.perplexity_base_url,
        )

    raise ValueError(
        f"Unknown RESEARCH_PROVIDER={provider!r}. Supported: 'perplexity'. "
        "Add a new ResearchProvider implementation to plug in other apps."
    )


def build_research_tool(provider: ResearchProvider | None = None) -> StructuredTool:
    provider = provider or get_research_provider()

    def _run(query: str) -> str:
        return provider.research(query)

    return StructuredTool.from_function(
        func=_run,
        name="research",
        description=(
            "Get a grounded research answer for a specific question "
            "(market, competitors, facts). Returns the answer only — "
            "use when you need evidence, not vibes."
        ),
        args_schema=ResearchQuery,
    )
