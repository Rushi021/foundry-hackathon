"""Tool registry: capabilities the system knows about, connected or not.

Add a tool by registering a ToolSpec here and listing its id on the agents allowed
to use it in ``foundry.agents.registry``. Routing picks new tools up automatically.
"""

from __future__ import annotations

from typing import Any

from foundry.tools.base import ToolSpec
from foundry.tools.research import build_research_tool


def _build_web_search(run: Any):
    return build_research_tool(run.research_provider(), run_id=run.run_id, cache=run.research_cache)


TOOLS: dict[str, ToolSpec] = {
    spec.id: spec
    for spec in (
        ToolSpec(
            id="web_search",
            description="Current information, news, facts, markets and competitors",
            build=_build_web_search,
        ),
        ToolSpec(
            id="gmail",
            description="Read or send the user's email",
            consequential=True,
        ),
        ToolSpec(
            id="google_calendar",
            description="Read the user's calendar or schedule meetings",
            consequential=True,
        ),
        ToolSpec(
            id="github",
            description="Repositories, issues and pull requests",
            consequential=True,
        ),
    )
}
