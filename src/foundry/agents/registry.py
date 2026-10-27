"""Agent registry: who exists, what they own, and what they are authorized to use.

Add an agent by registering an AgentSpec. The router builds its destination options
from routable agents, so the workflow graph does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from foundry.agents import prompts


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    # Shown to the decision router as the option description (keep <= 300 chars).
    route_description: str
    persona: str
    # Settings attribute holding this agent's "provider:model".
    llm_setting: str
    tools: tuple[str, ...] = ()
    memory_scopes: frozenset[str] = frozenset()
    # Delegation: results come back to this agent, which integrates them.
    can_delegate_to: tuple[str, ...] = ()
    # Escalation: hand the whole request over; the target answers the user.
    can_escalate_to: tuple[str, ...] = ()
    routable: bool = True
    max_rounds: int = 4
    temperature: float = 0.2


AGENTS: dict[str, AgentSpec] = {
    spec.id: spec
    for spec in (
        AgentSpec(
            id="direct",
            name="Direct Assistant",
            route_description=(
                "Straightforward requests: general knowledge, explaining concepts, everyday "
                "questions, quick lookups of current information or news. No business "
                "strategy and no coordination of work."
            ),
            persona=prompts.DIRECT,
            llm_setting="assistant_llm",
            tools=("web_search", "gmail", "google_calendar"),
            memory_scopes=frozenset({"shared"}),
            can_escalate_to=("cofounder", "coo", "research"),
            max_rounds=3,
        ),
        AgentSpec(
            id="cofounder",
            name="Co-Founder",
            route_description=(
                "Strategic thinking partner: startup ideas, evaluating opportunities, business "
                "strategy, product direction, prioritization, trade-offs, business models, "
                "whether to pursue something, or decisions using the user's goals and history."
            ),
            persona=prompts.COFOUNDER,
            llm_setting="founder_llm",
            tools=("web_search",),
            memory_scopes=frozenset({"private", "shared"}),
            can_delegate_to=("research", "coo", "engineering", "marketing"),
            max_rounds=5,
            temperature=0.4,
        ),
        AgentSpec(
            id="coo",
            name="COO / Chief of Agents",
            route_description=(
                "Operational execution across workstreams: coordinate several teams, break an "
                "approved objective into tasks, plan and track multi-step work such as "
                "preparing a launch across engineering and marketing."
            ),
            persona=prompts.COO,
            llm_setting="agent_llm",
            memory_scopes=frozenset({"shared"}),
            can_delegate_to=("research", "engineering", "marketing", "cofounder"),
            max_rounds=6,
        ),
        AgentSpec(
            id="research",
            name="Research Agent",
            route_description=(
                "Dedicated research deliverable: in-depth market research, competitor analysis, "
                "industry landscape or evidence gathering on a specific topic."
            ),
            persona=prompts.RESEARCH,
            llm_setting="agent_llm",
            tools=("web_search",),
            memory_scopes=frozenset({"shared"}),
        ),
        AgentSpec(
            id="engineering",
            name="Engineering Agent",
            route_description=(
                "Engineering work: technical requirements, architecture, implementation plans, "
                "engineering readiness, investigating technical problems."
            ),
            persona=prompts.ENGINEERING,
            llm_setting="agent_llm",
            tools=("github",),
            memory_scopes=frozenset({"shared"}),
        ),
        AgentSpec(
            id="marketing",
            name="Marketing Agent",
            route_description=(
                "Marketing work: positioning statements, messaging, launch announcements, "
                "channel plans, content drafts."
            ),
            persona=prompts.MARKETING,
            llm_setting="agent_llm",
            memory_scopes=frozenset({"shared"}),
        ),
    )
}


def routable_agents() -> dict[str, AgentSpec]:
    return {k: v for k, v in AGENTS.items() if v.routable}


def memory_scopes_by_agent() -> dict[str, frozenset[str]]:
    return {k: v.memory_scopes for k, v in AGENTS.items()}
