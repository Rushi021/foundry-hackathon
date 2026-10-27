"""Test doubles: scripted chat models, scripted router decisions, isolated storage."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from foundry.config import Settings
from foundry.context import SessionStore
from foundry.memory import MemoryStore
from foundry.routing.models import RoutingDecision


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedModel:
    def __init__(self, purpose: str, script: list[Any]) -> None:
        self.purpose = purpose
        self.script = list(script)
        self.calls: list[list[Any]] = []
        self.bound: list[str] = []

    def bind_tools(self, tools: list[Any]) -> ScriptedModel:
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError(f"Unscripted model call for '{self.purpose}'")
        step = self.script.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, Exception):
            raise step
        return step

    def system_prompt(self, index: int = 0) -> str:
        return self.calls[index][0].content


class Models:
    """Model factory keyed by purpose (agent id, 'router', 'resolver')."""

    def __init__(self, **scripts: list[Any]) -> None:
        self.models = {k: ScriptedModel(k, v) for k, v in scripts.items()}

    def __call__(self, model_spec: str, *, purpose: str, temperature: float = 0.2) -> ScriptedModel:
        if purpose not in self.models:
            raise AssertionError(f"Unexpected model use: '{purpose}'")
        return self.models[purpose]

    def __getitem__(self, purpose: str) -> ScriptedModel:
        return self.models[purpose]


class ScriptedDecider:
    name = "jev"

    def __init__(self, *decisions: RoutingDecision | Exception) -> None:
        self.decisions = list(decisions)
        self.states: list[dict[str, Any]] = []

    def decide(self, state, *, destinations, tools) -> RoutingDecision:
        self.states.append(state)
        step = self.decisions.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def decision(destination: str, confidence: float = 0.95, **kw: Any) -> RoutingDecision:
    return RoutingDecision(destination=destination, confidence=confidence, provider="jev", **kw)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        foundry_data_dir=str(tmp_path),
        typesafe_api_key=None,
        perplexity_api_key=None,
        supabase_url=None,
        supabase_service_role_key=None,
    )


@pytest.fixture
def memory_store(settings) -> MemoryStore:
    return MemoryStore(settings.foundry_data_dir)


@pytest.fixture
def session_store(settings) -> SessionStore:
    return SessionStore(settings.foundry_data_dir)


class FakeResearch:
    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def research(self, query: str) -> str:
        self.queries.append(query)
        return f"researched: {query}"


@pytest.fixture
def research(monkeypatch) -> FakeResearch:
    fake = FakeResearch()
    monkeypatch.setattr("foundry.runtime.get_research_provider", lambda settings=None: fake)
    return fake
