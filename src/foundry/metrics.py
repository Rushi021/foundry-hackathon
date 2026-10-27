"""Per-run efficiency metrics: model calls, tokens, tools, delegations, memory, failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    router_provider: str | None = None
    router_fallback_reason: str | None = None
    decision_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls_by_purpose: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    delegations: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    memory_reads: list[dict[str, Any]] = field(default_factory=list)
    memory_denials: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0

    def record_model_call(self, purpose: str, usage: dict[str, Any] | None) -> None:
        self.model_calls += 1
        self.model_calls_by_purpose[purpose] = self.model_calls_by_purpose.get(purpose, 0) + 1
        self.add_usage(usage)

    def record_decision_call(self, usage: dict[str, Any] | None) -> None:
        self.decision_calls += 1
        self.add_usage(usage)

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
