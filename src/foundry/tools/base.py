"""Tool registry primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool


class ToolError(RuntimeError):
    """Raised by a tool when it could not do its job; the runtime records the failure."""


@dataclass(frozen=True)
class ToolSpec:
    id: str
    description: str
    # Builds the LangChain tool for one run; None means the capability is registered
    # but not connected yet (agents are told so instead of pretending).
    build: Callable[[Any], StructuredTool] | None = None
    # Consequential tools (send email, create event, open PR) never execute without
    # explicit user approval.
    consequential: bool = False

    @property
    def available(self) -> bool:
        return self.build is not None
