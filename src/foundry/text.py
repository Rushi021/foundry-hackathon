"""Helpers for reading LangChain messages."""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return str(content)


def usage_of(message: Any) -> dict[str, Any] | None:
    usage = getattr(message, "usage_metadata", None)
    return dict(usage) if usage else None
