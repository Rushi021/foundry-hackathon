"""Real-time step progress for terminal verification (pre-Supabase)."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

_verbose: ContextVar[bool] = ContextVar("foundry_progress_verbose", default=True)


def set_verbose(enabled: bool) -> None:
    _verbose.set(enabled)


def emit(step: str, **payload: Any) -> None:
    """Print a step line to the terminal and (when streaming) push a custom event."""
    event = {"step": step, **payload}
    if _verbose.get():
        print(_format(event), flush=True)
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)
    except Exception:
        # Outside a LangGraph stream context — terminal print is enough.
        pass


def _format(event: dict[str, Any]) -> str:
    step = event.get("step", "?")
    run_id = event.get("run_id")
    prefix = f"[foundry:{run_id[:8]}]" if isinstance(run_id, str) and run_id else "[foundry]"
    details = {k: v for k, v in event.items() if k not in {"step", "run_id"} and v is not None}
    if not details:
        return f"{prefix} ▶ {step}"

    # Prefer short, readable fields; dump nested blobs as compact JSON.
    parts: list[str] = []
    for key, value in details.items():
        if isinstance(value, str) and len(value) > 400:
            value = value[:400] + "…"
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        parts.append(f"{key}={value}")
    return f"{prefix} ▶ {step} | " + " | ".join(parts)
