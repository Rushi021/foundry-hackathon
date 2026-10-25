"""Optional Supabase run logging — soft no-op when unset."""

from __future__ import annotations

import logging
from typing import Any

from foundry.config import Settings, get_settings

logger = logging.getLogger(__name__)


def log_workflow_run(
    *,
    run_id: str,
    user_prompt: str,
    route: str | None,
    route_rationale: str | None,
    response: str | None,
    latency_ms: int | None,
    metadata: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> bool:
    settings = settings or get_settings()
    if not settings.supabase_configured:
        logger.debug("Supabase not configured — skipping workflow_runs insert.")
        return False

    try:
        from supabase import create_client

        client = create_client(
            settings.supabase_url,  # type: ignore[arg-type]
            settings.supabase_service_role_key,  # type: ignore[arg-type]
        )
        payload = {
            "id": run_id,
            "user_prompt": user_prompt,
            "route": route,
            "route_rationale": route_rationale,
            "response": response,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }
        client.table("workflow_runs").insert(payload).execute()
        return True
    except Exception:
        logger.exception("Failed to log workflow run to Supabase (soft-fail).")
        return False
