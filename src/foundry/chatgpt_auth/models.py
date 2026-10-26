"""List models available to the signed-in ChatGPT account."""

from __future__ import annotations

from typing import Any

import httpx

from foundry.chatgpt_auth.constants import API_BASE
from foundry.chatgpt_auth.oauth import get_valid_access_token


def list_models(*, visibility: str = "list") -> list[dict[str, Any]]:
    """
    Return account models for the picker (``visibility == "list"`` by default).

    Each item: ``{slug, display_name, …}`` preserving server order.
    """
    token = get_valid_access_token()
    # trust_env=False avoids local HTTP(S)_PROXY intercepts that 403 api.openai.com
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        res = client.get(
            f"{API_BASE}/models",
            headers={"Authorization": f"Bearer {token}"},
        )
    if res.status_code != 200:
        raise RuntimeError(f"List models failed ({res.status_code}): {res.text[:500]}")
    payload = res.json()
    models = payload.get("models") or payload.get("data") or []
    out: list[dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        # SIWC catalog uses visibility == "list"; Completions-style lists omit it.
        if "visibility" in m and visibility and m.get("visibility") != visibility:
            continue
        slug = m.get("slug") or m.get("id")
        if not slug:
            continue
        out.append(
            {
                "slug": slug,
                "display_name": m.get("display_name") or m.get("name") or slug,
                "visibility": m.get("visibility"),
            }
        )
    return out
