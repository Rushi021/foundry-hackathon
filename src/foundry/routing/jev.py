"""Jev (TypeSafe AI) decision provider — typed questions in, calibrated probabilities out.

API: POST {base_url}/systemone with ``model``, ``state`` and named ``questions``
(https://docs.typesafe.ai/api/system-one). All routing questions go in one round trip.
"""

from __future__ import annotations

from typing import Any

import httpx

from foundry.routing.models import (
    CLARIFY,
    CLARIFY_OPTION,
    COMPLEXITY_OPTIONS,
    NO_TOOL,
    RoutingDecision,
)


class JevError(RuntimeError):
    pass


def build_questions(destinations: dict[str, str], tools: dict[str, str]) -> dict[str, Any]:
    return {
        "destination": {
            "type": "choice",
            "instructions": (
                "Who should handle current_request, read in the context of the recent "
                "conversation? Prefer the simplest destination that can fully handle it."
            ),
            "criteria": {**destinations, CLARIFY: CLARIFY_OPTION},
        },
        "complexity": {
            "type": "choice",
            "instructions": "How complex is current_request to execute?",
            "criteria": COMPLEXITY_OPTIONS,
        },
        "tool": {
            "type": "choice",
            "instructions": "Which external capability, if any, does current_request most need?",
            "criteria": {NO_TOOL: "No external tool; can be answered from knowledge.", **tools},
        },
        "needs_memory": {
            "type": "noul",
            "instructions": (
                "Does a good answer depend on the user's own stored history — their projects, "
                "goals, preferences or prior decisions — beyond the recent conversation shown?"
            ),
        },
        "needs_clarification": {
            "type": "noul",
            "instructions": (
                "Is essential information missing from current_request that cannot be "
                "inferred from the recent conversation?"
            ),
            "criteria": {
                "true": "Refers to an unknown person, object or detail that is required to act.",
                "false": "The request is self-contained or its references are resolvable.",
            },
        },
    }


class JevDecisionProvider:
    name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1",
        timeout_s: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = base_url.rstrip("/") + "/systemone"
        self._timeout = timeout_s
        self._client = client

    def decide(
        self,
        state: dict[str, Any],
        *,
        destinations: dict[str, str],
        tools: dict[str, str],
    ) -> RoutingDecision:
        body = {
            "model": self._model,
            "state": state,
            "questions": build_questions(destinations, tools),
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._client or httpx.Client(timeout=self._timeout)
        try:
            res = client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise JevError(f"Jev request failed: {exc}") from exc
        finally:
            if self._client is None:
                client.close()
        if res.status_code != 200:
            raise JevError(f"Jev returned HTTP {res.status_code}: {res.text[:300]}")
        return parse_response(res.json(), destinations=destinations, tools=tools)


def parse_response(
    payload: dict[str, Any],
    *,
    destinations: dict[str, str],
    tools: dict[str, str],
) -> RoutingDecision:
    try:
        answers = payload["answers"]
        dest = answers["destination"]
        choice = dest["choice"]
        probabilities = {k: float(v) for k, v in (dest.get("probabilities") or {}).items()}
        complexity = answers["complexity"]["choice"]
        tool = answers["tool"]["choice"]
        memory = float(answers["needs_memory"]["noul"])
        clarify = float(answers["needs_clarification"]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"Unexpected Jev response shape: {str(payload)[:300]}") from exc

    if choice not in destinations and choice != CLARIFY:
        raise JevError(f"Jev chose unknown destination {choice!r}")
    return RoutingDecision(
        destination=choice,
        # Probability of the chosen option is the most interpretable confidence signal.
        confidence=probabilities.get(choice, float(dest.get("confidence", 0.0))),
        probabilities=probabilities,
        complexity=complexity if complexity in COMPLEXITY_OPTIONS else "conversational",
        memory_probability=memory,
        clarification_probability=clarify,
        tool_hint=tool if tool in tools else NO_TOOL,
        provider="jev",
        usage=payload.get("usage"),
    )
