"""Provider-agnostic chat model factory (API key or ChatGPT plan usage)."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ChatMessage, SystemMessage

from foundry.chatgpt_auth.store import load_active_credentials

# ChatGPT plan usage (SIWC) rejects standard API Completions model IDs.
# Account catalog uses plan slugs (e.g. gpt-5.6-luna, gpt-6.1-sol).
_CHATGPT_FAST_DEFAULT = "gpt-5.6-luna"
_CHATGPT_STRONG_DEFAULT = "gpt-6.1-sol"

# Names commonly set for OPENAI_API_KEY billing that fail on the plan-usage route.
_API_KEY_MODEL_PREFIXES = (
    "gpt-3.5",
    "gpt-4",
    "gpt-4o",
    "o1",
    "o3",
    "o4",
)


def chatgpt_plan_ready() -> bool:
    """True when a local ChatGPT session has subscription-sharing permission."""
    creds = load_active_credentials()
    return bool(
        creds
        and (creds.access_token or creds.refresh_token)
        and creds.plan_usage_enabled()
    )


def resolve_chatgpt_model(model: str) -> str:
    """
    Map API-key model IDs to ChatGPT plan catalog slugs.

    ``gpt-4o-mini`` / ``gpt-4o`` etc. are not valid on the subscription-sharing
    Responses route and return ``model_not_found``.
    """
    lowered = model.strip().lower()
    if any(lowered == p or lowered.startswith(p + "-") or lowered.startswith(p)
           for p in _API_KEY_MODEL_PREFIXES):
        if any(tag in lowered for tag in ("mini", "nano", "small")):
            return _CHATGPT_FAST_DEFAULT
        return _CHATGPT_STRONG_DEFAULT
    return model


def get_chat_model(model_spec: str, *, temperature: float = 0.2) -> BaseChatModel:
    """
    Create a chat model from a ``provider:model`` string.

    Examples:
        ``openai:gpt-4o``
        ``anthropic:claude-sonnet-4-5``

    When ChatGPT plan usage is signed in (and billing is not forced to
    ``api_key``), OpenAI models use the OAuth access token against the
    Responses API (``store=false``, streamed).
    """
    if ":" not in model_spec:
        raise ValueError(
            f"Invalid model spec '{model_spec}'. Use 'provider:model' "
            "(e.g. 'openai:gpt-4o-mini')."
        )
    provider, model = model_spec.split(":", 1)
    billing = (os.environ.get("FOUNDRY_LLM_BILLING") or "auto").strip().lower()

    use_chatgpt = (
        provider == "openai"
        and billing != "api_key"
        and (billing == "chatgpt" or chatgpt_plan_ready())
    )
    if use_chatgpt:
        return _chatgpt_responses_model(resolve_chatgpt_model(model))

    return init_chat_model(model_spec, temperature=temperature)


def _chatgpt_responses_model(model: str) -> BaseChatModel:
    """
    ChatGPT subscription-sharing path: Responses API + OAuth bearer.

    Preview requirements (OpenAI SIWC docs): ``store=false``, ``stream=true``;
    omit unsupported fields (e.g. temperature); use developer/instructions
    instead of ``role: system`` input items.
    """
    from langchain_openai import ChatOpenAI

    from foundry.chatgpt_auth.oauth import get_valid_access_token

    class _ChatGPTPlanModel(ChatOpenAI):
        """ChatOpenAI tuned for Sign in with ChatGPT plan usage."""

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict:
            messages = self._convert_input(input_).to_messages()
            rewritten = [
                ChatMessage(content=m.content, role="developer")
                if isinstance(m, SystemMessage)
                else m
                for m in messages
            ]
            payload = super()._get_request_payload(rewritten, stop=stop, **kwargs)
            # SIWC preview rejects these Completions-oriented fields.
            for key in (
                "temperature",
                "top_p",
                "top_logprobs",
                "max_output_tokens",
                "max_tokens",
                "max_completion_tokens",
                "user",
                "metadata",
                "truncation",
            ):
                payload.pop(key, None)
            payload["store"] = False
            payload["stream"] = True
            return payload

    return _ChatGPTPlanModel(
        model=model,
        api_key=get_valid_access_token,
        base_url="https://api.openai.com/v1",
        use_responses_api=True,
        store=False,
        streaming=True,
        max_retries=0,
    )
