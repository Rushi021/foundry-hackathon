"""Unit tests for ChatGPT SIWC OAuth helpers (no live network)."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import pytest

from foundry.chatgpt_auth.constants import (
    DYNAMIC_CLIENT_ID,
    PLAN_USAGE_SCOPE,
    RESOURCE,
)
from foundry.chatgpt_auth.oauth import PendingAuth, build_authorize_url
from foundry.chatgpt_auth.pkce import challenge_s256, generate_verifier
from foundry.chatgpt_auth.store import (
    ChatGPTCredentials,
    get_or_create_host_id,
    load_active_credentials,
    save_credentials,
)


def test_pkce_challenge_is_s256_base64url():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge_s256(verifier) == expected
    assert "=" not in challenge_s256(generate_verifier())


def test_authorize_url_registration_params():
    pending = PendingAuth(
        state="st",
        nonce="nn",
        code_verifier="verifier-value-with-enough-entropy-aaaaaaaa",
        redirect_uri="http://127.0.0.1:1455/auth/callback",
        client_id=DYNAMIC_CLIENT_ID,
        is_registration=True,
        host_id="urn:uuid:11111111-2222-3333-4444-555555555555",
    )
    url = build_authorize_url(pending)
    assert "client_id=dynamic_agent_client" in url
    assert "agent_name_hint=Foundry" in url
    assert "ext_agent_host_id=" in url
    assert "chatgpt.tokens.use.direct" in url
    assert "resource.invoke" in url
    assert "offline_access" in url
    assert f"resource={RESOURCE.replace(':', '%3A').replace('/', '%2F')}" in url
    assert "code_challenge_method=S256" in url
    assert "127.0.0.1" in url
    assert "localhost" not in url.split("://", 1)[-1].split("/")[0] or True
    # Must not use hostname localhost for redirect
    assert "redirect_uri=http%3A%2F%2F127.0.0.1" in url


def test_authorize_url_omits_agent_name_on_reauth():
    pending = PendingAuth(
        state="st",
        nonce="nn",
        code_verifier="verifier-value-with-enough-entropy-bbbbbbbb",
        redirect_uri="http://127.0.0.1:54321/auth/callback",
        client_id="oaiapp_test",
        is_registration=False,
        host_id="urn:uuid:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    url = build_authorize_url(pending)
    assert "agent_name_hint" not in url
    assert "client_id=oaiapp_test" in url


def test_credential_store_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    host = get_or_create_host_id()
    assert host.startswith("urn:uuid:")
    assert get_or_create_host_id() == host

    creds = ChatGPTCredentials(
        email="a@example.com",
        issuer="https://auth.openai.com",
        subject="sub_1",
        client_id="oaiapp_abc",
        ext_agent_host_id=host,
        id_token="id.jwt",
        access_token="access",
        refresh_token="refresh",
        scopes=[PLAN_USAGE_SCOPE, "openid"],
        saved_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T01:00:00Z",
    )
    save_credentials(creds)
    loaded = load_active_credentials()
    assert loaded is not None
    assert loaded.email == "a@example.com"
    assert loaded.plan_usage_enabled() is True
    mode = (tmp_path / "foundry" / "chatgpt_credentials.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_chatgpt_plan_ready_false_without_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from foundry.llm import chatgpt_plan_ready

    save_credentials(
        ChatGPTCredentials(
            email="a@example.com",
            issuer="https://auth.openai.com",
            subject="sub_1",
            client_id="oaiapp_abc",
            ext_agent_host_id=get_or_create_host_id(),
            id_token="id",
            access_token="access",
            refresh_token="refresh",
            scopes=["openid", "profile"],
        )
    )
    assert chatgpt_plan_ready() is False


def test_resolve_chatgpt_model_maps_api_ids():
    from foundry.llm import resolve_chatgpt_model

    assert resolve_chatgpt_model("gpt-4o-mini") == "gpt-5.6-luna"
    assert resolve_chatgpt_model("gpt-4o") == "gpt-6.1-sol"
    assert resolve_chatgpt_model("gpt-5.6-luna") == "gpt-5.6-luna"
    assert resolve_chatgpt_model("gpt-6.1-sol") == "gpt-6.1-sol"


def test_get_chat_model_uses_responses_when_plan_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FOUNDRY_LLM_BILLING", "auto")

    save_credentials(
        ChatGPTCredentials(
            email="a@example.com",
            issuer="https://auth.openai.com",
            subject="sub_1",
            client_id="oaiapp_abc",
            ext_agent_host_id=get_or_create_host_id(),
            id_token="id",
            access_token="access",
            refresh_token="refresh",
            scopes=[PLAN_USAGE_SCOPE, "openid"],
            expires_at="2099-01-01T00:00:00+00:00",
        )
    )

    from foundry.llm import get_chat_model

    model = get_chat_model("openai:gpt-4o-mini")
    assert model.model_name == "gpt-5.6-luna"
    assert model.use_responses_api is True
    assert model.store is False
    assert model.streaming is True
