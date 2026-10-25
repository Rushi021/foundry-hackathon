"""OAuth authorization-code + PKCE flow for ChatGPT plan usage (SIWC OSS)."""

from __future__ import annotations

import html
import http.server
import socket
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from foundry.chatgpt_auth.constants import (
    AGENT_NAME,
    AUTHORIZE_URL,
    CALLBACK_HOST,
    CALLBACK_PATH,
    DEFAULT_CALLBACK_PORT,
    DYNAMIC_CLIENT_ID,
    OPENID_CONFIGURATION_URL,
    PLAN_USAGE_SCOPE,
    RESOURCE,
    SCOPES,
    TOKEN_URL,
)
from foundry.chatgpt_auth.id_token import validate_id_token
from foundry.chatgpt_auth.pkce import challenge_s256, generate_verifier, random_token
from foundry.chatgpt_auth.store import (
    ChatGPTCredentials,
    clear_tokens,
    get_or_create_host_id,
    load_active_credentials,
    load_all_credentials,
    save_credentials,
    utc_now_iso,
)


@dataclass
class PendingAuth:
    state: str
    nonce: str
    code_verifier: str
    redirect_uri: str
    client_id: str  # dynamic_agent_client or previously issued id
    is_registration: bool
    host_id: str


@dataclass
class CallbackResult:
    code: str | None = None
    state: str | None = None
    client_id: str | None = None
    scope: str | None = None
    error: str | None = None
    error_description: str | None = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: CallbackResult | None = None
    expected_path: str = CALLBACK_PATH
    done: threading.Event | None = None

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != self.expected_path:
            self.send_response(404)
            self.end_headers()
            return

        qs = urllib.parse.parse_qs(parsed.query)
        result = CallbackResult(
            code=(qs.get("code") or [None])[0],
            state=(qs.get("state") or [None])[0],
            client_id=(qs.get("client_id") or [None])[0],
            scope=(qs.get("scope") or [None])[0],
            error=(qs.get("error") or [None])[0],
            error_description=(qs.get("error_description") or [None])[0],
        )
        _CallbackHandler.result = result

        if result.error:
            body = (
                "<html><body><h1>Authorization declined</h1>"
                f"<p>{html.escape(result.error_description or result.error)}</p>"
                "<p>You can close this window.</p></body></html>"
            )
        else:
            body = (
                "<html><body><h1>Signed in to Foundry</h1>"
                "<p>You can close this window and return to the terminal.</p>"
                "</body></html>"
            )
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        if self.done:
            self.done.set()


def _pick_port(preferred: int = DEFAULT_CALLBACK_PORT) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((CALLBACK_HOST, port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("Could not bind a loopback callback port on 127.0.0.1")


def build_authorize_url(pending: PendingAuth) -> str:
    params: dict[str, str] = {
        "client_id": pending.client_id,
        "response_type": "code",
        "redirect_uri": pending.redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": pending.state,
        "nonce": pending.nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge_s256(pending.code_verifier),
        "ext_agent_host_id": pending.host_id,
    }
    if pending.is_registration:
        params["agent_name_hint"] = AGENT_NAME
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(
    *,
    code: str,
    client_id: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": RESOURCE,
    }
    with httpx.Client(timeout=60.0) as client:
        res = client.post(TOKEN_URL, data=data)
    if res.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed ({res.status_code}): {res.text[:500]}"
        )
    return res.json()


def refresh_tokens(creds: ChatGPTCredentials) -> ChatGPTCredentials:
    """Rotate access + refresh tokens; persist the replacement set atomically."""
    if not creds.refresh_token:
        raise RuntimeError("No refresh token available; run `foundry login` again.")
    data = {
        "grant_type": "refresh_token",
        "refresh_token": creds.refresh_token,
        "client_id": creds.client_id,
        "resource": RESOURCE,
    }
    with httpx.Client(timeout=60.0) as client:
        res = client.post(TOKEN_URL, data=data)
    if res.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed ({res.status_code}): {res.text[:500]}"
        )
    payload = res.json()
    scopes = _parse_scopes(payload.get("scope") or " ".join(creds.scopes))
    saved_at = utc_now_iso()
    expires_in = int(payload.get("expires_in", 3600))
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    updated = ChatGPTCredentials(
        email=creds.email,
        issuer=creds.issuer,
        subject=creds.subject,
        client_id=creds.client_id,
        ext_agent_host_id=creds.ext_agent_host_id,
        id_token=payload.get("id_token") or creds.id_token,
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token") or creds.refresh_token,
        token_type=payload.get("token_type", "Bearer"),
        expires_in=expires_in,
        scopes=scopes,
        saved_at=saved_at,
        expires_at=expires_at,
    )
    save_credentials(updated, make_active=True)
    return updated


_refresh_lock = threading.Lock()


def get_valid_access_token(*, skew_seconds: int = 120) -> str:
    """
    Return a usable access token for the active ChatGPT account.

    Refreshes when within ``skew_seconds`` of expiry. Requires
    ``chatgpt.tokens.use.direct`` on the granted scopes.
    """
    with _refresh_lock:
        creds = load_active_credentials()
        if creds is None or not creds.access_token:
            raise RuntimeError(
                "Not signed in with ChatGPT. Run: python -m foundry login"
            )
        if not creds.plan_usage_enabled():
            raise RuntimeError(
                "ChatGPT plan usage was not granted. Re-run login and approve "
                f"'{PLAN_USAGE_SCOPE}', or set OPENAI_API_KEY for API billing."
            )
        exp = creds.expires_at_dt()
        now = datetime.now(timezone.utc)
        if exp is None or now >= (exp - timedelta(seconds=skew_seconds)):
            creds = refresh_tokens(creds)
        return creds.access_token


def _parse_scopes(scope_value: str) -> list[str]:
    return sorted({s for s in scope_value.replace("+", " ").split() if s})


def _revocation_endpoint() -> str | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.get(OPENID_CONFIGURATION_URL)
        if res.status_code != 200:
            return None
        return res.json().get("revocation_endpoint")
    except httpx.HTTPError:
        return None


def revoke_refresh_token(creds: ChatGPTCredentials) -> bool:
    """Best-effort remote revocation. Returns True if confirmed."""
    endpoint = _revocation_endpoint()
    if not endpoint or not creds.refresh_token:
        return False
    data = {
        "token": creds.refresh_token,
        "token_type_hint": "refresh_token",
        "client_id": creds.client_id,
    }
    with httpx.Client(timeout=30.0) as client:
        res = client.post(endpoint, data=data)
    return res.status_code == 200


def login(
    *,
    open_browser: bool = True,
    timeout: float = 300.0,
    port: int | None = None,
    new_account: bool = False,
) -> ChatGPTCredentials:
    """
    Run dynamic client registration (or reauthorization) via loopback PKCE.

    Opens the system browser, waits for the callback, exchanges the code,
    validates the ID token, and stores credentials under ``~/.config/foundry/``.

    By default, reuses a saved issued ``client_id`` (returning sign-in).
    Pass ``new_account=True`` to register another ChatGPT account via
    ``dynamic_agent_client``.
    """
    host_id = get_or_create_host_id()
    existing = None if new_account else load_active_credentials()
    if existing is None and not new_account:
        saved = load_all_credentials()
        if len(saved) == 1:
            existing = next(iter(saved.values()))

    if (
        existing
        and existing.client_id
        and existing.client_id != DYNAMIC_CLIENT_ID
        and not new_account
    ):
        client_id = existing.client_id
        is_registration = False
    else:
        client_id = DYNAMIC_CLIENT_ID
        is_registration = True
        existing = None
    listen_port = port if port is not None else _pick_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{listen_port}{CALLBACK_PATH}"
    pending = PendingAuth(
        state=random_token(),
        nonce=random_token(),
        code_verifier=generate_verifier(),
        redirect_uri=redirect_uri,
        client_id=client_id,
        is_registration=is_registration,
        host_id=host_id,
    )

    done = threading.Event()
    _CallbackHandler.result = None
    _CallbackHandler.expected_path = CALLBACK_PATH
    _CallbackHandler.done = done

    server = http.server.HTTPServer((CALLBACK_HOST, listen_port), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    auth_url = build_authorize_url(pending)
    # Attach login hints for returning accounts (omit from logs that print the URL).
    if not is_registration and existing:
        hint_params: dict[str, str] = {}
        if existing.id_token:
            hint_params["id_token_hint"] = existing.id_token
        if existing.email:
            hint_params["login_hint"] = existing.email
        if hint_params:
            auth_url = auth_url + "&" + urllib.parse.urlencode(hint_params)

    print("Continue with ChatGPT in your browser.")
    # Redact id_token_hint from the printed URL (credential security).
    printable = auth_url
    if "id_token_hint=" in printable:
        printable = printable.split("id_token_hint=")[0] + "id_token_hint=<redacted>"
    print(f"If it does not open, visit:\n  {printable}")
    if open_browser:
        webbrowser.open(auth_url)

    if not done.wait(timeout):
        server.server_close()
        raise TimeoutError(
            f"No OAuth callback received within {timeout:.0f}s on {redirect_uri}"
        )
    thread.join(timeout=5)
    server.server_close()

    result = _CallbackHandler.result
    if result is None:
        raise RuntimeError("OAuth callback completed without a result.")
    if result.state != pending.state:
        raise RuntimeError("OAuth state mismatch (possible CSRF). Aborting.")
    if result.error:
        raise RuntimeError(
            f"Authorization error: {result.error}"
            + (f" — {result.error_description}" if result.error_description else "")
        )
    if not result.code:
        raise RuntimeError("Authorization callback missing code.")

    issued_client_id = result.client_id or pending.client_id
    if pending.is_registration:
        if not result.client_id or result.client_id == DYNAMIC_CLIENT_ID:
            raise RuntimeError(
                "Dynamic registration did not return an issued client_id."
            )
        issued_client_id = result.client_id
    elif result.client_id and result.client_id != pending.client_id:
        raise RuntimeError(
            "Callback client_id does not match the selected account registration."
        )

    token_payload = exchange_code(
        code=result.code,
        client_id=issued_client_id,
        code_verifier=pending.code_verifier,
        redirect_uri=pending.redirect_uri,
    )
    id_token = token_payload.get("id_token")
    if not isinstance(id_token, str) or not id_token:
        raise RuntimeError("Token response missing id_token.")

    claims = validate_id_token(
        id_token,
        audience=issued_client_id,
        nonce=pending.nonce,
    )
    if (
        existing
        and not pending.is_registration
        and claims["sub"] != existing.subject
    ):
        raise RuntimeError(
            "Signed-in identity does not match the selected ChatGPT account."
        )

    scopes = _parse_scopes(token_payload.get("scope") or result.scope or "")
    saved_at = utc_now_iso()
    expires_in = int(token_payload.get("expires_in", 3600))
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    creds = ChatGPTCredentials(
        email=claims.get("email"),
        issuer=claims.get("iss") or "https://auth.openai.com",
        subject=claims["sub"],
        client_id=issued_client_id,
        ext_agent_host_id=host_id,
        id_token=id_token,
        access_token=token_payload["access_token"],
        refresh_token=token_payload.get("refresh_token") or "",
        token_type=token_payload.get("token_type", "Bearer"),
        expires_in=expires_in,
        scopes=scopes,
        saved_at=saved_at,
        expires_at=expires_at,
    )
    save_credentials(creds, make_active=True)

    if not creds.plan_usage_enabled():
        print(
            "Signed in, but ChatGPT plan usage was not granted "
            f"(missing scope {PLAN_USAGE_SCOPE}). "
            "Re-run login and approve plan sharing, or set OPENAI_API_KEY."
        )
    else:
        print(
            f"Signed in as {creds.email or creds.subject}. "
            "ChatGPT plan usage enabled."
        )
        print("Manage limits: ChatGPT → Settings → Usage")
    return creds


def logout(*, revoke: bool = True) -> None:
    """Revoke (best-effort) and clear tokens for the active account."""
    creds = load_active_credentials()
    if creds is None:
        print("No active ChatGPT session.")
        return
    revoked = False
    if revoke and creds.refresh_token:
        try:
            revoked = revoke_refresh_token(creds)
        except Exception as exc:  # noqa: BLE001
            print(f"Remote revocation failed ({exc}); clearing local tokens.")
    clear_tokens(creds.client_id, remove_registration=False)
    if revoked:
        print("Signed out; refresh token revoked.")
    else:
        print(
            "Signed out locally. If remote revocation was not confirmed, "
            "disconnect the app in ChatGPT → Settings → Usage."
        )


def status() -> dict[str, Any]:
    creds = load_active_credentials()
    if creds is None:
        return {"signed_in": False}
    return {
        "signed_in": bool(creds.access_token or creds.refresh_token),
        "email": creds.email,
        "subject": creds.subject,
        "client_id": creds.client_id,
        "plan_usage": creds.plan_usage_enabled(),
        "scopes": creds.scopes,
        "expires_at": creds.expires_at,
        "host_id": creds.ext_agent_host_id,
    }

