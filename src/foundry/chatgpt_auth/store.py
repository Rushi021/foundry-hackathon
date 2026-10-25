"""Local credential + host-ID persistence for ChatGPT plan usage."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from foundry.chatgpt_auth.constants import (
    ACTIVE_ACCOUNT_FILENAME,
    CONFIG_DIR_NAME,
    CREDENTIALS_FILENAME,
    HOST_ID_FILENAME,
    PLAN_USAGE_SCOPE,
)


def config_dir() -> Path:
    """Return ``~/.config/foundry`` (XDG-style), creating it if needed."""
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / CONFIG_DIR_NAME
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    return path


def _atomic_write(path: Path, data: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise

def get_or_create_host_id() -> str:
    """Persist and return a stable ``urn:uuid:…`` host identifier."""
    path = config_dir() / HOST_ID_FILENAME
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    host_id = f"urn:uuid:{uuid.uuid4()}"
    _atomic_write(path, host_id + "\n")
    return host_id


@dataclass
class ChatGPTCredentials:
    """One ChatGPT account registration + token set (official SIWC shape)."""

    email: str | None
    issuer: str
    subject: str
    client_id: str
    ext_agent_host_id: str
    id_token: str
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scopes: list[str] = field(default_factory=list)
    saved_at: str = ""
    expires_at: str = ""  # ISO UTC; computed from saved_at + expires_in

    def plan_usage_enabled(self) -> bool:
        return PLAN_USAGE_SCOPE in self.scopes

    def expires_at_dt(self) -> datetime | None:
        if not self.expires_at:
            return None
        return datetime.fromisoformat(self.expires_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatGPTCredentials:
        return cls(
            email=data.get("email"),
            issuer=data["issuer"],
            subject=data["subject"],
            client_id=data["client_id"],
            ext_agent_host_id=data["ext_agent_host_id"],
            id_token=data["id_token"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=int(data.get("expires_in", 3600)),
            scopes=list(data.get("scopes") or []),
            saved_at=data.get("saved_at") or "",
            expires_at=data.get("expires_at") or "",
        )


def _credentials_path() -> Path:
    return config_dir() / CREDENTIALS_FILENAME


def _active_path() -> Path:
    return config_dir() / ACTIVE_ACCOUNT_FILENAME


def load_all_credentials() -> dict[str, ChatGPTCredentials]:
    """Map issued ``client_id`` → credential record."""
    path = _credentials_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    accounts = raw.get("accounts", raw) if isinstance(raw, dict) else {}
    out: dict[str, ChatGPTCredentials] = {}
    for client_id, payload in accounts.items():
        if isinstance(payload, dict):
            out[client_id] = ChatGPTCredentials.from_dict(payload)
    return out


def save_credentials(creds: ChatGPTCredentials, *, make_active: bool = True) -> None:
    accounts = {cid: c.to_dict() for cid, c in load_all_credentials().items()}
    accounts[creds.client_id] = creds.to_dict()
    _atomic_write(_credentials_path(), json.dumps({"accounts": accounts}, indent=2) + "\n")
    if make_active:
        set_active_client_id(creds.client_id)


def set_active_client_id(client_id: str) -> None:
    _atomic_write(_active_path(), json.dumps({"client_id": client_id}) + "\n")


def get_active_client_id() -> str | None:
    path = _active_path()
    if not path.exists():
        accounts = load_all_credentials()
        if len(accounts) == 1:
            return next(iter(accounts))
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("client_id")


def load_active_credentials() -> ChatGPTCredentials | None:
    client_id = get_active_client_id()
    if not client_id:
        return None
    return load_all_credentials().get(client_id)


def clear_tokens(client_id: str, *, remove_registration: bool = False) -> None:
    """Clear access/refresh/id tokens; optionally drop the registration entirely."""
    accounts = load_all_credentials()
    if client_id not in accounts:
        return
    if remove_registration:
        del accounts[client_id]
    else:
        c = accounts[client_id]
        c.access_token = ""
        c.refresh_token = ""
        c.id_token = ""
        c.scopes = []
        c.expires_at = ""
        accounts[client_id] = c
    _atomic_write(
        _credentials_path(),
        json.dumps({"accounts": {k: v.to_dict() for k, v in accounts.items()}}, indent=2)
        + "\n",
    )
    active = get_active_client_id()
    if active == client_id:
        remaining = list(accounts.keys())
        if remaining:
            set_active_client_id(remaining[0])
        elif _active_path().exists():
            _active_path().unlink()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
