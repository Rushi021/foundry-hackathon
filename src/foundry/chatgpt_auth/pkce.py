"""PKCE (S256) helpers for the ChatGPT OAuth flow."""

from __future__ import annotations

import base64
import hashlib
import secrets


def generate_verifier(nbytes: int = 64) -> str:
    """Return a high-entropy PKCE code verifier (base64url, no padding)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(nbytes)).rstrip(b"=").decode("ascii")


def challenge_s256(verifier: str) -> str:
    """Return the S256 code_challenge for ``verifier`` (base64url, no padding)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def random_token(nbytes: int = 32) -> str:
    """Return a URL-safe random token for ``state`` / ``nonce``."""
    return secrets.token_urlsafe(nbytes)
