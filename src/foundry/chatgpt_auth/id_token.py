"""Validate ChatGPT OIDC ID tokens against OpenAI JWKS."""

from __future__ import annotations

from typing import Any

import jwt
from jwt import PyJWKClient

from foundry.chatgpt_auth.constants import ISSUER, JWKS_URL

_jwks_client: PyJWKClient | None = None


def _client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, lifespan=3600)
    return _jwks_client


def validate_id_token(
    id_token: str,
    *,
    audience: str,
    nonce: str,
) -> dict[str, Any]:
    """
    Verify signature, issuer, audience (issued client_id), exp, and nonce.

    Returns the validated JWT claims. Raises ``jwt.InvalidTokenError`` on failure.
    """
    signing_key = _client().get_signing_key_from_jwt(id_token)
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256", "ES256"],
        audience=audience,
        issuer=ISSUER,
        options={
            "require": ["sub", "exp", "iat"],
        },
        leeway=5,
    )
    token_nonce = claims.get("nonce")
    if token_nonce != nonce:
        raise jwt.InvalidTokenError("ID token nonce did not match the authorization attempt.")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise jwt.InvalidTokenError("ID token missing subject.")
    return claims
