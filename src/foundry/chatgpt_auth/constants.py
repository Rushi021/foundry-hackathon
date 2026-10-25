"""OpenAI Sign in with ChatGPT (subscription-sharing) constants."""

from __future__ import annotations

AGENT_NAME = "Foundry"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OPENID_CONFIGURATION_URL = "https://auth.openai.com/.well-known/openid-configuration"
JWKS_URL = "https://auth.openai.com/.well-known/jwks.json"
API_BASE = "https://api.openai.com/v1"
RESOURCE = "https://api.openai.com/v1"

# Identity + ChatGPT plan usage (subscription sharing)
SCOPES = (
    "openid profile email "
    "offline_access resource.invoke chatgpt.tokens.use.direct"
)
PLAN_USAGE_SCOPE = "chatgpt.tokens.use.direct"

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/auth/callback"
DEFAULT_CALLBACK_PORT = 1455

ISSUER = "https://auth.openai.com"

CONFIG_DIR_NAME = "foundry"
HOST_ID_FILENAME = "host_id"
CREDENTIALS_FILENAME = "chatgpt_credentials.json"
ACTIVE_ACCOUNT_FILENAME = "chatgpt_active.json"
