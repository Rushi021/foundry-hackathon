from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM: "provider:model" — swap without code changes
    # Defaults target ChatGPT plan catalog slugs (SIWC). For API-key billing,
    # set openai:gpt-4o-mini / openai:gpt-4o (or any Completions model you pay for).
    # Fast model: LLM router fallback + context resolution.
    categorize_llm: str = "openai:gpt-5.6-luna"
    # Co-Founder agent.
    founder_llm: str = "openai:gpt-6.1-sol"
    # Direct Assistant.
    assistant_llm: str = "openai:gpt-5.6-luna"
    # COO and specialist agents.
    agent_llm: str = "openai:gpt-6.1-sol"
    # auto | chatgpt | api_key — prefer Sign in with ChatGPT plan usage when signed in
    foundry_llm_billing: str = "auto"

    # Decision router: auto (Jev if TYPESAFE_API_KEY is set, else LLM fallback) | jev | llm
    router_provider: str = "auto"
    typesafe_api_key: str | None = None
    jev_model: str = "jev-latest"
    jev_base_url: str = "https://api.typesafe.ai/v1"
    jev_timeout_s: float = 15.0
    # Below this probability for the chosen destination, routing is treated as uncertain.
    router_min_confidence: float = 0.45
    # At or above this, the request is likely missing essential information.
    router_clarify_threshold: float = 0.7
    # At or above this, the selected agent preloads authorized long-term memory.
    router_memory_threshold: float = 0.5

    # Local storage for sessions and memory
    foundry_data_dir: str = ".foundry"

    # Research
    research_provider: str = "perplexity"
    perplexity_api_key: str | None = None
    perplexity_model: str = "sonar"
    perplexity_base_url: str = "https://api.perplexity.ai"

    # Supabase (optional)
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def perplexity_configured(self) -> bool:
        return bool(self.perplexity_api_key)

    @property
    def jev_configured(self) -> bool:
        return bool(self.typesafe_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
