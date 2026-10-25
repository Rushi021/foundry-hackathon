from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM: "provider:model" — swap without code changes
    categorize_llm: str = "openai:gpt-4o-mini"
    founder_llm: str = "openai:gpt-4o"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
