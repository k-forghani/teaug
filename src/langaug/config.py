import logging
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field(default="gpt-5-nano", env="OPENAI_MODEL")
    temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    max_tokens: int | None = Field(default=None, env="OPENAI_MAX_TOKENS")
    reasoning_effort: str | None = Field(default=None, env="OPENAI_REASONING_EFFORT")


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    public_key: str = Field(..., env="LANGFUSE_PUBLIC_KEY")
    secret_key: str = Field(..., env="LANGFUSE_SECRET_KEY")
    base_url: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_BASE_URL")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO")
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
