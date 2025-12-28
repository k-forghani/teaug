import logging
import sys
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field(default="gpt-4.1-nano-2025-04-14", env="OPENAI_MODEL")
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
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_level = logger.level(record.levelname).name
            except ValueError:
                log_level = record.levelno

            frame = logging.currentframe()
            depth = 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(log_level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
