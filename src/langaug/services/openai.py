from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from loguru import logger
from pydantic import BaseModel

from langaug.config import get_settings
from langaug.services.base import BaseLLMService


class OpenAIService(BaseLLMService):
    def __init__(self) -> None:
        super().__init__()
        self._reasoning_effort: str | None = None

    def _create_client(self, **kwargs: Any) -> BaseChatModel:
        settings = get_settings()

        config: dict[str, Any] = {
            "api_key": settings.openai.api_key,
            "model": kwargs.get("model", settings.openai.model),
            "temperature": kwargs.get("temperature", settings.openai.temperature),
        }

        if "max_tokens" in kwargs:
            config["max_tokens"] = kwargs["max_tokens"]
        elif settings.openai.max_tokens is not None:
            config["max_tokens"] = settings.openai.max_tokens

        self._reasoning_effort = kwargs.get("reasoning_effort", settings.openai.reasoning_effort)

        logger.info("Creating OpenAI client with model: {}", config["model"])
        return ChatOpenAI(**config)

    def _get_callbacks(self) -> list[Any]:
        settings = get_settings()

        Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            base_url=settings.langfuse.base_url,
        )

        return [LangfuseCallbackHandler()]

    def invoke(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> BaseModel | str:
        callbacks = self._get_callbacks()
        client = self.client

        if self._reasoning_effort:
            client = client.with_config(
                {
                    "reasoning": {
                        "effort": self._reasoning_effort,
                    }
                }
            )

        if output_schema is not None:
            structured_client = client.with_structured_output(output_schema)
            response = structured_client.invoke(prompt, config={"callbacks": callbacks}, **kwargs)
            logger.debug("Structured response received: {}", type(response).__name__)
            return response

        response = client.invoke(prompt, config={"callbacks": callbacks}, **kwargs)
        logger.debug("Raw response received")
        return response.content
