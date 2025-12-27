import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMServiceConfig(BaseModel):
    temperature: float = Field(default=0.7)
    max_tokens: int | None = Field(default=None)


class BaseLLMService(ABC):
    _instance: "BaseLLMService | None" = None
    _client: BaseChatModel | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "BaseLLMService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def _create_client(self, **kwargs: Any) -> BaseChatModel:
        raise NotImplementedError

    @abstractmethod
    def _get_callbacks(self) -> list[Any]:
        raise NotImplementedError

    @property
    def client(self) -> BaseChatModel:
        if self._client is None:
            msg = "LLM client not initialized. Call configure() first."
            raise RuntimeError(msg)
        return self._client

    def configure(self, **kwargs: Any) -> "BaseLLMService":
        self._client = self._create_client(**kwargs)
        logger.info("LLM service configured: %s", self.__class__.__name__)
        return self

    def invoke(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> BaseModel | str:
        callbacks = self._get_callbacks()

        if output_schema is not None:
            structured_client = self.client.with_structured_output(output_schema)
            response = structured_client.invoke(prompt, config={"callbacks": callbacks}, **kwargs)
            logger.debug("Structured response received: %s", type(response).__name__)
            return response

        response = self.client.invoke(prompt, config={"callbacks": callbacks}, **kwargs)
        logger.debug("Raw response received")
        return response.content
