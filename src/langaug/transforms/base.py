from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from langaug.services.base import BaseLLMService
from langaug.utils.prompts import PromptLoader

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class TransformResult(BaseModel, Generic[OutputT]):
    success: bool
    output: OutputT | None = None
    error: str | None = None


class BaseTransform(ABC, Generic[InputT, OutputT]):
    def __init__(
        self,
        input_schema: type[InputT],
        output_schema: type[OutputT],
        prompt: str,
        llm_service: BaseLLMService | None = None,
        transform_id: str | None = None,
    ) -> None:
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._prompt = prompt
        self._llm_service = llm_service
        self._transform_id = transform_id or self.__class__.__name__

    @property
    def transform_id(self) -> str:
        return self._transform_id

    @property
    def input_schema(self) -> type[InputT]:
        return self._input_schema

    @property
    def output_schema(self) -> type[OutputT]:
        return self._output_schema

    def _render_prompt(self, record: InputT) -> str:
        return PromptLoader.render(self._prompt, record)

    @abstractmethod
    def _merge_output(self, input_record: InputT, llm_output: OutputT) -> OutputT:
        raise NotImplementedError

    def execute(self, record: InputT) -> TransformResult[OutputT]:
        logger.info("Executing transform: {}", self._transform_id)

        try:
            if self._llm_service is None:
                return self._execute_deterministic(record)

            prompt = self._render_prompt(record)
            llm_output = self._llm_service.invoke(prompt, output_schema=self._output_schema)
            merged = self._merge_output(record, llm_output)  # type: ignore[arg-type]

            return TransformResult(success=True, output=merged)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Transform {} failed: {}", self._transform_id, exc)
            return TransformResult(success=False, error=str(exc))

    def _execute_deterministic(self, record: InputT) -> TransformResult[OutputT]:
        raise NotImplementedError("Deterministic execution not implemented for this transform")
