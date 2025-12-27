import logging

from pydantic import BaseModel, Field

from langaug.services.base import BaseLLMService
from langaug.transforms.base import BaseTransform
from langaug.utils.prompts import PromptLoader

logger = logging.getLogger(__name__)


class ClassSwapInput(BaseModel):
    texts: list[str]
    source_labels: list[str | int]
    target_labels: list[str | int]
    label_descriptions: dict[str | int, str] | None = None


class ClassSwapOutput(BaseModel):
    texts: list[str] = Field(description="Transformed texts with swapped classes in order")


class ClassSwapTransform(BaseTransform[ClassSwapInput, ClassSwapOutput]):
    def __init__(
        self,
        llm_service: BaseLLMService,
        prompt: str = "class_swap",
        transform_id: str | None = None,
    ) -> None:
        super().__init__(
            input_schema=ClassSwapInput,
            output_schema=ClassSwapOutput,
            prompt=prompt,
            llm_service=llm_service,
            transform_id=transform_id,
        )

    def _render_prompt(self, record: ClassSwapInput) -> str:
        def to_label(value: str | int) -> str:
            if record.label_descriptions:
                if value in record.label_descriptions:
                    return record.label_descriptions[value]
                if str(value) in record.label_descriptions:
                    return record.label_descriptions[str(value)]  # type: ignore[index]
            return str(value)

        pairs = [
            {
                "text": text,
                "source_label": to_label(source),
                "target_label": to_label(target),
            }
            for text, source, target in zip(record.texts, record.source_labels, record.target_labels)
        ]

        all_labels = []
        if record.label_descriptions:
            all_labels = [{"key": key, "value": value} for key, value in record.label_descriptions.items()]

        return PromptLoader.render(self._prompt, {"items": pairs, "all_labels": all_labels})

    def _merge_output(self, input_record: ClassSwapInput, llm_output: ClassSwapOutput) -> ClassSwapOutput:
        return ClassSwapOutput(texts=llm_output.texts)
