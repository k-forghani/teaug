from loguru import logger
from pydantic import BaseModel, Field

from langaug.services.base import BaseLLMService
from langaug.transforms.base import BaseTransform
from langaug.utils.prompts import PromptLoader

class ClassSwapInput(BaseModel):
    texts: list[str]
    source_labels: list[str | int]
    target_labels: list[str | int]
    label_names: dict[str | int, str] | None = None
    label_descriptions: dict[str | int, str] | None = None


class ClassSwapItem(BaseModel):
    output: str = Field(description="Rewritten sentence aligned to the target sentiment.")
    target_label: str | int = Field(description="Target sentiment label for the rewritten sentence.")


class ClassSwapOutput(BaseModel):
    items: list[ClassSwapItem] = Field(
        description="Array of rewritten items aligned with the input order."
    )

    @property
    def texts(self) -> list[str]:
        return [item.output for item in self.items]


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
            if record.label_names:
                if value in record.label_names:
                    return record.label_names[value]
                if str(value) in record.label_names:
                    return record.label_names[str(value)]  # type: ignore[index]
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
        if record.label_names and record.label_descriptions:
            for key, name in record.label_names.items():
                description = record.label_descriptions.get(key)
                if description is None:
                    description = record.label_descriptions.get(str(key))
                all_labels.append(
                    {
                        "name": name,
                        "description": description or "",
                    }
                )
        elif record.label_descriptions:
            for key, description in record.label_descriptions.items():
                all_labels.append(
                    {
                        "name": str(key),
                        "description": description,
                    }
                )
        elif record.label_names:
            for name in record.label_names.values():
                all_labels.append({"name": name, "description": ""})

        return PromptLoader.render(self._prompt, {"items": pairs, "all_labels": all_labels})

    def _merge_output(self, input_record: ClassSwapInput, llm_output: ClassSwapOutput) -> ClassSwapOutput:
        return llm_output
