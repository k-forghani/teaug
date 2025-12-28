from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from langaug.transforms.base import BaseTransform, TransformResult

T = TypeVar("T", bound=BaseModel)


class PipelineResult(BaseModel, Generic[T]):
    success: bool
    final_output: T | None = None
    intermediate_results: list[dict[str, Any]] = []
    failed_at: str | None = None
    error: str | None = None


class Pipeline:
    def __init__(self, transforms: list[BaseTransform], pipeline_id: str | None = None) -> None:
        self._transforms = transforms
        self._pipeline_id = pipeline_id or "Pipeline"
        self._validate_chain()

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def transforms(self) -> list[BaseTransform]:
        return self._transforms.copy()

    def _validate_chain(self) -> None:
        for index in range(len(self._transforms) - 1):
            current_output = self._transforms[index].output_schema
            next_input = self._transforms[index + 1].input_schema

            current_fields = set(current_output.model_fields.keys())
            required_fields = {name for name, field in next_input.model_fields.items() if field.is_required()}

            if not required_fields.issubset(current_fields):
                missing = required_fields - current_fields
                msg = (
                    "Pipeline validation failed: Transform {} output missing fields required by Transform {}: {}"
                ).format(index, index + 1, missing)
                raise ValueError(msg)

        logger.info("Pipeline {} validated with {} transforms", self._pipeline_id, len(self._transforms))

    def execute(self, record: BaseModel) -> PipelineResult:
        logger.info("Executing pipeline: {}", self._pipeline_id)

        current_data = record
        intermediate: list[dict[str, Any]] = []

        for transform in self._transforms:
            result: TransformResult = transform.execute(current_data)  # type: ignore[type-arg]

            intermediate.append(
                {
                    "transform_id": transform.transform_id,
                    "success": result.success,
                    "output": result.output.model_dump() if result.output else None,
                    "error": result.error,
                }
            )

            if not result.success:
                logger.error("Pipeline {} failed at {}", self._pipeline_id, transform.transform_id)
                return PipelineResult(
                    success=False,
                    intermediate_results=intermediate,
                    failed_at=transform.transform_id,
                    error=result.error,
                )

            current_data = result.output

        logger.info("Pipeline {} completed successfully", self._pipeline_id)
        return PipelineResult(success=True, final_output=current_data, intermediate_results=intermediate)
