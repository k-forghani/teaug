from typing import Any, Callable, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from langaug.datasets.base import Dataset, DatasetMeta
from langaug.pipelines.base import Pipeline, PipelineResult
from langaug.samplers.base import BaseSampler

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class AugmentationReport(BaseModel):
    augmentor_id: str
    total_sampled: int
    successful: int
    failed: int
    iterations: int
    results: list[dict[str, Any]] = []


class Augmentor(Generic[InputT, OutputT]):
    def __init__(
        self,
        pipeline: Pipeline,
        sampler: BaseSampler[InputT],
        record_transformer: Callable[[InputT], BaseModel],
        output_transformer: Callable[[Any, InputT], OutputT],
        augmentor_id: str | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._sampler = sampler
        self._record_transformer = record_transformer
        self._output_transformer = output_transformer
        self._augmentor_id = augmentor_id or self.__class__.__name__

    @property
    def augmentor_id(self) -> str:
        return self._augmentor_id

    def _add_metadata(self, record: OutputT, source_index: int) -> OutputT:
        if hasattr(record, "meta"):
            meta = DatasetMeta(
                is_synthetic=True,
                pipeline_id=self._pipeline.pipeline_id,
                sampler_id=self._sampler.sampler_id,
                source_index=source_index,
            )
            return record.model_copy(update={"meta": meta})
        return record

    def preview(self, dataset: Dataset[InputT], count: int = 3) -> list[PipelineResult]:
        sampled = self._sampler.sample(dataset)[:count]
        results: list[PipelineResult] = []

        for _, record in sampled:
            transformed_input = self._record_transformer(record)
            result = self._pipeline.execute(transformed_input)
            results.append(result)

        return results

    def augment(self, dataset: Dataset[InputT], iterations: int = 1) -> tuple[Dataset[OutputT], AugmentationReport]:
        augmented_records: list[OutputT] = []
        report_results: list[dict[str, Any]] = []
        successful = 0
        failed = 0

        for iteration in range(iterations):
            sampled = self._sampler.sample(dataset)

            for source_index, record in sampled:
                transformed_input = self._record_transformer(record)
                result = self._pipeline.execute(transformed_input)

                report_results.append(
                    {
                        "iteration": iteration,
                        "source_index": source_index,
                        "success": result.success,
                        "error": result.error,
                    }
                )

                if result.success and result.final_output:
                    output_record = self._output_transformer(result.final_output, record)
                    output_record = self._add_metadata(output_record, source_index)
                    augmented_records.append(output_record)
                    successful += 1
                else:
                    failed += 1

        report = AugmentationReport(
            augmentor_id=self._augmentor_id,
            total_sampled=len(report_results),
            successful=successful,
            failed=failed,
            iterations=iterations,
            results=report_results,
        )

        output_schema = type(augmented_records[0]) if augmented_records else BaseModel  # type: ignore[assignment]
        augmented_dataset = Dataset(records=augmented_records, schema=output_schema)

        return augmented_dataset, report
