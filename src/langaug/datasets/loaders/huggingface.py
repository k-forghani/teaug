import logging
from typing import Any, TypeVar

from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import BaseModel

from langaug.datasets.base import BaseLoader, Dataset

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HuggingFaceLoader(BaseLoader[T]):
    def __init__(self, schema: type[T], field_mapping: dict[str, str] | None = None) -> None:
        super().__init__(schema)
        self._field_mapping = field_mapping or {}

    def load(
        self,
        source: str,
        split: str = "train",
        subset: str | None = None,
        limit: int | None = None,
    ) -> Dataset[T]:
        logger.info("Loading HuggingFace dataset: %s (split=%s, subset=%s)", source, split, subset)

        kwargs: dict[str, Any] = {"path": source, "split": split}
        if subset:
            kwargs["name"] = subset

        hf_dataset: HFDataset = load_dataset(**kwargs)  # type: ignore[assignment]

        if limit:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        records: list[T] = []
        for item in hf_dataset:
            mapped_item = self._apply_mapping(item)
            record = self._schema.model_validate(mapped_item)
            records.append(record)

        logger.info("Loaded %d records from HuggingFace", len(records))
        return Dataset(records=records, schema=self._schema)

    def _apply_mapping(self, item: dict[str, Any]) -> dict[str, Any]:
        if not self._field_mapping:
            return item

        mapped: dict[str, Any] = {}
        for target_field, source_field in self._field_mapping.items():
            if source_field in item:
                mapped[target_field] = item[source_field]

        for key, value in item.items():
            if key not in self._field_mapping.values() and key not in mapped:
                mapped[key] = value

        return mapped
