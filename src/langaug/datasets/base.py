import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class DatasetMeta(BaseModel):
    is_synthetic: bool = Field(default=False)
    pipeline_id: str | None = Field(default=None)
    sampler_id: str | None = Field(default=None)
    source_index: int | None = Field(default=None)


class BaseRecord(BaseModel):
    meta: DatasetMeta | None = Field(default=None)


class Dataset(Generic[T]):
    def __init__(self, records: list[T], schema: type[T]) -> None:
        self._records = records
        self._schema = schema
        logger.info("Dataset created with {} records", len(records))

    @property
    def records(self) -> list[T]:
        return self._records.copy()

    @property
    def schema(self) -> type[T]:
        return self._schema

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[T]:
        return iter(self._records)

    def __getitem__(self, index: int) -> T:
        return self._records[index]

    def derive(self, records: list[T]) -> "Dataset[T]":
        return Dataset(records=records, schema=self._schema)

    def merge(self, other: "Dataset[T]") -> "Dataset[T]":
        combined = self._records + other.records
        logger.info("Merged datasets: {} + {} = {}", len(self), len(other), len(combined))
        return Dataset(records=combined, schema=self._schema)

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in self._records:
                file.write(record.model_dump_json() + "\n")
        logger.info("Exported {} records to {}", len(self), path)

    @classmethod
    def from_jsonl(cls, path: Path, schema: type[T]) -> "Dataset[T]":
        records: list[T] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                records.append(schema.model_validate(data))
        logger.info("Loaded {} records from {}", len(records), path)
        return cls(records=records, schema=schema)


class BaseLoader(ABC, Generic[T]):
    def __init__(self, schema: type[T]) -> None:
        self._schema = schema

    @abstractmethod
    def load(self, source: Any) -> Dataset[T]:
        raise NotImplementedError
