import random
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from langaug.datasets.base import Dataset

T = TypeVar("T", bound=BaseModel)


class SamplerConfig(BaseModel):
    count: int
    seed: int | None = None
    exclude_indices: list[int] = []


class BaseSampler(ABC, Generic[T]):
    def __init__(self, config: SamplerConfig, sampler_id: str | None = None) -> None:
        self._config = config
        self._sampler_id = sampler_id or self.__class__.__name__
        if config.seed is not None:
            random.seed(config.seed)

    @property
    def sampler_id(self) -> str:
        return self._sampler_id

    def _pre_filter(self, dataset: Dataset[T]) -> list[tuple[int, T]]:
        return [(index, record) for index, record in enumerate(dataset) if index not in self._config.exclude_indices]

    @abstractmethod
    def _sample(self, candidates: list[tuple[int, T]]) -> list[tuple[int, T]]:
        raise NotImplementedError

    def _post_filter(self, sampled: list[tuple[int, T]]) -> list[tuple[int, T]]:
        return sampled

    def sample(self, dataset: Dataset[T]) -> list[tuple[int, T]]:
        logger.info("Sampling from dataset with {} records", len(dataset))

        candidates = self._pre_filter(dataset)
        sampled = self._sample(candidates)
        result = self._post_filter(sampled)

        logger.info("Final sample size: {}", len(result))
        return result


class RandomSampler(BaseSampler[T]):
    def _sample(self, candidates: list[tuple[int, T]]) -> list[tuple[int, T]]:
        count = min(self._config.count, len(candidates))
        return random.sample(candidates, count)


class StratifiedSampler(BaseSampler[T]):
    def __init__(self, config: SamplerConfig, stratify_field: str, sampler_id: str | None = None) -> None:
        super().__init__(config, sampler_id)
        self._stratify_field = stratify_field

    def _sample(self, candidates: list[tuple[int, T]]) -> list[tuple[int, T]]:
        groups: dict[Any, list[tuple[int, T]]] = {}

        for index, record in candidates:
            key = getattr(record, self._stratify_field)
            groups.setdefault(key, []).append((index, record))

        per_group = max(1, self._config.count // max(len(groups), 1))

        result: list[tuple[int, T]] = []
        for group_items in groups.values():
            count = min(per_group, len(group_items))
            result.extend(random.sample(group_items, count))

        return result[: self._config.count]


class ExclusiveSampler(BaseSampler[T]):
    def __init__(self, config: SamplerConfig, sampler_id: str | None = None) -> None:
        super().__init__(config, sampler_id)
        self._sampled_indices: set[int] = set()

    def _sample(self, candidates: list[tuple[int, T]]) -> list[tuple[int, T]]:
        available = [item for item in candidates if item[0] not in self._sampled_indices]
        count = min(self._config.count, len(available))
        sampled = random.sample(available, count)
        self._sampled_indices.update(idx for idx, _ in sampled)
        return sampled
