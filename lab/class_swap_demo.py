import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

import langaug
from langaug import (
    ClassSwapInput,
    ClassSwapTransform,
    DatasetMeta,
    ExclusiveSampler,
    HuggingFaceLoader,
    Pipeline,
    SamplerConfig,
)
from langaug.datasets.base import Dataset


class SentimentRecord(BaseModel):
    text: str
    label: int
    meta: DatasetMeta | None = None


class AugmentedSentimentRecord(BaseModel):
    text: str
    label: int
    original_text: str | None = None
    source_label: int | None = None
    meta: DatasetMeta | None = None


CLASS_DEFINITIONS = {
    0: {
        "name": "Very Negative",
        "description": "intense dissatisfaction, frustration, or strong criticism",
    },
    1: {
        "name": "Negative",
        "description": "clear dislike, disappointment, or criticism without extremes",
    },
    2: {
        "name": "Neutral",
        "description": "factual or balanced tone with no strong positive/negative cues",
    },
    3: {
        "name": "Positive",
        "description": "clear approval, satisfaction, or praise without extremes",
    },
    4: {
        "name": "Very Positive",
        "description": "strong enthusiasm, admiration, or high praise",
    },
}
LABEL_NAMES = {key: value["name"] for key, value in CLASS_DEFINITIONS.items()}
LABEL_DESCRIPTIONS = {key: value["description"] for key, value in CLASS_DEFINITIONS.items()}
DATASET_SOURCE = "kforghani/sentipers"
DATASET_SPLIT = "train"
CACHE_CSV_PATH = Path("data/base/sentipers_train.csv")
OUTPUT_CSV_PATH = Path("data/output/augmented_sentipers_train.csv")
SEED = 42
LLM_TEMPERATURE = 1
LLM_MAX_TOKENS = 8192
BATCH_SIZE = 5
SAMPLE_SIZE = 4000
BASE_SAMPLE_SEED = 42
OUTPUT_SHUFFLE = True
OUTPUT_SHUFFLE_SEED = 42
OUTPUT_METADATA_FIELDS = ["is_synthetic", "source_label", "pipeline_id", "sampler_id", "source_index"]
OUTPUT_FIELDNAMES = ["text", "label", *OUTPUT_METADATA_FIELDS]
LOG_FILE_PATH = Path("data/output/class_swap_demo.log")


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        LOG_FILE_PATH,
        level="INFO",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
    )

TARGET_AUGMENTATION_COUNTS = {
    0: 1445,
    1: 1080,
    3: 225,
    4: 755,
}


def build_class_swap_input(records: list[SentimentRecord], target_label: int) -> ClassSwapInput:
    texts = [r.text for r in records]
    source_labels = [r.label for r in records]
    target_labels = [target_label] * len(records)
    return ClassSwapInput(
        texts=texts,
        source_labels=source_labels,
        target_labels=target_labels,
        label_names=LABEL_NAMES,
        label_descriptions=LABEL_DESCRIPTIONS,
    )


def main() -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    configure_logging()
    logger.info("Starting Class Swap Demo")
    logger.info("File logging enabled at {}", LOG_FILE_PATH)

    llm_service = langaug.OpenAIService().configure(
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    loader = HuggingFaceLoader(schema=SentimentRecord, field_mapping={"text": "text", "label": "label"})
    full_dataset = loader.load(source=DATASET_SOURCE, split=DATASET_SPLIT)
    logger.info("Loaded {} records from full dataset", len(full_dataset))

    full_labels = [record.label for record in full_dataset.records]
    full_counts = Counter(full_labels)

    label_to_records: dict[int, list[SentimentRecord]] = {label: [] for label in LABEL_DESCRIPTIONS}
    for record in full_dataset.records:
        label_to_records[record.label].append(record)

    total_records = len(full_dataset)
    raw_targets = {
        label: (full_counts.get(label, 0) / total_records) * SAMPLE_SIZE
        for label in LABEL_DESCRIPTIONS
    }
    targets = {label: int(raw_targets[label]) for label in LABEL_DESCRIPTIONS}
    remainder = SAMPLE_SIZE - sum(targets.values())
    if remainder > 0:
        fractional_order = sorted(
            LABEL_DESCRIPTIONS,
            key=lambda label: raw_targets[label] - targets[label],
            reverse=True,
        )
        for label in fractional_order[:remainder]:
            targets[label] += 1

    rng = random.Random(BASE_SAMPLE_SEED)
    sampled_records: list[SentimentRecord] = []
    for label in LABEL_DESCRIPTIONS:
        records = label_to_records[label]
        target = min(targets[label], len(records))
        sampled_records.extend(rng.sample(records, target))

    dataset = full_dataset.derive(sampled_records)
    sampled_counts = Counter(record.label for record in sampled_records)
    logger.info("Sampled {} records for base dataset", len(sampled_records))
    for label in sorted(LABEL_DESCRIPTIONS):
        logger.info(
            "Base sample {}: {} records",
            LABEL_DESCRIPTIONS[label],
            sampled_counts.get(label, 0),
        )

    CACHE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset.records:
            writer.writerow({"text": record.text, "label": record.label})
    logger.info("Cached base dataset CSV at {}", CACHE_CSV_PATH)

    random.seed(SEED)
    logger.info("Global random seed set to {}", SEED)

    class_swap = ClassSwapTransform(llm_service=llm_service)
    pipeline = Pipeline(transforms=[class_swap])
    logger.info("Pipeline initialized: {} (id={})", pipeline, pipeline.pipeline_id)

    label_to_records = {label: [r for r in dataset.records if r.label == label] for label in CLASS_DEFINITIONS}
    source_datasets = {label: dataset.derive(records) for label, records in label_to_records.items()}
    source_samplers = {
        label: ExclusiveSampler(SamplerConfig(count=BATCH_SIZE, seed=None), sampler_id=f"ExclusiveSampler[{label}]")
        for label in CLASS_DEFINITIONS
    }
    for label in sorted(CLASS_DEFINITIONS):
        logger.info(
            "Source class {} availability: {} records",
            LABEL_NAMES[label],
            len(label_to_records[label]),
        )

    all_augmented: list[AugmentedSentimentRecord] = []
    total_target = sum(TARGET_AUGMENTATION_COUNTS.values())
    logger.info(
        "Planned augmentation total: {} records across {} target classes",
        total_target,
        len(TARGET_AUGMENTATION_COUNTS),
    )

    exhausted_sources: set[int] = set()

    with tqdm(total=total_target, desc="Augmentation", unit="records") as overall_bar:
        for target_label in sorted(TARGET_AUGMENTATION_COUNTS):
            target_count = TARGET_AUGMENTATION_COUNTS[target_label]
            source_labels = [label for label in sorted(CLASS_DEFINITIONS) if label != target_label]
            active_sources = [label for label in source_labels if label not in exhausted_sources]

            logger.info(
                "Target {}: need {} augmented records (sources={})",
                LABEL_NAMES[target_label],
                target_count,
                [LABEL_NAMES[label] for label in active_sources],
            )

            processed = 0
            attempted = 0
            round_num = 0

            with tqdm(
                total=target_count,
                desc=f"Target {LABEL_DESCRIPTIONS[target_label]}",
                unit="records",
                leave=False,
            ) as target_bar:
                while processed < target_count and active_sources:
                    round_num += 1
                    successes_this_round = 0

                    for source_label in list(active_sources):
                        if processed >= target_count:
                            break

                        batch_count = min(BATCH_SIZE, target_count - processed)
                        sampler = source_samplers[source_label]
                        sampler._config.count = batch_count

                        logger.info(
                            "Target {} round {}: sampling {} from {} (processed={}/{})",
                            LABEL_NAMES[target_label],
                            round_num,
                            batch_count,
                            LABEL_NAMES[source_label],
                            processed,
                            target_count,
                        )

                        sampled = sampler.sample(source_datasets[source_label])
                        sampled_records = [record for _, record in sampled]

                        if not sampled_records:
                            exhausted_sources.add(source_label)
                            logger.warning(
                                "Source {} exhausted; skipping for remaining targets",
                                LABEL_NAMES[source_label],
                            )
                            continue

                        attempted += len(sampled_records)
                        batch_input = build_class_swap_input(sampled_records, target_label)
                        result = pipeline.execute(batch_input)
                        if not result.success or not result.final_output:
                            logger.error(
                                "Target {} round {} failed from {}: {}",
                                LABEL_NAMES[target_label],
                                round_num,
                                LABEL_NAMES[source_label],
                                result.error,
                            )
                            continue

                        expected_target = LABEL_NAMES[target_label]
                        accepted = 0
                        for (source_index, _), item in zip(sampled, result.final_output.items):
                            if item.target_label not in {
                                expected_target,
                                target_label,
                                str(target_label),
                            }:
                                logger.warning(
                                    "LLM target mismatch: expected {}, got {}",
                                    expected_target,
                                    item.target_label,
                                )
                                continue
                            meta = DatasetMeta(
                                is_synthetic=True,
                                pipeline_id=pipeline.pipeline_id,
                                sampler_id=sampler.sampler_id,
                                source_index=source_index,
                            )
                            all_augmented.append(
                                AugmentedSentimentRecord(
                                    text=item.output,
                                    label=target_label,
                                    source_label=source_label,
                                    original_text=None,
                                    meta=meta,
                                )
                            )
                            accepted += 1

                        if accepted < len(sampled_records):
                            logger.warning(
                                "Dropped {} records due to target mismatch for {} → {}",
                                len(sampled_records) - accepted,
                                LABEL_NAMES[source_label],
                                LABEL_NAMES[target_label],
                            )

                        processed += accepted
                        successes_this_round += accepted
                        target_bar.update(accepted)
                        overall_bar.update(accepted)

                    if successes_this_round == 0:
                        logger.error(
                            "Target {} stopping: no successful augmentations in round {}",
                            LABEL_NAMES[target_label],
                            round_num,
                        )
                        break

                    active_sources = [label for label in source_labels if label not in exhausted_sources]

            logger.info(
                "Target {} completed: {} augmented records generated (attempted={})",
                LABEL_NAMES[target_label],
                processed,
                attempted,
            )

    combined_records: list[SentimentRecord | AugmentedSentimentRecord] = []

    for record in dataset.records:
        combined_records.append(record)

    for augmented_record in all_augmented:
        combined_records.append(augmented_record)

    if OUTPUT_SHUFFLE:
        rng = random.Random(OUTPUT_SHUFFLE_SEED)
        rng.shuffle(combined_records)
        logger.info(
            "Shuffled combined records before export (seed={}, total={})",
            OUTPUT_SHUFFLE_SEED,
            len(combined_records),
        )

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for record in combined_records:
            meta = record.meta
            is_synthetic = bool(meta and meta.is_synthetic)
            writer.writerow(
                {
                    "text": record.text,
                    "label": record.label,
                    "is_synthetic": is_synthetic,
                    "source_label": getattr(record, "source_label", None),
                    "pipeline_id": meta.pipeline_id if meta else None,
                    "sampler_id": meta.sampler_id if meta else None,
                    "source_index": meta.source_index if meta else None,
                }
            )

    logger.info(
        "Augmented dataset saved to {} (original: {}, augmented: {}, total: {})",
        OUTPUT_CSV_PATH,
        len(dataset),
        len(all_augmented),
        len(combined_records),
    )


if __name__ == "__main__":
    main()
