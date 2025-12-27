import csv
import logging
from pathlib import Path

from pydantic import BaseModel

import langaug
from langaug import (
    ClassSwapInput,
    ClassSwapOutput,
    ClassSwapTransform,
    DatasetMeta,
    ExclusiveSampler,
    HuggingFaceLoader,
    Pipeline,
    SamplerConfig,
    setup_logging,
)
from langaug.datasets.base import Dataset

setup_logging("INFO")
logger = logging.getLogger(__name__)


class SentimentRecord(BaseModel):
    text: str
    label: int
    meta: DatasetMeta | None = None


class AugmentedSentimentRecord(BaseModel):
    text: str
    label: int
    original_text: str | None = None
    meta: DatasetMeta | None = None


LABEL_DESCRIPTIONS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

AUGMENTATION_RULES = [
    # Class 0 (Very Negative) - needs ~740 - from positive classes for contrast
    {"source": 3, "target": 0, "count": 370},
    {"source": 4, "target": 0, "count": 370},
    
    # Class 1 (Negative) - needs ~570 - from positive classes for contrast
    {"source": 3, "target": 1, "count": 300},
    {"source": 4, "target": 1, "count": 270},
    
    # Class 3 (Positive) - needs ~140 - from negative classes for contrast
    {"source": 1, "target": 3, "count": 115},
    {"source": 0, "target": 3, "count": 25},
    
    # Class 4 (Very Positive) - needs ~385 - from negative classes for contrast
    {"source": 1, "target": 4, "count": 300},
    {"source": 0, "target": 4, "count": 85},
]


def build_class_swap_input(records: list[SentimentRecord], target_label: int) -> ClassSwapInput:
    texts = [r.text for r in records]
    source_labels = [r.label for r in records]
    target_labels = [target_label] * len(records)
    return ClassSwapInput(
        texts=texts,
        source_labels=source_labels,
        target_labels=target_labels,
        label_descriptions=LABEL_DESCRIPTIONS,
    )


def main() -> None:
    logger.info("Starting Class Swap Demo")

    seed = 42
    llm_service = langaug.OpenAIService().configure(temperature=0.7, max_tokens=8192)

    cache_csv = Path("data/base/sentipers_train.csv")

    loader = HuggingFaceLoader(schema=SentimentRecord, field_mapping={"text": "text", "label": "label"})
    full_dataset = loader.load(source="kforghani/sentipers", split="train")
    base_records = full_dataset.records[:2000]
    dataset = full_dataset.derive(base_records)
    logger.info("Loaded %d records from full dataset, using first 2000 as base", len(full_dataset))

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with cache_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset.records:
            writer.writerow({"text": record.text, "label": record.label})
    logger.info("Cached base dataset CSV at %s", cache_csv)

    class_swap = ClassSwapTransform(llm_service=llm_service)
    pipeline = Pipeline(transforms=[class_swap])

    all_augmented: list[AugmentedSentimentRecord] = []

    for rule in AUGMENTATION_RULES:
        source_label = rule["source"]
        target_label = rule["target"]
        count = rule["count"]

        logger.info(
            "Augmenting %s → %s (%d records)",
            LABEL_DESCRIPTIONS[source_label],
            LABEL_DESCRIPTIONS[target_label],
            count,
        )

        source_records = [r for r in dataset.records if r.label == source_label]
        if len(source_records) < count:
            logger.warning(
                "Not enough source records for %s (available: %d, requested: %d)",
                LABEL_DESCRIPTIONS[source_label],
                len(source_records),
                count,
            )
            count = len(source_records)

        sampler = ExclusiveSampler(SamplerConfig(count=5, seed=seed))
        source_dataset = dataset.derive(source_records)
        
        processed = 0
        batch_num = 0
        
        while processed < count:
            batch_count = min(5, count - processed)
            sampler._config.count = batch_count
            
            sampled = sampler.sample(source_dataset)
            sampled_records = [record for _, record in sampled]
            
            if not sampled_records:
                logger.warning("No more records to sample for rule %s → %s", source_label, target_label)
                break

            batch_input = build_class_swap_input(sampled_records, target_label)

            result = pipeline.execute(batch_input)
            if not result.success or not result.final_output:
                logger.error(
                    "Class swap failed for rule %s → %s (batch %d): %s",
                    source_label,
                    target_label,
                    batch_num,
                    result.error,
                )
                batch_num += 1
                processed += len(sampled_records)
                continue

            for idx, text in enumerate(result.final_output.texts):
                source_index, _ = sampled[idx]
                meta = DatasetMeta(
                    is_synthetic=True,
                    pipeline_id=pipeline.pipeline_id,
                    sampler_id=sampler.sampler_id,
                    source_index=source_index,
                )
                all_augmented.append(
                    AugmentedSentimentRecord(
                        text=text,
                        label=target_label,
                        original_text=batch_input.texts[idx],
                        meta=meta,
                    )
                )
            
            processed += len(sampled_records)
            batch_num += 1
            logger.info("Processed batch %d (%d/%d records)", batch_num, processed, count)

            processed += len(sampled_records)
            batch_num += 1
            logger.info("Processed batch %d (%d/%d records)", batch_num, processed, count)

        logger.info("Generated %d augmented records for this rule", processed)

    output_csv = Path("data/output/augmented_sentipers_train.csv")
    combined_records: list[SentimentRecord | AugmentedSentimentRecord] = []

    for record in dataset.records:
        combined_records.append(record)

    for augmented_record in all_augmented:
        combined_records.append(augmented_record)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in combined_records:
            writer.writerow({"text": record.text, "label": record.label})

    logger.info("Augmented dataset saved to %s (original: %d, augmented: %d, total: %d)", 
                output_csv, len(dataset), len(all_augmented), len(combined_records))


if __name__ == "__main__":
    main()
