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
    0: "Very Negative - شدیداً منفی",
    1: "Negative - منفی",
    2: "Neutral - خنثی",
    3: "Positive - مثبت",
    4: "Very Positive - شدیداً مثبت",
}


def build_class_swap_input(records: list[SentimentRecord]) -> ClassSwapInput:
    polarity_flip = {0: 4, 1: 3, 2: 3, 3: 1, 4: 0}  # ensure target differs; neutral -> positive
    texts = [r.text for r in records]
    source_labels = [r.label for r in records]
    target_labels = [polarity_flip.get(r.label, 2) for r in records]
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
    dataset = loader.load(source="kforghani/sentipers", split="train", limit=10)
    logger.info("Loaded dataset with %d records", len(dataset))

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with cache_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset.records:
            writer.writerow({"text": record.text, "label": record.label})
    logger.info("Cached base dataset CSV at %s", cache_csv)

    logger.info("Loaded dataset with %d records", len(dataset))

    class_swap = ClassSwapTransform(llm_service=llm_service)
    pipeline = Pipeline(transforms=[class_swap])

    sampler = ExclusiveSampler(SamplerConfig(count=5, seed=seed))
    sampled = sampler.sample(dataset)
    sampled_records = [record for _, record in sampled]

    batch_input = build_class_swap_input(sampled_records)

    outputs: list[AugmentedSentimentRecord] = []

    result = pipeline.execute(batch_input)
    if not result.success or not result.final_output:
        logger.error("Class swap failed: %s", result.error)
    else:
        for idx, text in enumerate(result.final_output.texts):
            source_index, _ = sampled[idx]
            meta = DatasetMeta(
                is_synthetic=True,
                pipeline_id=pipeline.pipeline_id,
                sampler_id=sampler.sampler_id,
                source_index=source_index,
            )
            outputs.append(
                AugmentedSentimentRecord(
                    text=text,
                    label=int(batch_input.target_labels[idx]),
                    original_text=batch_input.texts[idx],
                    meta=meta,
                )
            )

    output_csv = Path("data/output/augmented_sentipers_train.csv")
    if outputs:
        combined_records: list[SentimentRecord | AugmentedSentimentRecord] = []
        
        for record in dataset.records:
            combined_records.append(record)
        
        for augmented_record in outputs:
            combined_records.append(augmented_record)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as file:
            fieldnames = ["text", "label"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in combined_records:
                writer.writerow({"text": record.text, "label": record.label})

        logger.info("Augmented dataset saved to %s (original: %d, augmented: %d, total: %d)", 
                    output_csv, len(dataset), len(outputs), len(combined_records))
    else:
        logger.warning("No outputs generated")


if __name__ == "__main__":
    main()
