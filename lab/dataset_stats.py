import random
from collections import Counter

from loguru import logger
from pydantic import BaseModel

from langaug import HuggingFaceLoader, setup_logging

setup_logging("INFO")


class SentimentRecord(BaseModel):
    text: str
    label: int


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
SAMPLE_SIZE = 4000
RANDOM_SEED = 42


def main() -> None:
    logger.info("Loading SentiPers dataset...")

    loader = HuggingFaceLoader(schema=SentimentRecord, field_mapping={"text": "text", "label": "label"})
    full_dataset = loader.load(source=DATASET_SOURCE, split=DATASET_SPLIT)

    full_labels = [record.label for record in full_dataset.records]
    full_counts = Counter(full_labels)

    label_to_records: dict[int, list[SentimentRecord]] = {label: [] for label in CLASS_DEFINITIONS}
    for record in full_dataset.records:
        label_to_records[record.label].append(record)

    total_records = len(full_dataset)
    raw_targets = {
        label: (full_counts.get(label, 0) / total_records) * SAMPLE_SIZE
        for label in CLASS_DEFINITIONS
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

    rng = random.Random(RANDOM_SEED)
    sampled_records: list[SentimentRecord] = []
    for label in CLASS_DEFINITIONS:
        records = label_to_records[label]
        target = min(targets[label], len(records))
        sampled_records.extend(rng.sample(records, target))

    sampled_labels = [record.label for record in sampled_records]
    sampled_counts = Counter(sampled_labels)

    logger.info("\n" + "=" * 70)
    logger.info("SENTIPERS TRAIN SPLIT - CLASS DISTRIBUTION")
    logger.info("=" * 70)

    logger.info("\nFull Dataset ({} records):", len(full_dataset))
    logger.info("-" * 70)
    for label in sorted(CLASS_DEFINITIONS.keys()):
        count = full_counts.get(label, 0)
        percentage = (count / len(full_dataset)) * 100
        label_display = f"{LABEL_NAMES[label]} — {LABEL_DESCRIPTIONS[label]}"
        logger.info(
            "  {}: {} {} ({}%)",
            label,
            label_display,
            count,
            round(percentage, 1),
        )

    logger.info("\nStratified Sample ({} records):", len(sampled_records))
    logger.info("-" * 70)
    for label in sorted(CLASS_DEFINITIONS.keys()):
        count = sampled_counts.get(label, 0)
        percentage = (count / len(sampled_records)) * 100 if sampled_records else 0.0
        label_display = f"{LABEL_NAMES[label]} — {LABEL_DESCRIPTIONS[label]}"
        logger.info(
            "  {}: {} {} ({}%)",
            label,
            label_display,
            count,
            round(percentage, 1),
        )

    logger.info("\nClass Imbalance Analysis (Stratified Sample):")
    logger.info("-" * 70)
    max_count = max(sampled_counts.values()) if sampled_counts else 0
    for label in sorted(CLASS_DEFINITIONS.keys()):
        count = sampled_counts.get(label, 0)
        deficit = max_count - count
        label_display = f"{LABEL_NAMES[label]} — {LABEL_DESCRIPTIONS[label]}"
        if deficit > 0:
            logger.info(
                "  {}: {} needs +{} to balance",
                label,
                label_display,
                deficit,
            )
        else:
            logger.info("  {}: {} (majority class)", label, label_display)

    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
