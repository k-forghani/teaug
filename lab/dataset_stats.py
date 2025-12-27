import logging
from collections import Counter

from pydantic import BaseModel

from langaug import HuggingFaceLoader, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


class SentimentRecord(BaseModel):
    text: str
    label: int


LABEL_DESCRIPTIONS = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive",
}


def main() -> None:
    logger.info("Loading SentiPers dataset...")

    loader = HuggingFaceLoader(schema=SentimentRecord, field_mapping={"text": "text", "label": "label"})
    full_dataset = loader.load(source="kforghani/sentipers", split="train")

    full_labels = [record.label for record in full_dataset.records]
    full_counts = Counter(full_labels)

    truncated_labels = [record.label for record in full_dataset.records[:2000]]
    truncated_counts = Counter(truncated_labels)

    print("\n" + "=" * 70)
    print("SENTIPERS TRAIN SPLIT - CLASS DISTRIBUTION")
    print("=" * 70)

    print(f"\nFull Dataset ({len(full_dataset)} records):")
    print("-" * 70)
    for label in sorted(LABEL_DESCRIPTIONS.keys()):
        count = full_counts.get(label, 0)
        percentage = (count / len(full_dataset)) * 100
        print(f"  {label}: {LABEL_DESCRIPTIONS[label]:<35} {count:>6} ({percentage:>5.1f}%)")

    print(f"\nTruncated Dataset (first 2000 records):")
    print("-" * 70)
    for label in sorted(LABEL_DESCRIPTIONS.keys()):
        count = truncated_counts.get(label, 0)
        percentage = (count / 2000) * 100
        print(f"  {label}: {LABEL_DESCRIPTIONS[label]:<35} {count:>6} ({percentage:>5.1f}%)")

    print("\nClass Imbalance Analysis (Truncated Dataset):")
    print("-" * 70)
    max_count = max(truncated_counts.values())
    for label in sorted(LABEL_DESCRIPTIONS.keys()):
        count = truncated_counts.get(label, 0)
        deficit = max_count - count
        if deficit > 0:
            print(f"  {label}: {LABEL_DESCRIPTIONS[label]:<35} needs +{deficit:>4} to balance")
        else:
            print(f"  {label}: {LABEL_DESCRIPTIONS[label]:<35} (majority class)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
