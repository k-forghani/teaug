import logging
from pathlib import Path

from pydantic import BaseModel

import langaug
from langaug import (
    ClassSwapInput,
    ClassSwapOutput,
    ClassSwapTransform,
    DatasetMeta,
    HuggingFaceLoader,
    Pipeline,
    RandomSampler,
    SamplerConfig,
    setup_logging,
)

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

    llm_service = langaug.OpenAIService().configure(temperature=0.7, max_tokens=8192, reasoning_effort="minimal")

    loader = HuggingFaceLoader(schema=SentimentRecord, field_mapping={"text": "text", "label": "label"})
    dataset = loader.load(source="kforghani/sentipers", split="train", limit=10)
    logger.info("Loaded dataset with %d records", len(dataset))

    class_swap = ClassSwapTransform(llm_service=llm_service)
    pipeline = Pipeline(transforms=[class_swap])

    sampler = RandomSampler(SamplerConfig(count=3))
    sampled = sampler.sample(dataset)
    sampled_records = [record for _, record in sampled]

    batch_input = build_class_swap_input(sampled_records)

    outputs: list[AugmentedSentimentRecord] = []

    result = pipeline.execute(batch_input)
    if not result.success or not result.final_output:
        logger.error("Class swap failed: %s", result.error)
    else:
        for idx, text in enumerate(result.final_output.texts):
            outputs.append(
                AugmentedSentimentRecord(
                    text=text,
                    label=int(batch_input.target_labels[idx]),
                    original_text=batch_input.texts[idx],
                )
            )

    output_path = Path("data/output/class_swap_results.jsonl")
    if outputs:
        from langaug.datasets.base import Dataset

        augmented_dataset = Dataset(records=outputs, schema=AugmentedSentimentRecord)
        augmented_dataset.to_jsonl(output_path)
        logger.info("Results saved to %s", output_path)
    else:
        logger.warning("No outputs generated")


if __name__ == "__main__":
    main()
