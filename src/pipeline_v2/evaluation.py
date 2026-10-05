"""Evaluation runner for document visual question answering."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import (
    AnnotationType,
    MultiPageDocumentInstance,
)

from pipeline_v2.metrics import compute_anls, compute_exact_match, compute_f1
from pipeline_v2.models.qwen import QwenVLModel

logger = get_logger(__name__)


def evaluate_dataset(
    dataset: Dataset[MultiPageDocumentInstance, DatasetConfig],
    model: QwenVLModel,
    output_file: Path | None = None,
) -> dict[str, Any]:
    """Runs model evaluation across all dataset splits and collects metrics."""
    results_by_split: dict[str, Any] = {}

    for split_key, split_iterator in dataset.split_iterators.items():
        logger.info(f"Evaluating split: {split_key.value}")
        split_records: list[dict[str, Any]] = []

        anls_scores: list[float] = []
        f1_scores: list[float] = []
        em_scores: list[float] = []

        start_time = time.perf_counter()

        sample_count = 0
        for sample_idx, sample in enumerate(split_iterator):
            sample_count += 1
            deck_name = sample.metadata.get("deck_name", sample.sample_id)
            logger.info(
                f"[{split_key.value}] Processing sample {sample_idx + 1}: deck={deck_name!r} "
                f"({len(sample.pages)} pages)"
            )

            # Load slide images in memory
            images = [page.load().require_content() for page in sample.pages]

            qa_annotation = sample.get_annotation_by_type(
                AnnotationType.multi_page_question_answering
            )
            if qa_annotation is None:
                logger.warning(f"No QA annotation found for sample {sample.sample_id}")
                continue

            for qa in qa_annotation.qa_pairs:
                question = qa.question_text
                gold_answer = qa.answer_text
                targets = [gold_answer]
                if getattr(qa, "alternative_answers", None):
                    targets.extend(qa.alternative_answers)

                prediction = model.predict(images=images, question=question)

                anls = compute_anls(prediction, targets)
                f1 = compute_f1(prediction, targets)
                em = compute_exact_match(prediction, targets)

                anls_scores.append(anls)
                f1_scores.append(f1)
                em_scores.append(em)

                record = {
                    "deck_name": deck_name,
                    "qa_id": qa.id,
                    "question": question,
                    "prediction": prediction,
                    "golden_answer": gold_answer,
                    "targets": targets,
                    "evidence_pages": qa.evidence_pages,
                    "anls": round(anls, 4),
                    "f1": round(f1, 4),
                    "em": round(em, 4),
                }
                split_records.append(record)

                logger.info(
                    f"Q: {question[:50]!r} | Pred: {prediction!r} | Gold: {gold_answer!r} "
                    f"| ANLS={anls:.2f} F1={f1:.2f}"
                )

        elapsed = time.perf_counter() - start_time
        num_questions = len(anls_scores)
        mean_anls = sum(anls_scores) / num_questions if num_questions else 0.0
        mean_f1 = sum(f1_scores) / num_questions if num_questions else 0.0
        mean_em = sum(em_scores) / num_questions if num_questions else 0.0

        split_summary = {
            "num_decks": sample_count,
            "num_questions": num_questions,
            "mean_anls": round(mean_anls, 4),
            "mean_f1": round(mean_f1, 4),
            "mean_em": round(mean_em, 4),
            "elapsed_seconds": round(elapsed, 2),
            "questions_per_second": round(num_questions / elapsed, 2)
            if elapsed > 0
            else 0.0,
            "records": split_records,
        }
        results_by_split[split_key.value] = split_summary

        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - [{split_key.value.upper()}]")
        print("=" * 60)
        print(f"Total Decks Evaluated:     {split_summary['num_decks']}")
        print(f"Total Questions Evaluated: {split_summary['num_questions']}")
        print(f"Average ANLS:              {mean_anls * 100:.2f}%")
        print(f"Average Token F1:          {mean_f1 * 100:.2f}%")
        print(f"Exact Match (EM):          {mean_em * 100:.2f}%")
        print(f"Elapsed Time:              {elapsed:.2f}s")
        print("=" * 60 + "\n")

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "model_id": model.config.model_id,
        "api_url": getattr(model.config, "api_url", None),
        "mock": getattr(model.config, "mock", False),
        "splits": results_by_split,
    }

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, indent=2))
        logger.info(f"Saved evaluation report to {output_file}")

    return report
