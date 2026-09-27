"""CiteVQA Evaluator implementation following official OpenDataLab evaluation standard."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atria_core.datasets import Dataset, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import AnnotationType, MultiPageDocumentInstance

from pipeline_v2.evaluators.base import BaseEvaluator
from pipeline_v2.metrics import compute_anls, compute_exact_match, compute_f1
from pipeline_v2.models.qwen import QwenVLModel
from pipeline_v2.parsers.docling import DoclingTransform

logger = get_logger(__name__)


def extract_cited_pages(prediction_text: str) -> list[int]:
    """Extracts 0-indexed page numbers cited in model prediction text (e.g. 'Page 1', '[Page 2]')."""
    found_pages: set[int] = set()
    # Match patterns like Page 1, Page: 2, [Page 3], p. 1
    matches = re.findall(
        r"(?:page|p\.)\s*:?\s*\[?(\d+)\]?", prediction_text, re.IGNORECASE
    )
    for m in matches:
        try:
            page_num = int(m)
            # Convert 1-indexed to 0-indexed if positive
            if page_num > 0:
                found_pages.add(page_num - 1)
        except ValueError:
            continue
    return sorted(found_pages)


class CiteVQAEvaluator(BaseEvaluator):
    """Evaluates Vision-Language & Document Parsing models on CiteVQA benchmark.

    Evaluates:
    - Textual Answer Metrics: ANLS, Token F1, Exact Match (EM).
    - Evidence Attribution Metrics:
        - Page Citation Accuracy (PCA)
        - Strict Attributed Accuracy (SAA): Credit only when both answer (ANLS >= 0.5)
          and page citation ground-truth match.
    """

    def evaluate(
        self,
        dataset: Dataset[MultiPageDocumentInstance, DatasetConfig],
        model: QwenVLModel,
        output_file: Path | None = None,
        docling_transform: DoclingTransform | None = None,
    ) -> dict[str, Any]:
        results_by_split: dict[str, Any] = {}

        for split_key, split_iterator in dataset.split_iterators.items():
            logger.info(f"[CiteVQA] Evaluating split: {split_key.value}")
            split_records: list[dict[str, Any]] = []

            scores_by_type: dict[str, dict[str, list[float]]] = {}

            start_time = time.perf_counter()
            sample_count = 0

            for sample_idx, sample in enumerate(split_iterator):
                if not sample.pages or sample.metadata.get("invalid_pdf"):
                    logger.warning(
                        f"[{split_key.value}] Skipping sample {sample_idx + 1}: empty or invalid PDF document"
                    )
                    continue

                try:
                    images = [page.load().require_content() for page in sample.pages]
                except Exception as err:  # noqa: BLE001
                    logger.warning(
                        f"[{split_key.value}] Skipping sample {sample_idx + 1}: could not load pages ({err})"
                    )
                    continue

                sample_count += 1
                doc_id = sample.metadata.get("doc_id", sample.sample_id)
                dataset_type = sample.metadata.get("dataset_type", "Single-Doc")
                logger.info(
                    f"[{split_key.value}] Processing sample {sample_idx + 1}: question_id={sample.sample_id!r} "
                    f"type={dataset_type!r} ({len(sample.pages)} total pages)"
                )

                # Optional Docling layout parsing integration
                docling_parsed_pages: list[str] = []
                if docling_transform is not None:
                    try:
                        for page in sample.pages:
                            doc = docling_transform(page)
                            docling_parsed_pages.append(doc.export_to_markdown())
                    except Exception as err:  # noqa: BLE001
                        logger.warning(
                            f"Docling parsing failed for {sample.sample_id}, falling back to visual QA: {err}"
                        )

                qa_annotation = sample.get_annotation_by_type(
                    AnnotationType.multi_page_question_answering
                )
                if qa_annotation is None:
                    logger.warning(
                        f"No QA annotation found for sample {sample.sample_id}"
                    )
                    continue

                for qa in qa_annotation.qa_pairs:
                    question = qa.question_text
                    gold_answer = qa.answer_text
                    targets = [gold_answer]
                    if getattr(qa, "alternative_answers", None):
                        targets.extend(qa.alternative_answers)

                    gold_evidence_pages = sorted(qa.evidence_pages or [])

                    # Construct specialized prompt for CiteVQA requesting answer & page citation
                    prompt = (
                        f"{question}\n"
                        "Provide a concise answer. If applicable, cite the 1-indexed page number(s) "
                        "containing the evidence (e.g. 'Answer: ... [Page 1]')."
                    )

                    prediction = model.predict(images=images, question=prompt)

                    anls = compute_anls(prediction, targets)
                    f1 = compute_f1(prediction, targets)
                    em = compute_exact_match(prediction, targets)

                    predicted_pages = extract_cited_pages(prediction)

                    # Compute Page Citation Accuracy & SAA
                    if not gold_evidence_pages:
                        page_match = True
                    else:
                        page_match = bool(
                            set(predicted_pages).intersection(set(gold_evidence_pages))
                        )

                    page_cite_acc = 1.0 if page_match else 0.0
                    saa = 1.0 if (anls >= 0.5 and page_match) else 0.0

                    anls_scores.append(anls)
                    f1_scores.append(f1)
                    em_scores.append(em)
                    saa_scores.append(saa)
                    page_cite_acc_scores.append(page_cite_acc)

                    type_entry = scores_by_type.setdefault(
                        dataset_type,
                        {"anls": [], "f1": [], "em": [], "saa": [], "pca": []},
                    )
                    type_entry["anls"].append(anls)
                    type_entry["f1"].append(f1)
                    type_entry["em"].append(em)
                    type_entry["saa"].append(saa)
                    type_entry["pca"].append(page_cite_acc)

                    record = {
                        "question_id": sample.sample_id,
                        "dataset_type": dataset_type,
                        "doc_id": doc_id,
                        "qa_id": qa.id,
                        "question": question,
                        "prediction": prediction,
                        "golden_answer": gold_answer,
                        "targets": targets,
                        "gold_evidence_pages": gold_evidence_pages,
                        "predicted_evidence_pages": predicted_pages,
                        "anls": round(anls, 4),
                        "f1": round(f1, 4),
                        "em": round(em, 4),
                        "saa": round(saa, 4),
                        "page_cite_acc": round(page_cite_acc, 4),
                    }
                    split_records.append(record)

                    logger.info(
                        f"Q: {question[:40]!r} | Pred: {prediction!r} | Gold: {gold_answer!r} "
                        f"| ANLS={anls:.2f} SAA={saa:.2f}"
                    )

            elapsed = time.perf_counter() - start_time
            num_questions = len(anls_scores)
            mean_anls = sum(anls_scores) / num_questions if num_questions else 0.0
            mean_f1 = sum(f1_scores) / num_questions if num_questions else 0.0
            mean_em = sum(em_scores) / num_questions if num_questions else 0.0
            mean_saa = sum(saa_scores) / num_questions if num_questions else 0.0
            mean_pca = (
                sum(page_cite_acc_scores) / num_questions if num_questions else 0.0
            )

            by_type_summary: dict[str, dict[str, float]] = {}
            for d_type, scores in scores_by_type.items():
                n_type = len(scores["anls"])
                by_type_summary[d_type] = {
                    "count": n_type,
                    "mean_anls": round(sum(scores["anls"]) / n_type, 4) if n_type else 0.0,
                    "mean_f1": round(sum(scores["f1"]) / n_type, 4) if n_type else 0.0,
                    "mean_em": round(sum(scores["em"]) / n_type, 4) if n_type else 0.0,
                    "mean_saa": round(sum(scores["saa"]) / n_type, 4) if n_type else 0.0,
                    "mean_page_citation_acc": round(sum(scores["pca"]) / n_type, 4) if n_type else 0.0,
                }

            split_summary = {
                "num_samples": sample_count,
                "num_questions": num_questions,
                "mean_anls": round(mean_anls, 4),
                "mean_f1": round(mean_f1, 4),
                "mean_em": round(mean_em, 4),
                "mean_saa": round(mean_saa, 4),
                "mean_page_citation_acc": round(mean_pca, 4),
                "by_dataset_type": by_type_summary,
                "elapsed_seconds": round(elapsed, 2),
                "questions_per_second": round(num_questions / elapsed, 2)
                if elapsed > 0
                else 0.0,
                "records": split_records,
            }
            results_by_split[split_key.value] = split_summary

            print("\n" + "=" * 60)
            print(f"EVALUATION RESULTS - [CITEVQA / {split_key.value.upper()}]")
            print("=" * 60)
            print(f"Total Question Samples Evaluated: {split_summary['num_questions']}")
            print(f"Average ANLS:                     {mean_anls * 100:.2f}%")
            print(f"Average Token F1:                 {mean_f1 * 100:.2f}%")
            print(f"Exact Match (EM):                 {mean_em * 100:.2f}%")
            print(f"Strict Attributed Accuracy (SAA): {mean_saa * 100:.2f}%")
            print(f"Page Citation Accuracy:           {mean_pca * 100:.2f}%")
            if by_type_summary:
                print("-" * 60)
                print("Breakdown by Dataset Type:")
                for d_type, stats in by_type_summary.items():
                    print(
                        f"  [{d_type}] (N={stats['count']}) "
                        f"SAA: {stats['mean_saa']*100:.2f}% | ANLS: {stats['mean_anls']*100:.2f}%"
                    )
            print(f"Elapsed Time:                     {elapsed:.2f}s")
            print("=" * 60 + "\n")

        report = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "dataset": "citevqa",
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
