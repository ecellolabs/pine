"""Stage 4: run the grounded agentic DocQA pipeline (parser -> index -> planner
-> orchestrator -> verifier -> official scorer) over a dataset, or over a
selection of it (``--max-samples N`` or a sample set such as ``s0``), writing
one fully traced folder per question.

Examples::

    # Replicate the reference experiment (3 fixed samples from sample_sets/s0.json)
    OPENROUTER_API_KEY=... uv run usage/04_agent_qa.py mmlongbench_doc --max-samples s0

    # First 2 documents of the split, every question of each document
    OPENROUTER_API_KEY=... uv run usage/04_agent_qa.py mmlongbench_doc --max-samples 2

    # Same, against a vLLM server (OpenAI-compatible) instead of OpenRouter
    uv run usage/04_agent_qa.py mmlongbench_doc --max-samples s0 \
        --api-url http://serv-3334:10001/v1 --text-model Qwen/Qwen2.5-7B-Instruct \
        --vision-model Qwen/Qwen2.5-VL-7B-Instruct

Afterwards build the HTML visual-samples page (first 20 runs) with ``usage/05_visual_samples.py``.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep run logs clean
import sys
from pathlib import Path
from typing import Any, cast

from atria_core.datasets import Dataset, DatasetBuilder, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import AnnotationType, DatasetSplitType, MultiPageDocumentInstance

from pipeline_v2.agent.common import (
    DEFAULT_API_URL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    AgentSettings,
    write_json,
)
from pipeline_v2.agent.runner import RunSpec, run_sample
from pipeline_v2.agent.visual_samples import build_visual_samples
from pipeline_v2.datasets import *  # noqa: F403  (registers the datasets)
from pipeline_v2.sampling import (
    SampleSet,
    dataset_load_kwargs,
    describe,
    resolve_max_samples,
)

logger = get_logger(__name__)


def build_run_specs(
    dataset: Dataset[MultiPageDocumentInstance, DatasetConfig],
    dataset_name: str,
    resolved: int | SampleSet | None,
    questions_per_doc: int | None,
) -> list[RunSpec]:
    """Turn dataset samples (+ an optional sample set) into one RunSpec per question."""
    specs: list[RunSpec] = []
    pdf_dir = Path(dataset.data_dir) / "pdfs"
    for split_key, split_iterator in dataset.split_iterators.items():
        for sample in split_iterator:
            doc_id = str(sample.metadata.get("doc_id", sample.sample_id))
            qa = sample.get_annotation_by_type(
                AnnotationType.multi_page_question_answering
            )
            if qa is None:
                logger.warning(f"no QA annotation for {doc_id}; skipped")
                continue
            pdf_path = pdf_dir / doc_id
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF for {doc_id} not found at {pdf_path}")
            wanted = (
                resolved.questions_for(doc_id)
                if isinstance(resolved, SampleSet)
                else None
            )
            chosen = 0
            for qa_pair in qa.qa_pairs:
                run_id = f"{split_key.value}_{doc_id.rsplit('.', 1)[0][:24]}_q{qa_pair.id:02d}"
                why = ""
                if wanted is not None:
                    match = next(
                        (
                            w
                            for w in wanted
                            if w.question is None or w.question == qa_pair.question_text
                        ),
                        None,
                    )
                    if match is None:
                        continue
                    run_id = (
                        match.run_id
                        if match.question is not None
                        else f"{match.run_id}_q{qa_pair.id:02d}"
                    )
                    why = match.why_chosen
                elif questions_per_doc is not None and chosen >= questions_per_doc:
                    break
                chosen += 1
                gold: dict[str, Any] = {
                    "answer": qa_pair.answer_text,
                    "answer_format": getattr(qa_pair, "answer_format", "Str"),
                    # the loader stores 0-indexed pages; the agent and the report use 1-indexed
                    "evidence_pages": [
                        int(p) + 1 for p in (qa_pair.evidence_pages or [])
                    ],
                    "evidence_sources": list(
                        getattr(qa_pair, "evidence_sources", []) or []
                    ),
                    "qa_id": qa_pair.id,
                    "split": split_key.value,
                }
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        doc_id=doc_id,
                        question=qa_pair.question_text,
                        pdf_path=pdf_path,
                        gold=gold,
                        doc_type=str(sample.metadata.get("doc_type", "")),
                        why_chosen=why,
                        dataset=dataset_name,
                    )
                )
    if isinstance(resolved, SampleSet):
        found = {s.run_id for s in specs}
        for w in resolved.samples:
            if w.question is not None and w.run_id not in found:
                logger.warning(
                    f"sample {w.run_id!r}: question not found verbatim in {w.doc_id}; "
                    "check the text in the sample-set file"
                )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="mmlongbench_doc",
        help="Registered dataset name (default: mmlongbench_doc).",
    )
    parser.add_argument(
        "--max-samples",
        type=str,
        default=None,
        help="Integer N = first N documents; a name like 's0' = the exact samples listed in sample_sets/s0.json (a .json path also works).",
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=None,
        help="With an integer --max-samples: run only the first K questions of each document (default: all).",
    )
    parser.add_argument(
        "--split", choices=[s.value for s in DatasetSplitType], default=None
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root directory (default: atria cache).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./agent_runs"),
        help="Where run folders are written (default: ./agent_runs).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("QWEN_API_URL") or DEFAULT_API_URL,
        help=f"OpenAI-compatible base URL (default: $QWEN_API_URL or {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY") or os.getenv("QWEN_API_KEY") or "",
        help="API key (default: $OPENROUTER_API_KEY or $QWEN_API_KEY). Never written to the run folders.",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=os.getenv("TEXT_MODEL") or DEFAULT_TEXT_MODEL,
        help="Planner / orchestrator / verifier / page-summary model.",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=os.getenv("VISION_MODEL") or DEFAULT_VISION_MODEL,
        help="Model used for page images.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Plan/act/verify rounds per question (default: 2).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=12,
        help="Tool-call budget per round (default: 12).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="LLM response cache (default: <output-dir>/.cache/llm). Identical requests are never paid for twice.",
    )
    parser.add_argument(
        "--docling-url",
        type=str,
        default=os.getenv("DOCLING_API_URL"),
        help="Use the remote Docling API instead of the in-process library.",
    )
    parser.add_argument(
        "--rapidocr-backend",
        type=str,
        default=os.getenv("RAPIDOCR_BACKEND", "torch"),
        help="RapidOCR inference backend for in-process Docling (torch | onnxruntime).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not (re)build <output-dir>/visual_samples.html at the end.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "No API key: pass --api-key or set OPENROUTER_API_KEY / QWEN_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(2)

    resolved = resolve_max_samples(args.max_samples)
    load_kwargs = dataset_load_kwargs(resolved, args.name)
    split = DatasetSplitType(args.split) if args.split else None
    logger.info(f"Loading dataset {args.name!r}: {describe(resolved)}")
    dataset = cast(
        Dataset[MultiPageDocumentInstance, DatasetConfig],
        DatasetBuilder()
        .load(args.name, data_dir=args.data_dir, split=split, **load_kwargs)
        .build(),
    )
    specs = build_run_specs(dataset, args.name, resolved, args.questions_per_doc)
    if not specs:
        print("Nothing to run (no matching samples).", file=sys.stderr)
        sys.exit(1)

    settings = AgentSettings(
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        text_model=args.text_model,
        vision_model=args.vision_model,
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        cache_dir=args.cache_dir,
        docling_url=args.docling_url,
        rapidocr_backend=args.rapidocr_backend,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": args.name,
        "selection": describe(resolved),
        "sample_set": resolved.name if isinstance(resolved, SampleSet) else None,
        "settings": settings.public(),
        "runs": [],
    }
    logger.info(f"{len(specs)} sample run(s) -> {args.output_dir.resolve()}")
    results = []
    for i, spec in enumerate(specs, 1):
        logger.info(f"[{i}/{len(specs)}] {spec.run_id}: {spec.question!r}")
        try:
            result = run_sample(spec, settings)
            manifest["runs"].append(
                {
                    "run_id": spec.run_id,
                    "doc_id": spec.doc_id,
                    "question": spec.question,
                    "status": "ok",
                    "score": result["score_official_rule"],
                    "answer": result["raw_system_answer"],
                }
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"run {spec.run_id} failed")
            manifest["runs"].append(
                {
                    "run_id": spec.run_id,
                    "doc_id": spec.doc_id,
                    "question": spec.question,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        write_json(args.output_dir / "manifest.json", manifest)

    scored = [r["score_official_rule"] for r in results]
    print("\n" + "=" * 70)
    print(f"AGENT QA RUNS  dataset={args.name}  {describe(resolved)}")
    print("=" * 70)
    for entry in manifest["runs"]:
        if entry["status"] == "ok":
            print(
                f"  {entry['run_id']:<40} score={entry['score']:.2f}  answer={entry['answer']!r}"
            )
        else:
            print(f"  {entry['run_id']:<40} FAILED: {entry['error'][:80]}")
    if scored:
        print(
            f"  mean official-rule score: {sum(scored) / len(scored):.3f} over {len(scored)} question(s)"
        )
    print(f"  total cost: ${sum(r['cost']['cost_usd'] for r in results):.4f}")
    print("=" * 70)
    if not args.no_html and results:
        out = build_visual_samples(args.output_dir)
        print(f"  dashboard: {out}")


if __name__ == "__main__":
    main()
