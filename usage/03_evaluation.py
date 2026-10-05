"""CLI entrypoint to run baseline Qwen Vision-Language model evaluation on SlideVQA via API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from atria_core.datasets import Dataset, DatasetBuilder, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType, MultiPageDocumentInstance

from pipeline_v2.datasets import *
from pipeline_v2.evaluation import evaluate_dataset
from pipeline_v2.models.qwen import (
    DEFAULT_QWEN_API_URL,
    DEFAULT_QWEN_MODEL_ID,
    QwenVLConfig,
    QwenVLModel,
)
from pipeline_v2.sampling import dataset_load_kwargs, describe, resolve_max_samples

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen Vision-Language baseline evaluation on SlideVQA via API."
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="slidevqa",
        help="Registered dataset name (default: 'slidevqa').",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible vLLM server URL (e.g. 'http://serv-3334:10001/v1'). "
            "Defaults to $QWEN_API_URL or $VLM_BASE_URL or "
            f"'{DEFAULT_QWEN_API_URL}'."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Model repo ID or served name (e.g. 'Qwen/Qwen2.5-VL-7B-Instruct' or 'Qwen/Qwen3-VL-8B-Instruct'). "
            f"Defaults to $QWEN_MODEL_ID or $VLM_MODEL or '{DEFAULT_QWEN_MODEL_ID}'."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API authorization key if required by proxy/server (default: 'dummy').",
    )
    parser.add_argument(
        "--split",
        choices=[s.value for s in DatasetSplitType],
        default="validation",
        help="Dataset split to evaluate (default: 'validation').",
    )
    parser.add_argument(
        "--max-samples",
        type=str,
        default=None,
        help=(
            "Integer N = first N decks (ideal for fast pilot runs); a name like "
            "'s0' = the exact documents listed in sample_sets/s0.json."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per question (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request HTTP timeout in seconds (default: 120.0).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("./evaluation_results/slidevqa_qwen_eval.json"),
        help="Path to save evaluation summary and predictions JSON.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock inference for dry-run verification without network/server calls.",
    )
    args = parser.parse_args()

    # Build model configuration
    config_kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "mock": args.mock,
    }
    if args.api_url:
        config_kwargs["api_url"] = args.api_url
    if args.model_id:
        config_kwargs["model_id"] = args.model_id
    if args.api_key:
        config_kwargs["api_key"] = args.api_key

    config = QwenVLConfig(**config_kwargs)
    model = QwenVLModel(config=config)

    # Pre-flight health check on API endpoint
    if not args.mock:
        logger.info(
            f"Testing connection to Qwen Vision API at {config.api_url} (model: {config.model_id})..."
        )
        ok, conn_msg = model.check_connection()
        if not ok:
            print("\n" + "!" * 80, file=sys.stderr)
            print(f"[PREFLIGHT FAILED] {conn_msg}", file=sys.stderr)
            print(
                "Please specify the active vLLM vision endpoint using:", file=sys.stderr
            )
            print("  --api-url http://<active-node>:<port>/v1", file=sys.stderr)
            print("Or export one of:", file=sys.stderr)
            print(
                "  export QWEN_API_URL=http://<active-node>:<port>/v1", file=sys.stderr
            )
            print(
                "  export VLM_BASE_URL=http://<active-node>:<port>/v1", file=sys.stderr
            )
            print("!" * 80 + "\n", file=sys.stderr)
            sys.exit(1)
        logger.info(f"[PREFLIGHT OK] {conn_msg}")

    split = DatasetSplitType(args.split) if args.split is not None else None
    logger.info(
        f"Loading dataset {args.name!r} (split={args.split}, max_samples={args.max_samples})"
    )
    resolved = resolve_max_samples(args.max_samples)
    logger.info(f"Selection: {describe(resolved)}")
    dataset = cast(
        Dataset[MultiPageDocumentInstance, DatasetConfig],
        DatasetBuilder()
        .load(args.name, split=split, **dataset_load_kwargs(resolved, args.name))
        .build(),
    )

    evaluate_dataset(dataset=dataset, model=model, output_file=args.output_file)


if __name__ == "__main__":
    main()
