"""Stage 06: STaR sampling (and the base-model evaluation) over the stage-05 examples.

For every example of --split, sends its 10 candidate pages exactly as ``--navigator direct``
does, draws --k thinking samples, scores each final answer (the official rules when they
pass fully, otherwise the LLM judge, default gpt-oss-120b, the project judge) and, for answerable examples with no correct
sample, samples again with a hint naming the gold pages.

    # base accuracy on star_dev: one sample, no hint (step 4 of the STaR plan)
    OPENROUTER_API_KEY=sk-or-... uv run usage/06_star_sample.py --data-dir data/vidore_dev \
        --split dev --k 1 --no-rationalize

    # STaR samples for training: answer model on the cluster's vLLM server, judge on OpenRouter
    uv run usage/06_star_sample.py --split train \
        --api-url http://<node>:<port>/v1 --model Qwen/Qwen3.5-9B \
        --judge-api-url https://openrouter.ai/api/v1 --judge-api-key $OPENROUTER_API_KEY

Writes <data-dir>/star/<split>_<tag>.jsonl (one row per example: every sample with its
reasoning, answer, judge verdict and reason, the official rule score for audit, and
whether it was hinted; status and the chosen sample) and
<split>_<tag>.manifest.json (settings, judge accuracy, rule accuracy, pass@k, counts, tokens, cost). The tag
defaults to k<k>[_norat]. Re-running skips examples already in the JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from pine.agent.judge import make_judge_chat
from pine.training.chat import THINK_MAX_TOKENS, make_thinking_chat
from pine.training.star import StarSettings, load_rows, run_star, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/vidore_v3"))
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--k", type=int, default=4, help="Samples per example.")
    p.add_argument(
        "--no-rationalize",
        action="store_true",
        help="Skip the hinted second pass (use for evaluation).",
    )
    p.add_argument("--limit", type=int, default=None, help="First N examples only.")
    p.add_argument("--tag", default=None, help="Output name (default k<k>[_norat]).")
    p.add_argument(
        "--api-url",
        default=os.getenv("QWEN_API_URL") or "https://openrouter.ai/api/v1",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("QWEN_API_KEY") or os.getenv("OPENROUTER_API_KEY", ""),
    )
    p.add_argument(
        "--model",
        default="qwen/qwen3.5-9b",
        help="Answer model (OpenRouter id, or e.g. Qwen/Qwen3.5-9B / a LoRA name on vLLM).",
    )
    p.add_argument("--max-tokens", type=int, default=THINK_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=900.0, help="Seconds per call.")
    p.add_argument("--workers", type=int, default=8, help="Parallel examples.")
    p.add_argument(
        "--judge-model",
        default="openai/gpt-oss-120b",
        help="LLM judge that decides whether a sample is correct.",
    )
    p.add_argument(
        "--judge-api-url",
        default=None,
        help="OpenAI-compatible endpoint of the judge (default: --api-url).",
    )
    p.add_argument("--judge-api-key", default=None, help="Default: --api-key.")
    p.add_argument(
        "--judge-all",
        action="store_true",
        help="Judge every sample, also those the official rules already score 1.0.",
    )
    p.add_argument(
        "--judge-max-tokens",
        type=int,
        default=2000,
        help="Reply budget of the judge (a reasoning judge thinks before the verdict).",
    )
    p.add_argument(
        "--judge-extra-body",
        type=json.loads,
        default={"reasoning_effort": "low"},
        help="JSON passed to the judge call (default: low reasoning effort for gpt-oss).",
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    judge_url = args.judge_api_url or args.api_url
    judge_key = args.judge_api_key if args.judge_api_key is not None else args.api_key
    for url, key in ((args.api_url, args.api_key), (judge_url, judge_key)):
        if not key and "openrouter.ai" in url:
            p.error("set OPENROUTER_API_KEY or pass --api-url for a vLLM server")

    src = args.data_dir / f"star_{args.split}.jsonl"
    records = list(load_rows(src))
    if not records:
        p.error(f"no examples in {src}; run usage/05_training.py first")
    records = records[: args.limit] if args.limit else records
    tag = args.tag or f"k{args.k}" + ("_norat" if args.no_rationalize else "")
    out = args.data_dir / "star" / f"{args.split}_{tag}.jsonl"

    settings = StarSettings(
        data_dir=args.data_dir,
        k=args.k,
        rationalize=not args.no_rationalize,
        workers=args.workers,
        model=args.model,
        judge_model=args.judge_model,
        judge_all=args.judge_all,
    )
    sample_fn = make_thinking_chat(
        args.api_url,
        args.api_key,
        args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    judge = make_judge_chat(
        judge_url,
        judge_key,
        args.judge_model,
        max_tokens=args.judge_max_tokens,
        extra_body=args.judge_extra_body,
    )
    rows = run_star(records, out, sample_fn, judge, settings)
    summary = summarize(rows, len(records))
    manifest = {
        "settings": {
            **{
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(settings).items()
            },
            "split": args.split,
            "limit": args.limit,
            "api_url": args.api_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "judge_api_url": judge_url,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_extra_body": args.judge_extra_body,
        },
        **summary,
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"samples: {out.resolve()}")


if __name__ == "__main__":
    main()
