"""Stage 05: build STaR training data from ViDoRe V3 (English documents only).

Downloads the five English-document subsets (hr, finance_en, industrial, pharmaceuticals,
computer_science): question tables, page Markdown (no images) and the original PDFs. Keeps
English short-answer questions whose gold pages sit in one document, picks 10 candidate pages
per question (gold + BM25 distractors; about 15% unanswerable with the gold pages left out),
and rewrites the long gold answer into MMLongBench's short form with a Qwen model.

    # on the cluster (vLLM serving Qwen3.5-9B): no API cost
    uv run usage/05_training.py --api-url http://<node>:<port>/v1 --model Qwen/Qwen3.5-9B

    # on OpenRouter (about 1,100 short calls, a few cents)
    OPENROUTER_API_KEY=sk-or-... uv run usage/05_training.py

    # download and select only; rewrite later (the rewrite cache is resumable)
    uv run usage/05_training.py --no-rewrite

    # quick look: 5 questions of one subset
    uv run usage/05_training.py --subsets hr --limit 5 --no-rewrite

Outputs in --out-dir (default data/vidore_v3): star_train.jsonl, star_dev.jsonl,
manifest.json, short_answers.jsonl, raw/. Re-running skips everything already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pine.training.vidore import ENGLISH_SUBSETS, BuildSettings, build, make_chat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, default=Path("data/vidore_v3"))
    p.add_argument(
        "--subsets",
        nargs="+",
        default=list(ENGLISH_SUBSETS),
        choices=list(ENGLISH_SUBSETS),
        help="ViDoRe V3 subsets with English documents (default: all five).",
    )
    p.add_argument(
        "--pages", type=int, default=10, help="Candidate pages per question."
    )
    p.add_argument(
        "--unanswerable-share",
        type=float,
        default=0.15,
        help="Share of questions sent without their gold pages (answer 'Not answerable').",
    )
    p.add_argument(
        "--dev-share",
        type=float,
        default=0.10,
        help="Share of documents held out as dev.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Questions per subset (testing)."
    )
    p.add_argument(
        "--api-url",
        default=os.getenv("QWEN_API_URL") or "https://openrouter.ai/api/v1",
        help="OpenAI-compatible endpoint of the rewrite model (vLLM on the cluster).",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("QWEN_API_KEY") or os.getenv("OPENROUTER_API_KEY", ""),
    )
    p.add_argument(
        "--model",
        default="qwen/qwen3.5-9b",
        help="Rewrite model (OpenRouter id, or e.g. Qwen/Qwen3.5-9B on vLLM).",
    )
    p.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip the short-answer rewrite; answerable questions wait in the cache step.",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel rewrite calls.")
    p.add_argument(
        "--exclude-docs",
        type=Path,
        default=None,
        help="JSON with an 'exclude' list of doc_ids to leave out (usage/05b_overlap.py).",
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if not args.no_rewrite and not args.api_key and "openrouter.ai" in args.api_url:
        p.error(
            "set OPENROUTER_API_KEY, pass --api-url for a vLLM server, or --no-rewrite"
        )
    settings = BuildSettings(
        out_dir=args.out_dir,
        subsets=tuple(args.subsets),
        pages=args.pages,
        unanswerable_share=args.unanswerable_share,
        dev_share=args.dev_share,
        limit=args.limit,
        workers=args.workers,
        shorten_model="" if args.no_rewrite else args.model,
        exclude_docs=tuple(
            json.loads(args.exclude_docs.read_text())["exclude"]
            if args.exclude_docs
            else ()
        ),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chat = (
        None if args.no_rewrite else make_chat(args.api_url, args.api_key, args.model)
    )
    manifest = build(settings, chat)
    print(json.dumps({k: v for k, v in manifest.items() if k != "settings"}, indent=2))
    print(
        f"STaR files: {(args.out_dir / 'star_train.jsonl').resolve()} (+ star_dev.jsonl)"
    )


if __name__ == "__main__":
    main()
