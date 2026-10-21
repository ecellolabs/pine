"""B1 baseline: answer MMLongBench-Doc by putting the document straight in the prompt.

No index, planner, navigator or verifier. It reuses the questions of a finished pipeline
batch (so the question set is identical), the pipeline's scorer, and the Docling text the
pipeline already parsed. Run it against the same vLLM server to see what PINE adds.

    # 5-question pilot first (checks the server accepts many images per prompt)
    uv run usage/B1_mmlongbench_baseline.py --limit 5 \
        --data-dir /netscratch/akhtar/data/mmlongbench_doc \
        --api-url http://serv-3346:10001/v1

    # all pages as images, thinking on (closest to how a model card is likely measured)
    uv run usage/B1_mmlongbench_baseline.py --mode all --modality image --thinking ...

    # flat retrieval: BM25 top-5 pages as text
    uv run usage/B1_mmlongbench_baseline.py --mode bm25 --modality text --top-k 5 ...

Re-running the same command resumes: finished questions are skipped, errored ones retried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pine.agent.judge import make_judge_chat
from pine.agent.baseline import (
    BaselineSettings,
    load_specs,
    make_openai_chat,
    pine_scores,
    run_baseline,
    summarize,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--spec-batch",
        type=Path,
        default=Path("agent_runs/full_cluster"),
        help="Finished pipeline batch whose questions and gold answers are reused.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="MMLongBench-Doc dataset root (contains pdfs/).",
    )
    p.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Docling text cache (default <spec-batch>/_doc_cache; native PDF text if absent).",
    )
    p.add_argument("--api-url", default=os.getenv("QWEN_API_URL", ""))
    p.add_argument("--api-key", default=os.getenv("QWEN_API_KEY", "EMPTY"))
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--mode", choices=["all", "bm25"], default="all")
    p.add_argument("--modality", choices=["image", "text"], default="image")
    p.add_argument("--top-k", type=int, default=5, help="Pages sent in bm25 mode.")
    p.add_argument(
        "--max-pages",
        type=int,
        default=60,
        help="All mode: longer documents are cut to their BM25 top pages.",
    )
    p.add_argument(
        "--max-side", type=int, default=896, help="Longest page-image side in pixels."
    )
    p.add_argument(
        "--thinking", action="store_true", help="Turn Qwen thinking mode on."
    )
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--judge-model",
        default=None,
        help="Score with an LLM judge (same judge as pipeline step 6). Default: rule-based scorer.",
    )
    p.add_argument("--judge-api-url", default=None, help="Default: --api-url.")
    p.add_argument("--judge-api-key", default=None, help="Default: --api-key.")
    p.add_argument(
        "--judge-mode", choices=["judge", "extract", "both"], default="judge"
    )
    p.add_argument("--judge-disable-thinking", action="store_true")
    p.add_argument("--judge-max-tokens", type=int, default=300)
    p.add_argument("--judge-extra-body", type=json.loads, default=None)
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument(
        "--limit", type=int, default=None, help="Only the first N questions (pilot)."
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.api_url:
        sys.exit("Pass --api-url (or set QWEN_API_URL), e.g. http://serv-3346:10001/v1")

    specs = load_specs(args.spec_batch, args.data_dir / "pdfs")
    if not specs:
        sys.exit(f"No questions found under {args.spec_batch}/*/00_input/question.json")
    if args.limit:
        specs = specs[: args.limit]
    missing = [s.doc_id for s in specs if not s.pdf_path.exists()]
    if missing:
        sys.exit(f"PDF not found for {len(missing)} document(s), e.g. {missing[0]}")

    variant = f"{args.mode}_{args.modality}_{'think' if args.thinking else 'nothink'}"
    if args.mode == "bm25":
        variant += f"_k{args.top_k}"
    out_dir = args.out_dir or Path("agent_runs") / f"baseline_{variant}"
    parsed = args.parsed_dir or (args.spec_batch / "_doc_cache")
    settings = BaselineSettings(
        model=args.model,
        mode=args.mode,
        modality=args.modality,
        top_k=args.top_k,
        max_pages=args.max_pages,
        max_side=args.max_side,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        parsed_dir=parsed if parsed.is_dir() else None,
    )
    print(f"{len(specs)} questions -> {out_dir}  [{variant}]")
    if settings.parsed_dir is None:
        print(
            "note: no Docling text cache found, text mode will use native PDF text only"
        )

    chat = make_openai_chat(args.api_url, args.api_key, args.model, args.timeout)
    if args.judge_model:
        settings.judge_chat = make_judge_chat(
            args.judge_api_url or args.api_url,
            args.judge_api_key or args.api_key,
            args.judge_model,
            args.timeout,
            args.judge_disable_thinking,
            args.judge_max_tokens,
            args.judge_extra_body,
        )
        settings.judge_mode = args.judge_mode

    def progress(i: int, n: int, rec: dict) -> None:
        if rec["error"]:
            print(f"[{i}/{n}] {rec['run_id']}: ERROR {rec['error'][:160]}", flush=True)
        elif i % 25 == 0 or i == n:
            print(
                f"[{i}/{n}] last: {rec['run_id']} score={rec['score']:.2f}",
                flush=True,
            )

    records = run_baseline(specs, settings, chat, out_dir / "results.jsonl", progress)
    summary = summarize(records, len(specs))
    summary["variant"] = variant
    summary["settings"] = {k: str(v) for k, v in vars(settings).items()}
    pine = pine_scores(args.spec_batch)
    shared = [r for r in records if r["run_id"] in pine]
    if shared:
        summary["pine_accuracy_same_questions"] = sum(
            pine[r["run_id"]] for r in shared
        ) / len(shared)
        summary["baseline_accuracy_same_questions"] = sum(
            r["score"] for r in shared
        ) / len(shared)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(
        f"BASELINE {variant}: {summary['questions_scored']} of {summary['questions_total']} scored"
    )
    print(
        f"  accuracy                 {summary['accuracy']:.3f}   (errors counted as 0: {summary['accuracy_failed_counted_zero']:.3f})"
    )
    print(
        f"  answerable / unanswerable {summary['accuracy_answerable']:.3f} / {summary['accuracy_unanswerable']:.3f}"
    )
    print(f"  approx F1                {summary['f1_approx']:.3f}")
    print(
        f"  single / multi page      {summary['single_page']:.3f} / {summary['multi_page']:.3f}"
    )
    for fmt, (n, a) in summary["by_answer_format"].items():
        print(f"  {fmt:<8} n={n:<4} acc={a:.3f}")
    if summary["gold_page_in_context"] is not None:
        print(f"  gold page was in the prompt: {summary['gold_page_in_context']:.3f}")
    if summary["documents_cut_to_max_pages"]:
        print(
            f"  questions where the document was cut to fit: {summary['documents_cut_to_max_pages']}"
        )
    if summary["truncated_in_thinking"]:
        print(
            f"  replies cut off while thinking: {summary['truncated_in_thinking']} (raise --max-tokens)"
        )
    if shared:
        print(
            f"  PINE on the same questions: {summary['pine_accuracy_same_questions']:.3f}"
        )
    print(
        f"  tokens: {summary['prompt_tokens']:,} prompt, {summary['completion_tokens']:,} completion"
    )
    print(f"  results: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
