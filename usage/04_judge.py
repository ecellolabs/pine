"""Stage 4: rescore saved answers with an LLM (answer extraction and/or judge).

Reads the answers an earlier stage already wrote, so nothing is rerun on the models or the
documents. SOURCE is a pipeline batch (``agent_runs/full_cluster``) or a baseline folder
(``agent_runs/baseline_all_image_think``).

    # 100-answer calibration first, then read <out>/flips.md by hand
    uv run usage/04_judge.py agent_runs/full_cluster --limit 100 \
        --api-url https://openrouter.ai/api/v1 --model <judge model> --api-key $OPENROUTER_API_KEY

    # a model on your own vLLM server instead
    uv run usage/04_judge.py agent_runs/full_cluster --api-url http://<node>:10001/v1 \
        --model Qwen/Qwen3.5-9B --disable-thinking

Writes <SOURCE>/judge_<tag>/items.jsonl, summary.json and flips.md. Re-running resumes.
Use a judge that is not the model being scored: a model grading its own answers is biased.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from pine.agent.judge import (
    load_items,
    make_judge_chat,
    run_judge,
    summarize,
    write_flips,
)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", type=Path, help="Pipeline batch or baseline folder.")
    p.add_argument("--api-url", default=os.getenv("JUDGE_API_URL", ""))
    p.add_argument(
        "--api-key",
        default=None,
        help="Default: $JUDGE_API_KEY, else $OPENROUTER_API_KEY; a claude-* judge reads "
        "$ANTHROPIC_API_KEY instead of the OpenRouter key.",
    )
    p.add_argument("--model", required=True, help="Judge model name on that endpoint.")
    p.add_argument("--mode", choices=["extract", "judge", "both"], default="both")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only N answers, a fixed random sample (calibration).",
    )
    p.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send enable_thinking=false (Qwen3.x on vLLM).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Reply budget; use ~2000 for a reasoning judge such as gpt-oss.",
    )
    p.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help='JSON of extra request fields, e.g. \'{"reasoning_effort": "low"}\'.',
    )
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--tag", default=None, help="Output folder suffix (default: model name)."
    )
    args = p.parse_args()

    claude = args.model.startswith("claude-")
    if args.api_key is None:
        args.api_key = os.getenv("JUDGE_API_KEY") or (
            "" if claude else os.getenv("OPENROUTER_API_KEY", "")
        )
    if not args.api_url and not claude:
        sys.exit("Pass --api-url (or set JUDGE_API_URL).")
    items = load_items(args.source)
    if not items:
        sys.exit(f"No answers found in {args.source}")
    if args.limit and args.limit < len(items):
        # a fixed random sample, so the calibration is not just the first documents
        items = sorted(
            random.Random(0).sample(items, args.limit), key=lambda i: i.run_id
        )
    tag = args.tag or re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out_dir = args.source / f"judge_{tag}"
    print(
        f"{len(items)} answers from {args.source} -> {out_dir}  [{args.mode}, judge {args.model}]"
    )

    chat = make_judge_chat(
        args.api_url,
        args.api_key,
        args.model,
        args.timeout,
        args.disable_thinking,
        args.max_tokens,
        args.extra_body,
    )

    def progress(i: int, n: int) -> None:
        if i % 100 == 0 or i == n:
            print(f"[{i}/{n}]", flush=True)

    records = run_judge(
        items, chat, out_dir / "items.jsonl", args.mode, args.workers, progress
    )
    summary = summarize(records, len(items))
    summary["judge_model"] = args.model
    summary["mode"] = args.mode
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_flips(records, out_dir / "flips.md")

    print("\n" + "=" * 70)
    print(
        f"RESCORED {summary['items_scored']} of {summary['items_total']} answers  (judge: {args.model})"
    )
    print(f"  rules (original)       {pct(summary['accuracy_rules_original'])}")
    if summary["accuracy_llm_extraction"] is not None:
        print(f"  LLM extraction + rules {pct(summary['accuracy_llm_extraction'])}")
    if summary["accuracy_llm_judge"] is not None:
        print(f"  LLM judge              {pct(summary['accuracy_llm_judge'])}")
    for name in ("extraction", "judge"):
        k = f"{name}_vs_rules"
        if k in summary:
            s = summary[k]
            print(
                f"  {name} vs rules: agree {pct(s['agree'])} | rules wrong, new right {s['rules_wrong_new_right']} | rules right, new wrong {s['rules_right_new_wrong']}"
            )
    for fmt, v in summary["by_format"].items():
        print(
            f"  {fmt:<8} n={v['n']:<4} rules {pct(v['rules'])}  extraction {pct(v['extraction'])}  judge {pct(v['judge'])}"
        )
    if summary["empty_answers"]:
        print(f"  empty answers (counted wrong): {summary['empty_answers']}")
    if summary["extraction_unparsed"] or summary["judge_unparsed"]:
        print(
            f"  unparsed replies: extraction {summary['extraction_unparsed']}, judge {summary['judge_unparsed']}"
        )
    print(f"  read the disagreements: {out_dir / 'flips.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
