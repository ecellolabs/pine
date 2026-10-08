"""Stage 3: build (or rebuild) ``visual_samples.html``, a self-contained page
that visualises every step (parser, index, planner, orchestrator, verifier,
evaluation, model calls, logs) of the first N runs in an agent-runs folder
produced by ``usage/02_agent_qa.py``.  ``runs_dir`` is a batch folder
(``agent_runs/s0_2026-10-08``) or the root ``agent_runs`` (newest batch).

    uv run usage/03_visual_samples.py agent_runs
    uv run usage/03_visual_samples.py agent_runs/s0_2026-10-08
    uv run usage/03_visual_samples.py agent_runs --max-runs 5 --out ~/Desktop/visual_samples.html

The page needs no server: every run's logs, traces, page Markdown, thumbnails
and model calls are inlined, so the single file can be shared as is.  Because
everything is inlined it is capped at ``--max-runs`` (default 20) runs.
Optionally ``--report report.md`` renders a Markdown report to HTML/PDF next to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pine.agent.report_pdf import markdown_to_pdf
from pine.agent.visual_samples import DEFAULT_MAX_RUNS, build_visual_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        nargs="?",
        default=Path("./agent_runs"),
        help="Batch folder, or the root agent_runs folder (newest batch) (default: ./agent_runs).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=DEFAULT_MAX_RUNS,
        help=f"Visualise at most this many runs (default: {DEFAULT_MAX_RUNS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: <runs_dir>/visual_samples.html).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional Markdown report to render to HTML and PDF (PDF needs Chrome).",
    )
    args = parser.parse_args()

    out = build_visual_samples(args.runs_dir, args.out, max_runs=args.max_runs)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if args.report:
        pdf = markdown_to_pdf(args.report, args.report.with_suffix(".pdf"))
        if pdf:
            print(f"wrote {pdf}")
        else:
            print(
                f"wrote {args.report.with_suffix('.html')} (no Chrome found, PDF skipped)"
            )


if __name__ == "__main__":
    main()
