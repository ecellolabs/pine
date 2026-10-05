"""End-to-end driver for one sample: parser -> index -> planner -> orchestrator
-> verifier -> official scorer.  Each step writes its own numbered sub-folder
with a log; ``run.log`` is the combined log; ``summary.md`` is human readable.

Folder layout produced for every sample (``<output_dir>/<run_id>/``)::

    00_input/              question.json, PDF copy, gold_evidence_page_*.jpg
    01_parser/             parser.log, page_stats.json, pages/page_NNN.md
    02_index/              index.log, outline.md, index.json, page_summaries.jsonl
    03_planner/            planner.log, plan_roundN.json
    04_orchestrator/       orchestrator.log, trace_roundN.jsonl, tool_results/,
                           images_viewed/, anomalies_roundN.json, ledger_roundN.json
    05_evidence_verifier/  verifier.log, verification_roundN.json
    06_evaluation/         eval.log, result.json
    run.log, driver.log, llm_calls.jsonl, cost.json, summary.md
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline_v2.agent.common import (
    STEP_DIRS,
    AgentSettings,
    CostLedger,
    read_json,
    render_page,
    setup_logger,
    write_json,
)
from pipeline_v2.agent.index import run_index
from pipeline_v2.agent.orchestrator import run_orchestrator
from pipeline_v2.agent.parser import run_parser
from pipeline_v2.agent.planner import run_planner
from pipeline_v2.agent.scoring import run_eval
from pipeline_v2.agent.verifier import run_verifier


@dataclass
class RunSpec:
    """One question on one document, with its gold annotation."""

    run_id: str
    doc_id: str
    question: str
    pdf_path: Path
    gold: dict[str, Any]
    doc_type: str = ""
    why_chosen: str = ""
    dataset: str = "mmlongbench_doc"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pdf_path"] = str(self.pdf_path)
        return data


def _summary_md(
    spec: RunSpec,
    index: dict[str, Any],
    final: dict[str, Any],
    verification: dict[str, Any],
    result: dict[str, Any],
    cost: dict[str, Any],
    timings: dict[str, Any],
    run_dir: Path,
    tool_calls_total: int,
) -> str:
    gold = spec.gold
    anomalies: list[dict[str, Any]] = []
    for f in sorted((run_dir / STEP_DIRS[4]).glob("anomalies_round*.json")):
        anomalies += read_json(f)
    st = read_json(run_dir / STEP_DIRS[1] / "page_stats.json")
    flags = "; ".join(
        f"p{s['page']}: {s['flag'].split(' (')[0]}"
        for s in st["page_stats"]
        if s["flag"]
    )
    lines = [
        f"# {spec.run_id}",
        "",
        f"**Document:** {spec.doc_id} ({spec.doc_type}, {index['n_pages']} pages)  ",
        f"**Question:** {spec.question}  ",
        (
            f"**Gold:** `{gold['answer']}` ({gold['answer_format']}; evidence pages "
            f"{gold['evidence_pages']}; sources {gold['evidence_sources']})  "
        ),
        f"**Why chosen:** {spec.why_chosen}",
        "",
        "## Outcome",
        "",
        "| | |",
        "|---|---|",
        f"| System answer | `{result['raw_system_answer']}` |",
        (
            f"| Official-rule score | {result['score_official_rule']:.2f} "
            f"({'CORRECT' if result['correct'] else 'WRONG'}) |"
        ),
        f"| Agent status / verifier decision | {final['status']} / {verification['decision']} |",
        f"| Cited pages (gold hit) | {result['cited_pages']} ({result['cited_page_hit']}) |",
        f"| Gold page visited by agent | {result['gold_page_visited_by_agent']} |",
        f"| Rounds / tool calls | {final['round']} / {tool_calls_total} |",
        (
            f"| LLM calls / tokens in+out / cost | {cost['llm_calls']} / "
            f"{cost['prompt_tokens']}+{cost['completion_tokens']} / ${cost['cost_usd']:.4f} |"
        ),
        (
            f"| Wall time parser / index / agent loop | {timings['01_parser']}s / "
            f"{timings['02_index']}s / {timings['03-05_agent_loop']}s |"
        ),
        "",
        "## Step-by-step",
        "",
        "| Step | Folder | Key files |",
        "|---|---|---|",
        "| 0 Input | 00_input | question.json, PDF copy, gold_evidence_page_*.jpg |",
        "| 1 Parser (Docling+RapidOCR) | 01_parser | parser.log, page_stats.json, pages/page_NNN.md |",
        "| 2 Index (PageIndex-style tree) | 02_index | index.log, outline.md, index.json, page_summaries.jsonl, llm_calls.jsonl |",
        "| 3 Planner | 03_planner | planner.log, plan_roundN.json |",
        "| 4 Orchestrator (tool calling) | 04_orchestrator | orchestrator.log, trace_roundN.jsonl, tool_results/, images_viewed/, anomalies_roundN.json |",
        "| 5 Evidence + Verifier | 05_evidence_verifier | verifier.log, verification_roundN.json |",
        "| 6 Evaluation (official rules) | 06_evaluation | eval.log, result.json |",
        "",
        "## Parser diagnostics",
        "",
        (
            f"- Docling status {st['docling_status']}, {st['elapsed_s']}s, {st['total_tables']} "
            f"tables, {st['total_pictures']} pictures, {st['pages_flagged']}/{st['pages']} pages flagged."
        ),
        f"- Flags: {flags[:1500] or 'none'}",
        "",
        "## Index diagnostics",
        "",
        (
            f"- {len(index['sections'])} top-level sections; "
            f"{index['pages_summarised_from_image']} pages summarised from the image; "
            f"hierarchy fallback used: {index['hierarchy_fallback_used']}; "
            f"validation problems: {index['validation_problems_final'] or 'none'}; "
            f"summary failures: {len(index['page_summary_failures'])}."
        ),
        "",
        "## Orchestrator anomalies",
        "",
    ]
    lines += [
        f"- {a['type']}: "
        f"{json.dumps({k: v for k, v in a.items() if k != 'type'}, ensure_ascii=False)[:300]}"
        for a in anomalies
    ] or ["- none"]
    verdict = verification["llm_verdict"]
    lines += [
        "",
        "## Verifier",
        "",
        f"- LLM verdict: {verdict.get('verdict')} — {verdict.get('explanation')}",
        f"- Quote checks: {[(q['page'], q['source'], q['sim_page_text']) for q in verification['quote_checks']]}",
        f"- Gaps: {verification['gaps']}",
        "",
    ]
    return "\n".join(lines)


def run_sample(spec: RunSpec, settings: AgentSettings) -> dict[str, Any]:
    """Run the whole pipeline for one sample and return the evaluation result."""
    run_dir = settings.output_dir / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "run.log"
    log = setup_logger("run", run_dir / "driver.log", run_log)
    timings: dict[str, Any] = {}
    ledger = CostLedger(run_dir)
    gold = spec.gold

    # ---- 00 input
    d0 = run_dir / STEP_DIRS[0]
    d0.mkdir(exist_ok=True)
    write_json(
        d0 / "question.json",
        {
            "run_id": spec.run_id,
            "dataset": spec.dataset,
            "doc_id": spec.doc_id,
            "question": spec.question,
            "why_chosen": spec.why_chosen,
            "gold": gold,
            "text_model": settings.text_model,
            "vision_model": settings.vision_model,
            "settings": settings.public(),
            "requested_model_note": (
                "Proposal names Qwen2.5-VL-7B-Instruct; the vision model actually "
                "used is recorded in 'vision_model'."
            ),
        },
    )
    if not (d0 / spec.doc_id).exists():
        shutil.copy(spec.pdf_path, d0 / spec.doc_id)
    for p in gold.get("evidence_pages", []):
        out = d0 / f"gold_evidence_page_{p:03d}.jpg"
        if not out.exists():
            out.write_bytes(render_page(spec.pdf_path, p, max_side=1024))
    log.info(
        "RUN %s | doc=%s | Q=%r | gold=%r (%s) pages=%s",
        spec.run_id,
        spec.doc_id,
        spec.question,
        gold["answer"],
        gold["answer_format"],
        gold["evidence_pages"],
    )

    # ---- 01 parser
    t = time.perf_counter()
    run_parser(
        spec.pdf_path,
        run_dir / STEP_DIRS[1],
        run_log,
        rapidocr_backend=settings.rapidocr_backend,
        docling_url=settings.docling_url,
    )
    timings["01_parser"] = round(time.perf_counter() - t, 1)

    # ---- 02 index
    t = time.perf_counter()
    index = run_index(
        settings,
        spec.pdf_path,
        run_dir / STEP_DIRS[1],
        run_dir / STEP_DIRS[2],
        run_log,
        ledger,
    )
    timings["02_index"] = round(time.perf_counter() - t, 1)
    cost_after_index = ledger.summary()

    # ---- 03-05 loop
    gaps: list[str] | None = None
    feedback: str | None = None
    final: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    tool_calls_total = 0
    t_loop = time.perf_counter()
    for round_no in range(1, settings.max_rounds + 1):
        log.info("===== ROUND %d =====", round_no)
        plan = run_planner(
            settings,
            spec.question,
            index,
            run_dir / STEP_DIRS[3],
            run_log,
            ledger,
            round_no=round_no,
            gaps=gaps,
        )
        final = run_orchestrator(
            settings,
            spec.question,
            spec.pdf_path,
            index,
            plan,
            run_dir / STEP_DIRS[1],
            run_dir / STEP_DIRS[4],
            run_log,
            ledger,
            round_no=round_no,
            prior_feedback=feedback,
        )
        tool_calls_total += final["tool_calls_used"]
        verification = run_verifier(
            settings,
            spec.question,
            final,
            run_dir / STEP_DIRS[1],
            index,
            run_dir / STEP_DIRS[5],
            run_log,
            ledger,
            round_no=round_no,
            max_rounds=settings.max_rounds,
        )
        if verification["decision"] != "REPLAN":
            break
        gaps = verification["gaps"]
        feedback = (
            verification["llm_verdict"].get("explanation", "")
            + " Gaps: "
            + "; ".join(gaps)
        )
    timings["03-05_agent_loop"] = round(time.perf_counter() - t_loop, 1)
    timings["tool_calls_total"] = tool_calls_total
    timings["rounds"] = final["round"]

    # ---- 06 eval
    cost = ledger.summary()
    cost["index_build_share"] = cost_after_index
    result = run_eval(
        {"run_id": spec.run_id, "question": spec.question},
        gold,
        verification,
        final,
        run_dir / STEP_DIRS[6],
        run_log,
        cost,
        timings,
    )
    write_json(run_dir / "cost.json", cost)
    (run_dir / "summary.md").write_text(
        _summary_md(
            spec,
            index,
            final,
            verification,
            result,
            cost,
            timings,
            run_dir,
            tool_calls_total,
        ),
        encoding="utf-8",
    )
    log.info(
        "DONE %s -> answer=%r score=%.2f cost=$%.4f",
        spec.run_id,
        result["raw_system_answer"],
        result["score_official_rule"],
        cost["cost_usd"],
    )
    return result
