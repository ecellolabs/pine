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
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pine.agent.common import (
    STEP_DIRS,
    AgentSettings,
    CostLedger,
    read_json,
    redacted_argv,
    render_page,
    setup_logger,
    write_json,
)
from pine.agent.index import run_index
from pine.agent.judge import make_judge_chat, score_answer
from pine.agent.lean_index import run_lean_index
from pine.agent.orchestrator import run_orchestrator
from pine.agent.parser import run_parser
from pine.agent.planner import run_planner
from pine.agent.scoring import run_eval
from pine.agent.verifier import run_verifier


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
            f"| Score ({result['score_source']}) | {result['score']:.2f} "
            f"({'CORRECT' if result['correct'] else 'WRONG'}) |"
        ),
        f"| Rule-based score (audit) | {result['score_official_rule']:.2f} |",
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
        f"- Quote checks: {[(q['page'], q['source'], q['similarity']) for q in verification['quote_checks']]}",
        f"- Code checks: numeric={verification['numeric_check']['status']} vision={[(c['page'], c['status']) for c in verification.get('vision_checks', [])]} check_pages={verification.get('check_pages', [])}"
        + (
            f" (kept round {verification['round']} after round {verification['fallback_from_round']})"
            if verification.get("fallback_from_round")
            else ""
        ),
        f"- Gaps: {verification['gaps']}",
        "",
    ]
    return "\n".join(lines)


def _restore_shared(shared_dir: Path, run_dir: Path) -> bool:
    """Copy a document's already built parser and index folders into a run folder."""
    if not (shared_dir / ".complete").exists():
        return False
    for name in (STEP_DIRS[1], STEP_DIRS[2]):
        shutil.copytree(shared_dir / name, run_dir / name, dirs_exist_ok=True)
    return True


def _publish_shared(shared_dir: Path, run_dir: Path) -> None:
    """Keep this run's parser and index folders so the document's other questions reuse them."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    for name in (STEP_DIRS[1], STEP_DIRS[2]):
        shutil.copytree(run_dir / name, shared_dir / name, dirs_exist_ok=True)
    (shared_dir / ".complete").touch()


def judge_final_answer(
    settings: AgentSettings,
    question: str,
    gold: dict[str, Any],
    verification: dict[str, Any],
    log: Any,
) -> dict[str, Any] | None:
    """Step 6 judging: the LLM scores the final answer, or None when no judge is configured."""
    if not settings.judge_model:
        return None
    claude = settings.judge_model.startswith("claude-")
    chat = settings.judge_chat or make_judge_chat(
        settings.judge_api_url or settings.api_url,
        # a Claude judge must not receive the agent's key; empty reads ANTHROPIC_API_KEY
        settings.judge_api_key or ("" if claude else settings.api_key),
        settings.judge_model,
        settings.request_timeout_s,
        settings.judge_disable_thinking,
        settings.judge_max_tokens,
        settings.judge_extra_body,
    )
    judged = score_answer(
        question,
        str(gold["answer"]),
        gold["answer_format"],
        verification["final_answer"],
        chat,
        settings.judge_mode,
    )
    if judged["error"]:
        log.error("LLM judge failed, using the rule-based score: %s", judged["error"])
    return judged


def run_sample(
    spec: RunSpec, settings: AgentSettings, shared_dir: Path | None = None
) -> dict[str, Any]:
    """Run the whole pipeline for one sample and return the evaluation result.

    ``shared_dir`` is a per-document folder: the parser and index of a document do not
    depend on the question, so the first question builds them and the others copy them."""
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
            try:
                out.write_bytes(render_page(spec.pdf_path, p, max_side=1024))
            except Exception as exc:  # noqa: BLE001 - annotation may cite a page the PDF lacks
                logging.getLogger("run").warning(
                    "gold evidence page %s cannot be rendered: %s", p, exc
                )
    log.info("DRIVER %s | output=%s", redacted_argv() or "<library call>", run_dir)
    log.info(
        "RUN %s | doc=%s | Q=%r | gold=%r (%s) pages=%s",
        spec.run_id,
        spec.doc_id,
        spec.question,
        gold["answer"],
        gold["answer_format"],
        gold["evidence_pages"],
    )

    reused = shared_dir is not None and _restore_shared(shared_dir, run_dir)
    if reused:
        log.info("parser and index reused from %s", shared_dir)

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
    if settings.index_mode == "lean":
        index = run_lean_index(
            settings,
            spec.pdf_path,
            run_dir / STEP_DIRS[1],
            run_dir / STEP_DIRS[2],
            run_log,
            ledger,
        )
    else:
        index = run_index(
            settings,
            spec.pdf_path,
            run_dir / STEP_DIRS[1],
            run_dir / STEP_DIRS[2],
            run_log,
            ledger,
        )
    timings["02_index"] = round(time.perf_counter() - t, 1)
    if shared_dir is not None and not reused:
        _publish_shared(shared_dir, run_dir)
    cost_after_index = ledger.summary()

    # ---- 03-05 loop
    gaps: list[str] | None = None
    feedback: str | None = None
    prior: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    force_pages: list[int] = []
    # an answer accepted before a coverage round, kept in case that round does worse
    fallback: tuple[dict[str, Any], dict[str, Any]] | None = None
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
            parser_dir=run_dir / STEP_DIRS[1],
            prior=prior,
            force_pages=force_pages,
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
            pdf_path=spec.pdf_path,
            previous=verification or None,
        )
        if verification["decision"] != "REPLAN":
            break
        if final["status"] == "answered" and verification.get(
            "decision_before_coverage"
        ) in ("ACCEPT", "ACCEPT_WEAK_GROUNDING"):
            fallback = (final, verification)
        # the pages the verifier asked for, then this round's evidence pages, so evidence
        # spread over pages stays together in the next round
        force_pages = list(
            dict.fromkeys(
                verification.get("check_pages", [])
                + verification.get("cited_pages", [])
            )
        )
        gaps = verification["gaps"]
        prior = {
            "candidate_sections": plan.get("candidate_sections"),
            "anchor_pages": plan.get("anchor_pages"),
            "pages_read": final.get("pages_read"),
            "tool_calls_used": final.get("tool_calls_used"),
            "status": final.get("status"),
        }
        feedback = (
            verification["llm_verdict"].get("explanation", "")
            + " Gaps: "
            + "; ".join(gaps)
        )
    accepted = verification.get("decision") in ("ACCEPT", "ACCEPT_WEAK_GROUNDING")
    # a later round that reached the same answer and had it rejected is evidence against it
    rejected_again = (
        fallback is not None
        and final["status"] == "answered"
        and not accepted
        and final["answer"].strip().lower() == fallback[0]["answer"].strip().lower()
    )
    if fallback is not None and rejected_again:
        log.info(
            "the same answer was rejected in a later round: not falling back to it"
        )
    if (
        fallback is not None
        and not rejected_again
        and not (accepted and final["status"] == "answered")
    ):
        # the coverage round found nothing better: keep the answer it was meant to check
        final, kept = fallback
        verification = {
            **kept,
            "decision": kept["decision_before_coverage"],
            "final_answer": kept["answer_in"],
            "fallback_from_round": verification.get("round"),
        }
        log.info(
            "round %s did not improve on round %s: keeping %r",
            verification["fallback_from_round"],
            kept["round"],
            kept["answer_in"],
        )
    timings["03-05_agent_loop"] = round(time.perf_counter() - t_loop, 1)
    timings["tool_calls_total"] = tool_calls_total
    timings["rounds"] = verification.get("fallback_from_round") or final["round"]

    # ---- 06 eval
    cost = ledger.summary()
    cost["index_build_share"] = cost_after_index
    judged = judge_final_answer(settings, spec.question, gold, verification, log)
    result = run_eval(
        {"run_id": spec.run_id, "question": spec.question},
        gold,
        verification,
        final,
        run_dir / STEP_DIRS[6],
        run_log,
        cost,
        timings,
        judged=judged,
        judge_mode=settings.judge_mode,
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
        "DONE %s -> answer=%r score=%.2f (%s) cost=$%.4f",
        spec.run_id,
        result["raw_system_answer"],
        result["score"],
        result["score_source"],
        cost["cost_usd"],
    )
    return result
