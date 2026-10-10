"""Step 3 - Planner (M3): a Pydantic AI agent that turns the question into a
search strategy (typed ``Plan``) and re-plans from verifier gaps.

Candidate regions are **not inferred from section titles**.  The agent has one
tool, ``locate_regions(queries)``, backed by ``locate.RegionLocator``: it resolves
"Figure N" / "Table N" references to the page that carries the caption and ranks
pages and sections lexically (BM25) for free-text queries.  The planner is told
to call it first and to take ``candidate_sections`` / ``anchor_pages`` from its
output.  As a guarantee the same locator is run in code after the model answers
and merged into the plan (reference sections and pages always come first), so
the navigator receives accurate regions even if the model ignored the tool.

Writes 03_planner/plan_round{N}.json, locate_round{N}.json, llm_calls.jsonl,
planner.log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from pine.agent.common import AgentSettings, CostLedger, setup_logger, write_json
from pine.agent.llm import traced_model
from pine.agent.locate import RegionLocator, enriched_outline
from pine.agent.schemas import Plan

PLAN_INSTRUCTIONS = """You are the PLANNER of a document question-answering agent. The agent can navigate a {n_pages}-page document
titled "{doc_title}" using a table of contents, keyword search over page text, reading page text, and inspecting page images.

Document outline (section id, title, page range, and the figures/tables whose captions are in that section):
{outline}

Procedure:
1. FIRST call locate_regions with 3-6 short keyword queries. Include the exact figure/table reference if the question names one (e.g. "Figure 1"), plus the key terms of the question.
   The tool resolves figure/table references to the page that carries the caption and ranks pages and sections by lexical match. It is the source of truth for where things are.
2. THEN produce the Plan: the question type, the expected answer format (Int | Float | Str | List | None),
   whether visual inspection of charts/figures/images is needed, ordered concrete sub-goals, the search queries you used,
   candidate_sections = the section ids returned by locate_regions (best first), and anchor_pages = its "pages to open first".
Do not guess sections from titles; use the tool's result."""


def outline_lines(index: dict[str, Any]) -> str:
    """Outline shown to the planner and the navigator (sections + their figures/tables)."""
    return enriched_outline(index)


@dataclass
class PlanDeps:
    question: str
    locator: RegionLocator
    log: logging.Logger
    calls: list[dict[str, Any]] = field(default_factory=list)


def build_planner(model: Any, index: dict[str, Any]) -> Agent[PlanDeps, Plan]:
    agent: Agent[PlanDeps, Plan] = Agent(
        model,
        deps_type=PlanDeps,
        output_type=ToolOutput(Plan, name="Plan"),
        instructions=PLAN_INSTRUCTIONS.format(
            n_pages=index["n_pages"],
            doc_title=index["doc_title"],
            outline=outline_lines(index),
        ),
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=700),
    )

    @agent.tool
    def locate_regions(ctx: RunContext[PlanDeps], queries: list[str]) -> str:
        """Find where the evidence for the question should be. Resolves "Figure N" /
        "Table N" references to the page whose caption carries them and ranks pages and
        sections by keyword match for the given queries. Returns the best sections
        (ids, best first) and the pages to open first.

        Args:
            queries: 3-6 short keyword queries (include exact figure/table references).
        """
        deps = ctx.deps
        result = deps.locator.locate(deps.question, queries)
        deps.calls.append(result)
        deps.log.info(
            "locate_regions(%s) -> sections=%s anchor_pages=%s refs=%s",
            queries,
            result["sections"],
            result["anchor_pages"],
            [(r["ref"], r["pages"], r["via"]) for r in result["refs"]],
        )
        return result["explanation"]

    return agent


def _merge_locator(plan: dict[str, Any], loc: dict[str, Any]) -> dict[str, Any]:
    """Guarantee: resolved reference sections/pages first, then the model's choice,
    then the locator's lexical candidates."""
    ref_secs = loc["ref_sections"]
    model_secs = [s for s in plan.get("candidate_sections") or [] if isinstance(s, str)]
    plan["candidate_sections"] = list(
        dict.fromkeys(ref_secs + model_secs + loc["sections"])
    )[: max(3, len(ref_secs) + len(model_secs))]
    ref_pages = [p for r in loc["refs"] for p in r["pages"]]
    model_pages = [p for p in plan.get("anchor_pages") or [] if isinstance(p, int)]
    plan["anchor_pages"] = list(
        dict.fromkeys(ref_pages + model_pages + loc["anchor_pages"])
    )[: max(4, len(ref_pages))]
    plan["locator"] = {
        "refs": loc["refs"],
        "ref_sections": ref_secs,
        "sections": loc["sections"],
        "anchor_pages": loc["anchor_pages"],
        "ranked_pages": loc["ranked_pages"],
        "ranked_sections": loc["ranked_sections"],
    }
    return plan


def run_planner(
    settings: AgentSettings,
    question: str,
    index: dict[str, Any],
    step_dir: Path,
    run_log: Path,
    ledger: CostLedger,
    round_no: int = 1,
    gaps: list[str] | None = None,
    parser_dir: Path | None = None,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``parser_dir`` gives the locator the page Markdown (full-text BM25); ``prior``
    is the previous round's outcome (candidate_sections, anchor_pages, pages_read,
    tool_calls_used, status) so a re-plan can change course."""
    log = setup_logger("step3.planner", step_dir / "planner.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    pages_md: dict[int, str] = {}
    if parser_dir is not None:
        for p in range(1, int(index.get("n_pages") or 0) + 1):
            f = parser_dir / "pages" / f"page_{p:03d}.md"
            if f.exists():
                pages_md[p] = f.read_text(encoding="utf-8")
    locator = RegionLocator(index, pages_md)
    model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="03_planner",
    )
    model.purpose = f"plan_round{round_no}"
    model.round_no = round_no
    agent = build_planner(model, index)
    deps = PlanDeps(question=question, locator=locator, log=log)

    prompt = f"Question: {question}"
    if gaps:
        prompt += (
            "\n\nThe previous round FAILED verification. Open gaps reported by the verifier:\n- "
            + "\n- ".join(gaps)
        )
    if prior:
        prompt += (
            f"\n\nPrevious round: candidate_sections={prior.get('candidate_sections')} "
            f"anchor_pages={prior.get('anchor_pages')} pages actually read="
            f"{prior.get('pages_read')} tool calls={prior.get('tool_calls_used')} "
            f"outcome={prior.get('status')!r}."
        )
        if gaps:
            prompt += (
                " Re-plan to close the gaps: if the previous pages were never opened, "
                "keep them as anchor pages; otherwise choose different sections / queries "
                "/ visual inspection."
            )
    plan: dict[str, Any]
    try:
        plan = agent.run_sync(
            prompt,
            deps=deps,
            usage_limits=UsageLimits(request_limit=4),
        ).output.model_dump()
        plan["_error"] = None
    except Exception as exc:  # noqa: BLE001
        log.error("planner failed (PLANNER FAILURE): %s -> using default plan", exc)
        plan = {
            "question_type": "unknown",
            "expected_answer_format": "Str",
            "needs_visual_inspection": True,
            "sub_goals": [question],
            "search_queries": [question],
            "candidate_sections": [],
            "anchor_pages": [],
            "_error": str(exc),
        }
    # code-side guarantee (also covers a model that never called the tool)
    loc = locator.locate(question, list(plan.get("search_queries") or []))
    plan = _merge_locator(plan, loc)
    plan["locator"]["tool_called"] = bool(deps.calls)
    plan["round"] = round_no
    plan["gaps_in"] = gaps or []
    write_json(step_dir / f"plan_round{round_no}.json", plan)
    write_json(
        step_dir / f"locate_round{round_no}.json",
        {"tool_calls": deps.calls, "final": loc},
    )
    log.info(
        "plan round %d: type=%s format=%s visual=%s sub_goals=%s sections=%s anchor_pages=%s (locator tool called: %s)",
        round_no,
        plan.get("question_type"),
        plan.get("expected_answer_format"),
        plan.get("needs_visual_inspection"),
        plan.get("sub_goals"),
        plan.get("candidate_sections"),
        plan.get("anchor_pages"),
        bool(deps.calls),
    )
    return plan
