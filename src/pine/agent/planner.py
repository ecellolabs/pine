"""Step 3 - Planner (M3): a Pydantic AI agent that decomposes the question into
sub-goals / facet queries (typed ``Plan``); re-plans from verifier gaps.

Writes 03_planner/plan_round{N}.json, llm_calls.jsonl, planner.log
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings

from pine.agent.common import AgentSettings, CostLedger, setup_logger, write_json
from pine.agent.llm import traced_model
from pine.agent.schemas import Plan

PLAN_INSTRUCTIONS = """You are the PLANNER of a document question-answering agent. The agent can navigate a {n_pages}-page document
titled "{doc_title}" using a table of contents, keyword search over page text, reading page text, and inspecting page images.

Document outline (section titles with page ranges):
{outline}

Produce a search strategy: the question type, the expected answer format (Int | Float | Str | List | None),
whether visual inspection of charts/figures/images is needed, ordered concrete sub-goals,
3-6 short keyword search queries, and the outline section ids most likely to hold the evidence."""


def outline_lines(index: dict[str, Any]) -> str:
    lines: list[str] = []

    def walk(nodes: list[dict[str, Any]], depth: int) -> None:
        for s in nodes:
            lines.append(
                f"{'  ' * depth}[{s['id']}] {s['title']} (pages {s['start_page']}-{s['end_page']})"
            )
            if s.get("subsections"):
                walk(s["subsections"], depth + 1)

    walk(index["sections"], 0)
    return "\n".join(lines)


def run_planner(
    settings: AgentSettings,
    question: str,
    index: dict[str, Any],
    step_dir: Path,
    run_log: Path,
    ledger: CostLedger,
    round_no: int = 1,
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    log = setup_logger("step3.planner", step_dir / "planner.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="03_planner",
    )
    model.purpose = f"plan_round{round_no}"
    agent: Agent[None, Plan] = Agent(
        model,
        output_type=ToolOutput(Plan, name="Plan"),
        instructions=PLAN_INSTRUCTIONS.format(
            n_pages=index["n_pages"],
            doc_title=index["doc_title"],
            outline=outline_lines(index),
        ),
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=600),
    )
    prompt = f"Question: {question}"
    if gaps:
        prompt += (
            "\n\nThe previous round FAILED verification. Open gaps reported by the verifier:\n- "
            + "\n- ".join(gaps)
            + "\nRe-plan to close these gaps (different sections / queries / visual inspection)."
        )
    plan: dict[str, Any]
    try:
        plan = agent.run_sync(prompt).output.model_dump()
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
            "_error": str(exc),
        }
    plan["round"] = round_no
    plan["gaps_in"] = gaps or []
    write_json(step_dir / f"plan_round{round_no}.json", plan)
    log.info(
        "plan round %d: type=%s format=%s visual=%s sub_goals=%s sections=%s",
        round_no,
        plan.get("question_type"),
        plan.get("expected_answer_format"),
        plan.get("needs_visual_inspection"),
        plan.get("sub_goals"),
        plan.get("candidate_sections"),
    )
    return plan
