"""Step 4 (``--navigator direct``): one reasoning call over the planner's candidate pages.

Replaces the navigator and the verifier. The planner's pages (resolved figure/table
references, its anchor pages, then the locator's BM25 ranking) are topped up to
``answer_pages`` pages and sent as page images, in page order, to one model call with
thinking on. The prompt is the B1 baseline's (``baseline.INSTRUCTION``), so this run and
``B1_mmlongbench_baseline.py`` differ only in which pages the model sees.

Writes into ``04_orchestrator/`` so the visual page shows it: ``trace_round1.jsonl``,
``final_round1.json``, ``images_viewed/`` (the pages sent), ``llm_calls.jsonl``,
``orchestrator.log``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from pine.agent.baseline import INSTRUCTION
from pine.agent.common import (
    AgentSettings,
    CostLedger,
    append_jsonl,
    render_page,
    setup_logger,
    write_json,
)
from pine.agent.llm import traced_model
from pine.agent.locate import RegionLocator

ANSWER_MAX_TOKENS = 12000  # the B1 baseline's thinking budget
PAGE_MAX_SIDE = 896  # the B1 baseline's image size


def candidate_pages(
    plan: dict[str, Any], locator: RegionLocator, question: str, n_pages: int, k: int
) -> list[int]:
    """``k`` pages in page order: the plan's anchor pages, then the locator's ranking,
    then BM25 over all pages for the question and the plan's queries, then early pages."""
    queries = [question, *(plan.get("search_queries") or [])]
    ranked = [r["page"] for r in (plan.get("locator") or {}).get("ranked_pages", [])]
    for q in queries:
        ranked += [p for p, _ in locator.rank_pages(q, k=k)]
    picked = list(
        dict.fromkeys(
            p
            for p in [
                *(plan.get("anchor_pages") or []),
                *ranked,
                *range(1, n_pages + 1),
            ]
            if isinstance(p, int) and 1 <= p <= n_pages
        )
    )
    return sorted(picked[:k])


def run_direct(
    settings: AgentSettings,
    question: str,
    pdf_path: Path,
    index: dict[str, Any],
    plan: dict[str, Any],
    parser_dir: Path,
    step_dir: Path,
    run_log: Path,
    ledger: CostLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Answer from the candidate pages; returns ``(final, verification)`` in the shapes
    ``scoring.run_eval`` reads (no verifier runs: the decision is ``DIRECT``)."""
    log = setup_logger("step4.direct", step_dir / "orchestrator.log", run_log)
    img_dir = step_dir / "images_viewed"
    img_dir.mkdir(parents=True, exist_ok=True)
    n_pages = int(index["n_pages"])
    pages_md = {
        p: f.read_text(encoding="utf-8")
        for p in range(1, n_pages + 1)
        if (f := parser_dir / "pages" / f"page_{p:03d}.md").exists()
    }
    pages = candidate_pages(
        plan, RegionLocator(index, pages_md), question, n_pages, settings.answer_pages
    )
    content: list[Any] = []
    for p in pages:
        jpeg = render_page(pdf_path, p, max_side=PAGE_MAX_SIDE)
        (img_dir / f"round1_page{p:03d}_01.jpg").write_bytes(jpeg)
        content += [f"[Page {p}]", BinaryContent(data=jpeg, media_type="image/jpeg")]
    user = f"{INSTRUCTION}\n\nQuestion: {question}"
    content.append(user)
    log.info(
        "answering from %d pages %s with %s", len(pages), pages, settings.answer_model
    )
    append_jsonl(
        step_dir / "trace_round1.jsonl",
        {"event": "start", "round": 1, "system": "", "user": user, "pages_sent": pages},
    )

    # the answerer always thinks, whatever --no-reasoning says for the planner and index
    thinking = dataclasses.replace(settings, reasoning=True, thinking_sampling=True)
    model = traced_model(
        thinking,
        settings.answer_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="04_orchestrator",
    )
    model.purpose = "direct_answer"
    model.round_no = 1
    agent: Agent[None, str] = Agent(
        model,
        # OpenRouter leaves thinking to the provider's default unless it is asked for;
        # other servers ignore the key (vLLM's Qwen3.5 template thinks by default)
        model_settings=OpenRouterModelSettings(
            temperature=0.0,
            max_tokens=ANSWER_MAX_TOKENS,
            openrouter_reasoning={"enabled": True},
        ),
    )
    error = None
    try:
        answer = asyncio.run(agent.run(content)).output.strip()
    except Exception as exc:  # noqa: BLE001 - a failed call is scored as no answer
        log.error("direct answer call failed: %s", exc)
        answer, error = "", str(exc)
    answer = answer or "Not answerable"
    status = (
        "not_answerable" if answer.lower().startswith("not answerable") else "answered"
    )
    final = {
        "status": status,
        "answer": answer,
        "evidence_pages": [],
        "reasoning": "",
        "salvaged": error is not None,
        "round": 1,
        "tool_calls_used": 0,
        "pages_read": pages,
        "image_inspections": [],
        "anomalies": [{"type": "call_failed", "detail": error}] if error else [],
    }
    write_json(step_dir / "final_round1.json", final)
    append_jsonl(step_dir / "trace_round1.jsonl", {"event": "final", **final})
    log.info("direct answer: %r (status %s)", answer, status)
    verification = {
        "decision": "DIRECT",
        "final_answer": answer,
        "cited_pages": [],
        "llm_verdict": {"verdict": None, "explanation": "no verifier (direct mode)"},
        "quote_checks": [],
        "gaps": [],
    }
    return final, verification
