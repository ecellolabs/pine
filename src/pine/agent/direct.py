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
import re
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


def split_reasoning(text: str) -> tuple[str, str]:
    """``(reasoning, final answer)`` of the answer call's text.

    OpenRouter returns the thinking separately, so the text is the answer. A vLLM server
    without a reasoning parser (the cluster's ``serve_vllm.sh``) returns it inline: the
    thinking ends at ``</think>`` or, with the Qwen3.5 template, at a gap of three newlines
    (``baseline.clean_reply``). Text with neither marker is taken as the answer."""
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text)
    for marker in ("</think>", "\n\n\n"):
        if marker in text:
            reasoning, answer = text.rsplit(marker, 1)
            return reasoning.replace("<think>", "", 1).strip(), answer.strip()
    return "", text.strip()


def answer_prompt(question: str, hint: str | None = None) -> str:
    """The text after the page images. ``hint`` is only used to build STaR
    rationalizations (``pine.training.star``); inference never sends one."""
    hint_line = f"{hint}\n\n" if hint else ""
    return f"{INSTRUCTION}\n\n{hint_line}Question: {question}"


def prompt_parts(
    pdf_path: Path, pages: list[int], question: str, hint: str | None = None
) -> list[str | bytes]:
    """``[Page p]`` labels, JPEG pages and the instruction, in the order they are sent.
    Shared with the STaR training data so training and inference see the same prompt."""
    parts: list[str | bytes] = []
    for p in pages:
        parts += [f"[Page {p}]", render_page(pdf_path, p, max_side=PAGE_MAX_SIDE)]
    parts.append(answer_prompt(question, hint))
    return parts


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
    parts = prompt_parts(pdf_path, pages, question)
    content: list[Any] = []
    for p, jpeg in zip(pages, parts[1:-1:2]):
        assert isinstance(jpeg, bytes)
        (img_dir / f"round1_page{p:03d}_01.jpg").write_bytes(jpeg)
    for part in parts:
        content.append(
            BinaryContent(data=part, media_type="image/jpeg")
            if isinstance(part, bytes)
            else part
        )
    user = str(parts[-1])
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
    reasoning = ""
    try:
        # the cluster's vLLM returns the thinking inline; keep it out of the scored answer
        reasoning, answer = split_reasoning(asyncio.run(agent.run(content)).output)
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
        "reasoning": reasoning,
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
