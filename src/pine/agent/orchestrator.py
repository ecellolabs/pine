"""Step 4 - Agentic orchestrator (M4): a Pydantic AI agent navigating the index
with tools.

Tools (``@agent.tool``): get_outline, search_pages (BM25 over page markdown +
summaries), read_page, inspect_page_image (vision model), record_evidence
(evidence ledger, M5; rejects quotations that are not verbatim on the page).
The final answer is a typed ``FinalAnswer`` emitted through the ``final_answer``
output tool.  The run is driven with ``agent.iter`` so that every model turn
and tool result is written to the trace as it happens, the tool-call budget is
enforced (``UsageLimits``), and anomalies are recorded.

Writes:
  04_orchestrator/trace_round{N}.jsonl      every assistant turn and tool result
  04_orchestrator/tool_results/              full text returned by each tool call
  04_orchestrator/images_viewed/             the exact JPEGs sent to the vision model
  04_orchestrator/ledger_round{N}.json       evidence ledger entries recorded by the agent
  04_orchestrator/anomalies_round{N}.json    tool-calling failures (no tool call, bad args,
                                             rejected quote, repeated call, budget)
  04_orchestrator/llm_calls.jsonl, orchestrator.log
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from pine.agent.common import (
    AgentSettings,
    CostLedger,
    append_jsonl,
    render_page,
    setup_logger,
    write_json,
)
from pine.agent.llm import TracedModel, response_to_chat, traced_model
from pine.agent.locate import BM25, build_page_corpus, enriched_outline
from pine.agent.locate import tokens as _tokens
from pine.agent.schemas import FinalAnswer
from pine.agent.verifier import quote_in_text

DEFAULT_MAX_TOOL_CALLS = 12
READ_PAGE_MAX_CHARS = 5000
TOOL_NAMES = [
    "get_outline",
    "search_pages",
    "read_page",
    "inspect_page_image",
    "record_evidence",
    "final_answer",
]

SYSTEM_PROMPT = """You are the ORCHESTRATOR of a grounded document question-answering agent. You answer ONE question about a long document by navigating a hierarchical index with tools. You never answer from memory; every answer must be supported by a page of THIS document.

Document: "{doc_title}" ({n_pages} pages). Sections (id, title, pages | figures/tables whose captions are in the section):
{outline}

Plan from the planner (candidate_sections and anchor_pages were resolved from the index: a "Figure N" reference points at the page carrying that caption):
{plan}

Procedure:
1. Open the plan's anchor_pages FIRST in your first turn: call read_page (you can pass multiple pages at once like read_page([1, 2])) for text, inspect_page_image when the evidence is a figure/chart/table image.
2. If evidence is not in anchor pages, use search_pages / read_page to locate further candidate pages.
3. Call final_answer with status, concise answer, evidence_pages, and quote (copied verbatim from page text or image inspection output). Providing quote and quote_page in final_answer automatically records evidence in a single turn.
Budget: at most {budget} tool calls. Never repeat a tool call with identical arguments. Always respond by calling a tool."""

VISION_INSTRUCTIONS = (
    "You are a careful visual document reader. Answer ONLY from what is visible on "
    "this page image. Quote any text/numbers you rely on exactly. If the requested "
    "information is not on the page, say so explicitly."
)

FORCED_FINAL_INSTRUCTIONS = (
    "The tool budget of the navigation agent is exhausted. Using ONLY the evidence "
    "ledger and page notes below, give the best grounded final answer, or status "
    "not_answerable if nothing supports an answer."
)


# BM25 / tokens: see pine.agent.locate (shared with the planner's locator)


# deps
@dataclass
class NavDeps:
    """Everything the tools need; also collects the ledger, inspections and trace."""

    pdf_path: Path
    index: dict[str, Any]
    outline_md: str
    pages_md: dict[int, str]
    bm25: BM25
    step_dir: Path
    round_no: int
    log: logging.Logger
    vision_model: TracedModel
    trace_path: Path
    calls: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)
    image_inspections: list[dict[str, Any]] = field(default_factory=list)
    pages_read: list[int] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    seen_calls: Counter[str] = field(default_factory=Counter)
    plan: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pages(self) -> int:
        return int(self.index["n_pages"])

    def record(self, name: str, args: dict[str, Any], result: str) -> str:
        """Log a tool call (full result file, log line, trace event) and return the
        text to hand back to the model; repeated identical calls get a notice so
        the model stops looping."""
        self.calls += 1
        sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        self.seen_calls[sig] += 1
        if self.seen_calls[sig] > 1:
            result = (
                f"NOTE: you already called {name} with these exact arguments "
                f"({self.seen_calls[sig] - 1}x); the result is unchanged. Use it: call "
                "record_evidence / final_answer, or try different arguments.\n" + result
            )
            self.anomalies.append(
                {
                    "type": "REPEATED_CALL",
                    "turn": self.calls,
                    "tool": name,
                    "args": args,
                    "count": self.seen_calls[sig],
                }
            )
            self.log.warning("turn %d: REPEATED CALL %s", self.calls, sig[:120])
        results_dir = self.step_dir / "tool_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (
            results_dir / f"round{self.round_no}_call{self.calls:02d}_{name}.txt"
        ).write_text(
            f"ARGS: {json.dumps(args, ensure_ascii=False, default=str)}\n\n{result}",
            encoding="utf-8",
        )
        self.log.info(
            "turn %d: %s(%s) -> %s",
            self.calls,
            name,
            json.dumps(args, ensure_ascii=False, default=str)[:150],
            result[:160].replace("\n", " "),
        )
        append_jsonl(
            self.trace_path,
            {
                "event": "tool_result",
                "turn": self.calls,
                "tool": name,
                "args": args,
                "result": result
                if len(result) < 4000
                else result[:4000] + "...[see tool_results/]",
            },
        )
        return result


def _page_ok(deps: NavDeps, page: int) -> int:
    if not 1 <= page <= deps.n_pages:
        raise ModelRetry(f"page {page} is out of range 1..{deps.n_pages}")
    return page


# agent
def build_navigator(text_model: TracedModel) -> Agent[NavDeps, FinalAnswer]:
    """Create the navigation agent with its tools and the final_answer output tool."""
    agent: Agent[NavDeps, FinalAnswer] = Agent(
        text_model,
        deps_type=NavDeps,
        output_type=ToolOutput(
            FinalAnswer,
            name="final_answer",
            description=(
                "Finish. status='answered' with a concise answer and the evidence "
                "pages, or status='not_answerable' when the document does not contain "
                "the information after a thorough search."
            ),
        ),
        retries=3,
        model_settings=ModelSettings(temperature=0.0, max_tokens=600),
    )

    @agent.output_validator
    def effort_gate(ctx: RunContext[NavDeps], out: FinalAnswer) -> FinalAnswer:
        """An *answered* final needs recorded evidence. Abstentions are never
        bounced without cause, but must follow 'look before you abstain': zero-call
        abstentions and uninspected figure abstentions are bounced so the agent
        actually inspects candidate pages."""
        deps = ctx.deps
        if out.status == "not_answerable":
            if deps.calls == 0:
                anchors = deps.plan.get("anchor_pages", [])
                anchor_msg = (
                    f" (such as page(s) {', '.join(str(p) for p in anchors)})"
                    if anchors
                    else ""
                )
                raise ModelRetry(
                    f"You cannot abstain without inspecting any pages first. Call read_page or "
                    f"inspect_page_image on candidate pages{anchor_msg} before calling final_answer."
                )
            needs_vis = deps.plan.get("needs_visual_inspection", False)
            loc_refs = deps.plan.get("locator", {}).get("refs", [])
            fig_pages = sorted(
                set(
                    [p for r in loc_refs for p in r.get("pages", [])]
                    + (deps.plan.get("anchor_pages", []) if needs_vis else [])
                )
            )
            if needs_vis and fig_pages:
                inspected_pages = {i["page"] for i in deps.image_inspections}
                uninspected = [p for p in fig_pages if p not in inspected_pages]
                if uninspected:
                    raise ModelRetry(
                        f"This question requires visual inspection of figures/tables on page(s) "
                        f"{', '.join(str(p) for p in uninspected)}. Call inspect_page_image on "
                        f"these page(s) before calling final_answer."
                    )

        if out.status == "answered" and not deps.ledger:
            if out.quote:
                qp = out.quote_page or (
                    out.evidence_pages[0] if out.evidence_pages else None
                )
                if qp is not None:
                    p = _page_ok(deps, qp)
                    page_text = deps.pages_md.get(p, "")
                    in_text = quote_in_text(out.quote, page_text)
                    in_img = any(
                        i["page"] == p and quote_in_text(out.quote, i["result"])
                        for i in deps.image_inspections
                    )
                    if in_text or in_img:
                        entry = {
                            "page": p,
                            "quote": out.quote,
                            "candidate_answer": out.answer,
                            "note": out.reasoning
                            or "Inline evidence from final_answer",
                            "round": deps.round_no,
                            "source": "page_text" if in_text else "image_inspection",
                        }
                        deps.ledger.append(entry)
                        deps.log.info(
                            "effort_gate: auto-recorded inline evidence for page %d", p
                        )
                        return out
                    raise ModelRetry(
                        f"The quote is not found verbatim on page {p}. Copy the exact text "
                        "from read_page or inspect_page_image output."
                    )
            raise ModelRetry(
                "Call record_evidence with the page and an exact quotation (or provide "
                "quote and quote_page directly in final_answer) before calling final_answer."
            )
        return out

    @agent.tool
    def get_outline(ctx: RunContext[NavDeps]) -> str:
        """Return the hierarchical table of contents of the document (section ids,
        titles, page ranges and one-line page titles)."""
        return ctx.deps.record("get_outline", {}, ctx.deps.outline_md)

    @agent.tool
    def search_pages(ctx: RunContext[NavDeps], query: str) -> str:
        """Keyword (BM25) search over the text of all pages. Returns the top matching
        pages with snippets. Use short keyword queries.

        Args:
            query: Short keyword query.
        """
        deps = ctx.deps
        hits = deps.bm25.search(query, k=5)
        if not hits:
            result = f"No pages match '{query}'."
        else:
            out = [f"Top pages for '{query}':"]
            qt = set(_tokens(query))
            for p, s in hits:
                snippet = ""
                for line in deps.pages_md[p].splitlines():
                    if qt & set(_tokens(line)):
                        snippet = line.strip()[:200]
                        break
                ps = deps.index["pages"][str(p)]
                vis_tag = ""
                figs = ps.get("figures") or []
                tabs = ps.get("tables") or []
                if figs or tabs or ps.get("has_visual_elements"):
                    v_details = []
                    if figs:
                        v_details.append(f"Figures: {', '.join(figs)}")
                    if tabs:
                        v_details.append(f"Tables: {', '.join(tabs)}")
                    if not v_details:
                        v_details.append("Has Visuals")
                    vis_tag = f" [{'; '.join(v_details)}]"

                out.append(
                    f"- page {p} (score {s:.1f}): {ps['title']}{vis_tag} | "
                    f"{snippet or ps['summary'][:160]}"
                )
            result = "\n".join(out)
        return deps.record("search_pages", {"query": query}, result)

    @agent.tool
    def read_page(ctx: RunContext[NavDeps], page: int | list[int]) -> str:
        """Return the full Markdown text of one page or multiple pages (1-indexed).

        Args:
            page: Page number or list of page numbers (e.g. 2 or [2, 3]).
        """
        deps = ctx.deps
        pages = page if isinstance(page, list) else [page]
        out_blocks: list[str] = []
        for p_raw in pages:
            p = _page_ok(deps, p_raw)
            deps.pages_read.append(p)
            md = deps.pages_md[p].strip()
            head = f"[page {p}] title: {deps.index['pages'][str(p)]['title']}\n"
            if len(md) < 40:
                res = head + (
                    "(This page has no machine-readable text. "
                    "Use inspect_page_image to look at it.)"
                )
            else:
                if len(md) > READ_PAGE_MAX_CHARS:
                    md = md[:READ_PAGE_MAX_CHARS] + "\n...[truncated]"
                res = head + md
            out_blocks.append(res)
        result = "\n\n---\n\n".join(out_blocks)
        return deps.record("read_page", {"page": page}, result)

    @agent.tool
    async def inspect_page_image(
        ctx: RunContext[NavDeps], page: int, question: str
    ) -> str:
        """Look at the rendered image of a page with a vision model and ask it a
        specific question (use for charts, figures, tables, images, or pages whose
        text is empty).

        Args:
            page: Page number, 1-indexed.
            question: What to look for on the page.
        """
        deps = ctx.deps
        p = _page_ok(deps, page)
        jpeg = render_page(deps.pdf_path, p, max_side=1024, quality=80)
        img_dir = deps.step_dir / "images_viewed"
        img_dir.mkdir(parents=True, exist_ok=True)
        seq = len(deps.image_inspections) + 1
        img_path = img_dir / f"round{deps.round_no}_page{p:03d}_{seq:02d}.jpg"
        img_path.write_bytes(jpeg)
        vision_agent: Agent[None, str] = Agent(
            deps.vision_model,
            instructions=VISION_INSTRUCTIONS,
            model_settings=ModelSettings(temperature=0.0, max_tokens=350),
        )
        deps.vision_model.purpose = f"inspect_page_{p}"
        try:
            reply = await vision_agent.run(
                [
                    BinaryContent(data=jpeg, media_type="image/jpeg"),
                    f"Page {p} of the document. {question}",
                ]
            )
            text = reply.output.strip() or "(vision model returned empty content)"
        except Exception as exc:  # noqa: BLE001
            text = f"ERROR: vision model call failed: {exc}"
        deps.image_inspections.append(
            {"page": p, "question": question, "image": img_path.name, "result": text}
        )
        result = f"[vision model on page {p} image] {text}"
        return deps.record(
            "inspect_page_image", {"page": p, "question": question}, result
        )

    @agent.tool
    def record_evidence(
        ctx: RunContext[NavDeps],
        page: int,
        quote: str,
        candidate_answer: str,
        note: str = "",
    ) -> str:
        """Record a candidate answer in the evidence ledger with the page number and
        an exact supporting quotation copied from the page text or from the image
        inspection result. Quotations that are not found on the page are rejected.

        Args:
            page: Page number, 1-indexed.
            quote: Verbatim quotation from the page text or the image inspection output.
            candidate_answer: The answer this evidence supports.
            note: Optional note.
        """
        deps = ctx.deps
        p = _page_ok(deps, page)
        in_text, sim_text = quote_in_text(quote, deps.pages_md.get(p, ""))
        inspections = "\n".join(
            i["result"] for i in deps.image_inspections if i["page"] == p
        )
        in_img, sim_img = (
            quote_in_text(quote, inspections) if inspections else (False, 0.0)
        )
        args = {
            "page": p,
            "quote": quote,
            "candidate_answer": candidate_answer,
            "note": note,
        }
        if not (in_text or in_img):
            deps.anomalies.append(
                {
                    "type": "QUOTE_REJECTED",
                    "turn": deps.calls + 1,
                    "page": p,
                    "quote": quote[:300],
                    "sim_page_text": sim_text,
                    "sim_image_inspection": sim_img,
                }
            )
            deps.record(
                "record_evidence",
                args,
                f"REJECTED: quote not found verbatim on page {p} "
                f"(similarity {sim_text:.2f}).",
            )
            raise ModelRetry(
                f"The quote is not found verbatim on page {p}. Copy the exact text from "
                "read_page or inspect_page_image output (a short exact phrase is enough)."
            )
        entry = {
            **args,
            "round": deps.round_no,
            "source": "page_text" if in_text else "image_inspection",
        }
        deps.ledger.append(entry)
        result = f"Recorded evidence #{len(deps.ledger)} (page {p}, source {entry['source']})."
        return deps.record("record_evidence", args, result)

    return agent


def _request_anomalies(deps: NavDeps, parts: list[Any]) -> None:
    """Inspect retry prompts Pydantic AI injected (bad args / plain text / rejected quote)."""
    for part in parts:
        if isinstance(part, RetryPromptPart):
            content = part.model_response()
            if part.tool_name == "record_evidence" and "not found verbatim" in content:
                continue  # already recorded as QUOTE_REJECTED by the tool
            if part.tool_name == "final_answer":
                kind = "FINAL_WITHOUT_EVIDENCE"
            elif part.tool_name:
                kind = "BAD_TOOL_ARGS"
            else:
                kind = "NO_TOOL_CALL"
            deps.anomalies.append(
                {
                    "type": kind,
                    "turn": deps.calls,
                    "tool": part.tool_name,
                    "content": content[:400],
                }
            )
            deps.log.warning("%s: %s", kind, content[:160].replace("\n", " "))


def _response_event(deps: NavDeps, turn: int, response: ModelResponse) -> None:
    chat = response_to_chat(response)
    append_jsonl(
        deps.trace_path,
        {
            "event": "assistant",
            "turn": turn,
            "content": chat.get("content", ""),
            "tool_calls": chat.get("tool_calls", []),
        },
    )
    has_call = any(isinstance(p, ToolCallPart) for p in response.parts)
    has_text = any(isinstance(p, TextPart) for p in response.parts)
    if not has_call and has_text:
        deps.log.warning("turn %d: model replied in prose instead of a tool call", turn)


async def _navigate(
    agent: Agent[NavDeps, FinalAnswer],
    deps: NavDeps,
    instructions: str,
    user: str,
    max_tool_calls: int,
) -> tuple[FinalAnswer | None, str | None]:
    """Drive the agent graph node by node; returns (final, failure_reason)."""
    limits = UsageLimits(
        request_limit=max_tool_calls + 6, tool_calls_limit=max_tool_calls
    )
    turn = 0
    final: FinalAnswer | None = None
    failure: str | None = None
    try:
        async with agent.iter(
            user, deps=deps, instructions=instructions, usage_limits=limits
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    _request_anomalies(deps, list(node.request.parts))
                elif Agent.is_call_tools_node(node):
                    turn += 1
                    _response_event(deps, turn, node.model_response)
                elif Agent.is_end_node(node):
                    final = node.data.output
    except UsageLimitExceeded as exc:
        failure = f"BUDGET_EXHAUSTED: {exc}"
    except UnexpectedModelBehavior as exc:
        failure = f"MODEL_BEHAVIOR: {exc}"
    return final, failure


async def _forced_final(
    text_model: TracedModel, deps: NavDeps, question: str, reason: str
) -> FinalAnswer:
    """Budget exhausted / retries exhausted: ask once for a grounded final answer.

    Must run inside the same event loop as the navigation (the provider's async
    HTTP client is bound to it), hence async."""
    notes = (
        "\n".join(
            f"- page {e['page']}: answer={e['candidate_answer']!r} quote={e['quote']!r}"
            for e in deps.ledger
        )
        or "(no evidence recorded)"
    )
    pages = ", ".join(str(p) for p in sorted(set(deps.pages_read))) or "none"
    inspections = (
        "\n".join(
            f"- page {i['page']}: {i['result'][:300]}" for i in deps.image_inspections
        )
        or "(none)"
    )
    agent: Agent[None, FinalAnswer] = Agent(
        text_model,
        output_type=ToolOutput(FinalAnswer, name="FinalAnswer"),
        instructions=FORCED_FINAL_INSTRUCTIONS,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=400),
    )
    text_model.purpose = "forced_final_answer"
    try:
        result = await agent.run(
            f"Question: {question}\nReason: {reason}\nPages read: {pages}\n"
            f"Evidence ledger:\n{notes}\nImage inspections:\n{inspections}"
        )
        return result.output
    except Exception as exc:  # noqa: BLE001
        deps.log.error("forced final answer failed: %s", exc)
        return FinalAnswer(
            status="not_answerable",
            answer="Not answerable",
            evidence_pages=[],
            reasoning=f"agent failed: {reason}; {exc}",
        )


def _format_plan_as_markup(plan: dict[str, Any]) -> str:
    """Format the plan dictionary as clean Markdown markup instead of raw JSON."""
    lines: list[str] = []
    if "question_type" in plan or "expected_answer_format" in plan:
        qt = plan.get("question_type", "unspecified")
        fmt = plan.get("expected_answer_format", "unspecified")
        lines.append(f"- **Question Type**: {qt} (Expected Format: {fmt})")
    if "needs_visual_inspection" in plan:
        vis = plan.get("needs_visual_inspection")
        lines.append(f"- **Needs Visual Inspection**: {vis}")
    if plan.get("sub_goals"):
        lines.append("- **Sub-goals**:")
        for sg in plan["sub_goals"]:
            lines.append(f"  - {sg}")
    if plan.get("search_queries"):
        queries = ", ".join(f"`{q}`" for q in plan["search_queries"])
        lines.append(f"- **Search Queries**: {queries}")
    if plan.get("candidate_sections"):
        sections = ", ".join(str(s) for s in plan["candidate_sections"])
        lines.append(f"- **Candidate Sections**: {sections}")
    if plan.get("anchor_pages"):
        pages = ", ".join(str(p) for p in plan["anchor_pages"])
        lines.append(f"- **Anchor Pages**: {pages}")

    known_keys = {
        "question_type",
        "expected_answer_format",
        "needs_visual_inspection",
        "sub_goals",
        "search_queries",
        "candidate_sections",
        "anchor_pages",
    }
    for k, v in plan.items():
        if k not in known_keys and not k.startswith("_"):
            title = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"- **{title}**: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"- **{title}**: {v}")

    return "\n".join(lines)


def run_orchestrator(
    settings: AgentSettings,
    question: str,
    pdf_path: Path,
    index: dict[str, Any],
    plan: dict[str, Any],
    parser_dir: Path,
    step_dir: Path,
    run_log: Path,
    ledger: CostLedger,
    round_no: int = 1,
    prior_feedback: str | None = None,
) -> dict[str, Any]:
    log = setup_logger("step4.orchestrator", step_dir / "orchestrator.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "images_viewed").mkdir(exist_ok=True)
    (step_dir / "tool_results").mkdir(exist_ok=True)
    text_model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="04_orchestrator",
    )
    vision_model = traced_model(
        settings,
        settings.vision_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="04_orchestrator",
    )
    text_model.round_no = round_no
    vision_model.round_no = round_no
    max_tool_calls = settings.max_tool_calls

    pages_md = {
        p: (parser_dir / "pages" / f"page_{p:03d}.md").read_text(encoding="utf-8")
        for p in range(1, index["n_pages"] + 1)
    }
    bm25_docs = build_page_corpus(index, pages_md)
    bm25 = BM25(bm25_docs)
    outline_path = step_dir.parent / "02_index" / "outline.md"
    outline_md = (
        outline_path.read_text(encoding="utf-8") if outline_path.exists() else ""
    )
    trace_path = step_dir / f"trace_round{round_no}.jsonl"
    trace_path.write_text("")
    deps = NavDeps(
        pdf_path=pdf_path,
        index=index,
        outline_md=outline_md,
        pages_md=pages_md,
        bm25=bm25,
        step_dir=step_dir,
        round_no=round_no,
        log=log,
        vision_model=vision_model,
        trace_path=trace_path,
        plan=plan,
    )

    plan_view = {
        k: v
        for k, v in plan.items()
        if not k.startswith("_") and k not in ("round", "gaps_in", "locator")
    }
    system = SYSTEM_PROMPT.format(
        doc_title=index["doc_title"],
        n_pages=index["n_pages"],
        outline=enriched_outline(index),
        plan=_format_plan_as_markup(plan_view),
        budget=max_tool_calls,
    )
    user = f"Question: {question}"
    if prior_feedback:
        user += (
            f"\n\nThis is round {round_no}. The verifier rejected the previous round: "
            f"{prior_feedback}\nClose these gaps before answering."
        )
    append_jsonl(
        trace_path,
        {"event": "start", "round": round_no, "system": system, "user": user},
    )

    agent = build_navigator(text_model)
    text_model.purpose = f"orchestrator_round{round_no}"

    async def _round() -> tuple[FinalAnswer | None, bool]:
        """Navigation and, if needed, the forced final answer on ONE event loop."""
        final_obj, failure = await _navigate(agent, deps, system, user, max_tool_calls)
        if not failure:
            return final_obj, False
        kind = failure.split(":", 1)[0]
        deps.anomalies.append(
            {"type": kind, "detail": failure[:400], "calls": deps.calls}
        )
        log.warning(
            "%s -> forcing a final answer from the evidence so far", failure[:200]
        )
        return await _forced_final(text_model, deps, question, failure), True

    final_obj, salvaged = asyncio.run(_round())
    assert final_obj is not None
    final: dict[str, Any] = {
        "status": final_obj.status,
        "answer": final_obj.answer.strip(),
        "evidence_pages": [
            p for p in final_obj.evidence_pages if 1 <= p <= deps.n_pages
        ],
        "reasoning": final_obj.reasoning,
        "salvaged": salvaged,
        "round": round_no,
        "tool_calls_used": deps.calls,
        "pages_read": sorted(set(deps.pages_read)),
        "image_inspections": deps.image_inspections,
        "ledger": deps.ledger,
        "anomalies": deps.anomalies,
    }
    write_json(step_dir / f"ledger_round{round_no}.json", deps.ledger)
    write_json(step_dir / f"anomalies_round{round_no}.json", deps.anomalies)
    public = {
        k: v for k, v in final.items() if k not in ("ledger", "image_inspections")
    }
    write_json(step_dir / f"final_round{round_no}.json", public)
    append_jsonl(
        trace_path,
        {"event": "final", **{k: v for k, v in public.items() if k != "anomalies"}},
    )
    log.info(
        "round %d finished: status=%s answer=%r pages=%s calls=%d anomalies=%d",
        round_no,
        final["status"],
        final["answer"],
        final["evidence_pages"],
        deps.calls,
        len(deps.anomalies),
    )
    return final


__all__ = ["BM25", "TOOL_NAMES", "NavDeps", "build_navigator", "run_orchestrator"]
