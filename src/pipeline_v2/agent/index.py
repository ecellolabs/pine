"""Step 2 - Index builder (M2, PageIndex-style vectorless hierarchy).

document -> sections -> pages.  Every page gets a typed ``PageSummary`` from a
Pydantic AI agent (text model on the Docling markdown; vision model on the page
image when the page has no usable text).  A second agent then groups the page
titles into a ``Hierarchy``; an output validator checks coverage and nesting
and asks the model to retry (``ModelRetry``) when the tree is invalid.  If it is
still invalid after the retries, the last attempt is repaired into a flat index
and the failure is recorded (``hierarchy_fallback_used``).

Writes:
  02_index/page_summaries.jsonl   one line per page (title, summary, keywords, source)
  02_index/index.json             the tree (validated) + validation report
  02_index/outline.md             human-readable outline
  02_index/llm_calls.jsonl        every model request/response for this step
  02_index/index.log
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _extract_visual_elements(md_text: str) -> tuple[list[str], list[str], bool]:
    """Extract figure captions, table titles/headers, and check for visual elements in page markdown."""
    figures: list[str] = []
    tables: list[str] = []
    for line in md_text.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        if re.search(r'\b(Figure|Fig)\.?\s*\d+[:\.]?', line_s, re.IGNORECASE):
            figures.append(line_s[:150])
        elif line_s.startswith("![") and "]" in line_s:
            caption = line_s[2:line_s.find("]")]
            if caption and caption.lower() != "image":
                figures.append(caption[:150])
        elif re.search(r'\bTable\s+\d+[:\.]?', line_s, re.IGNORECASE):
            tables.append(line_s[:150])
        elif line_s.startswith("|") and not line_s.startswith("|---") and not line_s.startswith("| ---"):
            cells = [c.strip() for c in line_s.split("|") if c.strip()]
            if cells and not tables and len(cells) > 1:
                tables.append(f"Table columns: {', '.join(cells[:5])}")

    has_visuals = len(figures) > 0 or len(tables) > 0 or ("<!-- image -->" in md_text) or ("![image" in md_text.lower())
    return figures, tables, has_visuals

from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings

from pipeline_v2.agent.common import (
    AgentSettings,
    CostLedger,
    read_json,
    render_page,
    setup_logger,
    write_json,
)
from pipeline_v2.agent.llm import traced_model
from pipeline_v2.agent.schemas import Hierarchy, PageSummary

SUMMARY_INSTRUCTIONS = (
    "You are indexing a long document for navigation. Given the Markdown text of "
    "ONE page, produce a short descriptive title, a 1-2 sentence summary of what "
    "information the page contains (topics, entities, numbers, tables, figures) "
    "and 5-10 keywords. Mention figure and table captions explicitly "
    "(e.g. 'Figure 1: ...') so they can be found later."
)
PAGE_SUMMARY_PROMPT = "Page {page} of {n_pages}.\n\nPAGE MARKDOWN:\n{text}\n"

IMAGE_SUMMARY_INSTRUCTIONS = (
    "You are indexing a long document for navigation. This page has no "
    "machine-readable text, so describe it from the image: headings, charts "
    "(type, series, axes, key values), tables, images, numbers. Produce a short "
    "title, a 2-3 sentence summary and 5-10 keywords."
)

HIERARCHY_INSTRUCTIONS = """You are building a table of contents (hierarchical index) for a {n_pages}-page document titled "{doc_title}".
You will be given one line per page: page number, title, short summary.

Group CONSECUTIVE pages into sections (and optional subsections) that reflect the document's structure.
Rules:
- Pages are 1-indexed and every page from 1 to {n_pages} must belong to exactly one leaf section.
- Sections must cover contiguous page ranges in order; no overlaps, no gaps.
- A subsection's page range must lie inside its parent's range.
- 3 to 12 top-level sections. Subsections optional."""


@dataclass
class HierarchyDeps:
    n_pages: int
    log: logging.Logger
    attempts: list[dict[str, Any]] = field(default_factory=list)


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _validate_tree(
    sections: list[dict[str, Any]], n_pages: int, log: logging.Logger
) -> list[str]:
    problems: list[str] = []
    covered: dict[int, int] = {}

    def walk(
        nodes: list[dict[str, Any]], depth: int, parent_range: tuple[int, int] | None
    ) -> None:
        for s in nodes:
            try:
                a, b = int(s["start_page"]), int(s["end_page"])
            except Exception:  # noqa: BLE001
                problems.append(
                    f"section {s.get('title')!r} has invalid page range "
                    f"{s.get('start_page')}-{s.get('end_page')}"
                )
                continue
            if a > b or a < 1 or b > n_pages:
                problems.append(
                    f"section {s['title']!r} range {a}-{b} outside 1..{n_pages} or reversed"
                )
            if parent_range and (a < parent_range[0] or b > parent_range[1]):
                problems.append(
                    f"subsection {s['title']!r} {a}-{b} escapes parent {parent_range}"
                )
            subs = s.get("subsections") or []
            if subs:
                walk(subs, depth + 1, (a, b))
            else:
                for p in range(a, b + 1):
                    covered[p] = covered.get(p, 0) + 1

    walk(sections, 0, None)
    missing = [p for p in range(1, n_pages + 1) if p not in covered]
    dup = [p for p, c in covered.items() if c > 1]
    if missing:
        problems.append(f"pages not covered by any leaf section: {missing}")
    if dup:
        problems.append(f"pages covered by more than one leaf section: {sorted(dup)}")
    for pr in problems:
        log.warning("index validation: %s", pr)
    return problems


def _assign_ids(sections: list[dict[str, Any]], prefix: str = "S") -> None:
    for i, s in enumerate(sections, 1):
        s["id"] = f"{prefix}{i}"
        if s.get("subsections"):
            _assign_ids(s["subsections"], prefix=f"{s['id']}.")


def _repair(sections: list[dict[str, Any]], n_pages: int) -> list[dict[str, Any]]:
    """Clamp, sort and fill gaps so navigation still works after an invalid tree."""
    repaired = []
    for s in sections:
        try:
            a, b = max(1, int(s["start_page"])), min(n_pages, int(s["end_page"]))
        except Exception as exc:  # noqa: BLE001
            log_fallback = logging.getLogger("step2.index")
            log_fallback.warning("dropping invalid section %r: %s", s, exc)
            continue
        if a <= b:
            repaired.append(
                {"title": s.get("title", "Section"), "start_page": a, "end_page": b}
            )
    repaired.sort(key=lambda s: s["start_page"])
    flat: list[dict[str, Any]] = []
    cursor = 1
    for s in repaired:
        if s["start_page"] > cursor:
            flat.append(
                {
                    "title": f"Pages {cursor}-{s['start_page'] - 1} (unassigned by LLM)",
                    "start_page": cursor,
                    "end_page": s["start_page"] - 1,
                }
            )
        if s["end_page"] >= cursor:
            flat.append({**s, "start_page": max(s["start_page"], cursor)})
            cursor = s["end_page"] + 1
    if cursor <= n_pages:
        flat.append(
            {
                "title": f"Pages {cursor}-{n_pages} (unassigned by LLM)",
                "start_page": cursor,
                "end_page": n_pages,
            }
        )
    return flat


def _outline_md(index: dict[str, Any]) -> str:
    lines = [f"# {index['doc_title']}  ({index['n_pages']} pages)", ""]

    def walk(nodes: list[dict[str, Any]], depth: int) -> None:
        for s in nodes:
            lines.append(
                f"{'  ' * depth}- [{s['id']}] {s['title']}  "
                f"(pages {s['start_page']}-{s['end_page']})"
            )
            if s.get("subsections"):
                walk(s["subsections"], depth + 1)
            else:
                for p in range(s["start_page"], s["end_page"] + 1):
                    ps = index["pages"][str(p)]
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
                            v_details.append("Visuals")
                        vis_tag = f" [{'; '.join(v_details)}]"
                    lines.append(
                        f"{'  ' * (depth + 1)}- p{p}: {ps['title']} [{ps['source']}]{vis_tag}"
                    )

    walk(index["sections"], 0)
    return "\n".join(lines) + "\n"


def _sections_to_dicts(hierarchy: Hierarchy) -> list[dict[str, Any]]:
    return [s.model_dump() for s in hierarchy.sections]


def run_index(
    settings: AgentSettings,
    pdf_path: Path,
    parser_dir: Path,
    step_dir: Path,
    run_log: Path,
    ledger: CostLedger,
) -> dict[str, Any]:
    log = setup_logger("step2.index", step_dir / "index.log", run_log)
    index_path = step_dir / "index.json"
    if index_path.exists():
        log.info("index already exists, skipping (idempotent)")
        return read_json(index_path)
    step_dir.mkdir(parents=True, exist_ok=True)

    text_model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="02_index",
    )
    vision_model = traced_model(
        settings,
        settings.vision_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="02_index",
    )
    summary_agent: Agent[None, PageSummary] = Agent(
        text_model,
        output_type=ToolOutput(PageSummary, name="PageSummary"),
        instructions=SUMMARY_INSTRUCTIONS,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=300),
    )
    image_summary_agent: Agent[None, PageSummary] = Agent(
        vision_model,
        output_type=ToolOutput(PageSummary, name="PageSummary"),
        instructions=IMAGE_SUMMARY_INSTRUCTIONS,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=300),
    )

    stats = read_json(parser_dir / "page_stats.json")
    n_pages = stats["pages"]
    pages: dict[str, dict[str, Any]] = {}
    summaries_path = step_dir / "page_summaries.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if summaries_path.exists():
        for line in summaries_path.read_text().splitlines():
            r = json.loads(line)
            done[r["page"]] = r
    failures: list[dict[str, Any]] = []
    with summaries_path.open("a", encoding="utf-8") as out:
        for st in stats["page_stats"]:
            p = st["page"]
            if p in done:
                pages[str(p)] = done[p]
                continue
            md = (parser_dir / "pages" / f"page_{p:03d}.md").read_text(encoding="utf-8")
            ext_figs, ext_tabs, has_vis = _extract_visual_elements(md)
            use_image = st["docling_chars"] < 40
            rec: dict[str, Any] = {
                "page": p,
                "source": "image" if use_image else "text",
                "docling_chars": st["docling_chars"],
                "tables_count": st["tables"],
                "pictures_count": st["pictures"],
            }
            try:
                if use_image:
                    vision_model.purpose = f"page_summary_p{p}"
                    jpeg = render_page(pdf_path, p, max_side=768, quality=70)
                    summary = image_summary_agent.run_sync(
                        [
                            BinaryContent(data=jpeg, media_type="image/jpeg"),
                            f"Page {p} of {n_pages} of the document.",
                        ]
                    ).output
                else:
                    text_model.purpose = f"page_summary_p{p}"
                    prompt_text = PAGE_SUMMARY_PROMPT.format(
                        page=p, n_pages=n_pages, text=_truncate(md, 3000)
                    )
                    if ext_figs or ext_tabs:
                        prompt_text += f"\n\nDETECTED VISUAL ASSETS ON THIS PAGE:\nFigures: {ext_figs}\nTables: {ext_tabs}\nMake sure to retain these exact figure and table titles in your response fields."
                    summary = summary_agent.run_sync(prompt_text).output

                merged_figs = list(dict.fromkeys(getattr(summary, "figures", []) + ext_figs))
                merged_tabs = list(dict.fromkeys(getattr(summary, "tables", []) + ext_tabs))
                rec.update(
                    {
                        "title": summary.title.strip()[:120],
                        "summary": summary.summary.strip()[:600],
                        "keywords": [str(k) for k in summary.keywords][:10],
                        "figures": merged_figs,
                        "tables": merged_tabs,
                        "has_visual_elements": getattr(summary, "has_visual_elements", False) or has_vis,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.error("page %d summary failed: %s", p, exc)
                failures.append({"page": p, "error": str(exc)})
                first_line = next(
                    (ln.strip("# ").strip() for ln in md.splitlines() if ln.strip()),
                    f"Page {p}",
                )
                rec.update(
                    {
                        "title": first_line[:120],
                        "summary": "(summary agent failed; fallback to first line)",
                        "keywords": [],
                        "figures": ext_figs,
                        "tables": ext_tabs,
                        "has_visual_elements": has_vis,
                        "error": str(exc),
                    }
                )
            log.info("page %3d [%s] title=%r figures=%r", p, rec["source"], rec["title"], rec.get("figures"))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pages[str(p)] = rec

    # hierarchy (agent + output validator that forces a retry on an invalid tree)
    doc_title = pages["1"]["title"]
    page_lines = "\n".join(
        f"p{p}: {pages[str(p)]['title']} -- {pages[str(p)]['summary'][:160]}"
        for p in range(1, n_pages + 1)
    )
    deps = HierarchyDeps(n_pages=n_pages, log=log)
    hierarchy_agent: Agent[HierarchyDeps, Hierarchy] = Agent(
        text_model,
        output_type=ToolOutput(Hierarchy, name="Hierarchy"),
        instructions=HIERARCHY_INSTRUCTIONS.format(
            n_pages=n_pages, doc_title=doc_title
        ),
        deps_type=HierarchyDeps,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=1500),
    )

    @hierarchy_agent.output_validator
    def check_tree(ctx: RunContext[HierarchyDeps], out: Hierarchy) -> Hierarchy:
        sections = _sections_to_dicts(out)
        problems = _validate_tree(sections, ctx.deps.n_pages, ctx.deps.log)
        ctx.deps.attempts.append(
            {
                "attempt": len(ctx.deps.attempts) + 1,
                "problems": problems,
                "n_sections": len(sections),
                "sections": sections,
            }
        )
        if problems:
            raise ModelRetry(
                "Your previous answer was invalid: "
                + "; ".join(problems[:4])
                + ". Fix it."
            )
        return out

    sections: list[dict[str, Any]] = []
    fallback_used = False
    text_model.purpose = "hierarchy"
    try:
        result = hierarchy_agent.run_sync(f"PAGES:\n{page_lines}", deps=deps)
        sections = _sections_to_dicts(result.output)
    except UnexpectedModelBehavior as exc:
        log.warning(
            "hierarchy invalid after retries (%s) -> repairing into a flat/clamped "
            "index (INDEX FAILURE recorded)",
            exc,
        )
        fallback_used = True
        last = deps.attempts[-1]["sections"] if deps.attempts else []
        sections = _repair(last, n_pages)
    except Exception as exc:  # noqa: BLE001
        log.error("hierarchy agent failed: %s -> flat index", exc)
        fallback_used = True
        deps.attempts.append(
            {"attempt": len(deps.attempts) + 1, "problems": [str(exc)]}
        )
        sections = _repair([], n_pages)
    _assign_ids(sections)

    index = {
        "doc": pdf_path.name,
        "doc_title": doc_title,
        "n_pages": n_pages,
        "text_model": settings.text_model,
        "vision_model": settings.vision_model,
        "pages_summarised_from_image": sum(
            1 for p in pages.values() if p["source"] == "image"
        ),
        "page_summary_failures": failures,
        "hierarchy_attempts": [
            {k: v for k, v in a.items() if k != "sections"} for a in deps.attempts
        ],
        "hierarchy_fallback_used": fallback_used,
        "validation_problems_final": _validate_tree(sections, n_pages, log),
        "sections": sections,
        "pages": pages,
    }
    write_json(index_path, index)
    (step_dir / "outline.md").write_text(_outline_md(index), encoding="utf-8")
    log.info(
        "index built: %d top-level sections, %d pages from image, fallback=%s, "
        "cost so far=$%.4f",
        len(sections),
        index["pages_summarised_from_image"],
        fallback_used,
        ledger.cost_usd,
    )
    return index
