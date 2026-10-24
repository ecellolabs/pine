"""Step 2 - the lean index (default, ``--index lean``): Docling structure in code, model
calls only for what text cannot carry (images). ``--index full`` keeps the original
model-summarised index (agent/index.py) for comparison; see INDEX.md for the experiments.

Measured on s2/s3: the full index's page summaries (one model call per page, 56-77% of all
model cost) and its model-built section tree did not change which pages the locator picks;
captions, page text and the readings of image-only pages did. The lean index keeps the
parts that mattered and builds the rest from the Docling parse:

- page records: title = the page's first Docling heading (else its first line), captions,
  picture/table counts, text length, figure/table mentions; no model call.
- outline: a section starts at each page that opens with a heading (Docling gives every
  heading level 1, so the outline is flat); no model call.
- image transcriptions: one vision call per large picture (>= 5% of the page, at most 4
  per page), cropped from the Docling box and de-duplicated by its pixels (a logo on every
  slide is read once), plus the whole page for pages without text. The text is kept in ``visual_text`` with its provenance, never merged into the page text:
  it is searchable, and the navigator sees it as a vision note, which the verifier checks
  like any other vision reading.

The locator ranks pages by their best passage (paragraph-sized) for a lean index.
Same file layout as the full index (02_index/index.json, outline.md, page_visuals.json).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.settings import ModelSettings

from pine.agent.captions import extract_page_visuals, write_visuals
from pine.agent.common import (
    AgentSettings,
    CostLedger,
    read_json,
    render_page,
    run_on_thread_loop,
    setup_logger,
    write_json,
)
from pine.agent.index import _assign_ids, _outline_md
from pine.agent.llm import CALL_PURPOSE, traced_model

BIG_PICTURE = 0.05  # share of the page area a picture needs to be worth reading
MAX_IMAGES_PER_PAGE = 4

TRANSCRIBE_INSTRUCTIONS = (
    "Transcribe this image from a document completely as plain text: every label, value, "
    "legend entry, axis title, row and column; colours where they encode meaning; for a "
    "diagram, its nodes and connections; for a photo, one line saying what it shows. Only "
    "what is visible; do not interpret."
)


def _headings(doc: dict[str, Any]) -> dict[int, list[str]]:
    """Docling headings per page, in reading order."""
    out: dict[int, list[str]] = {}
    for t in doc.get("texts") or []:
        if t.get("label") not in ("title", "section_header"):
            continue
        text = " ".join(str(t.get("text", "")).split())
        prov = t.get("prov") or []
        if text and prov:
            out.setdefault(int(prov[0]["page_no"]), []).append(text[:120])
    return out


def _first_line(md: str) -> str:
    for line in md.splitlines():
        line = line.strip("# ").strip()
        if len(line) >= 3 and not line.startswith(("<!--", "|")):
            return line[:120]
    return ""


def _picture_boxes(
    doc: dict[str, Any],
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Large pictures per page as (x0, y0, x1, y1) in points, origin top-left."""
    sizes = {
        int(k): (v["size"]["width"], v["size"]["height"])
        for k, v in (doc.get("pages") or {}).items()
    }
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    for item in doc.get("pictures") or []:
        for prov in item.get("prov") or []:
            p, bb = prov.get("page_no"), prov.get("bbox") or {}
            if p not in sizes or not bb:
                continue
            w, h = sizes[p]
            if bb.get("coord_origin") == "BOTTOMLEFT":
                y0, y1 = h - bb["t"], h - bb["b"]
            else:
                y0, y1 = bb["t"], bb["b"]
            box = (bb["l"], min(y0, y1), bb["r"], max(y0, y1))
            if (box[2] - box[0]) * (box[3] - box[1]) >= BIG_PICTURE * w * h:
                out.setdefault(p, []).append(box)
    return out


def _sections(
    n_pages: int, titles: dict[int, str], headed: set[int]
) -> list[dict[str, Any]]:
    """A section starts at each page that opens with a heading and runs to the next one."""
    starts = sorted(p for p in headed if 1 <= p <= n_pages)
    if not starts or starts[0] != 1:
        starts = [1, *starts]
    sections = []
    for i, a in enumerate(starts):
        b = (starts[i + 1] - 1) if i + 1 < len(starts) else n_pages
        sections.append(
            {
                "title": titles.get(a) or f"Page {a}",
                "start_page": a,
                "end_page": b,
                "subsections": [],
            }
        )
    _assign_ids(sections)
    return sections


def run_lean_index(
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
    stats = read_json(parser_dir / "page_stats.json")
    n_pages = int(stats["pages"])
    page_stats = {st["page"]: st for st in stats["page_stats"]}
    pages_md = {
        p: (parser_dir / "pages" / f"page_{p:03d}.md").read_text(encoding="utf-8")
        for p in page_stats
    }
    visuals = extract_page_visuals(
        parser_dir, pages_md, {p: st["pictures"] for p, st in page_stats.items()}
    )
    write_visuals(step_dir, visuals)
    doc_path = parser_dir / "docling_document.json"
    doc = read_json(doc_path) if doc_path.exists() else {}
    heads = _headings(doc)
    boxes = _picture_boxes(doc)

    pages: dict[str, dict[str, Any]] = {}
    titles: dict[int, str] = {}
    for p in sorted(page_stats):
        st, vis = page_stats[p], visuals[p]
        titles[p] = (heads.get(p) or [""])[0] or _first_line(pages_md[p]) or f"Page {p}"
        pages[str(p)] = {
            "page": p,
            "source": "image" if st["docling_chars"] < 40 else "text",
            "docling_chars": st["docling_chars"],
            "tables_count": st["tables"],
            "pictures_count": st["pictures"],
            "title": titles[p],
            "summary": "",
            "keywords": [],
            **vis.to_dict(),
            "has_visual_elements": vis.has_visual_elements or bool(st["pictures"]),
            "visual_text": "",
            "error": None,
        }

    # what gets a vision call
    jobs: list[tuple[int, tuple[float, float, float, float] | None]] = []
    for p in sorted(page_stats):
        if p in boxes:
            jobs += [(p, b) for b in boxes[p][:MAX_IMAGES_PER_PAGE]]
        elif page_stats[p]["docling_chars"] < 40:  # no text at all: read the whole page
            jobs.append((p, None))

    vision_model = traced_model(
        settings,
        settings.vision_model,
        step_dir=step_dir,
        ledger=ledger,
        logger=log,
        step_name="02_index",
    )
    reader: Agent[None, str] = Agent(
        vision_model,
        instructions=TRANSCRIBE_INSTRUCTIONS,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=600),
    )
    sem = asyncio.Semaphore(max(1, settings.index_concurrency))
    seen: set[str] = set()
    calls: list[int] = []
    failures: list[dict[str, Any]] = []

    async def read(
        i: int, p: int, box: tuple[float, float, float, float] | None
    ) -> tuple[int, str]:
        jpeg = render_page(pdf_path, p, max_side=1024, quality=80, clip=box)
        digest = hashlib.sha256(jpeg).hexdigest()
        if digest in seen:  # the same image again (a logo, a repeated slide element)
            return p, ""
        seen.add(digest)
        calls.append(p)
        async with sem:
            CALL_PURPOSE.set(f"transcribe_p{p}" + (f"_img{i}" if box else ""))
            try:
                reply = await reader.run(
                    [
                        BinaryContent(data=jpeg, media_type="image/jpeg"),
                        f"Page {p} of {n_pages}.",
                    ]
                )
                return p, reply.output.strip()
            except Exception as exc:  # noqa: BLE001 - a failed reading leaves the page text-only
                log.error("transcription of page %d failed: %s", p, exc)
                failures.append({"page": p, "error": str(exc)})
                return p, ""

    async def read_all() -> list[tuple[int, str]]:
        return list(
            await asyncio.gather(*(read(i, p, b) for i, (p, b) in enumerate(jobs)))
        )

    by_page: dict[int, list[str]] = {}
    for p, text in run_on_thread_loop(read_all()) if jobs else []:
        if text:
            by_page.setdefault(p, []).append(text)
    for p, texts in by_page.items():
        pages[str(p)]["visual_text"] = "\n".join(
            f"[image {i}] {t}" for i, t in enumerate(texts, 1)
        )[:4000]

    headed = {p for p, hs in heads.items() if hs}
    sections = _sections(n_pages, titles, headed)
    index: dict[str, Any] = {
        "doc": pdf_path.name,
        "doc_title": (heads.get(1) or [""])[0] or titles.get(1) or pdf_path.stem,
        "n_pages": n_pages,
        "index_kind": "lean",
        "text_model": None,
        "vision_model": settings.vision_model,
        "visual_calls": len(calls),
        "pages_with_visual_text": len(by_page),
        "pages_summarised_from_image": len(by_page),
        "page_summary_failures": failures,
        "hierarchy_attempts": [],
        "hierarchy_fallback_used": False,
        "validation_problems_final": [],
        "sections": sections,
        "pages": pages,
    }
    write_json(index_path, index)
    (step_dir / "outline.md").write_text(_outline_md(index), encoding="utf-8")
    log.info(
        "lean index built: %d sections from headings, %d vision calls, %d pages with "
        "visual text, %d failures, cost so far=$%.4f",
        len(sections),
        len(calls),
        len(by_page),
        len(failures),
        ledger.summary()["cost_usd"],
    )
    return index
