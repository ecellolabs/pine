"""Figure and table captions per page, taken from Docling's structured output
when available (``01_parser/docling_document.json``: every picture/table item is
linked to its caption and page), with a strict Markdown fallback.

Rules (checked against the RAPTOR paper, 2401.18059v1):

* a *caption* is a line that **starts** with ``Figure N:``/``Fig. N.``/``Table N:``
  (Docling emits captions as their own paragraph right after ``<!-- image -->``);
* an in-text *mention* ("... as shown in Figure 1 ...") is recorded separately as
  a number and never counted as a caption;
* captions proposed by the summary model are *hints* and are kept only when they
  occur verbatim on the page.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CAPTION_RE = re.compile(
    r"^(?:#+\s*)?(?:\*\*)?(Figure|Fig\.?|Table)\s*(\d+)[a-z]?\s*[:.]\s*(?:\*\*)?\s*(\S.*)$",
    re.IGNORECASE,
)
MENTION_RE = re.compile(r"\b(Figure|Fig\.?|Table)\s*(\d+)\b", re.IGNORECASE)


@dataclass
class PageVisuals:
    figures: list[str] = field(default_factory=list)  # true captions on this page
    tables: list[str] = field(default_factory=list)  # true captions on this page
    mentions_figures: list[int] = field(
        default_factory=list
    )  # numbers mentioned in text
    mentions_tables: list[int] = field(default_factory=list)
    has_visual_elements: bool = False
    captions_source: str = "markdown"  # "docling" | "markdown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def _clean_caption(kind: str, num: str, rest: str) -> str:
    label = "Table" if kind.lower().startswith("tab") else "Figure"
    rest = rest.strip().rstrip("*").strip()
    return f"{label} {int(num)}: {rest}"[:200]


def captions_from_markdown(md: str) -> tuple[list[str], list[str], set[int]]:
    """Caption lines only (line starts with Figure/Table N:). Returns
    (figure captions, table captions, indices of the caption lines)."""
    figures: list[str] = []
    tables: list[str] = []
    caption_lines: set[int] = set()
    for i, raw in enumerate(md.splitlines()):
        m = CAPTION_RE.match(raw.strip())
        if not m:
            continue
        kind, num, rest = m.group(1), m.group(2), m.group(3)
        caption = _clean_caption(kind, num, rest)
        (tables if kind.lower().startswith("tab") else figures).append(caption)
        caption_lines.add(i)
    return list(dict.fromkeys(figures)), list(dict.fromkeys(tables)), caption_lines


def mentions_from_markdown(
    md: str, caption_lines: set[int] | None = None
) -> tuple[list[int], list[int]]:
    """Figure/table numbers referred to in running text (caption lines excluded)."""
    figs: set[int] = set()
    tabs: set[int] = set()
    for i, raw in enumerate(md.splitlines()):
        if caption_lines and i in caption_lines:
            continue
        for kind, num in MENTION_RE.findall(raw):
            (tabs if kind.lower().startswith("tab") else figs).add(int(num))
    return sorted(figs), sorted(tabs)


def verify_hints(hints: list[str], md: str, min_chars: int = 12) -> list[str]:
    """Keep model-proposed captions only if they occur verbatim on the page."""
    page = _norm(md)
    kept: list[str] = []
    for hint in hints:
        text = _norm(str(hint))
        probe = text[: max(min_chars, min(len(text), 60))]
        if len(text) >= min_chars and probe in page:
            kept.append(str(hint).strip()[:200])
    return list(dict.fromkeys(kept))


def docling_captions(parser_dir: Path) -> dict[int, tuple[list[str], list[str]]] | None:
    """Per-page (figure captions, table captions) from Docling's item/caption links.
    Returns None when the Docling JSON is not available (e.g. API parsing mode)."""
    path = parser_dir / "docling_document.json"
    if not path.exists():
        return None
    try:
        from docling_core.types.doc.document import DoclingDocument

        doc = DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  (old/foreign JSON -> fall back to Markdown)
        return None
    out: dict[int, tuple[list[str], list[str]]] = {}
    for item, is_table in [(t, True) for t in doc.tables] + [
        (p, False) for p in doc.pictures
    ]:
        caption = item.caption_text(doc).strip()
        pages = sorted({pr.page_no for pr in item.prov})
        if not caption or not pages:
            continue
        m = CAPTION_RE.match(caption)
        caption = (
            _clean_caption(m.group(1), m.group(2), m.group(3)) if m else caption[:200]
        )
        figs, tabs = out.setdefault(pages[0], ([], []))
        (tabs if is_table else figs).append(caption)
    return out


def extract_page_visuals(
    parser_dir: Path,
    pages_md: dict[int, str],
    pictures_per_page: dict[int, int] | None = None,
) -> dict[int, PageVisuals]:
    """Docling captions first, Markdown caption lines as fallback, mentions always."""
    structured = docling_captions(parser_dir)
    result: dict[int, PageVisuals] = {}
    for p, md in pages_md.items():
        md_figs, md_tabs, caption_lines = captions_from_markdown(md)
        if structured is not None and p in structured:
            figs, tabs = structured[p]
            source = "docling"
        elif structured is not None:
            figs, tabs, source = [], [], "docling"
            # Docling saw no captioned item here; trust a strict caption line if present
            if md_figs or md_tabs:
                figs, tabs, source = md_figs, md_tabs, "markdown"
        else:
            figs, tabs, source = md_figs, md_tabs, "markdown"
        m_figs, m_tabs = mentions_from_markdown(md, caption_lines)
        n_pics = (pictures_per_page or {}).get(p, 0)
        result[p] = PageVisuals(
            figures=list(dict.fromkeys(figs)),
            tables=list(dict.fromkeys(tabs)),
            mentions_figures=m_figs,
            mentions_tables=m_tabs,
            has_visual_elements=bool(figs or tabs or n_pics or "<!-- image -->" in md),
            captions_source=source,
        )
    return result


def caption_tokens(captions: list[str]) -> list[str]:
    """Normalised search tokens for captions, e.g. 'figure1', 'table12'."""
    toks: list[str] = []
    for cap in captions:
        m = CAPTION_RE.match(cap)
        if m:
            label = "table" if m.group(1).lower().startswith("tab") else "figure"
            toks.append(f"{label}{int(m.group(2))}")
    return toks


def write_visuals(step_dir: Path, visuals: dict[int, PageVisuals]) -> None:
    (step_dir / "page_visuals.json").write_text(
        json.dumps(
            {str(p): v.to_dict() for p, v in visuals.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
