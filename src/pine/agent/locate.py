"""Deterministic region location over the page index (``RegionLocator``).

The planner used to *infer* candidate sections from section titles alone, which
sends "Figure 1" questions to whichever section title shares a word with the
question.  This module finds candidate regions from the index itself:

* **Reference resolution** - "Figure N" / "Table N" / "Fig. N" named in the
  question are resolved to the page(s) whose *caption* starts with that label
  (captions come from the Docling structure, see ``captions.py``); if no caption
  matches, pages that *mention* the number are used as a fallback.
* **Lexical ranking** - BM25 over every page (title x3, captions x5, summary,
  keywords, full Markdown: the same corpus the navigator's ``search_pages`` tool
  uses) and over every section (title, page titles, summaries, keywords and
  captions of its page range) for free-text queries.

``locate()`` combines the two into ranked ``sections`` and ``anchor_pages`` with
an explanation for each, so the planner (via its ``locate_regions`` tool), the
code-side merge in ``planner.run_planner`` and the navigator's prompt all agree
on where the evidence should be.  ``BM25`` and ``tokens`` live here and are
re-used by the orchestrator's ``search_pages`` tool."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_tok = re.compile(r"[a-z0-9]+")
REF_RE = re.compile(r"\b(fig(?:ure)?|tab(?:le)?)\.?\s*(\d+)\b", re.IGNORECASE)


def tokens(s: str) -> list[str]:
    """Lower-case alphanumeric tokens plus ``figureN`` / ``tableN`` compound tokens
    (so 'Figure 1', 'Fig. 01' and 'figure1' all match)."""
    toks = _tok.findall(s.lower())
    matches = re.findall(r"\b(figure|fig|table|tab)\s*(\d+)\b", s.lower())
    for tag, num in matches:
        clean_num = str(int(num))
        num_pad = f"{int(num):02d}"
        if tag in ("figure", "fig"):
            for nv in (num, clean_num, num_pad):
                toks.append(f"figure{nv}")
                toks.append(f"fig{nv}")
        elif tag in ("table", "tab"):
            for nv in (num, clean_num, num_pad):
                toks.append(f"table{nv}")
                toks.append(f"tab{nv}")
    return list(dict.fromkeys(toks))


class BM25:
    """Okapi BM25 over ``{doc_id: text}``; ``search`` returns ``[(doc_id, score)]``."""

    def __init__(self, docs: dict[Any, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {p: Counter(tokens(t)) for p, t in docs.items()}
        self.len = {p: sum(c.values()) for p, c in self.tf.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        df: Counter[str] = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Any, float]]:
        q = tokens(query)
        scores: dict[Any, float] = {}
        for p, c in self.tf.items():
            s = 0.0
            for t in q:
                if t in c:
                    tf = c[t]
                    s += (
                        self.idf.get(t, 0)
                        * tf
                        * (self.k1 + 1)
                        / (
                            tf
                            + self.k1 * (1 - self.b + self.b * self.len[p] / self.avg)
                        )
                    )
            if s > 0:
                scores[p] = s
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


# --------------------------------------------------------------------------- index helpers
def walk_sections(index: dict[str, Any]) -> list[dict[str, Any]]:
    """All sections of the tree, depth first, each with a ``depth`` key."""
    out: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], depth: int) -> None:
        for s in nodes:
            out.append({**s, "depth": depth})
            if s.get("subsections"):
                walk(s["subsections"], depth + 1)

    walk(index.get("sections", []), 0)
    return out


def sections_for_pages(index: dict[str, Any], pages: list[int]) -> list[str]:
    """Ids of every section whose page range covers one of ``pages`` (document order)."""
    pset = set(pages)
    return [
        s["id"]
        for s in walk_sections(index)
        if any(s["start_page"] <= p <= s["end_page"] for p in pset)
    ]


def build_page_corpus(
    index: dict[str, Any], pages_md: dict[int, str]
) -> dict[int, str]:
    """The lexical corpus shared by the locator and the navigator's search tool:
    captions (and the title) are repeated so that 'Figure N' queries rank the page
    that carries the caption first; in-text mentions stay at natural weight."""
    docs: dict[int, str] = {}
    for p_str, ps in index["pages"].items():
        p = int(p_str)
        figs = " ".join(ps.get("figures") or [])
        tabs = " ".join(ps.get("tables") or [])
        kw = " ".join(ps.get("keywords") or [])
        header_boost = f"{ps.get('title', '')} " * 3 + f"{figs} {tabs} " * 5
        docs[p] = f"{header_boost} {ps.get('summary', '')} {kw} {pages_md.get(p, '')}"
    return docs


def _ref_labels(ps: dict[str, Any], key: str) -> list[int]:
    """Figure/table numbers whose caption sits on this page."""
    out: list[int] = []
    for cap in ps.get(key) or []:
        m = REF_RE.match(cap.strip())
        if m:
            out.append(int(m.group(2)))
    return out


def section_visual_tags(index: dict[str, Any], section: dict[str, Any]) -> str:
    """'Figure 1 (p2); Tables 1-3 (p7-9)' for the pages of ``section`` (captions only)."""
    figs: list[tuple[int, int]] = []
    tabs: list[tuple[int, int]] = []
    for p in range(section["start_page"], section["end_page"] + 1):
        ps = index["pages"].get(str(p)) or {}
        figs += [(n, p) for n in _ref_labels(ps, "figures")]
        tabs += [(n, p) for n in _ref_labels(ps, "tables")]
    parts: list[str] = []
    for label, items in (("Figure", figs), ("Table", tabs)):
        if not items:
            continue
        seen: dict[int, int] = {}
        for n, p in sorted(items):
            seen.setdefault(n, p)
        parts.append(
            (f"{label}s " if len(seen) > 1 else f"{label} ")
            + ", ".join(f"{n} (p{p})" for n, p in seen.items())
        )
    return "; ".join(parts)


def enriched_outline(index: dict[str, Any]) -> str:
    """Section outline with the figures/tables each section holds, e.g.
    ``[S1] Introduction and Overview (pages 1-3) | Figure 1 (p2)``."""
    lines: list[str] = []
    for s in walk_sections(index):
        tags = section_visual_tags(index, s)
        lines.append(
            f"{'  ' * s['depth']}[{s['id']}] {s['title']} (pages {s['start_page']}-{s['end_page']})"
            + (f" | {tags}" if tags else "")
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- locator
@dataclass
class RefHit:
    ref: str  # "Figure 1"
    pages: list[int]
    via: str  # caption | mention | none
    captions: list[str] = field(default_factory=list)


class RegionLocator:
    """Finds the pages and sections that should hold the evidence for a question."""

    def __init__(self, index: dict[str, Any], pages_md: dict[int, str] | None = None):
        self.index = index
        self.sections = walk_sections(index)
        self.page_docs = build_page_corpus(index, pages_md or {})
        self.page_bm25 = BM25(self.page_docs)
        sec_docs: dict[str, str] = {}
        for s in self.sections:
            parts = [s["title"]] * 3
            for p in range(s["start_page"], s["end_page"] + 1):
                ps = index["pages"].get(str(p)) or {}
                parts.append(ps.get("title", ""))
                parts.append(ps.get("summary", ""))
                parts.append(" ".join(ps.get("keywords") or []))
                parts.append(
                    " ".join((ps.get("figures") or []) + (ps.get("tables") or []))
                )
            sec_docs[s["id"]] = " ".join(parts)
        self.section_bm25 = BM25(sec_docs)

    # ---- figure / table references
    def resolve_refs(self, text: str) -> list[RefHit]:
        hits: list[RefHit] = []
        seen: set[tuple[str, int]] = set()
        for kind, num in REF_RE.findall(text):
            key = "figures" if kind.lower().startswith("fig") else "tables"
            n = int(num)
            if (key, n) in seen:
                continue
            seen.add((key, n))
            label = f"{'Figure' if key == 'figures' else 'Table'} {n}"
            cap_pages: list[int] = []
            caps: list[str] = []
            for p_str, ps in self.index["pages"].items():
                if n in _ref_labels(ps, key):
                    cap_pages.append(int(p_str))
                    caps += [
                        c[:120]
                        for c in ps.get(key) or []
                        if (m := REF_RE.match(c.strip())) and int(m.group(2)) == n
                    ]
            if cap_pages:
                hits.append(RefHit(label, sorted(set(cap_pages)), "caption", caps))
                continue
            mention_pages = sorted(
                int(p_str)
                for p_str, ps in self.index["pages"].items()
                if n in (ps.get(f"mentions_{key}") or [])
            )
            hits.append(
                RefHit(label, mention_pages, "mention" if mention_pages else "none")
            )
        return hits

    # ---- lexical ranking
    def rank_pages(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        return [(int(p), s) for p, s in self.page_bm25.search(query, k=k)]

    def rank_sections(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        return [(str(s), v) for s, v in self.section_bm25.search(query, k=k)]

    # ---- combined
    def locate(
        self,
        question: str,
        queries: list[str] | None = None,
        k_pages: int = 5,
        k_sections: int = 3,
    ) -> dict[str, Any]:
        """Candidate regions for ``question`` (+ optional extra queries).

        Returns ``sections`` (ids, best first), ``anchor_pages`` (pages to open
        first), ``refs`` (resolved figure/table references), per-query hits and a
        human-readable ``explanation``.  Reference hits always come first: a
        'Figure 1' question is anchored on the page whose caption is 'Figure 1'."""
        qs = [q for q in [question, *(queries or [])] if q and q.strip()]
        qs = list(dict.fromkeys(q.strip() for q in qs))
        refs = self.resolve_refs(question)
        ref_pages = [p for r in refs for p in r.pages]
        ref_sections = sections_for_pages(self.index, ref_pages)

        page_score: dict[int, float] = {}
        sec_score: dict[str, float] = {}
        per_query: list[dict[str, Any]] = []
        for q in qs:
            ph = self.rank_pages(q, k=k_pages)
            sh = self.rank_sections(q, k=k_sections)
            top_p = ph[0][1] if ph else 1.0
            top_s = sh[0][1] if sh else 1.0
            for p, s in ph:
                page_score[p] = page_score.get(p, 0.0) + s / max(top_p, 1e-9)
            for sid, s in sh:
                sec_score[sid] = sec_score.get(sid, 0.0) + s / max(top_s, 1e-9)
            per_query.append(
                {
                    "query": q,
                    "pages": [{"page": p, "score": round(s, 2)} for p, s in ph],
                    "sections": [{"id": sid, "score": round(s, 2)} for sid, s in sh],
                }
            )
        ranked_pages = [p for p, _ in sorted(page_score.items(), key=lambda x: -x[1])]
        ranked_secs = [s for s, _ in sorted(sec_score.items(), key=lambda x: -x[1])]
        # sections also implied by the top lexical pages (summary-level search can
        # miss a page whose body text matches strongly)
        page_secs = sections_for_pages(self.index, ranked_pages[:3])

        sections = list(dict.fromkeys(ref_sections + ranked_secs + page_secs))[
            : max(k_sections, len(ref_sections))
        ]
        anchor_pages = list(dict.fromkeys(ref_pages + ranked_pages))[
            : max(4, len(ref_pages))
        ]

        def page_line(p: int) -> str:
            ps = self.index["pages"].get(str(p)) or {}
            caps = (ps.get("figures") or []) + (ps.get("tables") or [])
            tag = f" [{'; '.join(c[:60] for c in caps)}]" if caps else ""
            return f"p{p}: {ps.get('title', '')}{tag}"

        sec_title = {s["id"]: s for s in self.sections}
        lines = ["Resolved figure/table references:"]
        if refs:
            for r in refs:
                if r.via == "caption":
                    lines.append(
                        f"- {r.ref} is on page(s) {r.pages} (caption: "
                        f"{'; '.join(c[:80] for c in r.captions[:2])})"
                    )
                elif r.via == "mention":
                    lines.append(
                        f"- {r.ref}: no caption found; mentioned in the text of page(s) {r.pages}"
                    )
                else:
                    lines.append(f"- {r.ref}: not found in this document")
        else:
            lines.append("- none in the question")
        lines.append("Best sections (ranked; reference sections first):")
        for sid in sections:
            sec = sec_title.get(sid, {})
            why = (
                "holds the referenced figure/table"
                if sid in ref_sections
                else "lexical match"
            )
            lines.append(
                f"- {sid}: {sec.get('title', '')} (pages {sec.get('start_page')}-{sec.get('end_page')}) - {why}"
            )
        lines.append("Pages to open first (anchor pages):")
        lines += [f"- {page_line(p)}" for p in anchor_pages]
        return {
            "question": question,
            "queries": qs,
            "refs": [r.__dict__ for r in refs],
            "ref_sections": ref_sections,
            "sections": sections,
            "anchor_pages": anchor_pages,
            "ranked_pages": [
                {"page": p, "score": round(page_score[p], 3)} for p in ranked_pages[:8]
            ],
            "ranked_sections": [
                {"id": s, "score": round(sec_score[s], 3)} for s in ranked_secs
            ],
            "per_query": per_query,
            "explanation": "\n".join(lines),
        }
