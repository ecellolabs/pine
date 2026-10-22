"""Typed outputs of every agent in the pipeline (Pydantic models).  Pydantic AI
validates model replies against these and asks the model to retry on
validation errors, so downstream code only ever sees well-formed data."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, get_origin

from pydantic import BaseModel, BeforeValidator, Field, model_validator


def _as_text(value: Any) -> Any:
    """Models send a List answer as a JSON array and a number as a number where the schema
    says string; the text form is what downstream code (scorer, quote check) reads."""
    if isinstance(value, (list, tuple)):
        return json.dumps([str(v) for v in value], ensure_ascii=False)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


# a free-text model field: the JSON schema still says "string"
Text = Annotated[str, BeforeValidator(_as_text)]


class ToolArgs(BaseModel):
    """Base of every model reply. Some models (qwen3.5 via OpenRouter) send nested
    array arguments as a JSON-encoded *string* (``"sections": "[{...}]"``); Pydantic
    rejects that and the retry repeats it, so decode such strings before validating.
    A ``null`` for an optional list (``"subsections": null``) means "none"."""

    @model_validator(mode="before")
    @classmethod
    def _decode_json_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fixed = dict(data)
        for name, field in cls.model_fields.items():
            value = fixed.get(name)
            if value is None and name in fixed and not field.is_required():
                del fixed[name]
            elif (
                isinstance(value, str)
                and value.lstrip().startswith("[")
                and get_origin(field.annotation) is list
            ):
                try:
                    fixed[name] = json.loads(value)
                except ValueError:
                    pass
        return fixed


class PageSummary(ToolArgs):
    """Index node content for one page."""

    title: str = Field(description="Short descriptive title for the page, <= 12 words")
    summary: str = Field(
        description=(
            "1-3 sentences: what information the page contains (topics, entities, "
            "numbers, tables, figures, charts with their axes and key values)"
        )
    )
    keywords: list[str] = Field(default_factory=list, description="5-10 keywords")
    figures: list[str] = Field(
        default_factory=list,
        description="Verbatim figure captions and figure titles on this page (e.g. 'Figure 1: Tree construction process')",
    )
    tables: list[str] = Field(
        default_factory=list,
        description="Table titles, numbers or main headers on this page (e.g. 'Table 2: Ablation results')",
    )
    has_visual_elements: bool = Field(
        default=False,
        description="True if the page contains charts, diagrams, plots, tables, or figures",
    )


class Section(ToolArgs):
    """A contiguous page range of the document; may contain subsections."""

    title: str
    start_page: int = Field(description="First page of the section (1-indexed)")
    end_page: int = Field(description="Last page of the section (1-indexed, inclusive)")
    subsections: list[Section] = Field(default_factory=list)


class Hierarchy(ToolArgs):
    """Table of contents: top-level sections covering every page exactly once."""

    sections: list[Section]


class Plan(ToolArgs):
    """Planner output: how to attack the question."""

    question_type: str = Field(
        description="factoid | lookup | counting | comparison | list | reasoning"
    )
    expected_answer_format: str = Field(description="Int | Float | Str | List | None")
    needs_visual_inspection: bool = Field(
        description=(
            "true if the answer is likely inside a chart/figure/image/table image "
            "rather than plain text"
        )
    )
    sub_goals: list[str] = Field(description="Ordered concrete facts to find")
    search_queries: list[str] = Field(
        description="3-6 short keyword queries for page search"
    )
    candidate_sections: list[str] = Field(
        default_factory=list,
        description=(
            "Section ids most likely to contain the evidence, best first: take them "
            "from the locate_regions tool result"
        ),
    )
    anchor_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Pages (1-indexed) the navigator should open first, from the locate_regions "
            "tool result (e.g. the page carrying the referenced figure's caption)"
        ),
    )


class FinalAnswer(ToolArgs):
    """The orchestrator's final output (emitted through the final_answer tool)."""

    status: Literal["answered", "not_answerable"]
    answer: Text = Field(
        description=(
            "Concise answer only (number, short string, or list). For "
            "not_answerable write 'Not answerable'."
        )
    )
    evidence_pages: list[int] = Field(
        default_factory=list, description="1-indexed pages"
    )
    quote: str = Field(
        default="",
        description="Verbatim exact quotation copied from the page text or image inspection output supporting the answer",
    )
    quote_page: int | None = Field(
        default=None, description="Page number (1-indexed) supporting the quote"
    )
    reasoning: str = ""


class NoteRef(ToolArgs):
    """One note the answer rests on."""

    note: int = Field(description="Note number, e.g. 2 for [note 2]")
    quote: Text = Field(
        default="",
        description="Short phrase copied exactly from that note that supports the answer",
    )


class Decision(ToolArgs):
    """The procedural navigator's one decision after the candidate pages were read."""

    action: Literal["answer", "search", "abstain"] = Field(
        description=(
            "answer: the notes contain the answer; search: other pages may hold it; "
            "abstain: the notes show the document does not contain it"
        )
    )
    answer: Text = Field(
        default="",
        description="Concise answer only (number, short string, or list); empty unless action is answer",
    )
    evidence: list[NoteRef] = Field(
        default_factory=list,
        description="The notes the answer rests on (required when action is answer)",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2-4 short keyword queries for other pages (action search)",
    )
    reasoning: Text = ""


class Verdict(ToolArgs):
    """Verifier output."""

    verdict: Literal["supported", "not_supported", "insufficient"]
    quote_found: bool = Field(
        description="Is the quoted evidence actually present in the cited page content?"
    )
    explanation: str = Field(description="1-2 sentences")
    gaps: list[str] = Field(
        default_factory=list,
        description="What is still missing or what should be checked (sections/pages/figures)",
    )
    check_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Up to 3 pages from the index view, not read yet, that may hold missing or "
            "contradicting evidence (always consider for an abstention and for a list or "
            "count answer)"
        ),
    )
