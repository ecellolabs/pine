"""Typed outputs of every agent in the pipeline (Pydantic models).  Pydantic AI
validates model replies against these and asks the model to retry on
validation errors, so downstream code only ever sees well-formed data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PageSummary(BaseModel):
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
        description="Verbatim figure captions and figure titles on this page (e.g. 'Figure 1: Tree construction process')"
    )
    tables: list[str] = Field(
        default_factory=list,
        description="Table titles, numbers or main headers on this page (e.g. 'Table 2: Ablation results')"
    )
    has_visual_elements: bool = Field(
        default=False,
        description="True if the page contains charts, diagrams, plots, tables, or figures"
    )


class Section(BaseModel):
    """A contiguous page range of the document; may contain subsections."""

    title: str
    start_page: int = Field(description="First page of the section (1-indexed)")
    end_page: int = Field(description="Last page of the section (1-indexed, inclusive)")
    subsections: list[Section] = Field(default_factory=list)


class Hierarchy(BaseModel):
    """Table of contents: top-level sections covering every page exactly once."""

    sections: list[Section]


class Plan(BaseModel):
    """Planner output: how to attack the question."""

    question_type: str = Field(
        description=(
            "factoid | lookup | counting | comparison | list | reasoning | "
            "possibly_unanswerable"
        )
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
        description="Section ids from the outline most likely to contain the evidence",
    )
    abstain_condition: str = Field(
        description="When the agent should answer 'Not answerable'"
    )


class FinalAnswer(BaseModel):
    """The orchestrator's final output (emitted through the final_answer tool)."""

    status: Literal["answered", "not_answerable"]
    answer: str = Field(
        description=(
            "Concise answer only (number, short string, or list). For "
            "not_answerable write 'Not answerable'."
        )
    )
    evidence_pages: list[int] = Field(
        default_factory=list, description="1-indexed pages"
    )
    reasoning: str = ""


class Verdict(BaseModel):
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
