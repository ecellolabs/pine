"""Unit tests for the Pydantic AI agent layer. No network: the orchestrator is
driven by scripted ``FunctionModel``s (no real model is ever called)."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

from pine.agent.captions import captions_from_markdown
from pine.agent.common import AgentSettings, CostLedger
from pine.agent.index import _repair, _validate_tree, run_index
from pine.agent.llm import traced_model
from pine.agent.orchestrator import BM25, TOOL_NAMES, run_orchestrator
from pine.agent.planner import run_planner
from pine.agent.schemas import Hierarchy
from pine.agent.verifier import decide, quote_in_text, run_verifier
from pine.agent.visual_samples import (
    build_visual_samples,
    discover_runs,
    resolve_runs_dir,
)

PAGE2 = (
    "## National Atmospheric Research Laboratory (NARL)\n"
    "NARL at Gadanki near Tirupati is a centre for atmospheric research."
)


def test_bm25_ranks_matching_page_first() -> None:
    docs = {
        1: "introduction to the report and methodology",
        2: "global ios breakdown pie chart ios 9 adoption",
        3: "android version breakdown lollipop",
    }
    hits = BM25(docs).search("ios breakdown pie chart")
    assert hits[0][0] == 2
    assert BM25(docs).search("zzz-not-present") == []


def test_quote_in_text_exact_and_fuzzy() -> None:
    found, sim = quote_in_text("NARL at Gadanki near Tirupati", PAGE2)
    assert found and sim == 1.0
    found, sim = quote_in_text("NARL at Gadanki near Tirupathi is a center", PAGE2)
    assert found and sim >= 0.8
    found, sim = quote_in_text("completely unrelated sentence about pie charts", PAGE2)
    assert not found


def test_validate_tree_and_repair() -> None:
    log = logging.getLogger("test")
    ok = [
        {"title": "A", "start_page": 1, "end_page": 2},
        {"title": "B", "start_page": 3, "end_page": 4},
    ]
    assert _validate_tree(ok, 4, log) == []
    bad = [
        {
            "title": "A",
            "start_page": 2,
            "end_page": 3,
            "subsections": [{"title": "A1", "start_page": 3, "end_page": 4}],
        },
        {"title": "B", "start_page": 3, "end_page": 4},
    ]
    problems = _validate_tree(bad, 4, log)
    assert any("escapes parent" in p for p in problems)
    assert any("not covered" in p and "1" in p for p in problems)
    assert any("more than one" in p for p in problems)
    repaired = _repair(bad, 4)
    assert _validate_tree(repaired, 4, log) == []
    assert repaired[0]["start_page"] == 1


def test_extract_visual_elements() -> None:
    sample_md = (
        "## Section Title\n\n"
        "Figure 1: Tree construction process in RAPTOR\n"
        "<!-- image -->\n\n"
        "Table 2: Ablation study results\n"
        "| Model | ANLS | Score |\n"
        "| --- | --- | --- |\n"
        "| Ours | 0.95 | 1.00 |\n"
        "As shown in Figure 3, results improve.\n"
    )
    figs, tabs, caption_lines = captions_from_markdown(sample_md)
    assert figs == ["Figure 1: Tree construction process in RAPTOR"]
    assert tabs == ["Table 2: Ablation study results"]
    assert len(caption_lines) == 2  # the in-text 'Figure 3' mention is not a caption


def test_decision_rule() -> None:
    assert decide("answered", "supported", True, True, 1, 2) == "ACCEPT"
    assert decide("answered", "supported", False, True, 1, 2) == "ACCEPT_WEAK_GROUNDING"
    assert decide("answered", "not_supported", False, True, 1, 2) == "REPLAN"
    assert decide("answered", "not_supported", False, True, 2, 2) == "ABSTAIN"
    assert decide("not_answerable", "supported", False, False, 1, 2) == "ACCEPT_ABSTAIN"
    # an abstention with open gaps is not accepted while rounds remain
    assert decide("not_answerable", "not_supported", False, False, 1, 2) == "REPLAN"
    assert (
        decide("not_answerable", "insufficient", False, False, 2, 2) == "ACCEPT_ABSTAIN"
    )
    assert decide("error", None, False, False, 1, 2) == "ABSTAIN"
    # a crashed verifier (verdict None) must not reject a quote-checked answer
    assert decide("answered", None, True, True, 2, 2) == "ACCEPT_WEAK_GROUNDING"
    assert decide("answered", None, False, True, 2, 2) == "ABSTAIN"
    assert decide("answered", None, False, True, 1, 2) == "REPLAN"


def test_settings_public_hides_key(tmp_path: Path) -> None:
    s = AgentSettings(output_dir=tmp_path, api_key="secret")
    assert "secret" not in json.dumps(s.public())
    assert s.llm_cache_dir == tmp_path / ".cache" / "llm"
    assert s.public()["agent_framework"] == "pydantic-ai"
    assert s.public()["reasoning"] is True
    assert s.public()["max_tokens_scale"] == 1.0


# fixtures
def _write_parsed_doc(run_dir: Path) -> tuple[Path, Path]:
    """A fake 2-page parsed document (01_parser) and index (02_index)."""
    parser_dir = run_dir / "01_parser"
    (parser_dir / "pages").mkdir(parents=True)
    (parser_dir / "pages" / "page_001.md").write_text(
        "# Table of contents\nIntro page."
    )
    (parser_dir / "pages" / "page_002.md").write_text(PAGE2)
    stats = {
        "pages": 2,
        "docling_status": "SUCCESS",
        "elapsed_s": 1.0,
        "ocr_engine": "x",
        "pages_flagged": 0,
        "total_docling_chars": 100,
        "total_pymupdf_chars": 100,
        "total_tables": 0,
        "total_pictures": 0,
        "page_stats": [
            {
                "page": p,
                "docling_chars": 60,
                "pymupdf_chars": 60,
                "tables": 0,
                "pictures": 0,
                "headings": 1,
                "flag": None,
            }
            for p in (1, 2)
        ],
    }
    (parser_dir / "page_stats.json").write_text(json.dumps(stats))
    index_dir = run_dir / "02_index"
    index_dir.mkdir()
    index = {
        "doc": "doc.pdf",
        "doc_title": "Test doc",
        "n_pages": 2,
        "sections": [{"id": "S1", "title": "All", "start_page": 1, "end_page": 2}],
        "pages": {
            "1": {
                "page": 1,
                "title": "Table of contents",
                "summary": "intro",
                "keywords": [],
                "source": "text",
            },
            "2": {
                "page": 2,
                "title": "NARL",
                "summary": "about NARL",
                "keywords": ["NARL"],
                "source": "text",
            },
        },
    }
    (index_dir / "index.json").write_text(json.dumps(index))
    (index_dir / "outline.md").write_text(
        "- [S1] All (pages 1-2)\n  - p1: Table of contents\n  - p2: NARL\n"
    )
    return parser_dir, index_dir


def _scripted_navigator(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """search -> read page 2 -> record a bad quote (rejected) -> record a good quote -> final."""
    assert [t.name for t in info.function_tools] == TOOL_NAMES[:-1]
    assert [t.name for t in info.output_tools] == ["final_answer"]
    n = sum(isinstance(m, ModelResponse) for m in messages)
    if n == 0:
        return ModelResponse(parts=[ToolCallPart("search_pages", {"query": "NARL"})])
    if n == 1:
        return ModelResponse(parts=[ToolCallPart("read_page", {"page": 2})])
    if n == 2:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "record_evidence",
                    {
                        "page": 2,
                        "quote": "this sentence is not on the page",
                        "candidate_answer": "x",
                    },
                )
            ]
        )
    if n == 3:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "record_evidence",
                    {
                        "page": 2,
                        "quote": "National Atmospheric Research Laboratory (NARL)",
                        "candidate_answer": "National Atmospheric Research Laboratory",
                    },
                )
            ]
        )
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_answer",
                {
                    "status": "answered",
                    "answer": "National Atmospheric Research Laboratory",
                    "evidence_pages": [2],
                    "reasoning": "page 2 defines NARL",
                },
            )
        ]
    )


def _prompted_stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Schema-aware stand-in for the text model in prompted-output agents."""
    name = info.output_tools[0].name if info.output_tools else ""
    retries = sum(
        isinstance(part, RetryPromptPart)
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
    )
    if name == "PageSummary":
        payload: dict[str, Any] = {
            "title": "Stub title",
            "summary": "Stub summary.",
            "keywords": ["stub"],
        }
    elif name == "Hierarchy":
        if retries == 0:  # first attempt is invalid (page 2 uncovered) -> ModelRetry
            payload = {"sections": [{"title": "A", "start_page": 1, "end_page": 1}]}
        else:
            payload = {"sections": [{"title": "All", "start_page": 1, "end_page": 2}]}
    elif name == "Plan":
        payload = {
            "question_type": "lookup",
            "expected_answer_format": "Str",
            "needs_visual_inspection": False,
            "sub_goals": ["find NARL"],
            "search_queries": ["NARL"],
            "candidate_sections": ["S1"],
        }
    elif name == "Verdict":
        payload = {
            "verdict": "supported",
            "quote_found": True,
            "explanation": "The page defines NARL.",
            "gaps": [],
        }
    else:
        payload = {
            "status": "not_answerable",
            "answer": "Not answerable",
            "evidence_pages": [],
            "reasoning": "stub",
        }
    if not info.output_tools:
        if not name:
            payload = {
                "title": "Stub title",
                "summary": "Stub summary.",
                "keywords": ["stub"],
            }
        return ModelResponse(parts=[TextPart(json.dumps(payload))])
    tool_name = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(tool_name, payload)])


def _settings(tmp_path: Path, factory: Any) -> AgentSettings:
    return AgentSettings(
        output_dir=tmp_path,
        api_key="test",
        model_factory=factory,
        max_tool_calls=6,
        navigator="agentic",
    )


def test_orchestrator_with_scripted_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())
    settings = _settings(
        tmp_path, lambda name: FunctionModel(_scripted_navigator, model_name=name)
    )
    ledger = CostLedger(run_dir)
    pdf = run_dir / "doc.pdf"
    pdf.write_bytes(
        b"%PDF-1.4 fake"
    )  # never rendered: the script does not inspect images
    plan = {
        "question_type": "lookup",
        "sub_goals": ["find NARL"],
        "search_queries": ["NARL"],
        "candidate_sections": ["S1"],
    }
    final = run_orchestrator(
        settings,
        "What is NARL?",
        pdf,
        index,
        plan,
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        ledger,
    )
    assert final["status"] == "answered"
    assert final["answer"] == "National Atmospheric Research Laboratory"
    assert final["evidence_pages"] == [2]
    assert final["pages_read"] == [2]
    assert (
        final["tool_calls_used"] == 4
    )  # search, read, rejected record, accepted record
    assert [e["source"] for e in final["ledger"]] == ["page_text"]
    assert [a["type"] for a in final["anomalies"]] == ["QUOTE_REJECTED"]
    assert ledger.calls == 5  # five model responses, all traced
    trace = (
        (run_dir / "04_orchestrator" / "trace_round1.jsonl").read_text().splitlines()
    )
    events = [json.loads(line)["event"] for line in trace]
    assert events[0] == "start" and events[-1] == "final"
    assert events.count("tool_result") == 4
    results = run_dir / "04_orchestrator" / "tool_results"
    assert (results / "round1_call02_read_page.txt").exists()
    calls = [
        json.loads(line)
        for line in (run_dir / "04_orchestrator" / "llm_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert calls and all(c["round"] == 1 for c in calls)
    assert all("system" in {m["role"] for m in c["request"]["messages"]} for c in calls)

    # verifier on top of this final (stubbed verdict: supported)
    ver_settings = _settings(
        tmp_path, lambda name: FunctionModel(_prompted_stub, model_name=name)
    )
    verification = run_verifier(
        ver_settings,
        "What is NARL?",
        final,
        parser_dir,
        index,
        run_dir / "05_evidence_verifier",
        run_dir / "run.log",
        ledger,
        1,
        2,
    )
    assert verification["any_quote_verified"] is True
    assert verification["llm_verdict"]["verdict"] == "supported"
    assert verification["decision"] == "ACCEPT"
    assert verification["final_answer"] == "National Atmospheric Research Laboratory"


def test_budget_exhaustion_forces_final(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    def looping(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:  # navigator: never finishes
            return ModelResponse(parts=[ToolCallPart("get_outline", {})])
        # forced-final agent (prompted output, no function tools)
        payload = {
            "status": "not_answerable",
            "answer": "Not answerable",
            "evidence_pages": [],
            "reasoning": "budget",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="t",
        model_factory=lambda name: FunctionModel(looping, model_name=name),
        max_tool_calls=3,
        navigator="agentic",
    )
    final = run_orchestrator(
        settings,
        "Q?",
        run_dir / "doc.pdf",
        index,
        {},
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    assert final["salvaged"] is True
    assert final["status"] == "not_answerable"
    kinds = {a["type"] for a in final["anomalies"]}
    assert "BUDGET_EXHAUSTED" in kinds and "REPEATED_CALL" in kinds
    assert final["tool_calls_used"] <= 3


def test_abstention_is_not_gated_but_answers_need_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())
    abstain = {
        "status": "not_answerable",
        "answer": "Not answerable",
        "evidence_pages": [2],
        "reasoning": "figure has no colours",
    }

    def abstain_after_one_look(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        n = sum(isinstance(m, ModelResponse) for m in messages)
        if n == 0:
            return ModelResponse(parts=[ToolCallPart("read_page", {"page": 2})])
        return ModelResponse(parts=[ToolCallPart("final_answer", abstain)])

    settings = _settings(
        tmp_path, lambda name: FunctionModel(abstain_after_one_look, model_name=name)
    )
    final = run_orchestrator(
        settings,
        "Q?",
        run_dir / "doc.pdf",
        index,
        {},
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    assert final["status"] == "not_answerable" and final["salvaged"] is False
    assert final["pages_read"] == [2] and final["evidence_pages"] == [2]
    assert final["anomalies"] == []

    def answer_without_evidence(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        answered = {"status": "answered", "answer": "red", "evidence_pages": [2]}
        return ModelResponse(parts=[ToolCallPart("final_answer", answered)])

    settings = _settings(
        tmp_path, lambda name: FunctionModel(answer_without_evidence, model_name=name)
    )
    final = run_orchestrator(
        settings,
        "Q2 (distinct prompt so the response cache cannot replay run 1)?",
        run_dir / "doc.pdf",
        index,
        {},
        parser_dir,
        run_dir / "04_orchestrator_b",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    kinds = [a["type"] for a in final["anomalies"]]
    assert kinds.count("FINAL_WITHOUT_EVIDENCE") >= 1 and "MODEL_BEHAVIOR" in kinds
    assert final["salvaged"] is True


def test_inline_final_answer_quote_must_exist_on_the_page(tmp_path: Path) -> None:
    """quote_in_text returns (found, similarity); a tuple is always truthy, so an
    invented inline quote used to be recorded as page_text evidence."""
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    def invented_quote(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {
                        "status": "answered",
                        "answer": "red",
                        "evidence_pages": [2],
                        "quote": "The sky over Gadanki is permanently crimson.",
                        "quote_page": 2,
                    },
                )
            ]
        )

    settings = _settings(
        tmp_path, lambda name: FunctionModel(invented_quote, model_name=name)
    )
    final = run_orchestrator(
        settings,
        "Q3 (distinct prompt so the response cache cannot replay earlier runs)?",
        run_dir / "doc.pdf",
        index,
        {},
        parser_dir,
        run_dir / "04_orchestrator_c",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    assert final["salvaged"] is True and not final["ledger"]


def test_json_encoded_array_arguments_are_decoded() -> None:
    """qwen3.5 sends nested array arguments as a JSON string; that must validate."""
    sections = json.dumps(
        [
            {"title": "A", "start_page": 1, "end_page": 2, "subsections": []},
            {
                "title": "B",
                "start_page": 3,
                "end_page": 4,
                "subsections": json.dumps(
                    [{"title": "B1", "start_page": 3, "end_page": 4}]
                ),
            },
        ]
    )
    h = Hierarchy.model_validate({"sections": sections})
    assert [s.title for s in h.sections] == ["A", "B"]
    assert h.sections[1].subsections[0].title == "B1"
    # "subsections": null means no subsections
    h = Hierarchy.model_validate(
        {
            "sections": [
                {"title": "A", "start_page": 1, "end_page": 2, "subsections": None}
            ]
        }
    )
    assert h.sections[0].subsections == []
    # a string that is not JSON still fails validation instead of being swallowed
    with pytest.raises(ValidationError):
        Hierarchy.model_validate({"sections": "[not json"})


def test_planner_and_index_with_stub_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, index_dir = _write_parsed_doc(run_dir)
    (index_dir / "index.json").unlink()  # force a rebuild
    settings = _settings(
        tmp_path, lambda name: FunctionModel(_prompted_stub, model_name=name)
    )
    ledger = CostLedger(run_dir)
    pdf = run_dir / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    index = run_index(settings, pdf, parser_dir, index_dir, run_dir / "run.log", ledger)
    assert index["n_pages"] == 2
    assert set(index["pages"]) == {"1", "2"}
    assert index["pages"]["1"]["title"] == "Stub title"
    # first hierarchy attempt was invalid -> ModelRetry -> second attempt valid
    assert [a["problems"] == [] for a in index["hierarchy_attempts"]] == [False, True]
    assert index["hierarchy_fallback_used"] is False
    assert index["validation_problems_final"] == []
    assert index["sections"][0]["id"] == "S1"
    assert (index_dir / "outline.md").exists()
    plan = run_planner(
        settings,
        "What is NARL?",
        index,
        run_dir / "03_planner",
        run_dir / "run.log",
        ledger,
    )
    assert plan["_error"] is None
    assert plan["sub_goals"] == ["find NARL"]
    assert (run_dir / "03_planner" / "plan_round1.json").exists()
    assert ledger.calls == 5  # 2 page summaries + 2 hierarchy attempts + plan


#  visual samples
def _fake_run(root: Path, run_id: str) -> None:
    d = root / run_id
    (d / "00_input").mkdir(parents=True)
    (d / "00_input" / "question.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "doc_id": "doc.pdf",
                "question": "Q?",
                "why_chosen": "test",
                "gold": {
                    "answer": "A",
                    "answer_format": "Str",
                    "evidence_pages": [1],
                    "evidence_sources": [],
                },
                "text_model": "t",
                "vision_model": "v",
            }
        )
    )
    _write_parsed_doc(d)
    for sub in (
        "03_planner",
        "04_orchestrator",
        "05_evidence_verifier",
        "06_evaluation",
    ):
        (d / sub).mkdir()


def test_visual_samples_html_is_self_contained_and_capped(tmp_path: Path) -> None:
    for name in ("run_b", "run_a", "run_c"):
        _fake_run(tmp_path, name)
    assert [p.name for p in discover_runs(tmp_path)] == ["run_a", "run_b", "run_c"]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"runs": [{"run_id": "run_c"}, {"run_id": "run_a"}, {"run_id": "run_b"}]}
        )
    )
    assert [p.name for p in discover_runs(tmp_path)] == ["run_c", "run_a", "run_b"]
    # a run listed twice in the manifest (re-run appended) is shown once
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "run_c"},
                    {"run_id": "run_c"},
                    {"run_id": "run_a"},
                    {"run_id": "run_b"},
                ]
            }
        )
    )
    assert [p.name for p in discover_runs(tmp_path)] == ["run_c", "run_a", "run_b"]
    out = build_visual_samples(tmp_path)
    html = out.read_text()
    assert out.name == "visual_samples.html"
    assert "run_a" in html and "run_b" in html and "run_c" in html
    assert "<script src=" not in html and 'href="http' not in html
    capped = build_visual_samples(tmp_path, tmp_path / "two.html", max_runs=2)
    html2 = capped.read_text()
    assert '"run_c"' in html2 and '"run_a"' in html2 and '"run_b"' not in html2
    with pytest.raises(FileNotFoundError):
        build_visual_samples(tmp_path / "empty")


def test_visual_samples_resolves_newest_batch_folder(tmp_path: Path) -> None:
    root = tmp_path / "agent_runs"
    older = root / "s0_2026-10-07"
    newer = root / "s0_2026-10-08"
    for batch in (older, newer):
        _fake_run(batch, "run_x")
        (batch / "manifest.json").write_text(
            json.dumps({"batch": batch.name, "runs": [{"run_id": "run_x"}]})
        )
    os.utime(older / "manifest.json", (1, 1))
    assert resolve_runs_dir(root) == newer
    assert resolve_runs_dir(newer) == newer
    out = build_visual_samples(root)
    assert out == newer / "visual_samples.html"
    assert '"batch": "s0_2026-10-08"' in out.read_text()


def test_batch_dir_naming(tmp_path: Path) -> None:
    from pine.agent.common import batch_label, new_batch_dir
    from pine.sampling import resolve_max_samples

    assert batch_label(resolve_max_samples("s0")) == "s0"
    assert batch_label(3) == "first3"
    assert batch_label(None) == "full"
    first = new_batch_dir(tmp_path, "s0", date="2026-10-08")
    second = new_batch_dir(tmp_path, "s0", date="2026-10-08")
    third = new_batch_dir(tmp_path, "s0", date="2026-10-08")
    assert [first.name, second.name, third.name] == [
        "s0_2026-10-08",
        "s0_2026-10-08_2",
        "s0_2026-10-08_3",
    ]
    assert first.is_dir() and third.is_dir()


#  locator (planner candidate regions)
def _raptor_like_index() -> dict[str, Any]:
    pages = {
        "1": {
            "title": "Introduction",
            "summary": "RAPTOR clusters chunks of text.",
            "keywords": ["raptor", "clusters"],
            "figures": [],
            "tables": [],
            "mentions_figures": [1],
            "mentions_tables": [],
        },
        "2": {
            "title": "Tree construction",
            "summary": "How the tree is built.",
            "keywords": ["tree"],
            "figures": [
                "Figure 1: Tree construction process: RAPTOR recursively clusters chunks of text"
            ],
            "tables": [],
            "mentions_figures": [],
            "mentions_tables": [],
        },
        "3": {
            "title": "Related work",
            "summary": "Prior retrieval systems.",
            "keywords": ["retrieval"],
            "figures": [],
            "tables": [],
            "mentions_figures": [],
            "mentions_tables": [],
        },
        "4": {
            "title": "Text clustering and summarization",
            "summary": "Clustering with GMMs; nodes and clusters.",
            "keywords": ["clustering", "nodes", "clusters"],
            "figures": ["Figure 2: Clustering"],
            "tables": [],
            "mentions_figures": [2],
            "mentions_tables": [],
        },
        "5": {
            "title": "Retrieval mechanisms",
            "summary": "Tree traversal and collapsed tree.",
            "keywords": ["retrieval"],
            "figures": [],
            "tables": ["Table 1: NarrativeQA results"],
            "mentions_figures": [],
            "mentions_tables": [1, 5],
        },
    }
    return {
        "doc_title": "RAPTOR",
        "n_pages": 5,
        "sections": [
            {
                "id": "S1",
                "title": "Introduction and Overview",
                "start_page": 1,
                "end_page": 3,
                "subsections": [],
            },
            {
                "id": "S2",
                "title": "Text Clustering and Summarization",
                "start_page": 4,
                "end_page": 4,
                "subsections": [],
            },
            {
                "id": "S3",
                "title": "Retrieval Mechanisms",
                "start_page": 5,
                "end_page": 5,
                "subsections": [],
            },
        ],
        "pages": pages,
    }


def test_locator_resolves_figure_reference_to_caption_page() -> None:
    from pine.agent.locate import RegionLocator, enriched_outline

    loc = RegionLocator(
        _raptor_like_index(),
        {
            1: "As shown in Figure 1, RAPTOR clusters",
            2: "Figure 1: Tree construction process",
        },
    )
    q = "In Figure 1's demonstration, what are the color of the nodes that appear in all three clusters?"
    refs = loc.resolve_refs(q)
    assert [(r.ref, r.pages, r.via) for r in refs] == [("Figure 1", [2], "caption")]
    out = loc.locate(q, ["Figure 1", "clusters", "node colors"])
    assert (
        out["sections"][0] == "S1"
    )  # the caption page's section, not the title match (S2)
    assert out["anchor_pages"][0] == 2
    assert "Figure 1 is on page(s) [2]" in out["explanation"]
    # mention fallback when no caption carries the number, and 'not found' otherwise
    assert [(r.pages, r.via) for r in loc.resolve_refs("Table 5")] == [([5], "mention")]
    assert loc.resolve_refs("Fig. 9")[0].via == "none"
    # a page named in the question is the first anchor (a page beyond the document is not)
    named = loc.locate("What is in the table on page 3 and on page 99?")
    assert named["anchor_pages"][0] == 3 and 99 not in named["anchor_pages"]
    assert "page 3 is named in the question" in named["explanation"]
    # free-text ranking still works and the outline shows captions per section
    assert loc.rank_sections("clustering GMM nodes")[0][0] == "S2"
    assert (
        "[S1] Introduction and Overview (pages 1-3) | Figure 1 (p2)"
        in enriched_outline(_raptor_like_index())
    )
    assert "Table 1 (p5)" in enriched_outline(_raptor_like_index())


def test_planner_merges_locator_even_when_model_skips_tool(tmp_path: Path) -> None:
    from pine.agent.locate import RegionLocator
    from pine.agent.planner import _merge_locator

    index = _raptor_like_index()
    loc = RegionLocator(index).locate("What does Figure 1 show?", ["Figure 1"])
    plan = {
        "candidate_sections": ["S2", "S3"],
        "anchor_pages": [],
        "search_queries": ["Figure 1"],
    }
    merged = _merge_locator(plan, loc)
    assert (
        merged["candidate_sections"][0] == "S1" and "S2" in merged["candidate_sections"]
    )
    assert merged["anchor_pages"][0] == 2
    assert merged["locator"]["ref_sections"] == ["S1"]

    # end to end with a stub model that answers without calling the tool
    settings = _settings(
        tmp_path, lambda name: FunctionModel(_prompted_stub, model_name=name)
    )
    plan2 = run_planner(
        settings,
        "What does Figure 1 show?",
        index,
        tmp_path / "03_planner",
        tmp_path / "run.log",
        CostLedger(tmp_path),
    )
    assert plan2["_error"] is None
    assert plan2["candidate_sections"][0] == "S1" and plan2["anchor_pages"][0] == 2
    assert plan2["locator"]["tool_called"] is False
    assert (tmp_path / "03_planner" / "locate_round1.json").exists()


def test_claude_models_do_not_receive_temperature(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    settings = _settings(tmp_path, lambda name: FunctionModel(record, model_name=name))
    for name in ("claude-opus-5-5", "qwen/qwen3.5-9b"):
        step_dir = tmp_path / name.replace("/", "_")
        step_dir.mkdir()
        model = traced_model(
            settings,
            name,
            step_dir=step_dir,
            ledger=CostLedger(tmp_path),
            logger=logging.getLogger("t"),
            step_name="t",
        )
        original = model.wrapped.request

        async def spy(messages, model_settings, params, _o=original):  # type: ignore[no-untyped-def]
            seen.append(dict(model_settings or {}))
            return await _o(messages, model_settings, params)

        model.wrapped.request = spy  # type: ignore[method-assign]
        Agent(
            model, model_settings=ModelSettings(temperature=0.0, max_tokens=50)
        ).run_sync("hi")
    assert "temperature" not in seen[0] and seen[1].get("temperature") == 0.0


def test_api_keys_are_masked_in_logged_command() -> None:
    from pine.agent.common import redacted_argv

    cmd = redacted_argv(
        [
            "02_agent_qa.py",
            "--api-key",
            "sk-secret",
            "--vision-api-key=sk-other",
            "--x",
            "1",
        ]
    )
    assert "sk-secret" not in cmd and "sk-other" not in cmd
    assert cmd == "02_agent_qa.py --api-key *** --vision-api-key=*** --x 1"


def test_final_answer_accepts_list_and_number_answers() -> None:
    from pine.agent.schemas import FinalAnswer

    as_list = FinalAnswer(status="answered", answer=["7", "5", 2])  # type: ignore[arg-type]
    as_int = FinalAnswer(status="answered", answer=5)  # type: ignore[arg-type]
    assert json.loads(as_list.answer) == ["7", "5", "2"] and as_int.answer == "5"


def test_answer_numbers_must_follow_from_the_quotes() -> None:
    from pine.agent.verifier import numeric_check

    quotes = ["Foreign Companies: 32.17%, Foreign Institutional Investors: 12.79%"]
    assert numeric_check("54.96", quotes)["status"] == "mismatch"
    assert numeric_check("44.96%", quotes)["status"] == "ok"
    assert numeric_check("32%", ["Hispanic | 16", "White | 16"])["status"] == "ok"
    # a count of listed items cannot be checked against the quote's numbers
    assert (
        numeric_check("7", ["- Eusoff Hall\n- Raffles Hall"])["status"] == "unchecked"
    )


def test_rejoined_list_quote_is_found_piece_by_piece() -> None:
    from pine.agent.verifier import quote_on_page

    page = "Metrics\n- Followers\n- Tweet Clicks\n- Video Views"
    assert quote_on_page("Followers, Tweet Clicks, Video Views", page)[0]
    assert not quote_on_page("Followers, Tweet Clicks, Engagement Rate", page)[0]


def test_procedural_navigator_reads_anchors_and_cites_notes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    def decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # the answer cites note 1 (page 2) with a phrase that is not on the page
        args = {
            "action": "answer",
            "answer": "National Atmospheric Research Laboratory",
            "evidence": [{"note": 1, "quote": "NARL is in Delhi"}, {"note": 9}],
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="test",
        model_factory=lambda name: FunctionModel(decide, model_name=name),
    )
    plan = {
        "anchor_pages": [2],
        "needs_visual_inspection": False,
        "sub_goals": ["NARL"],
    }
    final = run_orchestrator(
        settings,
        "What is NARL?",
        run_dir / "doc.pdf",
        index,
        plan,
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    assert final["status"] == "answered" and final["pages_read"] == [2]
    assert final["evidence_pages"] == [2] and final["anomalies"] == []
    # the unknown note 9 is dropped; the invented phrase is replaced by the note's text
    [entry] = final["ledger"]
    assert entry["source"] == "page_text" and entry["quote"] in PAGE2


def test_replan_needs_a_page_not_yet_covered() -> None:
    from pine.agent.verifier import replan_has_new_target

    assert replan_has_new_target(["inspect the chart on page 15"], {15}, set(), 21)
    assert not replan_has_new_target(["inspect the chart on page 15"], {15}, {15}, 21)
    assert not replan_has_new_target(["Verify the exact spelling"], {1}, set(), 21)
    assert decide("answered", "not_supported", True, True, 1, 2, False) == "ABSTAIN"


def test_vision_readings_are_checked_against_the_text_layer() -> None:
    from pine.agent.verifier import check_vision_evidence, text_corroboration

    page = (
        "Categories of Shareholders " + "x " * 120 + "Foreign Companies 32.17 FII 12.79"
    )
    quote = "Foreign Companies held 32.17%, Foreign Institutional Investors 12.79%"
    # a sum of two values that stand in the page text is confirmed without a model call
    assert text_corroboration("44.96", quote, page)["status"] == "confirmed"
    assert text_corroboration("54.96", quote, page)["status"] == "not_in_text"
    # a reading the text cannot back is reported, never disputed (no second model call)
    final = {"answer": "54.96", "ledger": [{"page": 3, "quote": quote}]}
    checks = check_vision_evidence(
        final, [{"entry": 1, "page": 3, "source": "image_inspection"}], {3: page}
    )
    assert [(c["page"], c["status"]) for c in checks] == [(3, "vision_only")]


def test_list_answer_gets_a_coverage_round_on_the_next_page(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())
    quote = "National Atmospheric Research Laboratory (NARL)"
    final = {
        "status": "answered",
        "answer": '["NARL"]',
        "evidence_pages": [2],
        "ledger": [
            {"page": 2, "quote": quote, "candidate_answer": "x", "source": "page_text"}
        ],
        "pages_read": [2],
        "image_inspections": [],
        "tool_calls_used": 1,
    }
    settings = _settings(
        tmp_path, lambda name: FunctionModel(_prompted_stub, model_name=name)
    )
    v = run_verifier(
        settings,
        "Which labs?",
        final,
        parser_dir,
        index,
        run_dir / "05_evidence_verifier",
        run_dir / "run.log",
        CostLedger(run_dir),
        1,
        2,
    )
    assert v["decision_before_coverage"] == "ACCEPT" and v["decision"] == "REPLAN"
    assert v["coverage_pages"] == [1] and v["check_pages"] == [1]


def test_vision_evidence_is_not_relabelled_as_page_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())
    # the phrase stands in the page text, but the cited note was a vision reading
    entry = {
        "page": 2,
        "quote": "NARL",
        "candidate_answer": "x",
        "source": "image_inspection",
    }
    final = {
        "status": "answered",
        "answer": "x",
        "evidence_pages": [2],
        "ledger": [entry],
        "pages_read": [2],
        "image_inspections": [{"page": 2, "result": "a chart of rainfall"}],
        "tool_calls_used": 2,
    }
    settings = _settings(
        tmp_path, lambda name: FunctionModel(_prompted_stub, model_name=name)
    )
    v = run_verifier(
        settings,
        "q",
        final,
        parser_dir,
        index,
        run_dir / "05_evidence_verifier",
        run_dir / "run.log",
        CostLedger(run_dir),
        2,
        2,
    )
    assert v["quote_checks"][0]["source"] is None


def test_reasoning_roles_think_while_other_roles_do_not(tmp_path: Path) -> None:
    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="t",
        reasoning=False,
        reasoning_roles=["navigator"],
        model_factory=lambda name: FunctionModel(_prompted_stub, model_name=name),
    )
    kw = {
        "step_dir": tmp_path,
        "ledger": CostLedger(tmp_path),
        "logger": logging.getLogger("t"),
    }
    nav = traced_model(settings, "m", step_name="04", role="navigator", **kw)
    ver = traced_model(settings, "m", step_name="05", role="verifier", **kw)
    assert nav.run_settings.reasoning and nav.run_settings.thinking_sampling
    assert nav.run_settings.max_tokens_scale > 1
    assert not ver.run_settings.reasoning and not ver.run_settings.thinking_sampling


def test_lean_index_builds_from_the_parse_and_ranks_by_passage(tmp_path: Path) -> None:
    from pine.agent.lean_index import run_lean_index
    from pine.agent.locate import RegionLocator

    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    settings = AgentSettings(output_dir=tmp_path, api_key="t")
    index = run_lean_index(
        settings,
        run_dir / "doc.pdf",
        parser_dir,
        run_dir / "02_lean",
        run_dir / "run.log",
        CostLedger(run_dir),
    )
    assert index["index_kind"] == "lean" and index["visual_calls"] == 0
    covered = [
        p for s in index["sections"] for p in range(s["start_page"], s["end_page"] + 1)
    ]
    assert covered == [1, 2]
    pages_md = {
        p: (parser_dir / "pages" / f"page_{p:03d}.md").read_text() for p in (1, 2)
    }
    assert (
        RegionLocator(index, pages_md).rank_pages("NARL atmospheric research")[0][0]
        == 2
    )
