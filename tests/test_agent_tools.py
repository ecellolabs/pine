"""Unit tests for the Pydantic AI agent layer. No network: the orchestrator is
driven by scripted ``FunctionModel``s (no real model is ever called)."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pipeline_v2.agent.common import AgentSettings, CostLedger
from pipeline_v2.agent.captions import captions_from_markdown
from pipeline_v2.agent.index import _repair, _validate_tree, run_index
from pipeline_v2.agent.orchestrator import BM25, TOOL_NAMES, run_orchestrator
from pipeline_v2.agent.planner import run_planner
from pipeline_v2.agent.verifier import decide, quote_in_text, run_verifier
from pipeline_v2.agent.visual_samples import build_visual_samples, discover_runs

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


def test_settings_public_hides_key(tmp_path: Path) -> None:
    s = AgentSettings(output_dir=tmp_path, api_key="secret")
    assert "secret" not in json.dumps(s.public())
    assert s.llm_cache_dir == tmp_path / ".cache" / "llm"
    assert s.public()["agent_framework"] == "pydantic-ai"


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
            "abstain_condition": "never",
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
    tool_name = info.output_tools[0].name if info.output_tools else "output"
    return ModelResponse(parts=[ToolCallPart(tool_name, payload)])


def _settings(tmp_path: Path, factory: Any) -> AgentSettings:
    return AgentSettings(
        output_dir=tmp_path, api_key="test", model_factory=factory, max_tool_calls=6
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
    assert (run_dir / "04_orchestrator" / "llm_calls.jsonl").exists()

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


def test_premature_abstention_is_gated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    def lazy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Abstains immediately; after the gate's retry it reads pages, then abstains."""
        n = sum(isinstance(m, ModelResponse) for m in messages)
        abstain = {
            "status": "not_answerable",
            "answer": "Not answerable",
            "evidence_pages": [],
            "reasoning": "nothing",
        }
        if n == 0:
            return ModelResponse(parts=[ToolCallPart("final_answer", abstain)])
        if n == 1:
            return ModelResponse(parts=[ToolCallPart("read_page", {"page": 1})])
        if n == 2:
            return ModelResponse(parts=[ToolCallPart("read_page", {"page": 2})])
        return ModelResponse(parts=[ToolCallPart("final_answer", abstain)])

    settings = _settings(tmp_path, lambda name: FunctionModel(lazy, model_name=name))
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
    assert final["pages_read"] == [1, 2]
    assert [a["type"] for a in final["anomalies"]] == ["PREMATURE_FINAL"]


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
