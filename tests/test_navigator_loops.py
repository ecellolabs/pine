import json
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from test_agent_tools import _write_parsed_doc

from pine.agent.common import AgentSettings, CostLedger
from pine.agent.orchestrator import run_orchestrator


def _streamlined_scripted_navigator(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Turn 1: batch read pages [1, 2] -> Turn 2: final answer with inline quote directly (2 turns total)."""
    n = sum(isinstance(m, ModelResponse) for m in messages)
    if n == 0:
        return ModelResponse(parts=[ToolCallPart("read_page", {"page": [1, 2]})])
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_answer",
                {
                    "status": "answered",
                    "answer": "National Atmospheric Research Laboratory",
                    "evidence_pages": [2],
                    "quote": "National Atmospheric Research Laboratory (NARL)",
                    "quote_page": 2,
                    "reasoning": "page 2 defines NARL",
                },
            )
        ]
    )


def test_batch_read_pages_and_inline_final_answer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="test",
        model_factory=lambda name: FunctionModel(
            _streamlined_scripted_navigator, model_name=name
        ),
        max_tool_calls=6,
    )
    ledger = CostLedger(run_dir)
    pdf = run_dir / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    plan = {
        "candidate_sections": ["S1"],
        "anchor_pages": [1, 2],
        "search_queries": ["NARL"],
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

    # 1. Answer is returned correctly
    assert final["status"] == "answered"
    assert final["answer"] == "National Atmospheric Research Laboratory"

    # 2. Batch read_page succeeded and recorded both pages
    assert 1 in final["pages_read"] and 2 in final["pages_read"]

    # 3. Evidence was auto-recorded from final_answer in ledger without a separate record_evidence turn
    assert len(final["ledger"]) == 1
    assert final["ledger"][0]["page"] == 2
    assert (
        final["ledger"][0]["quote"] == "National Atmospheric Research Laboratory (NARL)"
    )

    # 4. Total navigator tool calls recorded was reduced to 1 call!
    assert final["tool_calls_used"] == 1


def _zero_call_abstain_scripted_navigator(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Turn 0 tries to abstain -> effort_gate bounces it -> Turn 1 reads page -> Turn 2 abstains."""
    n = sum(isinstance(m, ModelResponse) for m in messages)
    if n == 0:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {"status": "not_answerable", "answer": "Not answerable"},
                )
            ]
        )
    if n == 1:
        return ModelResponse(parts=[ToolCallPart("read_page", {"page": [1]})])
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_answer", {"status": "not_answerable", "answer": "Not answerable"}
            )
        ]
    )


def test_zero_call_abstention_bounced(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="test",
        model_factory=lambda name: FunctionModel(
            _zero_call_abstain_scripted_navigator, model_name=name
        ),
        max_tool_calls=6,
    )
    ledger = CostLedger(run_dir)
    import pymupdf

    pdf = run_dir / "doc.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    plan = {
        "candidate_sections": ["S1"],
        "anchor_pages": [1],
        "search_queries": ["NonExistent"],
    }

    final = run_orchestrator(
        settings,
        "Is there a unicorn on page 1?",
        pdf,
        index,
        plan,
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        ledger,
    )

    # Zero call abstention was bounced; agent was forced to read page 1 before abstaining
    assert final["status"] == "not_answerable"
    assert 1 in final["pages_read"]
    assert final["tool_calls_used"] >= 1


def _figure_uninspected_scripted_navigator(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """Turn 1 reads text -> Turn 2 tries to abstain -> effort_gate forces inspect_page_image -> Turn 3 inspects -> Turn 4 abstains."""
    n = sum(isinstance(m, ModelResponse) for m in messages)
    if n == 0:
        return ModelResponse(parts=[ToolCallPart("read_page", {"page": [1]})])
    if n == 1:
        # Tries to abstain without inspect_page_image on page 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {"status": "not_answerable", "answer": "Not answerable"},
                )
            ]
        )
    if n == 2:
        # After bounce, calls inspect_page_image
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_page_image",
                    {"page": 1, "question": "What is in Figure 1?"},
                )
            ]
        )
    return ModelResponse(
        parts=[
            ToolCallPart(
                "final_answer", {"status": "not_answerable", "answer": "Not answerable"}
            )
        ]
    )


def test_uninspected_figure_abstention_bounced(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    parser_dir, _ = _write_parsed_doc(run_dir)
    index = json.loads((run_dir / "02_index" / "index.json").read_text())

    settings = AgentSettings(
        output_dir=tmp_path,
        api_key="test",
        model_factory=lambda name: FunctionModel(
            _figure_uninspected_scripted_navigator, model_name=name
        ),
        max_tool_calls=6,
    )
    ledger = CostLedger(run_dir)
    import pymupdf

    pdf = run_dir / "doc.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    plan = {
        "needs_visual_inspection": True,
        "candidate_sections": ["S1"],
        "anchor_pages": [1],
        "search_queries": ["Figure 1"],
    }

    final = run_orchestrator(
        settings,
        "What does Figure 1 show?",
        pdf,
        index,
        plan,
        parser_dir,
        run_dir / "04_orchestrator",
        run_dir / "run.log",
        ledger,
    )

    # Abstention without inspect_page_image was bounced; agent was forced to inspect page 1 image
    assert final["status"] == "not_answerable"
    assert len(final["image_inspections"]) == 1
    assert final["image_inspections"][0]["page"] == 1
