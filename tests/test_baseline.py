"""Baseline helpers: page selection, reply cleaning, and a full run with a stub model."""

import json
from pathlib import Path

import pymupdf
import pytest

from pine.agent.baseline import (
    BaselineSettings,
    Spec,
    build_messages,
    clean_reply,
    load_done,
    run_baseline,
    select_pages,
    summarize,
)


def _pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    return path


def _spec(pdf: Path, run_id: str, question: str, answer: str, fmt: str = "Str") -> Spec:
    return Spec(
        run_id=run_id,
        doc_id=pdf.name,
        question=question,
        gold={
            "answer": answer,
            "answer_format": fmt,
            "evidence_pages": [2],
            "evidence_sources": [],
        },
        pdf_path=pdf,
    )


def test_select_pages_all_and_bm25() -> None:
    texts = {
        1: "intro words",
        2: "revenue was seven",
        3: "appendix",
        4: "more appendix",
    }
    all_pages = BaselineSettings(mode="all", max_pages=10)
    assert select_pages("revenue", texts, all_pages) == [1, 2, 3, 4]
    capped = BaselineSettings(mode="all", max_pages=2)
    assert 2 in select_pages("revenue", texts, capped)
    assert len(select_pages("revenue", texts, capped)) == 2
    top1 = BaselineSettings(mode="bm25", top_k=1)
    assert select_pages("revenue", texts, top1) == [2]
    # no page matches: still returns k pages
    assert select_pages("zzz", texts, BaselineSettings(mode="bm25", top_k=2)) == [1, 2]


def test_clean_reply() -> None:
    assert clean_reply("long thought</think>\n42 ", True) == ("42", False)
    assert clean_reply("long thought</think>\n42<|im_end|>", True) == ("42", False)
    # vLLM drops </think> and leaves a three-newline gap before the answer
    assert clean_reply("1. step\n\n2. check\n\n\n51", True) == ("51", False)
    assert clean_reply("a\n\n\nb\n\n\nTim Ziemer", True) == ("Tim Ziemer", False)
    assert clean_reply("still thinking, no gap", True) == ("", True)
    assert clean_reply("thought\n\n\n42", True, "length") == ("", True)
    assert clean_reply(" 7 ", False) == ("7", False)
    assert clean_reply("7\n\n\n8", False) == ("7\n\n\n8", False)


def test_build_messages_image_and_text() -> None:
    img = build_messages("Q?", [3], {3: "data:image/jpeg;base64,AA"}, None)
    kinds = [c["type"] for c in img[0]["content"]]
    assert kinds == ["text", "image_url", "text"]
    txt = build_messages("Q?", [3], None, {3: "page three"})
    assert "page three" in txt[0]["content"][0]["text"]


def test_run_baseline_with_stub_model(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "doc.pdf", ["alpha", "the capital is Paris", "omega"])
    specs = [
        _spec(pdf, "q1", "What is the capital?", "Paris"),
        _spec(pdf, "q2", "Who wrote it?", "Not answerable", "None"),
    ]
    seen: list[list[dict]] = []

    def stub(messages, max_tokens, temperature, thinking):  # type: ignore[no-untyped-def]
        seen.append(messages)
        q = messages[0]["content"][-1]["text"]
        return ("Paris" if "capital" in q else "Not answerable"), 100, 3, "stop"

    settings = BaselineSettings(mode="bm25", modality="text", top_k=1, workers=2)
    out = tmp_path / "results.jsonl"
    records = run_baseline(specs, settings, stub, out)
    assert [r["score_official_rule"] for r in records] == [1.0, 1.0]
    assert records[0]["pages_sent"] == [2] and records[0]["gold_page_in_context"]
    summary = summarize(records, 2)
    assert summary["accuracy"] == 1.0 and summary["questions_scored"] == 2

    # resume: nothing left to do, no new model calls
    calls = len(seen)
    run_baseline(specs, settings, stub, out)
    assert len(seen) == calls


def test_errors_are_recorded_and_retried(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "doc.pdf", ["only page"])
    spec = _spec(pdf, "q1", "Anything?", "x")
    settings = BaselineSettings(mode="all", modality="text", workers=1)
    out = tmp_path / "results.jsonl"

    def boom(messages, max_tokens, temperature, thinking):  # type: ignore[no-untyped-def]
        raise RuntimeError("server down")

    records = run_baseline([spec], settings, boom, out)
    assert records == []  # an errored question is not counted as done
    assert json.loads(out.read_text().splitlines()[0])["error"].startswith(
        "RuntimeError"
    )
    assert load_done(out) == {}

    ok = run_baseline([spec], settings, lambda *a: ("x", 1, 1, "stop"), out)
    assert ok[0]["score_official_rule"] == 1.0


def test_missing_pdf_page_text_falls_back_to_native(tmp_path: Path) -> None:
    from pine.agent.baseline import page_texts

    pdf = _pdf(tmp_path / "doc.pdf", ["hello"])
    assert "hello" in page_texts(pdf, tmp_path / "nope")[1]
    with pytest.raises(Exception):  # noqa: B017 - any pymupdf error for a missing file
        page_texts(tmp_path / "missing.pdf", None)


def test_thinking_sampling_override(tmp_path: Path) -> None:
    from pydantic_ai.settings import ModelSettings

    from pine.agent.common import AgentSettings
    from pine.agent.llm import apply_thinking_sampling

    base = ModelSettings(temperature=0.0, max_tokens=350)
    off = AgentSettings(output_dir=tmp_path)
    assert apply_thinking_sampling(base, off) == base  # default: nothing changes

    on = AgentSettings(
        output_dir=tmp_path, thinking_sampling=True, presence_penalty=0.5
    )
    out = apply_thinking_sampling(base, on)
    assert out is not None
    assert out["temperature"] == 0.6 and out["top_p"] == 0.95
    assert out["presence_penalty"] == 0.5 and out["max_tokens"] == 350
    assert out["extra_body"] == {"top_k": 20}
    assert apply_thinking_sampling(None, on) is None
