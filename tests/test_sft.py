"""SFT set: same prompt as direct mode, no hint, thinking format, skipped examples."""

import json
from pathlib import Path
from typing import Any

import pymupdf

from pine.agent.direct import prompt_parts
from pine.training.sft import build_sft, gold_answer_text


def _setup(tmp_path: Path) -> dict[str, dict[str, Any]]:
    doc = pymupdf.open()
    for i in range(5):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(tmp_path / "doc.pdf")
    base = {
        "split": "train",
        "pdf": "doc.pdf",
        "question": "Which?",
        "candidate_pages": [2, 4],
        "answerable": True,
    }
    return {
        "a": {**base, "id": "a", "answer": '["x", "y"]', "answer_format": "List"},
        "b": {**base, "id": "b", "answer": "7", "answer_format": "Int"},
    }


def _row(rid: str, status: str, chosen: int | None) -> dict[str, Any]:
    return {
        "id": rid,
        "status": status,
        "chosen": chosen,
        "answerable": True,
        "answer_format": "List",
        "samples": [
            {"reasoning": "wrong", "answer": "[z]", "hinted": False},
            {"reasoning": " page 4 lists x and y ", "answer": "[x, y]", "hinted": True},
        ],
    }


def test_build_sft(tmp_path: Path) -> None:
    records = _setup(tmp_path)
    out = tmp_path / "sft"
    rows = [
        _row("a", "rationalized", 1),
        _row("b", "unrescued", None),
        _row("a", "x", 0),
    ]
    manifest = build_sft(records, rows, tmp_path, out)
    lines = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and manifest["examples"] == 1
    assert manifest["skipped"] == {"unrescued": 1, "duplicate": 1}
    ex = lines[0]

    parts = prompt_parts(tmp_path / "doc.pdf", [2, 4], "Which?")
    expected = "".join("<image>" if isinstance(p, bytes) else p for p in parts)
    user, assistant = ex["messages"][0]["content"], ex["messages"][1]["content"]
    assert user == expected and "Hint" not in user
    assert user.startswith("[Page 2]<image>[Page 4]<image>")
    assert ex["images"] == ["images/a_p002.jpg", "images/a_p004.jpg"]
    assert (out / "images/a_p004.jpg").read_bytes() == parts[3]
    assert assistant == "<think>\npage 4 lists x and y\n</think>\n\n[x, y]"

    build_sft(records, rows[:1], tmp_path, out, answer_source="gold")
    gold = json.loads((out / "train.jsonl").read_text())
    assert gold["messages"][1]["content"].endswith("</think>\n\n[x, y]")


def test_gold_answer_text() -> None:
    assert (
        gold_answer_text({"answer": '["DE", "ES"]', "answer_format": "List"})
        == "[DE, ES]"
    )
    assert (
        gold_answer_text({"answer": "8.8 million", "answer_format": "Str"})
        == "8.8 million"
    )
