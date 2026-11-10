"""STaR sampling: prompt parity with direct mode, reply parsing, judge scoring, rationalization, resume."""

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymupdf

from pine.agent.baseline import INSTRUCTION
from pine.agent.direct import prompt_parts
from pine.training.chat import Batch, Sample, split_reply
from pine.training.star import (
    StarSettings,
    build_messages,
    hint_text,
    pick,
    run_star,
    summarize,
)


def _pdf(path: Path, n: int) -> None:
    doc = pymupdf.open()
    for i in range(n):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)


def _rec(id_: str, answer: str, fmt: str, answerable: bool = True) -> dict[str, Any]:
    return {
        "id": id_,
        "split": "train",
        "subset": "hr",
        "pdf": "doc.pdf",
        "question": "How many?",
        "answer": answer,
        "answer_format": fmt,
        "answerable": answerable,
        "gold_pages": [3, 2],
        "candidate_pages": [1, 2, 4] if not answerable else [1, 2, 3],
    }


def fake_judge(prompt: str) -> tuple[str, int, int]:
    """Correct when the response's number or text appears in the gold answer (so "8.8"
    matches "8.8 million", which the official rules score 0)."""
    last = prompt.rsplit("---", 1)[1]
    gold = last.split("Gold answer: ", 1)[1].split("\n", 1)[0]
    response = last.split("Response: ", 1)[1].strip()
    verdict = "Correct" if response and response in gold else "Incorrect"
    return f"Extracted answer: {response}\nVerdict: {verdict}\nReason: test", 7, 3


def _texts(messages: list[dict[str, Any]]) -> list[str]:
    return [c["text"] for c in messages[0]["content"] if c["type"] == "text"]


def test_prompt_matches_direct_mode(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", 4)
    rec = _rec("a", "7", "Int")
    msgs = build_messages(rec, tmp_path)
    parts = prompt_parts(tmp_path / "doc.pdf", [1, 2, 3], "How many?")
    assert _texts(msgs) == [p for p in parts if isinstance(p, str)]
    assert _texts(msgs)[:3] == ["[Page 1]", "[Page 2]", "[Page 3]"]
    assert _texts(msgs)[-1] == f"{INSTRUCTION}\n\nQuestion: How many?"
    images = [c for c in msgs[0]["content"] if c["type"] == "image_url"]
    assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == parts[1]

    hinted = _texts(build_messages(rec, tmp_path, hint_text(rec["gold_pages"])))[-1]
    assert hinted == (
        f"{INSTRUCTION}\n\nHint: the evidence is on page(s) 2, 3.\n\nQuestion: How many?"
    )


def test_split_reply_server_formats() -> None:
    openrouter = SimpleNamespace(content="42", reasoning="count rows", model_extra={})
    assert split_reply(openrouter, "stop") == Sample("count rows", "42", "stop", False)
    vllm = SimpleNamespace(content=" 42\n", model_extra={"reasoning_content": "r"})
    assert split_reply(vllm, "stop").reasoning == "r"
    inline = SimpleNamespace(content="<think>a\nb</think>\n\n42", model_extra={})
    assert split_reply(inline, "stop") == Sample("a\nb", "42", "stop", False)
    gap = SimpleNamespace(content="thinking\n\n\n42<|im_end|>", model_extra={})
    assert split_reply(gap, "stop") == Sample("thinking", "42", "stop", False)
    cut = SimpleNamespace(content="still thinking", model_extra={})
    s = split_reply(cut, "length")
    assert s.truncated and s.answer == "" and s.reasoning == "still thinking"


def test_pick_shortest_correct() -> None:
    samples = [
        {"reasoning": "long long", "score": 1.0},
        {"reasoning": "x", "score": 0.0},
        {"reasoning": "short", "score": 1.0},
    ]
    assert pick(samples) == 2
    assert pick([{"reasoning": "", "score": 0.0}]) is None


def test_run_star_rationalizes_and_resumes(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", 4)
    records = [
        {**_rec("right", "8.8 million", "Str"), "question": "right"},
        {**_rec("hinted", "12", "Int"), "question": "hinted"},
        {**_rec("never", "5", "Int"), "question": "never"},
        {
            **_rec("none", "Not answerable", "None", answerable=False),
            "question": "none",
        },
    ]
    # the fake model's reply per question, without and with the hint
    replies = {
        "right": ("8.8", "8.8"),
        "hinted": ("3", "12"),
        "never": ("4", "4"),
        "none": ("Not answerable", "Not answerable"),
    }
    calls: list[tuple[str, bool]] = []

    def fake(messages: list[dict[str, Any]], n: int) -> Batch:
        text = _texts(messages)[-1]
        question = text.rsplit("Question: ", 1)[1]
        hinted = "Hint:" in text
        calls.append((question, hinted))
        answer = replies[question][hinted]
        out = [
            Sample("short", answer, "stop", False),
            Sample("much longer reasoning", answer, "stop", False),
        ]
        return Batch(samples=out[:n], prompt_tokens=10, completion_tokens=5, cost=0.01)

    settings = StarSettings(data_dir=tmp_path, k=2, workers=2)
    out = tmp_path / "star" / "train_k2.jsonl"
    rows = run_star(records, out, fake, fake_judge, settings)
    by_id = {r["id"]: r for r in rows}
    assert by_id["right"]["status"] == "correct" and by_id["right"]["chosen"] == 0
    first = by_id["right"]["samples"][0]
    assert (
        first["verdict"] == "Correct" and first["rule_score"] == 0.0
    )  # judge, not rules
    assert by_id["hinted"]["status"] == "rationalized"
    assert by_id["hinted"]["chosen"] == 2 and by_id["hinted"]["samples"][2]["hinted"]
    assert by_id["hinted"]["hint"] == "Hint: the evidence is on page(s) 2, 3."
    assert by_id["never"]["status"] == "unrescued"
    assert len(by_id["never"]["samples"]) == 4 and by_id["never"]["chosen"] is None
    assert by_id["none"]["status"] == "correct"
    assert ("none", True) not in calls  # unanswerable examples are never hinted

    n_calls = len(calls)
    rows = run_star(
        records, out, fake, fake_judge, settings
    )  # everything on disk: no calls
    assert len(calls) == n_calls and len(rows) == 4

    summary = summarize(rows, 4)
    assert summary["status"] == {"correct": 2, "rationalized": 1, "unrescued": 1}
    assert (
        summary["accuracy"] == 0.5
    )  # first unhinted sample right on "right" and "none"
    assert summary["rule_accuracy"] == 0.25  # the rules miss "8.8" vs "8.8 million"
    assert summary["pass_at_k"] == 0.5
    # the rules settle the 2 hinted "12"s and the 2 "Not answerable"s; the judge the other 8
    assert summary["scored_by"] == {"judge": 8, "rules": 4}
    assert summary["judge_tokens"] == 8 * 10  # 10 tokens per judge call
    assert summary["cost_usd"] == 0.06  # 4 first passes + 2 hinted passes


def test_truncated_and_failed_judge(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", 4)
    settings = StarSettings(data_dir=tmp_path, k=1, rationalize=False, workers=1)

    def cut(messages: list[dict[str, Any]], n: int) -> Batch:
        return Batch([Sample("thinking", "", "length", True)] * n)

    def no_judge(prompt: str) -> tuple[str, int, int]:
        raise AssertionError("a truncated sample is never judged")

    rows = run_star(
        [_rec("a", "7", "Int")], tmp_path / "a.jsonl", cut, no_judge, settings
    )
    assert rows[0]["status"] == "unrescued" and rows[0]["samples"][0]["score"] == 0.0

    def answer(messages: list[dict[str, Any]], n: int) -> Batch:
        return Batch([Sample("r", "7", "stop", False)] * n)

    def down(prompt: str) -> tuple[str, int, int]:
        raise ConnectionError("judge offline")

    out = tmp_path / "b.jsonl"
    assert run_star([_rec("b", "7 days", "Str")], out, answer, down, settings) == []
    rows = run_star([_rec("b", "7 days", "Str")], out, answer, fake_judge, settings)
    assert rows[0]["status"] == "correct"  # the failed example was retried


def test_judge_all_judges_rule_passes(tmp_path: Path) -> None:
    _pdf(tmp_path / "doc.pdf", 4)
    seen: list[str] = []

    def judge(prompt: str) -> tuple[str, int, int]:
        seen.append(prompt)
        return fake_judge(prompt)

    def answer(messages: list[dict[str, Any]], n: int) -> Batch:
        return Batch([Sample("r", "7", "stop", False)] * n)

    rec = _rec("a", "7", "Int")
    for judge_all, calls in ((False, 0), (True, 1)):
        settings = StarSettings(
            data_dir=tmp_path, k=1, rationalize=False, workers=1, judge_all=judge_all
        )
        seen.clear()
        rows = run_star([rec], tmp_path / f"{judge_all}.jsonl", answer, judge, settings)
        assert len(seen) == calls and rows[0]["status"] == "correct"
        assert rows[0]["samples"][0]["scored_by"] == ("judge" if judge_all else "rules")
