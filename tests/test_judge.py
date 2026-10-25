"""Judge stage: prompt filling, reply parsing, scoring, loading and a run with a stub model."""

import json
from pathlib import Path

from pine.agent.judge import (
    Item,
    extraction_prompt,
    judge_prompt,
    load_done,
    load_items,
    parse_extraction,
    parse_verdict,
    run_judge,
    score_extracted,
    summarize,
    write_flips,
)


def test_official_prompt_is_unchanged_and_appended_to() -> None:
    p = extraction_prompt("How many?", "There are 7.")
    assert p.startswith(
        "Given the question and analysis, you are tasked to extract answers"
    )
    assert p.endswith("Question: How many?\nAnalysis: There are 7.\n")
    assert "Fail to answer" in p


def test_judge_prompt_fills_every_field() -> None:
    it = Item("r", "Q?", "10", "Int", "It is 10.", 1.0)
    p = judge_prompt(it)
    assert p.rstrip().endswith("Response: It is 10.")
    assert "Gold answer: 10\nAnswer format: Int" in p
    assert "{question}" not in p and "{response}" not in p


def test_parsers() -> None:
    assert parse_extraction("Extracted answer: 10\nAnswer format: Integer") == (
        "10",
        "Integer",
    )
    assert parse_extraction("no structure here") == (None, None)
    text = "Extracted answer: ['a', 'b']\nVerdict: Correct\nReason: Same items."
    assert parse_verdict(text) == ("['a', 'b']", "Correct", "Same items.")
    assert parse_verdict("Verdict: [incorrect]")[1] == "Incorrect"
    assert parse_verdict("garbage")[1] is None


def test_score_extracted_uses_rules_and_fail_to_answer() -> None:
    assert score_extracted("10", "10", "Int") == 1.0
    assert score_extracted("Not answerable", "Fail to answer", "Str") == 0.0
    assert score_extracted("Not answerable", "Not answerable", "None") == 1.0


def _write_batch(root: Path) -> None:
    for i, (ans, gold, fmt, score) in enumerate(
        [
            ("The total is 10 items", "10", "Int", 1.0),
            ("a long sentence about Paris and more", "Paris", "Str", 0.0),
        ]
    ):
        d = root / f"q{i}" / "06_evaluation"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "run_id": f"q{i}",
                    "question": "Q?",
                    "gold_answer": gold,
                    "gold_answer_format": fmt,
                    "raw_system_answer": ans,
                    "score_official_rule": score,
                }
            )
        )


def test_load_items_from_batch_and_baseline(tmp_path: Path) -> None:
    _write_batch(tmp_path / "batch")
    items = load_items(tmp_path / "batch")
    assert [i.run_id for i in items] == ["q0", "q1"] and items[1].response.startswith(
        "a long"
    )
    base = tmp_path / "base"
    base.mkdir()
    (base / "results.jsonl").write_text(
        json.dumps(
            {
                "run_id": "b0",
                "question": "Q?",
                "gold_answer": "x",
                "gold_answer_format": "Str",
                "raw_answer": "x",
                "score_official_rule": 1.0,
                "error": None,
            }
        )
        + "\n"
        + json.dumps(
            {
                "run_id": "b1",
                "question": "Q?",
                "gold_answer": "y",
                "gold_answer_format": "Str",
                "error": "boom",
                "score_official_rule": 0.0,
            }
        )
        + "\n"
    )
    assert [i.run_id for i in load_items(base)] == ["b0"]  # errored records are skipped


def test_run_judge_with_stub_model_and_resume(tmp_path: Path) -> None:
    _write_batch(tmp_path / "batch")
    items = load_items(tmp_path / "batch")
    calls: list[str] = []

    def stub(prompt: str) -> tuple[str, int, int]:
        calls.append(prompt)
        if prompt.startswith("Given the question"):  # extraction stage
            return (
                (
                    "Extracted answer: Paris\nAnswer format: String"
                    if "Paris" in prompt
                    else "Extracted answer: 10\nAnswer format: Integer"
                ),
                50,
                8,
            )
        return (
            (
                "Extracted answer: Paris\nVerdict: Correct\nReason: Same."
                if "Paris" in prompt
                else "Extracted answer: 10\nVerdict: Correct\nReason: Equal."
            ),
            60,
            9,
        )

    out = tmp_path / "items.jsonl"
    recs = run_judge(items, stub, out, mode="both", workers=2)
    assert [r["extract_score"] for r in recs] == [1.0, 1.0]
    assert [r["judge_score"] for r in recs] == [1.0, 1.0]
    s = summarize(recs, 2)
    assert s["accuracy_rules_original"] == 0.5 and s["accuracy_llm_judge"] == 1.0
    assert s["judge_vs_rules"]["rules_wrong_new_right"] == 1
    assert s["judge_vs_rules"]["rules_right_new_wrong"] == 0
    write_flips(recs, tmp_path / "flips.md")
    assert "q1" in (tmp_path / "flips.md").read_text()

    n = len(calls)
    run_judge(items, stub, out, mode="both", workers=2)  # resume: nothing left to do
    assert len(calls) == n and len(load_done(out)) == 2


def test_empty_answer_and_errors_do_not_call_the_model(tmp_path: Path) -> None:
    items = [
        Item("e", "Q?", "5", "Int", "   ", 0.0),
        Item("f", "Q?", "5", "Int", "5", 1.0),
    ]

    def boom(prompt: str) -> tuple[str, int, int]:
        raise RuntimeError("server down")

    recs = run_judge(items, boom, tmp_path / "o.jsonl", mode="both", workers=1)
    assert [r["run_id"] for r in recs] == [
        "e"
    ]  # the empty one needs no call; the other errored
    assert recs[0]["empty"] and recs[0]["judge_score"] == 0.0


def _eval_inputs() -> tuple[dict, dict, dict]:
    gold = {
        "answer": "Paris",
        "answer_format": "Str",
        "evidence_pages": [2],
        "evidence_sources": [],
    }
    verification = {
        "final_answer": "The capital of France, mentioned on page two, is the city of Paris.",
        "decision": "ACCEPT",
        "cited_pages": [2],
    }
    final = {"status": "answered", "pages_read": [2], "image_inspections": []}
    return gold, verification, final


def test_step6_uses_the_judge_and_keeps_the_rule_score(tmp_path: Path) -> None:
    from pine.agent.scoring import run_eval

    gold, verification, final = _eval_inputs()
    judged = {
        "error": None,
        "judge_score": 1.0,
        "verdict": "Correct",
        "judge_reason": "Same city.",
    }
    r = run_eval(
        {"run_id": "r", "question": "Q?"},
        gold,
        verification,
        final,
        tmp_path / "s6",
        tmp_path / "run.log",
        {},
        {},
        judged=judged,
    )
    assert r["score"] == 1.0 and r["score_source"] == "llm_judge" and r["correct"]
    assert (
        r["score_official_rule"] < 0.5
    )  # the wordy answer fails the string rule: kept for audit
    assert r["judge"]["verdict"] == "Correct"


def test_step6_falls_back_to_rules_when_the_judge_fails_or_is_off(
    tmp_path: Path,
) -> None:
    from pine.agent.scoring import run_eval

    gold, verification, final = _eval_inputs()
    args = (
        {"run_id": "r", "question": "Q?"},
        gold,
        verification,
        final,
        tmp_path / "s6",
        tmp_path / "run.log",
        {},
        {},
    )
    failed = run_eval(*args, judged={"error": "ConnectionError: down"})
    assert failed["score_source"] == "rules_fallback"
    assert failed["score"] == failed["score_official_rule"]
    off = run_eval(*args)
    assert off["score_source"] == "rules" and off["judge"] is None


def test_step6_extract_mode_scores_with_the_rules_on_the_llm_extraction(
    tmp_path: Path,
) -> None:
    from pine.agent.scoring import run_eval

    gold, verification, final = _eval_inputs()
    judged = {"error": None, "extract_score": 1.0, "extracted": "Paris"}
    r = run_eval(
        {"run_id": "r", "question": "Q?"},
        gold,
        verification,
        final,
        tmp_path / "s6",
        tmp_path / "run.log",
        {},
        {},
        judged=judged,
        judge_mode="extract",
    )
    assert r["score_source"] == "llm_extraction+rules" and r["score"] == 1.0


def test_score_answer_never_raises() -> None:
    from pine.agent.judge import score_answer

    def boom(prompt: str) -> tuple[str, int, int]:
        raise RuntimeError("server down")

    out = score_answer("Q?", "Paris", "Str", "Paris", boom, "judge")
    assert out["error"].startswith("RuntimeError")
    ok = score_answer(
        "Q?",
        "Paris",
        "Str",
        "Paris",
        lambda p: ("Verdict: Correct\nReason: ok", 5, 3),
        "judge",
    )
    assert ok["judge_score"] == 1.0 and ok["prompt_tokens"] == 5
    assert (
        score_answer("Q?", "Paris", "Str", "  ", boom, "judge")["judge_score"] == 0.0
    )  # no call for empty


def test_baseline_run_one_uses_the_judge(tmp_path: Path) -> None:
    import pymupdf

    from pine.agent.baseline import BaselineSettings, Spec, run_one

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "the capital is Paris")
    pdf = tmp_path / "doc.pdf"
    doc.save(pdf)
    spec = Spec(
        "q1",
        pdf.name,
        "What is the capital?",
        {
            "answer": "Paris",
            "answer_format": "Str",
            "evidence_pages": [1],
            "evidence_sources": [],
        },
        pdf,
    )
    wordy = lambda *a: (
        "It is the city of Paris, the capital of France.",
        10,
        5,
        "stop",
    )
    settings = BaselineSettings(
        mode="all",
        modality="text",
        judge_chat=lambda p: ("Verdict: Correct\nReason: Paris.", 4, 2),
    )
    rec = run_one(spec, settings, wordy)
    assert rec["score"] == 1.0 and rec["score_source"] == "llm_judge"
    assert rec["score_official_rule"] < 1.0
    plain = run_one(spec, BaselineSettings(mode="all", modality="text"), wordy)
    assert (
        plain["score_source"] == "rules"
        and plain["score"] == plain["score_official_rule"]
    )


def test_runner_judging_step(tmp_path: Path) -> None:
    import logging

    from pine.agent.common import AgentSettings
    from pine.agent.runner import judge_final_answer

    gold, verification, _ = _eval_inputs()
    log = logging.getLogger("test")
    off = AgentSettings(output_dir=tmp_path)
    assert (
        judge_final_answer(off, "Q?", gold, verification, log) is None
    )  # no judge configured

    on = AgentSettings(
        output_dir=tmp_path,
        judge_model="judge-x",
        judge_chat=lambda p: (
            "Extracted answer: Paris\nVerdict: Correct\nReason: Same.",
            9,
            4,
        ),
    )
    out = judge_final_answer(on, "Q?", gold, verification, log)
    assert out is not None and out["judge_score"] == 1.0 and out["error"] is None

    def boom(prompt: str) -> tuple[str, int, int]:
        raise RuntimeError("down")

    bad = AgentSettings(output_dir=tmp_path, judge_model="judge-x", judge_chat=boom)
    failed = judge_final_answer(bad, "Q?", gold, verification, log)
    assert failed is not None and failed["error"].startswith("RuntimeError")


def test_judge_chat_merges_extra_body(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import openai

    seen: dict = {}

    class FakeCompletions:
        def create(self, **kw):  # type: ignore[no-untyped-def]
            seen.update(kw)
            msg = type("M", (), {"content": "thought</think>Verdict: Correct"})()
            usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})()
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": msg})()], "usage": usage},
            )()

    class FakeClient:
        def __init__(self, **kw):  # type: ignore[no-untyped-def]
            self.chat = type("Ch", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    from pine.agent.judge import make_judge_chat

    chat = make_judge_chat(
        "http://x/v1",
        "k",
        "m",
        disable_thinking=True,
        max_tokens=2000,
        extra_body={"reasoning_effort": "low"},
    )
    text, p, c = chat("hi")
    assert text == "Verdict: Correct" and (p, c) == (7, 3)
    assert seen["max_tokens"] == 2000 and seen["temperature"] == 0.0
    assert seen["extra_body"] == {
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_judge_prompt_says_the_percent_sign_is_optional() -> None:
    from pine.agent.judge import JUDGE_PROMPT

    assert "31, 31% and 0.31 are all the same value" in JUDGE_PROMPT
    assert "Gold answer: 31%" in JUDGE_PROMPT


def test_index_is_built_without_thinking_when_asked(tmp_path: Path) -> None:
    from pine.agent.common import AgentSettings
    from pine.agent.index import index_settings

    on = AgentSettings(
        output_dir=tmp_path,
        reasoning=True,
        thinking_sampling=True,
        max_tokens_scale=20.0,
    )
    assert index_settings(on) is on  # default: the index follows the run's settings
    off = AgentSettings(
        output_dir=tmp_path,
        reasoning=True,
        thinking_sampling=True,
        max_tokens_scale=20.0,
        index_reasoning=False,
    )
    idx = index_settings(off)
    assert (
        idx.reasoning is False
        and idx.thinking_sampling is False
        and idx.max_tokens_scale == 1.0
    )
    assert (
        off.reasoning is True and off.max_tokens_scale == 20.0
    )  # the run's own settings are untouched
    assert idx.text_model == off.text_model and idx.api_url == off.api_url


def test_settings_diff_and_result_score() -> None:
    from pine.agent.common import result_score, settings_diff

    old = {
        "reasoning": False,
        "max_tokens_scale": 1.0,
        "llm_cache_dir": "a",
        "only_old": 1,
    }
    new = {
        "reasoning": True,
        "max_tokens_scale": 1.0,
        "llm_cache_dir": "b",
        "only_new": 2,
    }
    assert settings_diff(old, new) == {
        "reasoning": (False, True)
    }  # cache dir and one-sided keys ignored
    assert settings_diff(old, old) == {}
    assert result_score(
        {"score": 1.0, "score_source": "llm_judge", "score_official_rule": 0.0}
    ) == (1.0, "llm_judge")
    # a result written before the judge existed has only the rule-based score
    assert result_score({"score_official_rule": 0.5}) == (0.5, "rules")
