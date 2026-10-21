"""Step 4 - LLM answer extraction and LLM judge over answers already saved on disk.

The rule-based scorer in ``scoring.py`` replaces the official GPT-4o extraction step with a
few regular expressions, so a correct but wordy answer can score 0. This stage rescores
answers that earlier stages saved, without touching the document or the agent:

``extract``  the official protocol: an LLM turns the free-form answer into a short answer
             (the official ``prompt_for_answer_extraction.md``), then the same rule-based
             ``eval_score`` as before scores it.
``judge``    an LLM compares the answer with the gold answer and returns a verdict
             (``prompts/answer_judge.md``). The judge never sees the document.

Both are written next to the original scores so the change can be audited, and both work on
a pipeline batch (``*/06_evaluation/result.json``) or a baseline folder (``results.jsonl``).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pine.agent.scoring import eval_score, extract_answer

PROMPT_DIR = Path(__file__).parent / "prompts"
EXTRACT_PROMPT = (PROMPT_DIR / "answer_extraction.md").read_text(encoding="utf-8")
JUDGE_PROMPT = (PROMPT_DIR / "answer_judge.md").read_text(encoding="utf-8")

# chat(prompt) -> (reply text, prompt_tokens, completion_tokens)
JudgeChat = Callable[[str], tuple[str, int, int]]

FAIL = "fail to answer"


@dataclass
class Item:
    """One answer to rescore."""

    run_id: str
    question: str
    gold_answer: str
    gold_format: str
    response: str
    original_score: float


def load_items(source: Path) -> list[Item]:
    """Answers of a baseline folder (``results.jsonl``) or a pipeline batch."""
    items: list[Item] = []
    results = source / "results.jsonl"
    if results.exists():
        for line in results.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is None:
                items.append(
                    Item(
                        r["run_id"],
                        r["question"],
                        str(r["gold_answer"]),
                        r["gold_answer_format"],
                        r.get("raw_answer", ""),
                        float(r["score_official_rule"]),
                    )
                )
    else:
        for f in sorted(source.glob("*/06_evaluation/result.json")):
            r = json.loads(f.read_text())
            items.append(
                Item(
                    r["run_id"],
                    r["question"],
                    str(r["gold_answer"]),
                    r["gold_answer_format"],
                    r.get("raw_system_answer", ""),
                    float(r["score_official_rule"]),
                )
            )
    return sorted(items, key=lambda i: i.run_id)


def extraction_prompt(question: str, response: str) -> str:
    """The official prompt with the question and answer appended at the end."""
    return f"{EXTRACT_PROMPT}\nQuestion: {question}\nAnalysis: {response}\n"


def judge_prompt(item: Item) -> str:
    return (
        JUDGE_PROMPT.replace("{question}", item.question)
        .replace("{gold_answer}", item.gold_answer)
        .replace("{answer_format}", item.gold_format)
        .replace("{response}", item.response)
    )


def parse_extraction(text: str) -> tuple[str | None, str | None]:
    """``Extracted answer`` and ``Answer format`` lines, or None when missing."""
    a = re.search(r"Extracted answer:\s*(.+)", text)
    f = re.search(r"Answer format:\s*\[?([A-Za-z]+)", text)
    return (a.group(1).strip() if a else None, f.group(1) if f else None)


def parse_verdict(text: str) -> tuple[str | None, str | None, str]:
    """(extracted answer, 'Correct' | 'Incorrect', reason); None for a missing part."""
    extracted, _ = parse_extraction(text)
    v = re.search(r"Verdict:\s*\[?\s*(Correct|Incorrect)", text, re.IGNORECASE)
    reason = re.search(r"Reason:\s*(.+)", text)
    return (
        extracted,
        v.group(1).capitalize() if v else None,
        reason.group(1).strip() if reason else "",
    )


def score_extracted(gold: str, extracted: str, fmt: str) -> float:
    """Official rule-based score of an extracted answer; 'Fail to answer' is always wrong."""
    if extracted.strip().lower() == FAIL:
        return 0.0
    return eval_score(gold, extracted, fmt)


def score_answer(
    question: str,
    gold_answer: str,
    gold_format: str,
    response: str,
    chat: JudgeChat,
    mode: str = "judge",
) -> dict[str, Any]:
    """Score one answer with the LLM. Never raises: a failure is returned as ``error``."""
    out: dict[str, Any] = {"mode": mode, "error": None}
    p_tok = c_tok = 0
    try:
        if not response.strip():
            # nothing to extract or judge (for example a reply cut off while thinking)
            out.update(
                extracted="",
                extract_score=0.0,
                verdict="Incorrect",
                judge_score=0.0,
                empty=True,
            )
            return out
        if mode in ("extract", "both"):
            text, p, c = chat(extraction_prompt(question, response))
            p_tok, c_tok = p_tok + p, c_tok + c
            extracted, _ = parse_extraction(text)
            out["extract_raw"] = text[:600]
            out["extract_parse_ok"] = extracted is not None
            if (
                extracted is None
            ):  # unusable reply: fall back to the rule-based cleaning
                extracted = extract_answer(response)
            out["extracted"] = extracted
            out["extract_score"] = score_extracted(gold_answer, extracted, gold_format)
        if mode in ("judge", "both"):
            item = Item("", question, gold_answer, gold_format, response, 0.0)
            text, p, c = chat(judge_prompt(item))
            p_tok, c_tok = p_tok + p, c_tok + c
            judged, verdict, reason = parse_verdict(text)
            out["judge_raw"] = text[:600]
            out["judge_extracted"] = judged
            out["verdict"] = verdict
            out["judge_reason"] = reason
            out["judge_score"] = (
                None if verdict is None else float(verdict == "Correct")
            )
    except Exception as exc:  # noqa: BLE001 - a judge failure must not stop a run
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
    out["prompt_tokens"], out["completion_tokens"] = p_tok, c_tok
    return out


def judge_item(item: Item, chat: JudgeChat, mode: str) -> dict[str, Any]:
    """Rescore one saved answer (stage 4)."""
    rec: dict[str, Any] = {
        "run_id": item.run_id,
        "gold_answer": item.gold_answer,
        "gold_format": item.gold_format,
        "response": item.response[:1500],
        "original_score": item.original_score,
    }
    rec.update(
        score_answer(
            item.question, item.gold_answer, item.gold_format, item.response, chat, mode
        )
    )
    return rec


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done[r["run_id"]] = r
    return done


def run_judge(
    items: list[Item],
    chat: JudgeChat,
    out_path: Path,
    mode: str = "both",
    workers: int = 16,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Rescore every item not already in ``out_path`` (resumable) and return all records."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [i for i in items if i.run_id not in done]
    records = dict(done)
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a") as out:
        futures = [pool.submit(judge_item, i, chat, mode) for i in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if rec["error"] is None:
                records[rec["run_id"]] = rec
            finished += 1
            if on_progress:
                on_progress(finished, len(todo))
    return [records[i.run_id] for i in items if i.run_id in records]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Accuracy under each scorer, how often they agree, and where they differ."""
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_format[r["gold_format"]].append(r)
    out: dict[str, Any] = {
        "items_scored": len(records),
        "items_total": total,
        "accuracy_rules_original": _mean(records, "original_score"),
        "accuracy_llm_extraction": _mean(records, "extract_score"),
        "accuracy_llm_judge": _mean(records, "judge_score"),
        "extraction_unparsed": sum(
            1 for r in records if r.get("extract_parse_ok") is False
        ),
        "judge_unparsed": sum(
            1
            for r in records
            if "verdict" in r and r["verdict"] is None and not r.get("empty")
        ),
        "empty_answers": sum(1 for r in records if r.get("empty")),
        "by_format": {
            k: {
                "n": len(v),
                "rules": _mean(v, "original_score"),
                "extraction": _mean(v, "extract_score"),
                "judge": _mean(v, "judge_score"),
            }
            for k, v in sorted(by_format.items())
        },
    }
    for name, key in (("extraction", "extract_score"), ("judge", "judge_score")):
        both = [r for r in records if r.get(key) is not None]
        if not both:
            continue
        rule_ok = [r["original_score"] >= 0.999 for r in both]
        new_ok = [r[key] >= 0.999 for r in both]
        c = Counter(zip(rule_ok, new_ok))
        out[f"{name}_vs_rules"] = {
            "agree": (c[(True, True)] + c[(False, False)]) / len(both),
            "rules_wrong_new_right": c[(False, True)],
            "rules_right_new_wrong": c[(True, False)],
            "both_right": c[(True, True)],
            "both_wrong": c[(False, False)],
        }
    out["prompt_tokens"] = sum(r.get("prompt_tokens", 0) for r in records)
    out["completion_tokens"] = sum(r.get("completion_tokens", 0) for r in records)
    return out


def write_flips(records: list[dict[str, Any]], path: Path, limit: int = 40) -> None:
    """Markdown of answers where the new score disagrees with the rules, for a manual read."""

    def block(title: str, rows: list[dict[str, Any]]) -> list[str]:
        lines = [f"## {title} ({len(rows)} total, first {min(limit, len(rows))})", ""]
        for r in rows[:limit]:
            lines += [
                f"- `{r['run_id']}` [{r['gold_format']}]",
                f"  - gold: {r['gold_answer'][:150]}",
                f"  - answer: {r['response'][:240].replace(chr(10), ' ')}",
                f"  - extracted: {str(r.get('extracted', r.get('judge_extracted')))[:150]}",
                f"  - reason: {r.get('judge_reason', '')[:200]}",
            ]
        return lines + [""]

    lines = ["# Where the new scorer and the rules disagree", ""]
    for key, label in (
        ("judge_score", "LLM judge"),
        ("extract_score", "LLM extraction + rules"),
    ):
        have = [r for r in records if r.get(key) is not None]
        up = [r for r in have if r["original_score"] < 0.999 <= r[key]]
        down = [r for r in have if r[key] < 0.999 <= r["original_score"]]
        if have:
            lines += block(f"{label}: rules said wrong, new says right", up)
            lines += block(f"{label}: rules said right, new says wrong", down)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_judge_chat(
    api_url: str,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    disable_thinking: bool = False,
    max_tokens: int = 300,
    extra_body: dict[str, Any] | None = None,
) -> JudgeChat:
    """An OpenAI-compatible chat call at temperature 0 (any hosted API or a vLLM server).

    ``extra_body`` passes server-specific fields, for example ``{"reasoning_effort": "low"}``
    for gpt-oss or ``{"chat_template_kwargs": {...}}``; a reasoning judge also needs a larger
    ``max_tokens``, because its thinking is counted before the verdict."""
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=3
    )
    body: dict[str, Any] = dict(extra_body or {})
    if disable_thinking:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    def chat(prompt: str) -> tuple[str, int, int]:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body=body or None,
        )
        usage = resp.usage
        text = resp.choices[0].message.content or ""
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return (
            text.strip(),
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    return chat
