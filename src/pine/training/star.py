"""STaR sampling over the ViDoRe examples (``vidore.py``): sample, score, rationalize.

For each example the answer model sees exactly what ``--navigator direct`` sends at inference
(``agent/direct.prompt_parts``): the 10 candidate pages as images in page order, then the B1
instruction and the question. Per example:

1. draw ``k`` thinking samples and score each final answer. The official rules
   (``scoring.eval_score``) go first: a full rule score (1.0) is correct without a judge call.
   Anything less goes to the LLM judge (``agent/judge.score_answer``, the project judge is
   gpt-oss-120b), whose verdict decides, so a right answer in another form ("8.8" for
   "8.8 million") is not lost. ``judge_all`` sends every sample to the judge instead. A sample
   that ran out of tokens is incorrect without either. ``scored_by`` says which one decided.
2. keep the shortest correct reasoning (status ``correct``);
3. if none is correct and the question is answerable, sample again with a hint naming the
   gold pages, never the answer (status ``rationalized`` if one is then correct). The SFT step
   stores such a sample with the hint removed, so the model learns to find the page itself.
   Unanswerable examples are never hinted: there is no page to point at.
4. otherwise the status is ``unrescued``.

Rows are appended to a JSONL file one example at a time, so a killed run resumes where it
stopped; an example whose model or judge call failed is not written and is retried on the
next run.
With ``k=1`` and no rationalization the same code is the base-model evaluation on star_dev.
"""

from __future__ import annotations

import base64
import json
import logging
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pine.agent.direct import prompt_parts
from pine.agent.judge import JudgeChat, score_answer
from pine.agent.scoring import eval_score, extract_answer
from pine.training.chat import Sample, SampleFn

log = logging.getLogger(__name__)


@dataclass
class StarSettings:
    data_dir: Path  # --out-dir of 05_training.py: star_*.jsonl and raw/ live here
    k: int = 4
    rationalize: bool = True
    workers: int = 8
    model: str = ""
    judge_model: str = ""
    # True: judge every sample, even those the rules already score 1.0
    judge_all: bool = False


def hint_text(gold_pages: list[int]) -> str:
    pages = ", ".join(str(p) for p in sorted(gold_pages))
    return f"Hint: the evidence is on page(s) {pages}."


def build_messages(
    rec: dict[str, Any], data_dir: Path, hint: str | None = None
) -> list[dict[str, Any]]:
    """One user turn in OpenAI chat format, built from ``direct.prompt_parts``."""
    content: list[dict[str, Any]] = []
    for part in prompt_parts(
        data_dir / rec["pdf"], rec["candidate_pages"], rec["question"], hint
    ):
        if isinstance(part, bytes):
            url = "data:image/jpeg;base64," + base64.b64encode(part).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def score_sample(
    rec: dict[str, Any], s: Sample, judge: JudgeChat, judge_all: bool = False
) -> dict[str, Any]:
    """``score`` 1.0 or 0.0: the rules when they pass fully, else the judge's verdict.
    Raises when the judge call fails, so the example is retried on the next run."""
    if s.truncated:
        return {
            "extracted": "",
            "score": 0.0,
            "rule_score": 0.0,
            "verdict": None,
            "scored_by": "truncated",
        }
    extracted = extract_answer(s.answer)
    gold, fmt = str(rec["answer"]), rec["answer_format"]
    rule_score = eval_score(gold, extracted, fmt)
    if rule_score >= 1.0 and not judge_all:
        return {
            "extracted": extracted,
            "score": 1.0,
            "rule_score": rule_score,
            "verdict": None,
            "scored_by": "rules",
        }
    j = score_answer(rec["question"], gold, fmt, s.answer, judge, mode="judge")
    if j["error"]:
        raise RuntimeError(f"judge failed: {j['error']}")
    return {
        "extracted": extracted,
        # an unparseable verdict counts as incorrect; the raw reply is kept to inspect
        "score": j["judge_score"] or 0.0,
        "rule_score": rule_score,
        "verdict": j["verdict"],
        "scored_by": "judge",
        "judge_extracted": j.get("judge_extracted"),
        "judge_reason": j.get("judge_reason"),
        "judge_raw": None if j["verdict"] else j.get("judge_raw"),
        "judge_tokens": j["prompt_tokens"] + j["completion_tokens"],
    }


def pick(samples: list[dict[str, Any]]) -> int | None:
    """Index of the shortest correct reasoning, or None."""
    correct = [i for i, s in enumerate(samples) if s["score"] >= 1.0]
    if not correct:
        return None
    return min(correct, key=lambda i: len(samples[i]["reasoning"]))


def process(
    rec: dict[str, Any],
    sample_fn: SampleFn,
    judge: JudgeChat,
    settings: StarSettings,
) -> dict[str, Any]:
    """Sample, score and, if needed, rationalize one example; returns its JSONL row."""
    row: dict[str, Any] = {
        "id": rec["id"],
        "split": rec["split"],
        "subset": rec["subset"],
        "answerable": rec["answerable"],
        "answer": rec["answer"],
        "answer_format": rec["answer_format"],
        "model": settings.model,
        "judge_model": settings.judge_model,
        "samples": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "judge_tokens": 0,
    }

    def draw(hint: str | None) -> None:
        batch = sample_fn(build_messages(rec, settings.data_dir, hint), settings.k)
        row["prompt_tokens"] += batch.prompt_tokens
        row["completion_tokens"] += batch.completion_tokens
        row["cost"] += batch.cost
        for s in batch.samples:
            scored = score_sample(rec, s, judge, settings.judge_all)
            row["judge_tokens"] += scored.pop("judge_tokens", 0)
            row["samples"].append({**asdict(s), **scored, "hinted": hint is not None})

    draw(None)
    chosen = pick(row["samples"])
    status = "correct" if chosen is not None else "unrescued"
    if chosen is None and settings.rationalize and rec["answerable"]:
        row["hint"] = hint_text(rec["gold_pages"])
        draw(row["hint"])
        chosen = pick(row["samples"])
        if chosen is not None:
            status = "rationalized"
    row["status"] = status
    row["chosen"] = chosen
    return row


def load_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def run_star(
    records: list[dict[str, Any]],
    out_path: Path,
    sample_fn: SampleFn,
    judge: JudgeChat,
    settings: StarSettings,
) -> list[dict[str, Any]]:
    """Process every record not yet in ``out_path``; returns all rows on disk."""
    done = {r["id"] for r in load_rows(out_path)}
    todo = [r for r in records if r["id"] not in done]
    log.info("%d examples done, %d to sample", len(done), len(todo))

    def one(rec: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return process(rec, sample_fn, judge, settings)
        except Exception as exc:  # noqa: BLE001 - retried on the next run
            log.warning("sampling failed for %s: %s", rec["id"], exc)
            return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(max_workers=settings.workers) as pool,
        out_path.open("a", encoding="utf-8") as f,
    ):
        for n, row in enumerate(pool.map(one, todo), 1):
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                log.info(
                    "[%d/%d] %s: %s (scores %s)",
                    n,
                    len(todo),
                    row["id"],
                    row["status"],
                    [round(s["score"], 2) for s in row["samples"]],
                )
    wanted = {r["id"] for r in records}
    return [r for r in load_rows(out_path) if r["id"] in wanted]


def summarize(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Counts for the manifest. ``accuracy`` is the score (rules, else judge) of each
    example's first unhinted sample (what one direct-mode call would score),
    ``rule_accuracy`` the official rules alone on the same sample; ``pass_at_k`` the share
    with a correct unhinted sample."""

    def first_score(r: dict[str, Any], key: str = "score") -> float:
        return float(r["samples"][0][key]) if r["samples"] else 0.0

    def unhinted_pass(r: dict[str, Any]) -> bool:
        return any(s["score"] >= 1.0 for s in r["samples"] if not s.get("hinted"))

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"format:{r['answer_format']}", []).append(r)
        groups.setdefault(f"subset:{r['subset']}", []).append(r)
        groups.setdefault(
            "answerable" if r["answerable"] else "unanswerable", []
        ).append(r)
    samples = [s for r in rows for s in r["samples"]]
    return {
        "examples": total,
        "done": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "accuracy": mean([first_score(r) for r in rows]),
        "rule_accuracy": mean([first_score(r, "rule_score") for r in rows]),
        "pass_at_k": mean([float(unhinted_pass(r)) for r in rows]),
        "by_group": {
            g: {
                "n": len(rs),
                "accuracy": mean([first_score(r) for r in rs]),
                "pass_at_k": mean([float(unhinted_pass(r)) for r in rs]),
            }
            for g, rs in sorted(groups.items())
        },
        "samples": len(samples),
        "truncated_samples": sum(1 for s in samples if s["truncated"]),
        "scored_by": dict(Counter(s.get("scored_by", "judge") for s in samples)),
        "unparsed_verdicts": sum(
            1
            for s in samples
            if s.get("scored_by", "judge") == "judge" and s["verdict"] is None
        ),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "cost_usd": round(sum(r["cost"] for r in rows), 4),  # answer model only
        "judge_tokens": sum(r["judge_tokens"] for r in rows),
    }
