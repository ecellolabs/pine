"""Direct-context baseline for MMLongBench-Doc: no index, no agent, no verifier.

The same model answers each question from page images or page text placed straight in the
prompt. It exists to answer one question: does the PINE pipeline beat simply giving the
model the document? Questions, gold answers and the scorer are the ones the pipeline uses
(``pine.agent.scoring``), so the two numbers are comparable.

Two choices make a variant: which pages are sent (``all`` pages, or the BM25 ``bm25``
top-k, which is the flat-retrieval baseline) and in what form (``image`` or ``text``).
A document longer than ``max_pages`` is cut down to its BM25 top ``max_pages``.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pine.agent.locate import BM25
from pine.agent.scoring import eval_score, extract_answer

INSTRUCTION = (
    "Answer the question using only the document pages above. Reply with the final "
    "answer only, as short as possible: a number, a short phrase, or a list written "
    "like [a, b] when several items are asked for. If the document does not contain the "
    "answer, reply exactly: Not answerable"
)

# chat(messages, max_tokens, temperature, thinking)
#   -> (content, prompt_tokens, completion_tokens, finish_reason)
ChatFn = Callable[[list[dict[str, Any]], int, float, bool], tuple[str, int, int, str]]


@dataclass
class BaselineSettings:
    model: str = "Qwen/Qwen3.5-9B"
    mode: str = "all"  # all | bm25
    modality: str = "image"  # image | text
    top_k: int = 5  # pages sent in bm25 mode
    max_pages: int = 60  # cap in all mode; longer documents fall back to BM25 top pages
    max_side: int = 896  # longest side of a page image in pixels
    thinking: bool = False
    max_tokens: int | None = None  # default 512 without thinking, 12000 with
    temperature: float | None = None  # default 0.0 without thinking, 0.6 with
    workers: int = 16
    parsed_dir: Path | None = (
        None  # <batch>/_doc_cache: Docling text the pipeline built
    )


@dataclass
class Spec:
    run_id: str
    doc_id: str
    question: str
    gold: dict[str, Any]
    pdf_path: Path


def load_specs(batch_dir: Path, pdf_dir: Path) -> list[Spec]:
    """The exact questions of a finished pipeline batch, in a stable order."""
    specs = []
    for q in sorted(batch_dir.glob("*/00_input/question.json")):
        j = json.loads(q.read_text())
        specs.append(
            Spec(
                run_id=j["run_id"],
                doc_id=j["doc_id"],
                question=j["question"],
                gold=j["gold"],
                pdf_path=pdf_dir / j["doc_id"],
            )
        )
    return specs


def page_texts(pdf_path: Path, parsed_dir: Path | None) -> dict[int, str]:
    """1-indexed page text: the pipeline's Docling markdown when cached, else native PDF text."""
    import pymupdf

    native: dict[int, str] = {}
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            native[i] = page.get_text()
    md_dir = parsed_dir / pdf_path.stem / "01_parser" / "pages" if parsed_dir else None
    if md_dir is None or not md_dir.is_dir():
        return native
    out = {}
    for p, text in native.items():
        f = md_dir / f"page_{p:03d}.md"
        md = f.read_text(encoding="utf-8") if f.exists() else ""
        out[p] = md if md.strip() else text
    return out


def select_pages(
    question: str, texts: dict[int, str], settings: BaselineSettings
) -> list[int]:
    """Pages to send, in page order."""
    n = len(texts)
    limit = settings.top_k if settings.mode == "bm25" else settings.max_pages
    if settings.mode == "all" and n <= limit:
        return sorted(texts)
    ranked = [p for p, _ in BM25(dict(texts)).search(question, k=limit)]
    for p in sorted(texts):  # pad with early pages when few pages match the query
        if len(ranked) >= min(limit, n):
            break
        if p not in ranked:
            ranked.append(p)
    return sorted(ranked[:limit])


def render_pages(pdf_path: Path, pages: list[int], max_side: int) -> dict[int, str]:
    """JPEG data URLs for 1-indexed pages, opening the PDF once."""
    import pymupdf
    from PIL import Image

    out = {}
    with pymupdf.open(pdf_path) as doc:
        for p in pages:
            page = doc[p - 1]
            scale = max_side / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            out[p] = f"data:image/jpeg;base64,{b64}"
    return out


def build_messages(
    question: str,
    pages: list[int],
    images: dict[int, str] | None,
    texts: dict[int, str] | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for p in pages:
        if images is not None:
            content.append({"type": "text", "text": f"[Page {p}]"})
            content.append({"type": "image_url", "image_url": {"url": images[p]}})
        elif texts is not None:
            content.append({"type": "text", "text": f"[Page {p}]\n{texts[p]}"})
    content.append({"type": "text", "text": f"{INSTRUCTION}\n\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def clean_reply(
    content: str, thinking: bool, finish_reason: str = "stop"
) -> tuple[str, bool]:
    """Final answer text and whether the reply ran out of tokens while still thinking.

    With thinking on, a finished reply is the reasoning followed by the answer. Servers
    differ in how they separate them: some keep ``</think>``, vLLM here drops it and leaves
    a blank gap of three newlines, so the answer is what follows the last such gap."""
    content = re.sub(r"<\|(?:im_end|endoftext)\|>", "", content)
    if not thinking:
        return content.strip(), False
    if finish_reason == "length":
        return "", True
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1].strip(), False
    if "\n\n\n" in content:
        return content.rsplit("\n\n\n", 1)[1].strip(), False
    return "", True


def make_openai_chat(api_url: str, api_key: str, model: str, timeout: float) -> ChatFn:
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=3
    )

    def chat(
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        thinking: bool,
    ) -> tuple[str, int, int, str]:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            resp.choices[0].finish_reason or "",
        )

    return chat


def run_one(spec: Spec, settings: BaselineSettings, chat: ChatFn) -> dict[str, Any]:
    """Answer one question and score it. Failures are recorded, not raised."""
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        "run_id": spec.run_id,
        "doc_id": spec.doc_id,
        "question": spec.question,
        "gold_answer": spec.gold["answer"],
        "gold_answer_format": spec.gold["answer_format"],
        "gold_evidence_pages_1idx": spec.gold["evidence_pages"],
        "gold_evidence_sources": spec.gold.get("evidence_sources", []),
        "error": None,
    }
    try:
        texts = page_texts(spec.pdf_path, settings.parsed_dir)
        pages = select_pages(spec.question, texts, settings)
        images = (
            render_pages(spec.pdf_path, pages, settings.max_side)
            if settings.modality == "image"
            else None
        )
        messages = build_messages(
            spec.question, pages, images, texts if images is None else None
        )
        max_tokens = settings.max_tokens or (12000 if settings.thinking else 512)
        temp = settings.temperature
        if temp is None:
            temp = 0.6 if settings.thinking else 0.0
        reply, p_tok, c_tok, finish = chat(
            messages, max_tokens, temp, settings.thinking
        )
        answer, truncated = clean_reply(reply, settings.thinking, finish)
        extracted = extract_answer(answer) if answer else ""
        score = eval_score(spec.gold["answer"], extracted, spec.gold["answer_format"])
        gold_pages = spec.gold["evidence_pages"]
        rec.update(
            n_pages_in_document=len(texts),
            pages_sent=pages,
            raw_answer=answer,
            raw_reply_tail=reply[-300:],
            finish_reason=finish,
            extracted_answer=extracted,
            score_official_rule=score,
            truncated_in_thinking=truncated,
            gold_page_in_context=bool(set(pages) & set(gold_pages))
            if gold_pages
            else None,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the batch
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        rec["score_official_rule"] = 0.0
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def load_done(results_path: Path) -> dict[str, dict[str, Any]]:
    """Finished records from a previous invocation; errored ones are retried."""
    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:
                    done[rec["run_id"]] = rec
    return done


def run_baseline(
    specs: list[Spec],
    settings: BaselineSettings,
    chat: ChatFn,
    results_path: Path,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every spec not already finished, appending each record to ``results_path``."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(results_path)
    todo = [s for s in specs if s.run_id not in done]
    records = dict(done)
    finished = 0
    with (
        ThreadPoolExecutor(max_workers=settings.workers) as pool,
        results_path.open("a") as out,
    ):
        futures = [pool.submit(run_one, s, settings, chat) for s in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if rec["error"] is None:
                records[rec["run_id"]] = rec
            finished += 1
            if on_progress:
                on_progress(finished, len(todo), rec)
    return [records[s.run_id] for s in specs if s.run_id in records]


def summarize(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Accuracy overall, by answer format and by evidence page count, plus an F1."""

    def acc(rows: list[dict[str, Any]]) -> float:
        return sum(r["score_official_rule"] for r in rows) / len(rows) if rows else 0.0

    unanswerable = [
        r for r in records if str(r["gold_answer"]).strip() == "Not answerable"
    ]
    answerable = [r for r in records if r not in unanswerable]
    answered = [
        r
        for r in records
        if r.get("extracted_answer", "").strip().lower() != "not answerable"
        and r.get("extracted_answer", "") != ""
    ]
    recall, precision = acc(answerable), acc(answered)
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_format[str(r["gold_answer_format"])].append(r)
    with_gold = [r for r in records if r["gold_evidence_pages_1idx"]]
    return {
        "questions_scored": len(records),
        "questions_total": total,
        "accuracy": acc(records),
        "accuracy_failed_counted_zero": sum(r["score_official_rule"] for r in records)
        / max(total, 1),
        "accuracy_answerable": recall,
        "accuracy_unanswerable": acc(unanswerable),
        "f1_approx": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "by_answer_format": {k: (len(v), acc(v)) for k, v in sorted(by_format.items())},
        "single_page": acc(
            [r for r in with_gold if len(r["gold_evidence_pages_1idx"]) == 1]
        ),
        "multi_page": acc(
            [r for r in with_gold if len(r["gold_evidence_pages_1idx"]) > 1]
        ),
        "gold_page_in_context": (
            sum(bool(r.get("gold_page_in_context")) for r in with_gold) / len(with_gold)
            if with_gold
            else None
        ),
        "truncated_in_thinking": sum(
            bool(r.get("truncated_in_thinking")) for r in records
        ),
        "documents_cut_to_max_pages": sum(
            1
            for r in records
            if len(r.get("pages_sent", [])) < r.get("n_pages_in_document", 0)
        ),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in records),
    }


def pine_scores(batch_dir: Path) -> dict[str, float]:
    """Per-question scores of a finished pipeline batch, for a paired comparison."""
    out = {}
    for f in batch_dir.glob("*/06_evaluation/result.json"):
        r = json.loads(f.read_text())
        out[r["run_id"]] = float(r["score_official_rule"])
    return out
