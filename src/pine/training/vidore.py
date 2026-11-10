"""ViDoRe V3 (English documents) turned into STaR examples in the ``--navigator direct`` format.

Each example is one question on one PDF with 10 candidate pages, like the planner hands the
answer model at inference: the gold pages plus the closest BM25 matches from the same document
(ViDoRe's own page Markdown). About 15% become unanswerable: the gold pages and the pages
either side of them are left out and the answer is ``Not answerable``. Scanned pages (no text
layer, one image covering the page) are out of scope: a question whose gold page is a scan is
dropped, and scans are never used as distractors. The long gold sentence is rewritten into MMLongBench's short form
(Int / Float / Str / List) by a Qwen model behind any OpenAI-compatible endpoint.

Layout under ``out_dir``::

    raw/<subset>/            queries.parquet, qrels.parquet, documents_metadata.parquet,
                             corpus_text.parquet (no images), pdfs/<file_name>.pdf
    short_answers.jsonl      rewrite cache, one line per question (resumable)
    star_train.jsonl, star_dev.jsonl   the examples (dev = about 10% of documents)
    manifest.json            settings, counts, dropped questions with the reason

Stages are idempotent: files that exist are not downloaded or rewritten again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pine.agent.locate import BM25

log = logging.getLogger(__name__)

# subsets whose documents are English (energy, physics and finance_fr are French documents)
ENGLISH_SUBSETS = (
    "hr",
    "finance_en",
    "industrial",
    "pharmaceuticals",
    "computer_science",
)
# question types with a short answer; anything also tagged open-ended is left out
SHORT_TYPES = {"extractive", "numerical", "boolean", "enumerative"}
REPO = "vidore/vidore_v3_{subset}"
TABLES = ("queries", "qrels", "documents_metadata")
FORMATS = ("Int", "Float", "Str", "List")
MAX_ANSWER_WORDS = 10  # per answer, or per item of a list; longer rewrites are dropped
# a page is a scan (out of scope) when its PDF text layer is nearly empty and one image
# covers most of it
SCAN_MAX_CHARS = 50
SCAN_MIN_IMAGE_SHARE = 0.6

SHORTEN_PROMPT = """Rewrite the reference answer of a document question into the short form used by the MMLongBench-Doc benchmark.

Question: {question}
Reference answer: {answer}

Rules:
- Keep only the answer itself: a number, a short phrase, yes/no, or a list.
- Copy the reference's own words; never abbreviate, paraphrase or add details.
- Several items (names, values, steps) are a List, never one comma-separated string.
- Copy numbers exactly as written in the reference; keep a % sign or a unit only when the question asks for it.
- answer_format is Int (a whole number), Float (a decimal number or percentage), Str (text, yes/no) or List (several items, written like ["a", "b"]).
- If the reference has no single short answer, reply {{"answer": null, "answer_format": null}}.

Reply with JSON only: {{"answer": "...", "answer_format": "Int|Float|Str|List"}}"""

# chat(prompt) -> reply text
ChatFn = Callable[[str], str]


@dataclass
class BuildSettings:
    out_dir: Path
    subsets: tuple[str, ...] = ENGLISH_SUBSETS
    pages: int = 10
    unanswerable_share: float = 0.15
    dev_share: float = 0.10
    limit: int | None = None  # first N kept questions per subset (for a quick test)
    workers: int = 8
    shorten_model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- download
def _read_table(repo: str, name: str, columns: list[str] | None = None) -> Any:
    """All parquet shards of one table of a Hub dataset (larger subsets are sharded)."""
    import pandas as pd
    from huggingface_hub import HfFileSystem

    shards = sorted(HfFileSystem().glob(f"datasets/{repo}/{name}/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet files for {repo}/{name}")
    return pd.concat(
        [pd.read_parquet(f"hf://{s}", columns=columns) for s in shards],
        ignore_index=True,
    )


def download_subset(subset: str, raw_dir: Path) -> Path:
    """Fetch one subset's tables, the corpus text (column read, no images) and its PDFs."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo = REPO.format(subset=subset)
    d = raw_dir / subset
    d.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        out = d / f"{name}.parquet"
        if not out.exists():
            _read_table(repo, name).to_parquet(out)
    corpus = d / "corpus_text.parquet"
    if not corpus.exists():
        _read_table(
            repo, "corpus", ["corpus_id", "doc_id", "page_number_in_doc", "markdown"]
        ).to_parquet(corpus)
    meta = pd.read_parquet(d / "documents_metadata.parquet")
    for file_name in meta.file_name:
        if not (d / "pdfs" / file_name).exists():
            hf_hub_download(
                repo, f"pdfs/{file_name}", repo_type="dataset", local_dir=str(d)
            )
    log.info("%s: downloaded to %s", subset, d)
    return d


# --------------------------------------------------------------------------- selection
def _hash_share(key: str) -> float:
    """Stable value in [0, 1) for a key, so splits and choices replicate across runs."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0x100000000


def keep_type(query_types: list[str]) -> bool:
    types = set(query_types)
    return bool(types & SHORT_TYPES) and "open-ended" not in types


def scanned_pages(pdf: Any) -> set[int]:
    """1-indexed pages of an open PyMuPDF document that are scans (see SCAN_*)."""
    import pymupdf

    out = set()
    for i, page in enumerate(pdf, 1):
        if len(page.get_text().strip()) >= SCAN_MAX_CHARS:
            continue
        area = page.rect.width * page.rect.height
        for info in page.get_image_info():
            box = pymupdf.Rect(info["bbox"]) & page.rect
            if box.width * box.height >= SCAN_MIN_IMAGE_SHARE * area:
                out.add(i)
                break
    return out


def gold_neighbourhood(gold: list[int]) -> set[int]:
    """Gold pages and the pages either side: an answer often runs over onto them."""
    return {p + d for p in gold for d in (-1, 0, 1)}


def candidate_pages(
    question: str,
    pages_md: dict[int, str],
    gold: list[int],
    k: int,
    answerable: bool,
    exclude: set[int] | None = None,
) -> list[int]:
    """``k`` pages in page order: the gold pages (answerable only), then BM25 matches of the
    question over the document's pages, then the remaining pages in order. ``exclude`` pages
    (scans) are never distractors. An unanswerable example contains no gold page and no
    page next to one."""
    banned = set(exclude or ()) - (set(gold) if answerable else set())
    if not answerable:
        banned |= gold_neighbourhood(gold)
    ranked = [p for p, _ in BM25(dict(pages_md)).search(question, k=len(pages_md))]
    order = [*(gold if answerable else []), *ranked, *sorted(pages_md)]
    picked = list(dict.fromkeys(p for p in order if p not in banned))
    return sorted(picked[:k])


def select_questions(
    subset: str, d: Path, settings: BuildSettings
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """English short-answer questions with their document, gold pages and candidate pages."""
    import pandas as pd
    import pymupdf

    queries = pd.read_parquet(d / "queries.parquet")
    qrels = pd.read_parquet(d / "qrels.parquet")
    corpus = pd.read_parquet(d / "corpus_text.parquet")
    meta = pd.read_parquet(d / "documents_metadata.parquet").set_index("doc_id")
    page_of = {
        int(r.corpus_id): (
            r.doc_id,
            int(r.page_number_in_doc) + 1,
        )  # 0-indexed upstream
        for r in corpus.itertuples()
    }
    doc_pages: dict[str, dict[int, str]] = {}
    for r in corpus.itertuples():
        doc_pages.setdefault(r.doc_id, {})[int(r.page_number_in_doc) + 1] = (
            r.markdown or ""
        )
    n_pdf: dict[str, int] = {}
    scans: dict[str, set[int]] = {}
    for doc_id, row in meta.iterrows():
        with pymupdf.open(d / "pdfs" / row.file_name) as pdf:
            n_pdf[str(doc_id)] = pdf.page_count
            scans[str(doc_id)] = scanned_pages(pdf)
    gold_by_q: dict[int, list[tuple[str, int, int]]] = {}
    for r in qrels.itertuples():
        if r.score > 0:
            doc_id, page = page_of[int(r.corpus_id)]
            gold_by_q.setdefault(int(r.query_id), []).append(
                (doc_id, page, int(r.score))
            )

    out: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    dropped["scanned_pages_left_out"] = sum(len(v) for v in scans.values())
    for q in queries.itertuples():
        if q.language != "english":
            continue
        if not keep_type(list(q.query_types)):
            dropped["long_answer_type"] += 1
            continue
        gold = gold_by_q.get(int(q.query_id), [])
        docs = {g[0] for g in gold}
        if len(docs) != 1:
            dropped["no_gold_page" if not docs else "gold_in_several_documents"] += 1
            continue
        doc_id = docs.pop()
        if n_pdf[doc_id] != len(doc_pages[doc_id]):
            dropped["pdf_page_count_mismatch"] += 1
            continue
        # fully relevant pages first, so a cap keeps the strongest evidence
        gold_pages = [p for _, p, _ in sorted(gold, key=lambda g: (-g[2], g[1]))]
        gold_pages = list(dict.fromkeys(gold_pages))
        if len(gold_pages) > settings.pages:
            dropped["more_gold_pages_than_slots"] += 1
            continue
        if set(gold_pages) & scans[doc_id]:
            dropped["gold_page_scanned"] += 1
            continue
        qid = f"{subset}_q{int(q.query_id)}"
        answerable = _hash_share("unanswerable:" + qid) >= settings.unanswerable_share
        usable = set(range(1, n_pdf[doc_id] + 1)) - scans[doc_id]
        if (
            not answerable
            and len(usable - gold_neighbourhood(gold_pages)) < settings.pages
        ):
            answerable = True  # too few other pages for a full distractor set
        out.append(
            {
                "id": qid,
                "subset": subset,
                "doc_id": doc_id,
                "pdf": str(Path("raw") / subset / "pdfs" / meta.loc[doc_id].file_name),
                "n_pages": n_pdf[doc_id],
                "split": "dev"
                if _hash_share("dev:" + doc_id) < settings.dev_share
                else "train",
                "question": q.query,
                "query_types": list(q.query_types),
                "content_type": list(q.content_type),
                "answer_long": q.answer,
                "answer": None if answerable else "Not answerable",
                "answer_format": None if answerable else "None",
                "answerable": answerable,
                "gold_pages": sorted(gold_pages),
                "gold_pages_fully_relevant": sorted({p for _, p, s in gold if s >= 2}),
                "candidate_pages": candidate_pages(
                    q.query,
                    doc_pages[doc_id],
                    gold_pages,
                    settings.pages,
                    answerable,
                    exclude=scans[doc_id],
                ),
            }
        )
        if settings.limit and len(out) >= settings.limit:
            break
    return out, dropped


# --------------------------------------------------------------------------- short answers
def parse_short(reply: str) -> tuple[str | None, str | None]:
    """``(answer, answer_format)`` from the rewrite reply, or ``(None, None)`` when unusable."""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None, None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    answer, fmt = j.get("answer"), j.get("answer_format")
    if answer in (None, "") or fmt not in FORMATS:
        return None, None
    items = answer if isinstance(answer, list) else _list_items(answer, fmt)
    if any(len(str(a).split()) > MAX_ANSWER_WORDS for a in items):
        return None, None
    if isinstance(answer, list):
        answer = json.dumps([str(a) for a in answer], ensure_ascii=False)
    return str(answer).strip(), fmt


def _list_items(answer: Any, fmt: str) -> list[Any]:
    """Items of a list answer written as a JSON string, else the answer itself."""
    if fmt == "List" and isinstance(answer, str):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            return [answer]
        if isinstance(parsed, list):
            return parsed
    return [answer]


def make_chat(api_url: str, api_key: str, model: str, timeout: float = 120.0) -> ChatFn:
    """Thinking-off chat on vLLM (chat template switch) and OpenRouter (reasoning flag)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=3
    )

    def chat(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning": {"enabled": False},
            },
        )
        return resp.choices[0].message.content or ""

    return chat


def shorten_answers(
    records: list[dict[str, Any]], cache_path: Path, chat: ChatFn | None, workers: int
) -> set[str]:
    """Fill ``answer`` / ``answer_format`` of answerable records from the cache, asking
    ``chat`` for the missing ones (None = cache only). Unusable rewrites stay None.
    Returns the ids that have a rewrite (usable or not)."""
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            cache[j["id"]] = j
    todo = [r for r in records if r["answerable"] and r["id"] not in cache]
    if chat is not None and todo:

        def one(r: dict[str, Any]) -> dict[str, Any]:
            try:
                reply = chat(
                    SHORTEN_PROMPT.format(
                        question=r["question"], answer=r["answer_long"]
                    )
                )
            except Exception as exc:  # noqa: BLE001 - retried on the next run
                log.warning("rewrite failed for %s: %s", r["id"], exc)
                return {}
            answer, fmt = parse_short(reply)
            return {
                "id": r["id"],
                "reply": reply,
                "answer": answer,
                "answer_format": fmt,
            }

        with (
            ThreadPoolExecutor(max_workers=workers) as pool,
            cache_path.open("a", encoding="utf-8") as f,
        ):
            for done, row in enumerate(pool.map(one, todo), 1):
                if row:
                    cache[row["id"]] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if done % 100 == 0:
                    log.info("rewrote %d/%d answers", done, len(todo))
    for r in records:
        if r["answerable"] and r["id"] in cache:
            r["answer"] = cache[r["id"]]["answer"]
            r["answer_format"] = cache[r["id"]]["answer_format"]
    return set(cache)


# --------------------------------------------------------------------------- build
def build(settings: BuildSettings, chat: ChatFn | None) -> dict[str, Any]:
    """Download, select, shorten and write the STaR files; returns the manifest."""
    out = settings.out_dir
    records: list[dict[str, Any]] = []
    dropped: dict[str, dict[str, int]] = {}
    for subset in settings.subsets:
        d = download_subset(subset, out / "raw")
        recs, drops = select_questions(subset, d, settings)
        records += recs
        dropped[subset] = dict(drops)
        log.info("%s: %d questions kept, dropped %s", subset, len(recs), dict(drops))
    rewritten = shorten_answers(
        records, out / "short_answers.jsonl", chat, settings.workers
    )
    ready = [r for r in records if r["answer"] is not None]
    missing = sum(1 for r in records if r["answerable"] and r["id"] not in rewritten)
    for split in ("train", "dev"):
        with (out / f"star_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in ready:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "settings": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(settings).items()
        },
        "examples": len(ready),
        "by_split": dict(Counter(r["split"] for r in ready)),
        "by_subset": dict(Counter(r["subset"] for r in ready)),
        "by_format": dict(Counter(r["answer_format"] for r in ready)),
        "unanswerable": sum(1 for r in ready if not r["answerable"]),
        "documents": len({(r["subset"], r["doc_id"]) for r in ready}),
        "not_rewritten_yet": missing,
        "rewrite_unusable": sum(
            1 for r in records if r["answerable"] and r["answer"] is None
        )
        - missing,
        "dropped": dropped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
