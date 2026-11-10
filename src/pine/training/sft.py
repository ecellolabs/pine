"""STaR samples (``star.py``) turned into an SFT set in ms-swift's multimodal chat format.

One line per example that has a chosen (correct) sample::

    {"id": ..., "messages": [{"role": "user", "content": "[Page 3]<image>[Page 7]<image>...
     <instruction>\\n\\nQuestion: ..."}, {"role": "assistant", "content": "<think>\\n...
     \\n</think>\\n\\n<answer>"}], "images": ["images/hr_q4_p003.jpg", ...]}

The user turn is built from ``direct.prompt_parts`` (the exact inference prompt; every image
becomes an ``<image>`` tag at its place) and never contains the STaR hint, also for
rationalized samples: the model has to learn to find the page itself. The assistant turn is
the chosen sample's reasoning in Qwen3.5's thinking format followed by its final answer
(``answer_source="model"``) or the gold answer (``"gold"``, which teaches the short form even
when the judge accepted a longer answer). Unrescued examples are left out.

Page images are written as JPEGs under ``<out_dir>/images`` (paths in the JSONL are
relative to ``out_dir``), so training needs no PDF renderer.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pine.agent.direct import prompt_parts

THINK_FORMAT = "<think>\n{reasoning}\n</think>\n\n{answer}"
CHARS_PER_TOKEN = 4  # rough estimate for the length report only
IMAGE_TOKENS = 600  # about one 896-px page at Qwen3.5's 32-px merged patches


def gold_answer_text(rec: dict[str, Any]) -> str:
    """The gold answer in the form the instruction asks for (a list as ``[a, b]``)."""
    answer = str(rec["answer"])
    if rec["answer_format"] == "List":
        try:
            items = json.loads(answer)
        except json.JSONDecodeError:
            return answer
        if isinstance(items, list):
            return "[" + ", ".join(str(x) for x in items) + "]"
    return answer


def sft_example(
    rec: dict[str, Any],
    sample: dict[str, Any],
    data_dir: Path,
    out_dir: Path,
    answer_source: str = "model",
) -> dict[str, Any]:
    """One ms-swift line; renders the record's pages into ``out_dir/images``."""
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    user = ""
    images: list[str] = []
    pages = iter(rec["candidate_pages"])
    for part in prompt_parts(
        data_dir / rec["pdf"], rec["candidate_pages"], rec["question"]
    ):
        if isinstance(part, bytes):
            target = image_dir / f"{rec['id']}_p{next(pages):03d}.jpg"
            target.write_bytes(part)
            images.append(str(target.relative_to(out_dir)))
            user += "<image>"
        else:
            user += part
    answer = gold_answer_text(rec) if answer_source == "gold" else sample["answer"]
    assistant = THINK_FORMAT.format(
        reasoning=sample["reasoning"].strip(), answer=answer.strip()
    )
    return {
        "id": rec["id"],
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": images,
    }


def build_sft(
    records: dict[str, dict[str, Any]],
    star_rows: Iterable[dict[str, Any]],
    data_dir: Path,
    out_dir: Path,
    answer_source: str = "model",
) -> dict[str, Any]:
    """Write ``out_dir/train.jsonl`` from the chosen samples; returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    status: Counter[str] = Counter()
    seen: set[str] = set()
    est_tokens: list[int] = []
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in star_rows:
            rid = row["id"]
            if rid in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(rid)
            if rid not in records:
                skipped["no_record"] += 1
                continue
            if row.get("chosen") is None:
                skipped[row.get("status", "unrescued")] += 1
                continue
            sample = row["samples"][row["chosen"]]
            ex = sft_example(records[rid], sample, data_dir, out_dir, answer_source)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept.append(row)
            status[row["status"]] += 1
            est_tokens.append(
                len(ex["images"]) * IMAGE_TOKENS
                + sum(len(m["content"]) for m in ex["messages"]) // CHARS_PER_TOKEN
            )
    est_tokens.sort()

    def pct(q: float) -> int | None:
        return est_tokens[int(q * (len(est_tokens) - 1))] if est_tokens else None

    manifest = {
        "examples": len(kept),
        "status": dict(status),
        "skipped": dict(skipped),
        "by_format": dict(Counter(r["answer_format"] for r in kept)),
        "unanswerable": sum(1 for r in kept if not r["answerable"]),
        "answer_source": answer_source,
        "estimated_tokens": {
            "p50": pct(0.5),
            "p90": pct(0.9),
            "max": est_tokens[-1] if est_tokens else None,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
