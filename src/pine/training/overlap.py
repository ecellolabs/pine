"""Document overlap between the ViDoRe training documents and MMLongBench-Doc.

MMLongBench's doc_ids are mostly hashes or arXiv ids, so names cannot be compared; the
documents are compared by text. Each document becomes a set of word 8-grams (shingles),
subsampled deterministically (a shingle is kept when its crc32 falls in 1/``SAMPLE`` of the
range, the same rule for every document, so the overlap estimate is unbiased). ViDoRe text
is its page Markdown (``raw/<subset>/corpus_text.parquet``); MMLongBench text is extracted
from the PDFs with PyMuPDF. Different extraction only adds noise: a shared document still
shares most of its 8-grams.

``containment`` = shared shingles / shingles of the smaller document, so a ViDoRe report
that includes a whole MMLongBench brochure is caught as well as an identical file.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SHINGLE = 8
SAMPLE = 8  # keep about 1 in 8 shingles
THRESHOLD = 0.3


def shingles(text: str, n: int = SHINGLE, sample: int = SAMPLE) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out: set[int] = set()
    for i in range(len(words) - n + 1):
        h = zlib.crc32(" ".join(words[i : i + n]).encode())
        if h % sample == 0:
            out.add(h)
    return out


def vidore_texts(raw_dir: Path) -> dict[str, str]:
    """``{doc_id: text of all pages}`` for every subset downloaded under ``raw_dir``."""
    import pandas as pd

    texts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for f in sorted(raw_dir.glob("*/corpus_text.parquet")):
        for r in pd.read_parquet(f).itertuples():
            texts[str(r.doc_id)].append((int(r.page_number_in_doc), r.markdown or ""))
    return {d: "\n".join(t for _, t in sorted(pages)) for d, pages in texts.items()}


def pdf_text(path: Path) -> str:
    import pymupdf

    # broken image colour spaces in some PDFs print errors that do not affect the text
    pymupdf.TOOLS.mupdf_display_errors(False)
    with pymupdf.open(path) as pdf:
        return "\n".join(page.get_text() for page in pdf)


def find_overlap(
    vidore: dict[str, str], other: dict[str, str], threshold: float = THRESHOLD
) -> dict[str, Any]:
    """Best ViDoRe match for every ``other`` document and the ViDoRe doc_ids at or above
    ``threshold`` (the ``exclude`` list for ``05_training.py --exclude-docs``)."""
    index: dict[int, list[str]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for doc_id, text in vidore.items():
        sh = shingles(text)
        sizes[doc_id] = len(sh)
        for h in sh:
            index[h].append(doc_id)
    matches: list[dict[str, Any]] = []
    for other_id, text in sorted(other.items()):
        sh = shingles(text)
        hits: Counter[str] = Counter()
        for h in sh:
            hits.update(index.get(h, ()))
        best: dict[str, Any] = {"other": other_id, "shingles": len(sh), "vidore": None}
        scored = [
            (n / min(len(sh), sizes[d]), d) for d, n in hits.items() if sizes[d] and sh
        ]
        if scored:
            score, doc_id = max(scored)
            best.update(vidore=doc_id, containment=round(score, 4), shared=hits[doc_id])
        matches.append(best)
    exclude = sorted(
        {m["vidore"] for m in matches if m.get("containment", 0) >= threshold}
    )
    return {
        "threshold": threshold,
        "vidore_documents": len(vidore),
        "other_documents": len(other),
        "exclude": exclude,
        "matches": sorted(matches, key=lambda m: -m.get("containment", 0)),
    }
