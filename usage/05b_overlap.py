"""Stage 05b: find ViDoRe training documents that are also MMLongBench-Doc documents.

Compares document text (word 8-grams), not names: MMLongBench's doc_ids are mostly hashes.
Needs the ViDoRe page text from stage 05 (``--no-rewrite`` is enough) and the MMLongBench
PDFs (atria's cache after ``usage/00_prepare_dataset.py mmlongbench_doc``). No model calls.

    uv run usage/05b_overlap.py --data-dir data/vidore_v3
    uv run usage/05_training.py --exclude-docs data/vidore_v3/mmlongbench_overlap.json ...

Writes <data-dir>/mmlongbench_overlap.json: the best ViDoRe match of every MMLongBench
document with its containment score, and ``exclude``, the ViDoRe doc_ids at or above
--threshold. Read the top matches by hand before trusting the threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pine.training.overlap import THRESHOLD, find_overlap, pdf_text, vidore_texts

ATRIA_PDFS = Path.home() / ".cache/atria/datasets/mmlongbench_doc/pdfs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/vidore_v3"))
    p.add_argument("--mmlongbench-pdfs", type=Path, default=ATRIA_PDFS)
    p.add_argument(
        "--expected",
        type=int,
        default=135,
        help="MMLongBench documents expected (warns when fewer PDFs are present).",
    )
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    log = logging.getLogger("overlap")

    vidore = vidore_texts(args.data_dir / "raw")
    if not vidore:
        p.error(f"no corpus_text.parquet under {args.data_dir / 'raw'}; run stage 05")
    pdfs = sorted(args.mmlongbench_pdfs.glob("*.pdf"))
    if len(pdfs) < args.expected:
        log.warning(
            "only %d of %d MMLongBench PDFs in %s: the check is partial",
            len(pdfs),
            args.expected,
            args.mmlongbench_pdfs,
        )
    other = {}
    for f in pdfs:
        try:
            other[f.name] = pdf_text(f)
        except Exception as exc:  # noqa: BLE001 - one broken PDF must not stop the check
            log.warning("cannot read %s: %s", f.name, exc)
    report = find_overlap(vidore, other, args.threshold)
    report["mmlongbench_pdfs"] = str(args.mmlongbench_pdfs)
    report["partial"] = len(other) < args.expected
    out = args.data_dir / "mmlongbench_overlap.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m in report["matches"][:5]:
        log.info(
            "%-50s best %s (containment %s)",
            m["other"],
            m["vidore"],
            m.get("containment"),
        )
    print(
        f"{len(report['exclude'])} ViDoRe documents at or above {args.threshold}: "
        f"{report['exclude']}\nreport: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
