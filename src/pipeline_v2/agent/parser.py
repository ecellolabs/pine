"""Step 1 - Parser (M2): Docling layout-aware OCR -> Markdown per page.

Default: the Docling library runs in-process on the whole PDF (RapidOCR for
pages without a text layer).  With ``docling_url`` the remote Docling API
service is used instead, page by page, through
``pipeline_v2.parsers.docling.DoclingTransform`` (the cluster setup).

Writes:
  01_parser/pages/page_NNN.md      Docling markdown for each page
  01_parser/docling_document.json  DoclingDocument (whole document, in-process mode)
  01_parser/page_stats.json        chars per page (docling vs native), tables, pictures, flags
  01_parser/parser.log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, cast

from pipeline_v2.agent.common import setup_logger, write_json

EMPTY_FLAG = (
    "EMPTY_OR_NEAR_EMPTY (page content not captured as text; needs image inspection)"
)
OCR_FLAG = "OCR_ONLY (no native text layer; text comes from RapidOCR)"
LOSS_FLAG = "TEXT_LOSS (docling captured <40% of native text)"


def _parse_in_process(
    pdf_path: Path, n_pages: int, backend: str, max_pages: int | None, log: Any
) -> tuple[dict[int, dict[str, Any]], str, float, dict[str, Any] | None]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    if backend not in ("onnxruntime", "openvino", "paddle", "torch"):
        raise ValueError(f"unsupported RapidOCR backend {backend!r}")
    opts.ocr_options = RapidOcrOptions(backend=cast(Any, backend))
    opts.ocr_options.force_full_page_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    log.info(
        "docling options: ocr=%s engine=%s backend=%s table_structure=%s",
        opts.do_ocr,
        type(opts.ocr_options).__name__,
        backend,
        opts.do_table_structure,
    )
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"page_range": (1, max_pages)} if max_pages else {}
    result = converter.convert(str(pdf_path), **kwargs)
    elapsed = time.perf_counter() - t0
    doc = result.document
    log.info(
        "docling status=%s elapsed=%.1fs (%.2fs/page)",
        result.status,
        elapsed,
        elapsed / max(n_pages, 1),
    )
    for err in result.errors:
        log.error("docling error: %s", err)
    pages: dict[int, dict[str, Any]] = {}
    for p in range(1, n_pages + 1):
        try:
            md = doc.export_to_markdown(page_no=p)
        except Exception as exc:  # noqa: BLE001
            log.error("export_to_markdown failed page=%d: %s", p, exc)
            md = ""
        pages[p] = {
            "md": md,
            "tables": sum(
                1 for t in doc.tables if any(pr.page_no == p for pr in t.prov)
            ),
            "pictures": sum(
                1 for t in doc.pictures if any(pr.page_no == p for pr in t.prov)
            ),
        }
    return pages, str(result.status), elapsed, doc.export_to_dict()


def _parse_via_api(
    pdf_path: Path, n_pages: int, docling_url: str, log: Any
) -> tuple[dict[int, dict[str, Any]], str, float, None]:
    from atria_core.types import MultiPageDocumentInstance

    from pipeline_v2.parsers.docling import DoclingTransform

    log.info("docling API mode: %s (page by page)", docling_url)
    t0 = time.perf_counter()
    instance = MultiPageDocumentInstance.from_pdf(pdf_path)
    pages: dict[int, dict[str, Any]] = {}
    status = "SUCCESS"
    with DoclingTransform(api_url=docling_url) as transform:
        for p, page in enumerate(instance.pages, 1):
            if p > n_pages:
                break
            try:
                doc = transform(page)
                pages[p] = {
                    "md": doc.export_to_markdown(),
                    "tables": len(doc.tables),
                    "pictures": len(doc.pictures),
                }
            except Exception as exc:  # noqa: BLE001
                log.error("docling API failed page=%d: %s", p, exc)
                pages[p] = {"md": "", "tables": 0, "pictures": 0}
                status = "PARTIAL_SUCCESS"
    elapsed = time.perf_counter() - t0
    log.info("docling API status=%s elapsed=%.1fs", status, elapsed)
    return pages, status, elapsed, None


def run_parser(
    pdf_path: Path,
    step_dir: Path,
    run_log: Path,
    max_pages: int | None = None,
    rapidocr_backend: str = "torch",
    docling_url: str | None = None,
) -> dict[str, Any]:
    log = setup_logger("step1.parser", step_dir / "parser.log", run_log)
    pages_dir = step_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    stats_path = step_dir / "page_stats.json"
    if stats_path.exists():
        log.info("parser output already exists, skipping (idempotent)")
        return json.loads(stats_path.read_text())

    import pymupdf

    mu = pymupdf.open(pdf_path)
    n_pages = len(mu)
    mu_text = {i + 1: mu[i].get_text() for i in range(n_pages)}
    log.info(
        "pdf=%s pages=%d pymupdf_text_chars_total=%d",
        pdf_path.name,
        n_pages,
        sum(len(t) for t in mu_text.values()),
    )

    if docling_url:
        pages, status, elapsed, doc_dict = _parse_via_api(
            pdf_path, n_pages, docling_url, log
        )
        engine = f"Docling API ({docling_url})"
    else:
        backend = os.environ.get("RAPIDOCR_BACKEND", rapidocr_backend)
        pages, status, elapsed, doc_dict = _parse_in_process(
            pdf_path, n_pages, backend, max_pages, log
        )
        engine = "RapidOCR (docling RapidOcrOptions)"
    if doc_dict is not None:
        (step_dir / "docling_document.json").write_text(
            json.dumps(doc_dict), encoding="utf-8"
        )

    page_stats: list[dict[str, Any]] = []
    for p in range(1, n_pages + 1):
        info = pages.get(p, {"md": "", "tables": 0, "pictures": 0})
        md = info["md"]
        (pages_dir / f"page_{p:03d}.md").write_text(md, encoding="utf-8")
        st: dict[str, Any] = {
            "page": p,
            "docling_chars": len(md.strip()),
            "pymupdf_chars": len(mu_text[p].strip()),
            "tables": info["tables"],
            "pictures": info["pictures"],
            "headings": sum(1 for line in md.splitlines() if line.startswith("#")),
            "flag": None,
        }
        if st["docling_chars"] < 40:
            st["flag"] = EMPTY_FLAG
        elif st["pymupdf_chars"] == 0:
            st["flag"] = OCR_FLAG
        elif st["docling_chars"] < 0.4 * st["pymupdf_chars"]:
            st["flag"] = LOSS_FLAG
        page_stats.append(st)
        log.info(
            "page %3d: docling_chars=%5d pymupdf_chars=%5d tables=%d pictures=%d headings=%d %s",
            p,
            st["docling_chars"],
            st["pymupdf_chars"],
            st["tables"],
            st["pictures"],
            st["headings"],
            st["flag"] or "",
        )

    summary = {
        "pdf": pdf_path.name,
        "pages": n_pages,
        "docling_status": status,
        "elapsed_s": round(elapsed, 1),
        "ocr_engine": engine,
        "pages_flagged": sum(1 for s in page_stats if s["flag"]),
        "total_docling_chars": sum(s["docling_chars"] for s in page_stats),
        "total_pymupdf_chars": sum(s["pymupdf_chars"] for s in page_stats),
        "total_tables": sum(s["tables"] for s in page_stats),
        "total_pictures": sum(s["pictures"] for s in page_stats),
        "page_stats": page_stats,
    }
    write_json(stats_path, summary)
    log.info(
        "parser done: flagged=%d/%d pages, tables=%d, pictures=%d",
        summary["pages_flagged"],
        n_pages,
        summary["total_tables"],
        summary["total_pictures"],
    )
    return summary
