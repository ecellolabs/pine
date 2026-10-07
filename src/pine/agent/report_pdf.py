"""Render a Markdown report to HTML and, when a Chrome/Chromium binary is
available, to PDF (headless print)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eee; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
img { max-width: 100%; }
.small { font-size: 9pt; color: #444; }
blockquote { border-left: 3px solid #bbb; margin: 6pt 0; padding: 2pt 8pt; color: #333; }
"""


CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def find_chrome() -> str | None:
    for cand in CHROME_CANDIDATES:
        if Path(cand).is_file() or shutil.which(cand):
            return cand
    return None


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path | None:
    """Write ``<pdf_path>.html`` always; write the PDF when Chrome is found and
    return its path, else return ``None``."""
    md = md_path.read_text(encoding="utf-8")
    html_body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Report</title>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    chrome = find_chrome()
    if chrome is None:
        return None
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            str(html_path),
        ],
        check=True,
        capture_output=True,
    )
    return pdf_path
