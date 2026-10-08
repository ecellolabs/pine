"""Shared infrastructure for the agent runs: settings, logging, JSON helpers,
page rendering and the cost ledger.  Model access lives in
``pine.agent.llm`` (Pydantic AI models with caching and tracing)."""

from __future__ import annotations

import io
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.models import Model

DEFAULT_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEXT_MODEL = "qwen/qwen-2.5-7b-instruct"
DEFAULT_VISION_MODEL = "qwen/qwen3-vl-8b-instruct"

STEP_DIRS = {
    0: "00_input",
    1: "01_parser",
    2: "02_index",
    3: "03_planner",
    4: "04_orchestrator",
    5: "05_evidence_verifier",
    6: "06_evaluation",
}


@dataclass
class AgentSettings:
    """Everything a run needs to know that is not the sample itself."""

    output_dir: Path
    api_url: str = DEFAULT_API_URL
    api_key: str = ""
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    max_rounds: int = 2
    max_tool_calls: int = 12
    cache_dir: Path | None = None
    docling_url: str | None = None
    rapidocr_backend: str = "torch"
    # Optional factory returning a Pydantic AI ``Model`` for a model name; used by
    # tests (FunctionModel / TestModel) and by anyone who wants a custom provider.
    model_factory: Callable[[str], Model] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir or (self.output_dir / ".cache" / "llm")

    def public(self) -> dict[str, Any]:
        """Settings without the API key, for writing into run folders."""
        return {
            "api_url": self.api_url,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "docling_url": self.docling_url,
            "rapidocr_backend": self.rapidocr_backend,
            "llm_cache_dir": str(self.llm_cache_dir),
            "agent_framework": "pydantic-ai",
        }


# --------------------------------------------------------------------------- batch folders
def batch_label(selection: Any) -> str:
    """Folder label for a ``--max-samples`` value: sample-set name, ``firstN`` or ``full``."""
    name = getattr(selection, "name", None)
    if name:
        return str(name)
    if isinstance(selection, int):
        return f"first{selection}"
    return "full"


def new_batch_dir(root: Path, label: str, date: str | None = None) -> Path:
    """Create and return ``root/<label>_<YYYY-MM-DD>``; if that folder already exists
    (a copy of the same experiment on the same day) append ``_2``, ``_3``, ..."""
    import datetime as _dt

    day = date or _dt.datetime.now().astimezone().date().isoformat()
    base = root / f"{label}_{day}"
    candidate = base
    n = 1
    while candidate.exists():
        n += 1
        candidate = root / f"{label}_{day}_{n}"
    candidate.mkdir(parents=True)
    return candidate


# --------------------------------------------------------------------------- logging
def setup_logger(
    name: str, log_file: Path, run_log: Path | None = None
) -> logging.Logger:
    """Logger writing to a per-step file, the combined run.log and stderr.

    Every line names the step logger and the ``module.py:function`` that emitted
    it, so ``run.log`` reads as an execution trace of the source files."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(module)s.py:%(funcName)s | %(message)s"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for path in [log_file, run_log] if run_log else [log_file]:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.setLevel(logging.INFO)
    logger.addHandler(stream)
    logger.propagate = False
    return logger


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------- page images
def render_page(
    pdf_path: Path, page_no: int, max_side: int = 1024, quality: int = 80
) -> bytes:
    """Render a 1-indexed page to JPEG bytes with the longest side <= max_side."""
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    rect = page.rect
    scale = max_side / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- cost ledger
class CostLedger:
    """Accumulates model usage per run; every call is also appended to llm_calls.jsonl."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.calls = 0
        self.cached = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.latency_s = 0.0

    def add(self, usage: dict[str, Any], latency: float, from_cache: bool) -> None:
        self.calls += 1
        if from_cache:
            self.cached += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.cost_usd += float(usage.get("cost", 0) or 0)
        self.latency_s += latency

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "served_from_cache": self.cached,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_latency_s": round(self.latency_s, 2),
        }
