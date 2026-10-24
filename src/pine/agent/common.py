"""Shared infrastructure for the agent runs: settings, logging, JSON helpers,
page rendering and the cost ledger.  Model access lives in
``pine.agent.llm`` (Pydantic AI models with caching and tracing)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import warnings
from collections.abc import Callable, Coroutine
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
    vision_api_url: str | None = None
    vision_api_key: str | None = None
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    max_rounds: int = 2
    max_tool_calls: int = 12
    # "lean" (default): Docling structure in code plus a transcription of each large image
    # (agent/lean_index.py); "full": the original model page summaries + model section tree
    # (agent/index.py), kept for comparison. See INDEX.md.
    index_mode: str = "lean"
    # "procedural": read the plan's anchor pages in code, then one typed decision (default);
    # "agentic": the navigator picks its own tool calls (the original design)
    navigator: str = "procedural"
    # index page summaries in flight at once (one OpenRouter / vLLM request each)
    index_concurrency: int = 8
    cache_dir: Path | None = None
    docling_url: str | None = None
    rapidocr_backend: str = "torch"
    # False sends OpenRouter ``reasoning: {enabled: false}`` (hybrid-thinking models).
    reasoning: bool = True
    # OpenRouter providers never to route to (e.g. Together rejects qwen3.5 with reasoning
    # off and a required tool call). Empty keeps OpenRouter's default routing.
    openrouter_ignore_providers: list[str] = field(default_factory=list)
    # roles that think even when ``reasoning`` is False: "planner", "navigator", "verifier"
    # (index summaries and vision readings keep ``reasoning``)
    reasoning_roles: list[str] = field(default_factory=list)
    # Multiplies every call's max_tokens; reasoning models need room to think before answering.
    max_tokens_scale: float = 1.0
    # True replaces every call's greedy decoding with the Qwen3.5 model card's thinking-mode
    # sampling (temperature 0.6, top_p 0.95, top_k 20): greedy decoding makes a thinking model
    # loop on itself until the token budget runs out. Off by default so old runs and caches
    # are unchanged.
    thinking_sampling: bool = False
    # False builds the page index without thinking even when ``reasoning`` is on: the index makes
    # about one short call per page, so thinking there is slow and adds nothing to the answers.
    index_reasoning: bool = True
    presence_penalty: float = 0.0
    # LLM judge for step 6. Off when judge_model is None (the rule-based scorer is then the
    # score). The URL and key default to the ones the agent uses, so one vLLM server can do both.
    judge_model: str | None = None
    judge_api_url: str | None = None
    judge_api_key: str | None = None
    judge_mode: str = "judge"  # judge | extract | both
    judge_disable_thinking: bool = False
    judge_max_tokens: int = 300  # raise for a reasoning judge: its thinking counts too
    judge_extra_body: dict[str, Any] | None = None  # e.g. {"reasoning_effort": "low"}
    judge_chat: Callable[[str], tuple[str, int, int]] | None = (
        None  # tests inject a stub
    )
    # Seconds a single model call may take; thinking calls on a slow GPU need far more than 180.
    request_timeout_s: float = 180.0
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
            "vision_api_url": self.vision_api_url,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "index_mode": self.index_mode,
            "navigator": self.navigator,
            "index_concurrency": self.index_concurrency,
            "docling_url": self.docling_url,
            "rapidocr_backend": self.rapidocr_backend,
            "reasoning": self.reasoning,
            "openrouter_ignore_providers": self.openrouter_ignore_providers,
            "reasoning_roles": self.reasoning_roles,
            "max_tokens_scale": self.max_tokens_scale,
            "thinking_sampling": self.thinking_sampling,
            "index_reasoning": self.index_reasoning,
            "presence_penalty": self.presence_penalty,
            "request_timeout_s": self.request_timeout_s,
            "judge_model": self.judge_model,
            "judge_api_url": self.judge_api_url,
            "judge_mode": self.judge_mode,
            "judge_disable_thinking": self.judge_disable_thinking,
            "judge_max_tokens": self.judge_max_tokens,
            "judge_extra_body": self.judge_extra_body,
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
    pdf_path: Path,
    page_no: int,
    max_side: int = 1024,
    quality: int = 80,
    clip: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Render a 1-indexed page to JPEG bytes with the longest side <= max_side.

    ``clip`` (x0, y0, x1, y1 in PDF points, origin top-left) renders only that region,
    so a figure gets the whole pixel budget."""
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    rect = page.rect if clip is None else pymupdf.Rect(*clip) & page.rect
    scale = max_side / max(rect.width, rect.height)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        alpha=False,
        clip=None if clip is None else rect,
    )
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


def run_on_thread_loop[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on this thread's persistent event loop, the one ``Agent.run_sync`` uses.

    ``asyncio.run`` closes its loop when done, which strands the pooled connections of the
    shared provider HTTP client: the next ``run_sync`` then fails with "Event loop is closed"."""
    try:
        with (
            warnings.catch_warnings()
        ):  # "no current event loop" is the case handled here
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


_SECRET_FLAGS = ("--api-key", "--vision-api-key", "--judge-api-key")


def redacted_argv(argv: list[str] | None = None) -> str:
    """The command line with the value of every ``--*api-key`` flag masked, so run
    logs and manifests never contain a key passed on the command line."""
    import sys

    args = list(sys.argv if argv is None else argv)
    out: list[str] = []
    mask_next = False
    for a in args:
        if mask_next:
            out.append("***")
            mask_next = False
        elif a in _SECRET_FLAGS:
            out.append(a)
            mask_next = True
        elif a.startswith(tuple(f + "=" for f in _SECRET_FLAGS)):
            out.append(a.split("=", 1)[0] + "=***")
        else:
            out.append(a)
    return " ".join(out)
