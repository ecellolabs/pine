"""Step 5 - Evidence ledger + Verifier (M5).

1. Programmatic quote-in-page check for every ledger entry (exact / fuzzy match
   against Docling page text, or against the vision-model inspection output when
   the quote came from an image).
2. Grounding check by a Pydantic AI agent (typed ``Verdict``): does the cited
   page content support the answer?  Returns supported / not_supported /
   insufficient with explicit gaps.
3. Decision: ACCEPT, ACCEPT_WEAK_GROUNDING, ABSTAIN, ACCEPT_ABSTAIN or
   REPLAN(gaps).  An abstention is only accepted when the verifier agrees the
   information is plausibly absent (or no rounds are left); otherwise the gaps
   go back to the planner.

Writes 05_evidence_verifier/verification_round{N}.json, llm_calls.jsonl, verifier.log
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings

from pine.agent.common import (
    AgentSettings,
    CostLedger,
    read_json,
    render_page,
    setup_logger,
    write_json,
)
from pine.agent.llm import traced_model
from pine.agent.locate import RegionLocator
from pine.agent.schemas import Verdict

VERIFY_INSTRUCTIONS = """You are the VERIFIER of a grounded document question-answering system. Decide whether the cited evidence from the document supports the proposed answer. Be strict: the answer must be directly supported by the evidence text, not by general knowledge.
Report whether the quoted evidence is actually present in the cited page content, a 1-2 sentence explanation, and concrete gaps (what is still missing or should be checked: sections, pages, figures).
Evidence may be spread over several pages (a table on one page, the entity it is about on another); a link shown in any note read this round counts. Evidence marked as a vision model description was read from an image: it is checked separately against the page image, so judge only whether it supports the answer.
In check_pages, name up to 3 pages from the index view that were not read yet and may hold missing or contradicting evidence; for an abstention and for a list or count answer, always look for such pages (a list often continues on the next page).
If the answer is computed (a sum, difference or count), redo the computation from the evidence; a wrong result is "not_supported".
Name every gap with concrete page numbers, taken from the index view or the pages read (for example "inspect the chart on page 15"); a gap that names no page to look at cannot be acted on.
If the proposed status is not_answerable: return "supported" only if the pages read make it plausible that the document does not contain the information; if the evidence entries already contain the requested information, return "not_supported"; otherwise "insufficient" with gaps naming where to look."""

VERIFY_PROMPT = """Question: {question}
Proposed answer: {answer}
Proposed status: {status}

Search history: {n_calls} tool calls; pages read: {pages_read}; page images inspected: {inspected}
{previous}
Where the index points for this question (figure/table captions, best sections and pages):
{index_view}

Checks already done in code (trust them; do not reject an answer only because image evidence cannot be verified when a check confirmed it):
{checks}

Evidence (ledger entries recorded by the agent):
{ledger}

Cited page content:
{pages}"""

REREAD_INSTRUCTIONS = """You read one region of a document page. Answer only from what is visible in the image. The region may hold only part of the answer: other parts can be on other pages of the document.
First list every value or fact visible here that is relevant to the question, one per line, each with its label exactly as shown.
Then end with exactly one line: "ANSWER: <short answer>" if this region alone answers the question, "ANSWER: PARTIAL" if it holds only part of what is needed, or "ANSWER: NOT_SHOWN" if nothing relevant to the question is visible."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%$., ]+", " ", s.lower())).strip()


def quote_in_text(quote: str, text: str) -> tuple[bool, float]:
    """Exact (normalised) containment, else best sliding-window similarity ratio."""
    q, t = _norm(quote), _norm(text)
    if not q or not t:
        return False, 0.0
    if q in t:
        return True, 1.0
    n = len(q)
    best = 0.0
    step = max(1, n // 4)
    for i in range(0, max(1, len(t) - n + 1), step):
        r = difflib.SequenceMatcher(None, q, t[i : i + n]).ratio()
        if r > best:
            best = r
            if best > 0.95:
                break
    return best >= 0.8, round(best, 3)


# separators a model uses when it re-joins page lines or stitches two sources together
_PIECE_SPLIT = re.compile(r"\n+|\s+\|\s+|;\s+|,\s+|\s*(?:\.\.\.|…)\s*")


def quote_on_page(
    quote: str, page_text: str, inspections: str = ""
) -> tuple[bool, float, str | None]:
    """Is ``quote`` on the page: whole, in the page text or in that page's vision output,
    else piece by piece (a list re-joined with commas or newlines, or a quote stitched
    from page text and vision output; every piece must be found).
    Returns (found, best whole-quote similarity, source)."""
    ok_t, sim_t = quote_in_text(quote, page_text)
    if ok_t:
        return True, sim_t, "page_text"
    ok_i, sim_i = quote_in_text(quote, inspections) if inspections else (False, 0.0)
    if ok_i:
        return True, sim_i, "image_inspection"
    sim = max(sim_t, sim_i)
    pieces = [s.strip(" -*•\t") for s in _PIECE_SPLIT.split(quote)]
    pieces = [s for s in pieces if len(_norm(s)) >= 3]
    if len(pieces) < 2:
        return False, sim, None
    in_text = [quote_in_text(s, page_text)[0] for s in pieces]
    if all(in_text):
        return True, sim, "page_text"
    if inspections and all(
        t or quote_in_text(s, inspections)[0] for s, t in zip(pieces, in_text)
    ):
        return True, sim, "image_inspection"
    return False, sim, None


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, 0.005 * abs(b))


def _follows(value: float, nums: list[float]) -> bool:
    """``value`` is one of ``nums`` or follows from two of them (sum, difference,
    product, ratio, percentage) or from all of them (sum, mean)."""
    if any(_close(value, n) for n in nums):
        return True
    if len(nums) < 2:
        return False
    derived = [sum(nums), sum(nums) / len(nums)]
    for i, a in enumerate(nums):
        for b in nums[i + 1 :]:
            derived += [a + b, abs(a - b), a * b]
            for x, y in ((a, b), (b, a)):
                if y:
                    derived += [x / y, 100 * x / y]
    return any(_close(value, d) for d in derived)


def numeric_check(answer: str, quotes: list[str]) -> dict[str, Any]:
    """Does every number in the answer appear in the verified quotes, or follow from two of
    their numbers (sum, difference, product, ratio, percentage) or from all of them (sum,
    mean)?

    Skipped (``status: unchecked``) when the answer has no number or the quotes hold fewer
    than two numbers, e.g. a count of listed items. ``mismatch`` lists the answer numbers that
    neither appear nor follow, with the sums the quotes do give, for the re-plan gap."""
    ans = _numbers(answer)
    nums = [n for q in quotes for n in _numbers(q)]
    if not ans or len(nums) < 2:
        return {"status": "unchecked", "answer_numbers": ans, "quote_numbers": nums}
    unexplained = [a for a in ans if not _follows(a, nums)]
    return {
        "status": "mismatch" if unexplained else "ok",
        "answer_numbers": ans,
        "quote_numbers": nums,
        "unexplained": unexplained,
        "sum_of_quote_numbers": round(sum(nums), 4),
    }


def decide(
    status: str,
    verdict: str | None,
    any_quote_ok: bool,
    has_ledger: bool,
    round_no: int,
    max_rounds: int,
    replan_useful: bool = True,
) -> str:
    """The acceptance rule, kept separate so it can be unit-tested. ``replan_useful`` False
    (the gaps name nothing new to look at) makes this round final, as if no round were left."""
    rounds_left = round_no < max_rounds and replan_useful
    if status == "error":
        return "ABSTAIN"
    if verdict is None and status == "answered":
        # the verifier could not run: that is no evidence against the answer, so a
        # quote-checked answer stands; otherwise behave as for an unsupported one
        if any_quote_ok:
            return "ACCEPT_WEAK_GROUNDING"
        return "REPLAN" if rounds_left else "ABSTAIN"
    if status == "not_answerable":
        if verdict == "supported" or not rounds_left:
            return "ACCEPT_ABSTAIN"
        return "REPLAN"
    if verdict == "supported":
        if any_quote_ok:
            return "ACCEPT"
        return "ACCEPT_WEAK_GROUNDING"
    return "REPLAN" if rounds_left else "ABSTAIN"


_PAGE_LIST = re.compile(
    r"\b(?:pages?|pp?\.)\s*((?:\d+\s*(?:-|–|to|,|and|&)\s*)*\d+)", re.IGNORECASE
)
_VISUAL_GAP = re.compile(
    r"inspect|image|figure|chart|graph|table|visual", re.IGNORECASE
)


def gap_pages(gap: str) -> set[int]:
    """Page numbers a gap names: "page 10", "pages 11, 13 and 15", "pages 1-17"."""
    pages: set[int] = set()
    for m in _PAGE_LIST.finditer(gap):
        for a, b in re.findall(r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?", m.group(1)):
            lo = int(a)
            hi = int(b) if b else lo
            pages.update(range(lo, min(hi, lo + 200) + 1))
    return pages


def replan_has_new_target(
    gaps: list[str], read: set[int], inspected: set[int], n_pages: int
) -> bool:
    """Another round is worth it only when a gap names a page the agent has not looked at
    (or, for a figure/table gap, whose image it has not inspected). A gap that names no
    page, or only pages already covered, would send the next round to the same place."""
    for gap in gaps:
        named = {p for p in gap_pages(gap) if 1 <= p <= n_pages}
        covered = inspected if _VISUAL_GAP.search(gap) else read | inspected
        if named - covered:
            return True
    return False


def _previous_line(previous: dict[str, Any] | None) -> str:
    if not previous:
        return ""
    v = previous.get("llm_verdict") or {}
    return (
        f"Previous round: verdict {v.get('verdict')!r}, decision {previous.get('decision')}, "
        f"gaps {previous.get('gaps') or []}. The search history above is this round's: "
        "if it already covered those gaps, do not ask for them again.\n"
    )


_CHECK_WORDS = {
    "confirmed_by_text": "confirmed: the values stand in the page's own text",
    "confirmed_by_reread": "confirmed by a second, blind reading of the image",
    "disputed": "DISPUTED by a second, blind reading of the image",
    "vision_only": "not checkable (no text layer, no second reading)",
}
# a verdict that rejects the answer for its image evidence, not for its content
_IMAGE_DOUBT = re.compile(
    r"vision|image|visual|chart|figure|graph|column|cannot be verified|not verif",
    re.IGNORECASE,
)


def _reject(verdict: dict[str, Any], gap: str) -> None:
    """Overrule a 'supported' verdict with a check done in code, keeping the reason."""
    verdict["verdict"] = "not_supported"
    verdict["explanation"] = f"{verdict.get('explanation', '')} Code check: {gap}."
    verdict["gaps"] = [gap, *(verdict.get("gaps") or [])]


# ---- checks on evidence that came from the vision model
# numbers that name things ("Figure 2", "page 15") are not readings
_REF_NUM = re.compile(
    r"\b(?:figure|fig\.?|table|page|p\.|section|chapter|slide|q)\s*\d+", re.IGNORECASE
)
_TEXT_LAYER_CHARS = (
    200  # below this a page has no usable text layer (scanned, image-only)
)
_ANSWER_LINE = re.compile(
    r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)


def _readings(text: str) -> list[float]:
    return _numbers(_REF_NUM.sub(" ", text))


def _distinctive(n: float) -> bool:
    """Single digits occur all over a page's text; only larger or decimal values can
    confirm a reading by being found there."""
    return abs(n) >= 10 or n != int(n)


def text_corroboration(answer: str, quote: str, page_text: str) -> dict[str, Any]:
    """Does the page's own text layer back a vision reading? Every number of the answer must
    be a value the vision quote reports that also stands in the page text, or follow from
    such values (a sum of two table cells). Free: no model call."""
    if len(page_text.strip()) < _TEXT_LAYER_CHARS:
        return {"status": "no_text_layer"}
    targets = _readings(answer)
    if not targets:
        return {"status": "no_values"}
    page_nums = _readings(page_text)
    backed = [
        n
        for n in _readings(quote)
        if _distinctive(n) and any(_close(n, p) for p in page_nums)
    ]
    unbacked = [t for t in targets if not _distinctive(t) or not _follows(t, backed)]
    return {
        "status": "not_in_text" if unbacked else "confirmed",
        "backed_values": backed,
        "unbacked": unbacked,
    }


def visual_regions(parser_dir: Path) -> dict[int, tuple[float, float, float, float]]:
    """Per page, the box (x0, y0, x1, y1, points, origin top-left) around all its Docling
    pictures and tables, with a margin; pages where that box is most of the page are left
    out (the whole page is rendered at a higher resolution instead)."""
    path = parser_dir / "docling_document.json"
    if not path.exists():
        return {}
    doc = read_json(path)
    sizes = {
        int(k): (v["size"]["width"], v["size"]["height"])
        for k, v in (doc.get("pages") or {}).items()
    }
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for item in (doc.get("pictures") or []) + (doc.get("tables") or []):
        for prov in item.get("prov") or []:
            p, bb = prov.get("page_no"), prov.get("bbox") or {}
            if p not in sizes or not bb:
                continue
            h = sizes[p][1]
            if bb.get("coord_origin") == "BOTTOMLEFT":
                y0, y1 = h - bb["t"], h - bb["b"]
            else:
                y0, y1 = bb["t"], bb["b"]
            boxes.setdefault(p, []).append((bb["l"], min(y0, y1), bb["r"], max(y0, y1)))
    out = {}
    for p, bs in boxes.items():
        w, h = sizes[p]
        m = 12.0
        x0 = max(0.0, min(b[0] for b in bs) - m)
        y0 = max(0.0, min(b[1] for b in bs) - m)
        x1 = min(w, max(b[2] for b in bs) + m)
        y1 = min(h, max(b[3] for b in bs) + m)
        if (x1 - x0) * (y1 - y0) < 0.6 * w * h:
            out[p] = (x0, y0, x1, y1)
    return out


def _same_answer(a: str, b: str) -> bool:
    na, nb = _readings(a), _readings(b)
    if na and nb:
        return all(any(_close(x, y) for y in nb) for x in na)
    ca, cb = _norm(a), _norm(b)
    return bool(ca and cb) and (ca in cb or cb in ca)


def compare_reread(
    answer: str, quote: str, reread: str, multi_page: bool
) -> tuple[str, str]:
    """Compare a blind second reading with the first. A definite, different answer or
    "not shown" disputes the first reading; a page that holds only part of a multi-page
    answer is compared on the values the first reading took from it."""
    lines = _ANSWER_LINE.findall(reread)
    said = lines[-1].strip() if lines else ""
    if said.upper().startswith("NOT_SHOWN"):
        return "disputed", "the second reading found nothing relevant on this page"
    definite = bool(said) and not said.upper().startswith("PARTIAL")
    if definite and not multi_page:
        if _same_answer(answer, said):
            return "confirmed", f"second reading: {said}"
        return "disputed", f"second reading answered {said!r}"
    # years ("in 2018") date a reading, they are not values read off the figure
    answer_nums = _readings(answer)
    values = [
        n
        for n in _readings(quote)
        if _distinctive(n)
        and not (
            n == int(n)
            and 1900 <= n <= 2100
            and not any(_close(n, a) for a in answer_nums)
        )
    ]
    seen = _readings(reread)
    if values:
        missing = [v for v in values if not _follows(v, seen)]
        if missing:
            return "disputed", f"second reading does not show {missing}"
        return "confirmed", "second reading shows the same values"
    if _same_answer(answer, reread):
        return "confirmed", "second reading mentions the answer"
    return "unclear", "nothing to compare"


def _reread(
    settings: AgentSettings,
    pdf_path: Path,
    page: int,
    clip: tuple[float, float, float, float] | None,
    question: str,
    step_dir: Path,
    ledger_cost: CostLedger,
    log: logging.Logger,
    round_no: int,
) -> str:
    """A second, blind reading: the question only (never the proposed answer), on a crop
    of the page's figures and tables (or the whole page at a higher resolution)."""
    model = traced_model(
        settings,
        settings.vision_model,
        step_dir=step_dir,
        ledger=ledger_cost,
        logger=log,
        step_name="05_verifier",
    )
    model.purpose = f"reread_p{page}_round{round_no}"
    model.round_no = round_no
    agent: Agent[None, str] = Agent(
        model,
        instructions=REREAD_INSTRUCTIONS,
        model_settings=ModelSettings(temperature=0.0, max_tokens=400),
    )
    jpeg = render_page(pdf_path, page, max_side=1536, quality=85, clip=clip)
    (step_dir / "images_reread").mkdir(parents=True, exist_ok=True)
    (step_dir / "images_reread" / f"round{round_no}_page{page:03d}.jpg").write_bytes(
        jpeg
    )
    return agent.run_sync(
        [BinaryContent(data=jpeg, media_type="image/jpeg"), f"Question: {question}"]
    ).output.strip()


def check_vision_evidence(
    settings: AgentSettings,
    final: dict[str, Any],
    quote_checks: list[dict[str, Any]],
    pages_md: dict[int, str],
    parser_dir: Path,
    pdf_path: Path | None,
    question: str,
    step_dir: Path,
    ledger_cost: CostLedger,
    log: logging.Logger,
    round_no: int,
) -> list[dict[str, Any]]:
    """Every page whose cited evidence came only from the vision model: first the text-layer
    check, then, if the text cannot back it, one blind re-read of the page's figures."""
    by_page: dict[int, list[str]] = {}
    for q in quote_checks:
        if q["source"] == "image_inspection":
            by_page.setdefault(q["page"], []).append(
                final["ledger"][q["entry"] - 1]["quote"]
            )
    text_pages = {q["page"] for q in quote_checks if q["source"] == "page_text"}
    cited = {q["page"] for q in quote_checks if q["source"]}
    multi_page = len(cited) > 1
    regions = visual_regions(parser_dir) if by_page else {}
    checks = []
    for page, quotes in sorted(by_page.items()):
        if page in text_pages:
            continue  # the same page is also cited from its text
        quote = "\n".join(quotes)
        check: dict[str, Any] = {"page": page}
        check["text"] = text_corroboration(
            final["answer"], quote, pages_md.get(page, "")
        )
        if check["text"]["status"] == "confirmed":
            check["status"] = "confirmed_by_text"
        elif pdf_path is None:
            check["status"] = "vision_only"
        else:
            try:
                reply = _reread(
                    settings,
                    pdf_path,
                    page,
                    regions.get(page),
                    question,
                    step_dir,
                    ledger_cost,
                    log,
                    round_no,
                )
            except Exception as exc:  # noqa: BLE001 - no second reading is no evidence
                log.error("re-read of page %d failed: %s", page, exc)
                check.update(status="vision_only", reread_error=str(exc))
            else:
                verdict, why = compare_reread(final["answer"], quote, reply, multi_page)
                check.update(
                    reread=reply,
                    crop=regions.get(page),
                    why=why,
                    status={
                        "confirmed": "confirmed_by_reread",
                        "disputed": "disputed",
                    }.get(verdict, "vision_only"),
                )
        log.info(
            "vision evidence page %d -> %s %s",
            page,
            check["status"],
            check.get("why", ""),
        )
        checks.append(check)
    return checks


def run_verifier(
    settings: AgentSettings,
    question: str,
    final: dict[str, Any],
    parser_dir: Path,
    index: dict[str, Any],
    step_dir: Path,
    run_log: Path,
    ledger_cost: CostLedger,
    round_no: int,
    max_rounds: int,
    pdf_path: Path | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``pdf_path`` enables the independent vision check of image-only answers;
    ``previous`` is the previous round's verification, so the verdict can build on it."""
    log = setup_logger("step5.verifier", step_dir / "verifier.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger_cost,
        logger=log,
        step_name="05_verifier",
        role="verifier",
    )

    pages_md = {
        p: (parser_dir / "pages" / f"page_{p:03d}.md").read_text(encoding="utf-8")
        for p in range(1, index["n_pages"] + 1)
    }
    plan_path = step_dir.parent / "03_planner" / f"plan_round{round_no}.json"
    plan = read_json(plan_path) if plan_path.exists() else {}
    located = RegionLocator(index, pages_md).locate(
        question, list(plan.get("search_queries") or [])
    )
    index_view = located["explanation"]
    inspections: dict[int, list[str]] = {}
    for ins in final.get("image_inspections", []):
        inspections.setdefault(ins["page"], []).append(ins["result"])

    # ---- 1. programmatic quote checks
    quote_checks = []
    for i, e in enumerate(final.get("ledger", []), 1):
        p = e["page"]
        page_text, seen = pages_md.get(p, ""), "\n".join(inspections.get(p, []))
        # an entry that names its source (the cited note's) is checked against that source
        # only: a short vision claim must not pass as page text by a loose match
        if e.get("source") == "image_inspection":
            page_text = ""
        elif e.get("source") == "page_text":
            seen = ""
        _, sim, src = quote_on_page(e["quote"], page_text, seen)
        quote_checks.append(
            {
                "entry": i,
                "page": p,
                "quote": e["quote"][:300],
                "candidate_answer": e["candidate_answer"],
                "similarity": sim,
                "source": src,
            }
        )
        log.info(
            "ledger #%d page %d quote_found=%s source=%s (sim=%.2f) quote=%r",
            i,
            p,
            bool(src),
            src,
            sim,
            e["quote"][:80],
        )
    cited_pages = final.get("evidence_pages") or [
        e["page"] for e in final.get("ledger", [])
    ]
    cited_pages = [p for p in cited_pages if 1 <= p <= index["n_pages"]]

    # ---- 2a. checks in code, before the verdict so the verifier sees their results: the
    # answer's numbers must follow from the quotes, and image evidence must be backed by
    # the page's text layer or by a blind second reading
    numeric: dict[str, Any] = {"status": "skipped"}
    vision_checks: list[dict[str, Any]] = []
    verified = [q for q in quote_checks if q["source"]]
    if final["status"] == "answered" and verified:
        numeric = numeric_check(
            final["answer"],
            [final["ledger"][q["entry"] - 1]["quote"] for q in verified],
        )
        vision_checks = check_vision_evidence(
            settings,
            final,
            quote_checks,
            pages_md,
            parser_dir,
            pdf_path,
            question,
            step_dir,
            ledger_cost,
            log,
            round_no,
        )
    check_lines = (
        [f"- numbers: {numeric['status']}"] if numeric["status"] != "skipped" else []
    )
    check_lines += [
        f"- image evidence on page {vc['page']}: {_CHECK_WORDS[vc['status']]}"
        + (f" ({vc['why']})" if vc.get("why") else "")
        for vc in vision_checks
    ]

    # ---- 2. grounding verdict by the verifier agent
    ledger_txt = (
        "\n".join(
            f"- page {e['page']}: answer={e['candidate_answer']!r} quote={e['quote']!r}"
            for e in final.get("ledger", [])
        )
        or "(no ledger entries recorded)"
    )
    page_txt = []
    for p in list(dict.fromkeys(cited_pages))[:5]:
        body = pages_md[p].strip()[:2500] or "(no text)"
        if p in inspections:
            body += (
                "\n[vision model description of this page]: "
                + " | ".join(inspections[p])[:1500]
            )
        page_txt.append(f"=== page {p} ===\n{body}")
    if not page_txt:
        page_txt.append("(no pages cited)")
    # what was read but not cited: a link between facts (which platform a table is
    # about) may stand on a page the answer did not cite
    uncited = [
        f"- page {n['page']} {'image reading' if n['source'] == 'image_inspection' else 'text'}: "
        + n["text"][:500].replace("\n", " ")
        for n in final.get("notes", [])
        if not n.get("cited")
    ]
    if uncited:
        page_txt.append(
            "\nOther notes read this round, not cited (a link between facts may stand here):\n"
            + "\n".join(uncited)[:4000]
        )
    agent: Agent[None, Verdict] = Agent(
        model,
        output_type=ToolOutput(Verdict, name="Verdict"),
        instructions=VERIFY_INSTRUCTIONS,
        retries=1,
        model_settings=ModelSettings(temperature=0.0, max_tokens=400),
    )
    model.purpose = f"verify_round{round_no}"
    model.round_no = round_no
    llm_verdict: dict[str, Any]
    try:
        llm_verdict = agent.run_sync(
            VERIFY_PROMPT.format(
                question=question,
                answer=final["answer"],
                status=final["status"],
                ledger=ledger_txt,
                pages="\n".join(page_txt),
                n_calls=final.get("tool_calls_used", 0),
                pages_read=final.get("pages_read") or "none",
                inspected=sorted(inspections) or "none",
                previous=_previous_line(previous),
                index_view=index_view,
                checks="\n".join(check_lines) or "(none)",
            )
        ).output.model_dump()
        llm_verdict["_error"] = None
    except Exception as exc:  # noqa: BLE001
        log.error("verifier agent failed (VERIFIER FAILURE): %s", exc)
        llm_verdict = {
            "verdict": "insufficient",
            "quote_found": False,
            "explanation": f"verifier agent failed: {exc}",
            "gaps": ["verifier could not run"],
            "_error": str(exc),
        }

    # Enforce backstop for abstentions (Fix 1: zero tool calls; Fix 2: missing figure image inspection)
    if final.get("status") == "not_answerable":
        if final.get("tool_calls_used", 0) == 0:
            anchors = plan.get("anchor_pages", [])
            gap_msg = (
                f"open anchor pages {', '.join(str(p) for p in anchors)}"
                if anchors
                else "read document pages"
            )
            llm_verdict["verdict"] = "insufficient"
            llm_verdict["explanation"] = (
                "Abstention rejected: navigator made 0 tool calls and did not inspect any pages."
            )
            llm_verdict["gaps"] = [gap_msg]
        else:
            needs_vis = plan.get("needs_visual_inspection", False)
            loc_refs = plan.get("locator", {}).get("refs", [])
            fig_pages = sorted(
                set(
                    [p for r in loc_refs for p in r.get("pages", [])]
                    + (plan.get("anchor_pages", []) if needs_vis else [])
                )
            )
            if needs_vis and fig_pages:
                inspected_pages = {
                    i["page"] for i in final.get("image_inspections", [])
                }
                uninspected = [p for p in fig_pages if p not in inspected_pages]
                if uninspected:
                    llm_verdict["verdict"] = "insufficient"
                    llm_verdict["explanation"] = (
                        f"Abstention rejected: figure question requires visual inspection of page(s) "
                        f"{', '.join(str(p) for p in uninspected)}."
                    )
                    llm_verdict["gaps"] = [
                        f"inspect_page_image on page {p} for visual evidence"
                        for p in uninspected
                    ]

    # ---- 2b. apply the code checks: a wrong sum is cheap to redo (on the last round the
    # verdict stands); image evidence that did not hold up is rejected; image evidence whose
    # value is printed in the page's own text is not rejected for coming from an image
    code_gaps: list[str] = []
    if (
        numeric["status"] == "mismatch"
        and round_no < max_rounds
        and llm_verdict.get("verdict") == "supported"
    ):
        code_gaps.append("numeric")
        _reject(
            llm_verdict,
            f"answer number(s) {numeric['unexplained']} neither appear in the quoted "
            f"evidence nor follow from its numbers {numeric['quote_numbers']} "
            f"(their sum is {numeric['sum_of_quote_numbers']}); recompute the answer "
            "from the evidence",
        )
    for vc in vision_checks:
        if vc["status"] == "disputed":
            code_gaps.append("vision")
            _reject(
                llm_verdict,
                f"the image evidence on page {vc['page']} did not hold up: {vc['why']}; "
                f"re-read the figure on page {vc['page']} for the original question, "
                "or abstain if it is not shown",
            )
    doubts_image = _IMAGE_DOUBT.search(
        " ".join([llm_verdict.get("explanation", ""), *(llm_verdict.get("gaps") or [])])
    )
    if (
        final["status"] == "answered"
        and llm_verdict.get("verdict") != "supported"
        and not llm_verdict.get("_error")
        and not code_gaps
        and numeric["status"] != "mismatch"
        and vision_checks
        and all(vc["status"] == "confirmed_by_text" for vc in vision_checks)
        and doubts_image
    ):
        llm_verdict["verdict"] = "supported"
        llm_verdict["explanation"] += (
            " Code check: the values read from the image are printed in the page's own "
            "text, so the image evidence stands."
        )
        llm_verdict["gaps"] = []
        log.info("image evidence confirmed by the text layer: verdict set to supported")
    if final["status"] == "answered" and llm_verdict.get("verdict") == "supported":
        keys = [(e["page"], _norm(e["quote"])) for e in final.get("ledger", [])]
        if (
            final["answer"].lstrip().startswith("[")
            and len(set(keys)) < len(keys)
            and round_no < max_rounds
        ):
            code_gaps.append("duplicate")
            _reject(
                llm_verdict,
                "the same quote is cited for different items of the list answer; each "
                f"item needs its own evidence (pages {sorted({k[0] for k in keys})})",
            )

    # ---- 2c. pages still worth a look (Idea: the planner's pages can miss). The verifier
    # names them; code checks they are unread and, for an abstention or a list/count
    # answer, adds the unread neighbours of the cited pages and the best unread pages
    n_pages = index["n_pages"]
    read = set(final.get("pages_read") or []) | set(inspections)
    named = [
        p
        for p in llm_verdict.get("check_pages") or []
        if isinstance(p, int) and 1 <= p <= n_pages and p not in read
    ]
    list_like = (
        plan.get("expected_answer_format") == "List"
        or plan.get("question_type") in ("counting", "list")
        or final["answer"].lstrip().startswith("[")
    )
    coverage: list[int] = []
    if final["status"] == "not_answerable" or list_like:
        neighbours = [
            q
            for p in cited_pages
            for q in (p + 1, p - 1)
            if 1 <= q <= n_pages and q not in read
        ]
        ranked = [r["page"] for r in located["ranked_pages"] if r["page"] not in read]
        coverage = [p for p in dict.fromkeys(neighbours + ranked) if p not in named][:2]
    check_pages = list(dict.fromkeys(named + coverage))[:3]
    if check_pages:
        llm_verdict["gaps"] = [
            *(llm_verdict.get("gaps") or []),
            (
                f"read pages {', '.join(map(str, check_pages))}, not read yet, for missing "
                "or further evidence"
            ),
        ]

    # ---- 3. decision
    any_quote_ok = any(q["source"] for q in quote_checks)
    replan_useful = (
        bool(code_gaps)
        or bool(check_pages)
        or replan_has_new_target(
            llm_verdict.get("gaps") or [],
            set(final.get("pages_read") or []),
            set(inspections),
            index["n_pages"],
        )
    )
    decision = decide(
        final["status"],
        None if llm_verdict.get("_error") else llm_verdict.get("verdict"),
        any_quote_ok,
        bool(final.get("ledger")),
        round_no,
        max_rounds,
        replan_useful,
    )
    decision_before_coverage = decision
    if (
        coverage
        and round_no < max_rounds
        and decision in ("ACCEPT", "ACCEPT_WEAK_GROUNDING", "ACCEPT_ABSTAIN")
    ):
        # one more look before an abstention or a possibly incomplete list is final; the
        # runner falls back to this round's answer if the next round does not do better
        decision = "REPLAN"
        log.info("coverage round: pages %s not read yet", coverage)
    if not replan_useful and round_no < max_rounds:
        log.info("no gap names a page not yet read/inspected: this round is final")
    if decision == "ACCEPT_WEAK_GROUNDING":
        log.warning(
            "answer accepted by the verifier but no quotation could be located in the cited "
            "page text/inspection -> weak grounding"
        )
    result = {
        "round": round_no,
        "status_in": final["status"],
        "answer_in": final["answer"],
        "cited_pages": cited_pages,
        "quote_checks": quote_checks,
        "any_quote_verified": any_quote_ok,
        "replan_useful": replan_useful,
        "numeric_check": numeric,
        "vision_checks": vision_checks,
        "check_pages": check_pages,
        "coverage_pages": coverage,
        "decision_before_coverage": decision_before_coverage,
        "llm_verdict": llm_verdict,
        "decision": decision,
        "gaps": llm_verdict.get("gaps") or [],
        "final_answer": final["answer"]
        if decision.startswith("ACCEPT") and final["status"] == "answered"
        else "Not answerable",
    }
    write_json(step_dir / f"verification_round{round_no}.json", result)
    log.info(
        "round %d verdict=%s quote_verified=%s -> decision=%s final=%r gaps=%s",
        round_no,
        llm_verdict.get("verdict"),
        any_quote_ok,
        decision,
        result["final_answer"],
        result["gaps"],
    )
    return result
