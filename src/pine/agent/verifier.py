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
import re
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput
from pydantic_ai.settings import ModelSettings

from pine.agent.common import AgentSettings, CostLedger, setup_logger, write_json
from pine.agent.llm import traced_model
from pine.agent.schemas import Verdict

VERIFY_INSTRUCTIONS = """You are the VERIFIER of a grounded document question-answering system. Decide whether the cited evidence from the document supports the proposed answer. Be strict: the answer must be directly supported by the evidence text, not by general knowledge.
Report whether the quoted evidence is actually present in the cited page content, a 1-2 sentence explanation, and concrete gaps (what is still missing or should be checked: sections, pages, figures).
If the proposed status is not_answerable: return "supported" only if the cited pages / search history make it plausible that the document does not contain the information; otherwise "insufficient" with gaps naming where to look."""

VERIFY_PROMPT = """Question: {question}
Proposed answer: {answer}
Proposed status: {status}

Evidence (ledger entries recorded by the agent):
{ledger}

Cited page content:
{pages}"""


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


def decide(
    status: str,
    verdict: str | None,
    any_quote_ok: bool,
    has_ledger: bool,
    round_no: int,
    max_rounds: int,
) -> str:
    """The acceptance rule, kept separate so it can be unit-tested."""
    rounds_left = round_no < max_rounds
    if status == "error":
        return "ABSTAIN"
    if status == "not_answerable":
        if verdict == "supported" or not rounds_left:
            return "ACCEPT_ABSTAIN"
        return "REPLAN"
    if verdict == "supported":
        if any_quote_ok:
            return "ACCEPT"
        return "ACCEPT_WEAK_GROUNDING"
    return "REPLAN" if rounds_left else "ABSTAIN"


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
) -> dict[str, Any]:
    log = setup_logger("step5.verifier", step_dir / "verifier.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    model = traced_model(
        settings,
        settings.text_model,
        step_dir=step_dir,
        ledger=ledger_cost,
        logger=log,
        step_name="05_verifier",
    )

    pages_md = {
        p: (parser_dir / "pages" / f"page_{p:03d}.md").read_text(encoding="utf-8")
        for p in range(1, index["n_pages"] + 1)
    }
    inspections: dict[int, list[str]] = {}
    for ins in final.get("image_inspections", []):
        inspections.setdefault(ins["page"], []).append(ins["result"])

    # ---- 1. programmatic quote checks
    quote_checks = []
    for i, e in enumerate(final.get("ledger", []), 1):
        p = e["page"]
        ok_text, r_text = quote_in_text(e["quote"], pages_md.get(p, ""))
        ok_img, r_img = (False, 0.0)
        if p in inspections:
            ok_img, r_img = quote_in_text(e["quote"], "\n".join(inspections[p]))
        src = "page_text" if ok_text else ("image_inspection" if ok_img else None)
        quote_checks.append(
            {
                "entry": i,
                "page": p,
                "quote": e["quote"][:300],
                "candidate_answer": e["candidate_answer"],
                "found_in_page_text": ok_text,
                "sim_page_text": r_text,
                "found_in_image_inspection": ok_img,
                "sim_image_inspection": r_img,
                "source": src,
            }
        )
        log.info(
            "ledger #%d page %d quote_found=%s (text sim=%.2f, image sim=%.2f) quote=%r",
            i,
            p,
            bool(src),
            r_text,
            r_img,
            e["quote"][:80],
        )
    cited_pages = final.get("evidence_pages") or [
        e["page"] for e in final.get("ledger", [])
    ]
    cited_pages = [p for p in cited_pages if 1 <= p <= index["n_pages"]]

    # ---- 2. grounding verdict by the verifier agent
    ledger_txt = (
        "\n".join(
            f"- page {e['page']}: answer={e['candidate_answer']!r} quote={e['quote']!r}"
            for e in final.get("ledger", [])
        )
        or "(no ledger entries recorded)"
    )
    page_txt = []
    for p in cited_pages[:3]:
        body = pages_md[p].strip()[:3500] or "(no text)"
        if p in inspections:
            body += (
                "\n[vision model description of this page]: "
                + " | ".join(inspections[p])[:1500]
            )
        page_txt.append(f"=== page {p} ===\n{body}")
    if not page_txt:
        page_txt.append("(no pages cited)")
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

    # ---- 3. decision
    any_quote_ok = any(q["source"] for q in quote_checks)
    decision = decide(
        final["status"],
        llm_verdict.get("verdict"),
        any_quote_ok,
        bool(final.get("ledger")),
        round_no,
        max_rounds,
    )
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
