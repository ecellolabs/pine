"""Step 6 - Evaluation with a port of the official MMLongBench-Doc scorer (eval/eval_score.py).

The official protocol is: free-form answer -> GPT-4o answer extraction -> rule-based score by answer_format.
Our agent emits a concise structured answer, so extraction is a light rule-based normalisation instead of GPT-4o
(documented deviation).  Scoring rules (Int / Float / Str / List / None, ANLS with 0.5 threshold) follow the official code.
"""

from __future__ import annotations

import ast
import contextlib
import math
import re
from pathlib import Path
from typing import Any

from pipeline_v2.agent.common import setup_logger, write_json


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gt: str, thresh: float = 0.5) -> float:
    if not pred and not gt:
        return 1.0
    d = levenshtein(pred, gt)
    m = max(len(pred), len(gt)) or 1
    s = 1 - d / m
    return s if s > thresh else 0.0


def is_float_equal(
    ref: float, pred: float, include_percentage: bool = True, is_close: bool = True
) -> bool:
    cands = [ref / 100, ref, ref * 100] if include_percentage else [ref]
    for c in cands:
        if is_close:
            if math.isclose(c, pred, rel_tol=0.01):
                return True
        else:
            if abs(c - pred) < 1e-9:
                return True
    # official also accepts rounding
    for c in cands:
        with contextlib.suppress(Exception):
            if round(pred, 2) == round(c, 2):
                return True
    return False


def _remove_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _clean_str(s: str) -> str:
    s = _remove_quotes(str(s)).lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)  # trailing parenthetical
    s = s.replace("$", "").replace("%", "").strip()
    return s


def _to_float(s: str) -> float | None:
    s = (
        str(s)
        .strip()
        .replace("$", "")
        .replace("%", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )
    s = s.replace("million", "").replace("billion", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _as_float(s: str) -> float:
    """Like ``_to_float`` but raises when no number is present."""
    value = _to_float(s)
    if value is None:
        raise ValueError(f"no number in {s!r}")
    return value


def extract_answer(prediction: str) -> str:
    """Light rule-based stand-in for the GPT-4o extraction step."""
    p = prediction.strip()
    if re.search(
        r"not answerable|cannot be answered|does not contain|no information",
        p,
        re.IGNORECASE,
    ):
        return "Not answerable"
    p = (
        re.sub(r"^(the )?(answer|final answer)( is)?[:\s]+", "", p, flags=re.IGNORECASE)
        .strip()
        .rstrip(".")
    )
    return p


def _parse_list(s: str) -> list[str]:
    s = s.strip()
    with contextlib.suppress(Exception):
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
    s = s.strip("[]")
    parts = [x.strip() for x in re.split(r",|;| and ", s) if x.strip()]
    return parts


def eval_score(gt: str, pred: str, answer_type: str) -> float:
    """Port of the official rule-based scorer keyed on answer_format."""
    if answer_type == "Int":
        try:
            return float(int(_as_float(gt)) == int(_as_float(pred)))
        except Exception:  # noqa: BLE001
            return 0.0
    if answer_type == "Float":
        try:
            gf, pf = _to_float(gt), _to_float(pred)
            if gf is None or pf is None:
                return 0.0
            return float(is_float_equal(gf, pf))
        except Exception:  # noqa: BLE001
            return 0.0
    if answer_type in ("Str", "None"):
        gs, ps = _clean_str(gt), _clean_str(pred)
        if gt.strip().lower() == "not answerable":
            return float(ps == "not answerable")
        if ps == "not answerable":
            return 0.0
        exact_types = (
            re.match(r"^(https?://|www\.)", gs)
            or re.search(r"\.(pdf|png|jpg|csv|ipynb|py)$", gs)
            or re.match(r"^page \d+", gs)
            or re.match(r"^\d{4}-\d{2}-\d{2}$", gs)
            or "@" in gs
        )
        if exact_types:
            return float(gs == ps)
        return anls(ps, gs)
    if answer_type == "List":
        if pred.strip().lower() == "not answerable":
            return 0.0
        gl, pl = _parse_list(gt), _parse_list(pred)
        if len(gl) != len(pl):
            return 0.0
        gl_c = sorted(_clean_str(x) for x in gl)
        pl_c = sorted(_clean_str(x) for x in pl)
        if all(_to_float(x) is not None for x in gl_c):
            try:
                return float(
                    all(
                        is_float_equal(_as_float(a), _as_float(b))
                        for a, b in zip(gl_c, pl_c)
                    )
                )
            except Exception:  # noqa: BLE001
                return 0.0
        return min(anls(p, g) for g, p in zip(gl_c, pl_c))
    return 0.0


def run_eval(
    run_meta: dict[str, Any],
    gold: dict[str, Any],
    verification: dict[str, Any],
    final: dict[str, Any],
    step_dir: Path,
    run_log: Path,
    cost: dict[str, Any],
    timings: dict[str, Any],
) -> dict[str, Any]:
    log = setup_logger("step6.eval", step_dir / "eval.log", run_log)
    step_dir.mkdir(parents=True, exist_ok=True)
    raw_pred = verification["final_answer"]
    extracted = extract_answer(raw_pred)
    score = eval_score(gold["answer"], extracted, gold["answer_format"])
    gold_pages = gold["evidence_pages"]
    cited = verification.get("cited_pages") or []
    pages_read = set(final.get("pages_read", [])) | {
        i["page"] for i in final.get("image_inspections", [])
    }
    result = {
        "run_id": run_meta["run_id"],
        "question": run_meta["question"],
        "gold_answer": gold["answer"],
        "gold_answer_format": gold["answer_format"],
        "gold_evidence_pages_1idx": gold_pages,
        "gold_evidence_sources": gold["evidence_sources"],
        "raw_system_answer": raw_pred,
        "extracted_answer": extracted,
        "score_official_rule": score,
        "correct": score >= 0.999,
        "agent_status": final["status"],
        "verifier_decision": verification["decision"],
        "cited_pages": cited,
        "cited_page_hit": bool(set(cited) & set(gold_pages)) if gold_pages else None,
        "gold_page_visited_by_agent": bool(pages_read & set(gold_pages))
        if gold_pages
        else None,
        "pages_visited": sorted(pages_read),
        "rounds": final.get("round", 1),
        "tool_calls_total": timings.get("tool_calls_total"),
        "cost": cost,
        "timings_s": timings,
        "scorer_note": "Official eval_score.py rules ported; GPT-4o answer extraction replaced by rule-based normalisation of the agent's concise answer.",
    }
    write_json(step_dir / "result.json", result)
    log.info(
        "GOLD=%r (%s, pages %s) | PRED=%r -> score=%.2f correct=%s | cited=%s hit=%s | visited gold page=%s",
        gold["answer"],
        gold["answer_format"],
        gold_pages,
        extracted,
        score,
        result["correct"],
        cited,
        result["cited_page_hit"],
        result["gold_page_visited_by_agent"],
    )
    return result
