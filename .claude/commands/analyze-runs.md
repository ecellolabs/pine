---
description: Analyse an agent_runs folder and explain, per component, what went wrong (parser, index, orchestration, tool calling, verifier)
argument-hint: [runs-dir]
---
Analyse the agent runs under `${1:-agent_runs}` (a batch folder `agent_runs/<selection>_<date>[_n]`; given the root, use the newest batch) and write the findings to `<batch>/analysis.md`.

For every run folder (listed in `manifest.json`; if there are more than 20, aggregate the scores from `06_evaluation/result.json` first and go deep only on the failures):
1. Read `summary.md`, `06_evaluation/result.json`, `05_evidence_verifier/verification_round*.json`, `04_orchestrator/anomalies_round*.json`, `04_orchestrator/trace_round*.jsonl`, `02_index/index.json` (keys `hierarchy_attempts`, `hierarchy_fallback_used`, `pages_summarised_from_image`) and `01_parser/page_stats.json` (flagged pages).
2. State whether the official-rule score is correct **and whether it is correct for the right reason** (gold page visited? quote verified? abstention after actually inspecting the candidate pages?).
3. Attribute every problem to one component: parser (text loss, OCR, missing chart values), index (invalid hierarchy, summaries missing figure/table captions, BM25 misses), orchestration (premature abstention, unmotivated page choice, wasted vision calls, planner ignored), tool calling (malformed calls, non-verbatim quotes), verifier (decision rule, quote check), evaluation (scorer caveats).
4. End with a prioritised list of fixes and the total cost/latency from `cost.json`.

Write in plain prose with one table per run; cite file paths so a reader can verify each claim. Do not modify the run folders.
