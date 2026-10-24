# PINE (Page-Index Navigation with Evidence-grounding) — working notes for Claude Code

Beyond Flat Retrieval: Hierarchical Index Navigation for Grounded Multi-Page Document
Question Answering. Multi-page DocVQA dataset pipeline (atria-core based) plus the
**grounded agentic DocQA pipeline** from the FYP proposal "Grounded Multi-Page Document QA with
Indexed Agentic Orchestration" (NUST SEECS, Talal Majeed & Momena Akhtar).

Renamed on 7 Oct 2026 from `pipeline-v2` (package `pipeline_v2`) to `pine`; imports are
`from pine...` and the distribution name in `pyproject.toml` is `pine`.

## Setup

```bash
uv sync                      # Python 3.13, installs docling + rapidocr + pymupdf too
uv run hf auth login         # only needed for gated datasets (slidevqa)
./ci/run_checks.sh           # mypy + ruff + pytest; must pass before pushing
```

Secrets go in the environment, never in files: `OPENROUTER_API_KEY` (or
`QWEN_API_KEY` + `QWEN_API_URL` for a vLLM server). Nothing under `samples/`
ever contains the key.

## Layout

| Path | Purpose |
|---|---|
| `src/pine/datasets/` | atria-core dataset loaders (`mmlongbench_doc`, `mpdocvqa`, `slidevqa`) |
| `src/pine/parsers/docling.py` | remote Docling API transform (cluster) |
| `src/pine/models/qwen.py` | OpenAI-compatible VLM client used by the B0 baseline |
| `src/pine/sampling.py` | `--max-samples` parsing: integer **or** sample-set name (`s0` -> `sample_sets/s0.json`) |
| `src/pine/agent/` | the agent, built on **Pydantic AI**: `llm` (provider model + cache + trace), `schemas` (typed outputs), `parser` (Docling+RapidOCR), `lean_index` (default index: Docling structure in code + a transcription of each large image; see INDEX.md), `index` (the original model-summarised index, `--index full`), `locate` (deterministic region locator: figure/table caption resolution + BM25 over pages and sections; also the navigator's search corpus), `planner` (agent with the `locate_regions` tool), `orchestrator` (`Agent` with `@agent.tool` tools, `final_answer` output tool, driven with `agent.iter`), `verifier` (ledger + quote check), `scoring` (official MMLongBench-Doc rules), `judge` (LLM answer extraction and LLM judge over saved answers; prompts in `agent/prompts/`), `baseline` (document-in-the-prompt baselines, no agent), `runner`, `visual_samples` (HTML page) |
| `usage/0N_*.py` | stateless CLI stages (00 dataset, 01 Docling preprocess, 02 agent, 03 visual page; `B0_evaluation.py` is the single-pass VLM baseline, not a stage); `02_agent_qa.py` runs the agent (full dataset or a selection), `03_visual_samples.py` builds the visual-samples page; `04_judge.py` rescores the saved answers of any batch or baseline folder with an LLM (no model rerun), `B1_mmlongbench_baseline.py` runs the document-in-the-prompt baselines |
| `sample_sets/*.json` | named, exact lists of (doc_id, question) so an experiment is replicable |
| `tests/` | pytest; `ci/` the check scripts |

## Replicating the reference experiment (sample set `s0`)

```bash
OPENROUTER_API_KEY=sk-or-... uv run usage/02_agent_qa.py mmlongbench_doc --max-samples s0
uv run usage/03_visual_samples.py agent_runs        # -> agent_runs/visual_samples.html
```

Every invocation writes one **batch folder** `agent_runs/<selection>_<YYYY-MM-DD>[_<n>]/`
(`s0_2026-10-08`; a second run of the same selection on the same day becomes
`s0_2026-10-08_2`; integer selections give `first3_…`, the full dataset `full_…`;
`--batch-name` overrides). The batch folder holds `<run_id>/00_input … 06_evaluation`,
`run.log`, `llm_calls.jsonl`, `cost.json`, `summary.md` per question, `manifest.json`
(batch name, start time, exact command, settings, per-run scores) and
`visual_samples.html` (self-contained, first 20 runs). The three `s0` samples cost
about $0.01 in total on OpenRouter.

**Cache rule.** The LLM response cache lives inside the batch folder
(`<batch>/.cache/llm`), so a new invocation never replays an earlier batch: every
batch shows the model's live behaviour. The old rule still applies whenever a cache is
reused on purpose (`--batch-name` pointing at an existing batch, or `--cache-dir`):
reset it before evaluating model behaviour, keep it only for pure rebuilds (HTML,
parser-only changes). Quote the expected cost to the user before any run that misses
the cache.

`usage/03_visual_samples.py agent_runs` rebuilds the page for the **newest** batch;
pass a batch folder to pick one. The page groups the agent loop per round
(planner → orchestrator → verifier), shows the exact system/user prompt of every model
call (index, planner, navigator turns, vision inspections, verifier), lists all calls in
execution order and prints `run.log`, whose lines carry `module.py:function`.

`--max-samples` decides the scope: omitted = the **full** dataset; integer `N` =
the first N documents (add `--questions-per-doc K` to limit questions per
document); a name such as `s0` = exactly the samples in `sample_sets/s0.json`.
Shipped sets: `s0` (3 reference samples, ~$0.01), `s1` (RAPTOR figure question only),
`s2` (`s0` + 7 documents picked for diversity: all 7 document types, every evidence source
and answer format, two image-only decks, two unanswerable questions, 17-76 pages per
document, 287 pages in total, ~$0.10 per batch).
To define a new replicable set, copy `sample_sets/s0.json`, change `name`, list
the exact `doc_id` (PDF file name) and verbatim `question` text, then use
`--max-samples <name>`. The visual page is for inspection of a few runs; for a
full run read `manifest.json` and the per-run `06_evaluation/result.json`.

Models: text roles default to `qwen/qwen-2.5-7b-instruct`, page images to
`qwen/qwen3-vl-8b-instruct` (Qwen2.5-VL-7B-Instruct is not served on OpenRouter;
on the cluster pass `--vision-model Qwen/Qwen2.5-VL-7B-Instruct --api-url http://<node>:<port>/v1`).

## Analysing an agent_runs folder

Pick the batch (`agent_runs/<selection>_<date>[_n>]/`, see its `manifest.json`), start with
`<batch>/<run>/summary.md`, then:

- parser issues -> `01_parser/page_stats.json` (`flag` per page, Docling vs native chars)
- index issues -> `02_index/index.json` (`hierarchy_attempts`, `hierarchy_fallback_used`, `validation_problems_final`), `outline.md`
- orchestration / tool calling -> `04_orchestrator/trace_round1.jsonl`, `anomalies_round1.json`, `tool_results/`, `images_viewed/`
- grounding -> `05_evidence_verifier/verification_round1.json` (`quote_checks`, `llm_verdict`, `decision`)
- score -> `06_evaluation/result.json` (`score_official_rule`, `cited_page_hit`, `gold_page_visited_by_agent`)
- cost/latency -> `cost.json`, `llm_calls.jsonl` (each record has `step`, `round`, `purpose`, the full request messages and the response)
- execution order -> `run.log` (`timestamp | level | stepN.name | module.py:function | message`; first line is the driver command)

Slash commands in `.claude/commands/`: `/run-agent`, `/visual-samples`, `/analyze-runs`.

## Agent framework: Pydantic AI

Every model interaction goes through `pydantic_ai.Agent`; keep it that way when extending:

- Outputs are typed (`agent/schemas.py`) and delivered as tool calls (`ToolOutput`): the 7B
  models echo JSON schemas when asked for prompted JSON, so `PromptedOutput` is avoided.
  The navigator's output tool is `final_answer`; its only gate (an `@agent.output_validator`)
  is that an *answered* final needs a ledger entry. Abstentions are never bounced: there is no
  minimum-page rule. The planner is a search strategy (queries, sections, visual or not); it has
  no abstain field. Candidate regions are **not inferred from section titles**: the planner's
  `locate_regions` tool (`agent/locate.py`, `RegionLocator`) resolves "Figure N"/"Table N" to the
  page whose Docling caption carries it and ranks pages/sections with BM25 for free-text queries;
  `run_planner` then re-runs the locator in code and merges it (reference sections and pages
  first) so `candidate_sections` / `anchor_pages` are right even if the 7B model skipped the tool.
  The navigator's prompt shows the same figure-aware outline and opens `anchor_pages` first.
  Locator output is written to `03_planner/locate_round{N}.json` and shown on the visual page.
  Validation failures are retried by Pydantic AI (`retries=`); structural checks live in
  `@agent.output_validator` functions that raise `ModelRetry` (see the hierarchy validator).
- Tools are `@agent.tool` functions taking `RunContext[NavDeps]`; the docstring is the tool
  description. A tool rejects bad input by raising `ModelRetry` (e.g. a non-verbatim quote).
- Models are built once per step with `agent.llm.traced_model(...)`: a `WrapperModel` that adds
  the sha256 response cache, `llm_calls.jsonl` tracing and cost accounting (OpenRouter cost via
  `OpenRouterModel`, any OpenAI-compatible server via `OpenAIChatModel`). Set `model.purpose`
  before a run so the trace is readable.
- Budgets use `UsageLimits(tool_calls_limit=...)`; budget/behaviour failures are caught and a
  single forced `FinalAnswer` is requested from the evidence so far.
- Tests never touch the network: `AgentSettings.model_factory` injects `FunctionModel`
  (scripted tool calls) or `TestModel` (schema-valid dummies). See `tests/test_agent_tools.py`.

## Navigator and verifier (current design)

- **Index (default `--index lean`)**: page records and a flat outline from Docling headings, passage-level BM25, and one vision transcription per large image (stored as `visual_text`, shown to the navigator as a vision note). `--index full` is the original page-summary index. Experiments and numbers: `INDEX.md`.
- **Procedural navigator (default, `--navigator procedural`)**: code reads the plan's anchor pages
  (forced and carried-over pages first, topped up to 6 with the locator's next ranked pages) and decides
  which pages get a vision reading (no text, referenced figure, image-led page, or visual page when the plan
  expects visual evidence; 4 slots, in locator order). Each reading is a numbered note; one typed `decide`
  call answers, searches once, or abstains, citing notes by number. The ledger is built in code from the
  cited notes with their source. `--navigator agentic` keeps the free tool loop.
- **Verifier**: code checks run first and are shown to the LLM verdict: quotes against the cited note's
  source, `numeric_check`, and for image-only evidence a text-layer check, else a blind re-read of a crop of
  the page's figures (question only; PARTIAL allowed for multi-page answers). A disputed reading rejects the
  answer. `check_pages` (typed) plus coverage pages (abstentions, list/count answers) become the next round's
  first anchors; a re-plan happens only when a gap names a page not yet covered.
- **Runner**: carries the previous round's cited pages forward; keeps an accepted answer if a coverage round
  does not improve on it (not if that round reached the same answer and had it rejected).
- OpenRouter + qwen3.5 with `--no-reasoning`: pass `--ignore-provider Together` (it rejects reasoning-off
  plus a required tool call). A `claude-*` judge uses the Anthropic SDK and reads `ANTHROPIC_API_KEY`.
- Sample sets: `s3` is ten questions on ten documents unused by `s0`/`s2`, for testing changes on unseen data.

## Scoring (step 6)

- Without `--judge-model` the score is the rule-based port of the official scorer
  (`agent/scoring.py`, `score_source: rules`).
- With `--judge-model` (any OpenAI-compatible endpoint, `--judge-api-url` defaults to the agent's) the
  LLM judge scores each final answer inside step 6: `result["score"]` is the judge's verdict
  (`score_source: llm_judge`), `result["score_official_rule"]` keeps the rule-based score for audit, and
  `result["judge"]` has the verdict, extracted answer and reason. If a judge call fails the run falls back
  to the rule-based score and says `rules_fallback`. `--judge-mode extract` uses the official GPT-4o
  extraction prompt followed by the rules instead (`agent/prompts/`, Apache-2.0 notice there).
- Use a judge that is not the model being scored. `usage/04_judge.py` rescoring saved answers from older
  runs without rerunning anything; the manifest, summary and visual page report `score`.
- **Project judge: gpt-oss (open weights), not Opus.** We score with `openai/gpt-oss-120b`, so results are
  reproducible with open models and need no `ANTHROPIC_API_KEY`. It is a reasoning model: give it a bigger
  reply budget and a low effort. Inside a run (same OpenRouter key as the agent):
  `--judge-model openai/gpt-oss-120b --judge-max-tokens 2000 --judge-extra-body '{"reasoning_effort": "low"}'`
  (on a vLLM server also pass `--judge-api-url http://<node>:<port>/v1`). To rescore saved answers without
  rerunning the agent: `uv run usage/04_judge.py <batch> --api-url https://openrouter.ai/api/v1 --model openai/gpt-oss-120b --max-tokens 2000 --extra-body '{"reasoning_effort": "low"}'` (key from `$OPENROUTER_API_KEY`).
  The `claude-*` judge path stays in the code but is not used for reported numbers; scores from different
  judges are not comparable, so re-score older batches with gpt-oss before comparing them.

## Conventions

- Stages are stateless: read from disk, write to disk, idempotent (skip existing outputs).
- Library code in `src/pine/` never parses CLI args; drivers live in `usage/`.
- Keep prompts and schemas in `agent/*.py` stable: the LLM cache is keyed on the exact
  request (messages + tool schemas + settings), so changing one invalidates cached (free) replays.
- Run `./ci/run_checks.sh` before committing; `ruff format src` fixes formatting.
- Commits and pull requests carry no AI attribution: no `Co-Authored-By` trailer, no
  "Generated with" footer. Author is the person running the session.
- Do not commit `agent_runs/` (PDF copies, CC BY-NC benchmark text) or any key.
