# pipeline-v2 — working notes for Claude Code

Multi-page DocVQA dataset pipeline (atria-core based) plus the **grounded agentic
DocQA pipeline** from the FYP proposal "Grounded Multi-Page Document QA with
Indexed Agentic Orchestration" (NUST SEECS, Talal Majeed & Momena Akhtar).

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
| `src/pipeline_v2/datasets/` | atria-core dataset loaders (`mmlongbench_doc`, `mpdocvqa`, `slidevqa`) |
| `src/pipeline_v2/parsers/docling.py` | remote Docling API transform (cluster) |
| `src/pipeline_v2/models/qwen.py` | OpenAI-compatible VLM client used by the B0 baseline |
| `src/pipeline_v2/sampling.py` | `--max-samples` parsing: integer **or** sample-set name (`s0` -> `sample_sets/s0.json`) |
| `src/pipeline_v2/agent/` | the agent, built on **Pydantic AI**: `llm` (provider model + cache + trace), `schemas` (typed outputs), `parser` (Docling+RapidOCR), `index` (PageIndex-style tree), `planner`, `orchestrator` (`Agent` with `@agent.tool` tools, `final_answer` output tool, driven with `agent.iter`), `verifier` (ledger + quote check), `scoring` (official MMLongBench-Doc rules), `runner`, `visual_samples` (HTML page) |
| `usage/0N_*.py` | stateless CLI stages; `04_agent_qa.py` runs the agent (full dataset or a selection), `05_visual_samples.py` builds the visual-samples page |
| `sample_sets/*.json` | named, exact lists of (doc_id, question) so an experiment is replicable |
| `tests/` | pytest; `ci/` the check scripts |

## Replicating the reference experiment (sample set `s0`)

```bash
OPENROUTER_API_KEY=sk-or-... uv run usage/04_agent_qa.py mmlongbench_doc --max-samples s0
uv run usage/05_visual_samples.py agent_runs        # -> agent_runs/visual_samples.html
```

This produces `agent_runs/<run_id>/00_input … 06_evaluation`, `run.log`,
`llm_calls.jsonl`, `cost.json`, `summary.md` per question, `agent_runs/manifest.json`
and `agent_runs/visual_samples.html` (self-contained, first 20 runs). The three
`s0` samples cost about $0.01 in total on OpenRouter; responses are cached under
`agent_runs/.cache/llm` so re-runs are free.

`--max-samples` decides the scope: omitted = the **full** dataset; integer `N` =
the first N documents (add `--questions-per-doc K` to limit questions per
document); a name such as `s0` = exactly the samples in `sample_sets/s0.json`.
To define a new replicable set, copy `sample_sets/s0.json`, change `name`, list
the exact `doc_id` (PDF file name) and verbatim `question` text, then use
`--max-samples <name>`. The visual page is for inspection of a few runs; for a
full run read `manifest.json` and the per-run `06_evaluation/result.json`.

Models: text roles default to `qwen/qwen-2.5-7b-instruct`, page images to
`qwen/qwen3-vl-8b-instruct` (Qwen2.5-VL-7B-Instruct is not served on OpenRouter;
on the cluster pass `--vision-model Qwen/Qwen2.5-VL-7B-Instruct --api-url http://<node>:<port>/v1`).

## Analysing an agent_runs folder

Start with `agent_runs/<run>/summary.md`, then:

- parser issues -> `01_parser/page_stats.json` (`flag` per page, Docling vs native chars)
- index issues -> `02_index/index.json` (`hierarchy_attempts`, `hierarchy_fallback_used`, `validation_problems_final`), `outline.md`
- orchestration / tool calling -> `04_orchestrator/trace_round1.jsonl`, `anomalies_round1.json`, `tool_results/`, `images_viewed/`
- grounding -> `05_evidence_verifier/verification_round1.json` (`quote_checks`, `llm_verdict`, `decision`)
- score -> `06_evaluation/result.json` (`score_official_rule`, `cited_page_hit`, `gold_page_visited_by_agent`)
- cost/latency -> `cost.json`, `llm_calls.jsonl`

Slash commands in `.claude/commands/`: `/run-agent`, `/visual-samples`, `/analyze-runs`.

## Agent framework: Pydantic AI

Every model interaction goes through `pydantic_ai.Agent`; keep it that way when extending:

- Outputs are typed (`agent/schemas.py`): `PromptedOutput(Model)` for the planner, verifier,
  page summaries and hierarchy; `ToolOutput(FinalAnswer, name="final_answer")` for the navigator.
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

## Conventions

- Stages are stateless: read from disk, write to disk, idempotent (skip existing outputs).
- Library code in `src/pipeline_v2/` never parses CLI args; drivers live in `usage/`.
- Keep prompts and schemas in `agent/*.py` stable: the LLM cache is keyed on the exact
  request (messages + tool schemas + settings), so changing one invalidates cached (free) replays.
- Run `./ci/run_checks.sh` before committing; `ruff format src` fixes formatting.
- Do not commit `agent_runs/` (PDF copies, CC BY-NC benchmark text) or any key.
