---
description: Run the agentic DocQA pipeline (full dataset, first N docs, or a sample set such as s0) and build the visual-samples page
argument-hint: [N | sample-set | nothing for the full dataset] [extra args]
---
Run the grounded agent pipeline on MMLongBench-Doc and build the visual-samples page.

Steps:
1. Check that `OPENROUTER_API_KEY` (or `QWEN_API_KEY` + `QWEN_API_URL`) is set in the environment. If not, stop and ask the user for it; never write a key into a file.
2. Run from the repo root. `$ARGUMENTS` is passed through: an integer = first N documents, a name such as `s0` = the exact samples in `sample_sets/s0.json`, nothing = the full dataset (warn the user about cost first: ~1,100 questions). Default to `s0` when the user does not say.
   ```bash
   uv run usage/02_agent_qa.py mmlongbench_doc --max-samples ${1:-s0}
   ```
   Every invocation writes a new batch folder `agent_runs/<selection>_<YYYY-MM-DD>[_<n>]/` with a cold LLM cache (so quote the expected cost first: s0 ≈ $0.01, s1 ≈ $0.005). Each PDF is parsed with Docling (1–4 s/page) and indexed inside that batch.
3. When it finishes, the printed table ends with `batch folder:` and `visual samples:`; confirm `<batch>/manifest.json` and `<batch>/visual_samples.html` exist, and report per-run score, answer, cost and any `FAILED` entries.
4. If a run failed, open `<batch>/<run_id>/run.log` and `driver.log`, explain the cause, and fix the code (not the data) before re-running.
