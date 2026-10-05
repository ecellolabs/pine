---
description: Rebuild agent_runs/visual_samples.html (self-contained page, first 20 runs) from an existing agent_runs folder
argument-hint: [runs-dir] [--max-runs N]
---
Rebuild the visual-samples page for the runs folder `${1:-agent_runs}`:

```bash
uv run usage/05_visual_samples.py ${1:-agent_runs}
```

Then verify: the file exists, is roughly 0.3–10 MB, contains no `sk-or-` string and no absolute `/Users/` or `/home/` path (`grep -c`), and open it in the browser pane if available to check that the run tabs render. Report the path, size and how many runs were included (the page is capped at 20 runs by default; pass `--max-runs` to change it).
