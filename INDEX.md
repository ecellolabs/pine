# The document index: why it changed, and the experiments behind it

PINE builds an index for each document in step 2 (`02_index/`). The locator reads it to decide which pages
the navigator opens. The planner and verifier read its outline, and the navigator reads its page flags to
decide which pages get an image reading. This file records how the index was measured, the alternatives
tried, and why the default is now the **lean index** (`--index lean`). The original index is still
available as `--index full`.

## 1. The original index (`--index full`, `agent/index.py`)

| Part | How it is built | Share of all model cost |
|---|---|---|
| Captions and figure/table mentions | Docling, in code | 0% |
| Page summaries (title, summary, keywords) | **one model call per page** (vision model for pages without text) | **56–68%** (Qwen) / **70–77%** (Opus) |
| Section tree | one or two model calls, validated, flat fallback | 3–5% |
| *Answering the question (planner, navigator, verifier)* | | *20–39%* |

Cost shares are from `llm_calls.jsonl` of the s2/s3 batches with Qwen3.5-9B (`s2_2026-10-22_7`,
`s3_2026-10-22_2`) and Opus 5.5 (`s2_opus55_v3`, `s3_opus55_v3`).

## 2. What each part actually does (offline ablation, no model calls)

The locator was re-run on saved parses, indexes and planner queries, with parts of the index removed. The
measure is whether a gold evidence page is among the pages read in the first pass. This covers 32 answerable
questions (s2 + s3, with Qwen-built and Opus-built indexes).

| Variant | Gold page read in first pass |
|---|---|
| Full index | 31/32 |
| Without summaries of pages that have text | **31/32** (identical; a few ranks move one place either way) |
| Without any summaries | 29/32 (loses the scanned slide deck: image-only pages have no other text) |
| Flat section tree (one section) | **31/32** (identical in every case) |

Findings:
- Pages are found through **Docling captions** plus **BM25 over the page text**.
- The page summaries, most of the cost, did not change which pages were read, except for pages without
  text.
- In the procedural navigator the section tree is only text in the planner's and verifier's prompts; it does
  not affect page selection.

## 3. Images in the documents

Docling's picture elements in the 20 s2/s3 documents:

| | Count |
|---|---|
| Pages | 725 |
| Docling pictures | 1,055 |
| … large (at least 5% of the page) | **448**, on 348 pages |
| … small (logos, icons, bullets) | 607 |
| Pages without any text | 65 |
| Questions per document in MMLongBench-Doc (1,091 / 135) | 8.1 |

Index-time readings are paid once per document and shared by about 8 questions. Question-time readings
are paid on every question.

## 4. Two lean designs

Both build the page records and the outline from Docling in code: the title is the page's first heading or
its first line; the record holds captions, picture and table counts, text length and mentions; a section
starts at each page that opens with a heading (Docling gives every heading level 1, so the outline is flat).
Both rank pages by their **best paragraph-sized passage** instead of the whole page. They differ in how
images become text:

| | `lean-cards` (tested, removed) | `lean-transcribe` (kept as `lean`) |
|---|---|---|
| Vision input | the whole page, once per visual page | each large picture, cropped from its Docling box (at most 4 per page) |
| Pages read | pages without text, image-led pages, pages with a large picture | pages with large pictures, plus pages without text (read whole) |
| Instruction | describe the visuals for search | transcribe everything visible: labels, values, legends, rows, columns, meaningful colours |
| Repeated images | read every time | read once (de-duplicated by pixels) |

In both, the text is stored in the page's `visual_text` with its provenance and **never merged into the
page text**. It is searchable, and the navigator sees it as a vision note for pages it reads without a fresh
image reading. The verifier checks it like any other vision reading (text-layer check).
The navigator still asks the vision model the actual question for the pages it selects.

Offline check of the lean structure and passage ranking without any image text (no model calls): gold page in
first pass **8/8 (s2) and 7/8 (s3), the same as the full index**. Passage ranking moved several gold pages up
(s3 run8 from rank 4 to 1, s2 run1 and run9 from 2 to 1), and a few slipped one place.

## 5. End-to-end comparison (Qwen3.5-9B, 20 questions)

Batches `agent_runs/s{2,3}_q9b_{full,lean-cards,lean-transcribe}` on branch `beta-index`, with identical
settings:

```bash
OPENROUTER_API_KEY=… uv run usage/02_agent_qa.py mmlongbench_doc --max-samples s2 \
  --text-model qwen/qwen3.5-9b --vision-model qwen/qwen3.5-9b \
  --no-reasoning --reasoning-roles navigator,verifier --ignore-provider Together \
  --index lean \
  --judge-model openai/gpt-oss-120b --judge-api-url https://openrouter.ai/api/v1 \
  --judge-max-tokens 2000 --judge-extra-body '{"reasoning_effort": "low"}' \
  --workers 1 --index-concurrency 4 --request-timeout 180
```

Every run started from the same Docling parse. Three lean batches hung on one thinking verifier call
(an OpenRouter stall); they were restarted with `--request-timeout 180` and resumed from their cache, so
results are unaffected.

| | full | lean-cards | **lean-transcribe** |
|---|---|---|---|
| Score, GPT-OSS judge (s2 + s3) | 7 + 7 = **14/20** | 7 + 7 = **14/20** | 8 + 6 = **14/20** |
| Score, rules (s2 + s3) | 8 + 4 = 12/20 | 8 + 6 = 14/20 | 9 + 5 = **14/20** |
| Gold page read (answerable) | 16/16 | 15/16 | 15/16 |
| Index model calls | 748 | 426 | 466 |
| **Index cost** (20 documents) | $0.159 | $0.103 (−35%) | **$0.076 (−52%)** |
| **Median index time per document** (s2 / s3) | 82 s / 127 s | 77 s / 98 s | **31 s / 47 s** |
| Question-loop cost | $0.076 | $0.099 | $0.090 |

Costs are at $0.17 / $0.25 per million tokens. The 15/16 for the lean variants is s3 run9 (Best Buy 10-K):
the gold page was not read, but both still answered **1383** correctly from another page with the same figure.

Questions where the variants differ:

| Question | full | lean-cards | lean-transcribe | Why |
|---|---|---|---|---|
| s2 run8 West Germany chart | ✗ | ✗ | **✓ 29** | the cropped chart transcription held the "Both 29%" cell |
| s3 run8 PwC infographic | ✗ | **✓ 1914** | **✓ 1914** | the number lives inside an image; the index now holds its text |
| s3 run3 Mali colour | ✓ "#6A5ACD" | ✓ "purple" | ✗ "light purple" | judge strictness (it accepted a hex code but rejected "light purple") |
| s3 run10 SAO counter | ✓ "5.00pm" | ✗ "8.30am - 5.00pm" | ✗ abstain | how the ambiguous question is read; the gold page was read in all three |

The other misses (s2 run7, run9; s3 run4, run6) are the same in all three variants, so the index did not cause
them. s3 run6 is the false-premise question; strict answers are on for Claude only.

## 6. Decision

- **`lean` (image transcription) is the default index.** It has the same judged accuracy as the full index, is
  better on the rule scorer (14 vs 12), and wins both questions whose answer sits inside an image. It costs
  about half as much and builds about 2.6× faster.
- **`lean-cards` was removed:** more expensive and slower than per-image transcription (whole-page images on
  every visual page), with no accuracy advantage.
- **`full` stays available** as `--index full` to reproduce earlier batches and for comparison.

### Caveats

- 20 questions with thinking sampling (temperature 0.6) are noisy, and judge totals moved by single
  questions. Confirm on a larger sample before drawing strong conclusions.
- Per-image transcription can make more calls than pages on brochures and decks: the 76-page NTU guide had 98
  large images. It is cheaper overall because text-heavy documents (papers, reports, 10-Ks) need few calls.
- A general transcription can miss question-specific detail. That is why the navigator still reads the
  selected pages' images with the actual question, and why stored transcriptions stay labelled as vision
  readings rather than document text.
