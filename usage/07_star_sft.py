"""Stage 07: write the SFT set (ms-swift format) from the STaR samples of stage 06.

Takes every example with a chosen (correct) sample, builds the exact direct-mode prompt
without the hint, and writes the chosen reasoning plus final answer as the assistant turn.
Page images are rendered once into the output folder, so training needs no PDFs.

    uv run usage/07_star_sft.py --data-dir data/vidore_v3 \
        --samples data/vidore_v3/star/train_k4.jsonl --name k4

    # assistant answer = the gold short answer instead of the model's own wording
    uv run usage/07_star_sft.py --samples data/vidore_v3/star/train_k4.jsonl --name k4_gold \
        --answer-source gold

Writes <data-dir>/sft/<name>/: train.jsonl (one ms-swift line per example; image paths are
relative to that folder), images/, manifest.json (counts, skipped examples, estimated
token lengths for choosing --max_length). Train from inside that folder (train/README.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pine.training.sft import build_sft
from pine.training.star import load_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/vidore_v3"))
    p.add_argument(
        "--samples",
        type=Path,
        nargs="+",
        required=True,
        help="STaR sample files from usage/06_star_sample.py (first file wins on duplicates).",
    )
    p.add_argument("--name", required=True, help="Output folder under <data-dir>/sft/.")
    p.add_argument(
        "--answer-source",
        choices=["model", "gold"],
        default="model",
        help="Final answer after the reasoning: the chosen sample's own, or the gold one.",
    )
    args = p.parse_args()

    records = {
        r["id"]: r
        for split in ("train", "dev")
        for r in load_rows(args.data_dir / f"star_{split}.jsonl")
    }
    rows = [r for path in args.samples for r in load_rows(path)]
    if not rows:
        p.error("no STaR samples in " + ", ".join(map(str, args.samples)))
    if dev := sorted({r["id"] for r in rows if r.get("split") == "dev"}):
        p.error(f"{len(dev)} dev examples in --samples (e.g. {dev[0]}); train only")
    out_dir = args.data_dir / "sft" / args.name
    manifest = build_sft(records, rows, args.data_dir, out_dir, args.answer_source)
    manifest["samples_files"] = [str(s) for s in args.samples]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"SFT set: {(out_dir / 'train.jsonl').resolve()}")


if __name__ == "__main__":
    main()
