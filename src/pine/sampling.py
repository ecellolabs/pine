"""Sample selection for `--max-samples`.

Two forms are accepted by every usage script:

* an integer, e.g. ``--max-samples 3`` -> the first N documents of the split
  (unchanged behaviour);
* a sample-set name or path, e.g. ``--max-samples s0`` -> the exact documents
  and questions listed in ``sample_sets/s0.json`` (or ``s0.json`` given as a
  path).  This is how a colleague replicates the same experiment.

A sample-set file looks like::

    {
      "name": "s0",
      "dataset": "mmlongbench_doc",
      "description": "...",
      "samples": [
        {"run_id": "run1_isro_text", "doc_id": "<pdf file name>",
         "question": "<exact question text>", "why_chosen": "..."}
      ]
    }

``question`` may be omitted to select every question of the document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SAMPLE_SETS_DIR = Path(__file__).resolve().parents[2] / "sample_sets"


@dataclass(frozen=True)
class SampleSpec:
    """One (document, question) pair to run."""

    run_id: str
    doc_id: str
    question: str | None = None
    why_chosen: str = ""


@dataclass
class SampleSet:
    """A named, explicit list of samples."""

    name: str
    dataset: str | None
    description: str
    samples: list[SampleSpec]
    source: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        """Unique document ids in first-seen order."""
        seen: list[str] = []
        for s in self.samples:
            if s.doc_id not in seen:
                seen.append(s.doc_id)
        return seen

    def questions_for(self, doc_id: str) -> list[SampleSpec]:
        return [s for s in self.samples if s.doc_id == doc_id]

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> SampleSet:
        samples = []
        for i, raw in enumerate(data.get("samples", []), 1):
            if "doc_id" not in raw:
                raise ValueError(
                    f"sample #{i} in {source or 'sample set'} has no doc_id"
                )
            samples.append(
                SampleSpec(
                    run_id=str(raw.get("run_id") or f"sample{i:02d}"),
                    doc_id=str(raw["doc_id"]),
                    question=raw.get("question"),
                    why_chosen=str(raw.get("why_chosen", "")),
                )
            )
        if not samples:
            raise ValueError(f"sample set {source or data.get('name')} has no samples")
        return cls(
            name=str(data.get("name") or (source.stem if source else "sample_set")),
            dataset=data.get("dataset"),
            description=str(data.get("description", "")),
            samples=samples,
            source=source,
            extra={
                k: v
                for k, v in data.items()
                if k not in ("name", "dataset", "description", "samples")
            },
        )

    @classmethod
    def load(cls, name_or_path: str, search_dir: Path | None = None) -> SampleSet:
        """Resolve ``s0`` -> ``<search_dir>/s0.json`` (or an explicit path)."""
        candidates = []
        raw = Path(name_or_path).expanduser()
        if raw.suffix == ".json" or raw.exists():
            candidates.append(raw)
        base = search_dir or SAMPLE_SETS_DIR
        candidates.append(base / f"{name_or_path}.json")
        candidates.append(Path.cwd() / "sample_sets" / f"{name_or_path}.json")
        for cand in candidates:
            if cand.is_file():
                data = json.loads(cand.read_text(encoding="utf-8"))
                return cls.from_dict(data, source=cand.resolve())
        raise FileNotFoundError(
            f"sample set {name_or_path!r} not found; looked at: "
            + ", ".join(str(c) for c in candidates)
        )


def resolve_max_samples(value: str | int | None) -> int | SampleSet | None:
    """Parse the ``--max-samples`` CLI value.

    ``None`` -> everything; an integer string -> that many documents; anything
    else -> a sample set loaded from ``sample_sets/<value>.json`` or a path.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = value.strip()
    if text.lstrip("-").isdigit():
        n = int(text)
        if n <= 0:
            raise ValueError(
                "--max-samples must be a positive integer or a sample-set name"
            )
        return n
    return SampleSet.load(text)


def dataset_load_kwargs(
    resolved: int | SampleSet | None, dataset_name: str
) -> dict[str, Any]:
    """Translate a resolved ``--max-samples`` into ``DatasetBuilder.load`` params."""
    if resolved is None:
        return {}
    if isinstance(resolved, int):
        return {"max_samples": resolved}
    if resolved.dataset and resolved.dataset != dataset_name:
        raise ValueError(
            f"sample set {resolved.name!r} is defined for dataset {resolved.dataset!r}, "
            f"not {dataset_name!r}"
        )
    if dataset_name != "mmlongbench_doc":
        raise ValueError(
            f"sample sets (--max-samples {resolved.name}) are only supported for "
            f"'mmlongbench_doc' at the moment, not {dataset_name!r}"
        )
    return {"doc_ids": resolved.doc_ids}


def describe(resolved: int | SampleSet | None) -> str:
    if resolved is None:
        return "all samples"
    if isinstance(resolved, int):
        return f"first {resolved} document(s)"
    return f"sample set {resolved.name!r} ({len(resolved.samples)} sample(s), {len(resolved.doc_ids)} document(s)) from {resolved.source}"
