"""Runs DoclingTransform over a dataset, writing each page's parsed output
next to the dataset's own data."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from atria_core.datasets import Dataset, DatasetBuilder, DatasetConfig
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType, MultiPageDocumentInstance

from pipeline_v2.datasets import *
from pipeline_v2.parsers.docling import (
    DoclingApiOptions,
    DoclingTransform,
)
from pipeline_v2.sampling import dataset_load_kwargs, describe, resolve_max_samples

logger = get_logger(__name__)


_worker_transform: DoclingTransform | None = None
_worker_out_dir: Path | None = None


def _init_worker(api_url: str, options: DoclingApiOptions, out_dir: Path) -> None:
    global _worker_transform, _worker_out_dir
    _worker_transform = DoclingTransform(api_url=api_url, options=options)
    _worker_out_dir = out_dir


def _process_sample(
    sample: MultiPageDocumentInstance,
    transform: DoclingTransform,
    out_dir: Path,
) -> None:
    logger.info(f"sample_id={sample.sample_id!r} key={sample.key!r}")
    sample_dir = out_dir / sample.key
    sample_dir.mkdir(parents=True, exist_ok=True)
    for page in sample.pages:
        out_path = sample_dir / f"{page.key}.json"
        if out_path.exists():
            logger.info(f"skip page_id={page.sample_id!r} -> {out_path}")
            continue
        document = transform(page)
        out_path.write_text(json.dumps(document.export_to_dict()))
        logger.info(f"page_id={page.sample_id!r} key={page.key!r} -> {out_path}")


def _process_sample_worker(sample: MultiPageDocumentInstance) -> None:
    assert _worker_transform is not None
    assert _worker_out_dir is not None
    _process_sample(sample, _worker_transform, _worker_out_dir)


@dataclass
class Preprocessor:
    """Parses every page of every sample in a dataset split via the Docling API,
    writing each page's docling output to `data_dir/docling/{sample.key}/{page.key}.json`.
    Skips pages whose output already exists, so a re-run resumes rather than
    reprocessing."""

    out_dir: Path
    api_url: str
    options: DoclingApiOptions = field(default_factory=DoclingApiOptions)
    num_workers: int = 1

    def _process_sample(
        self,
        sample: MultiPageDocumentInstance,
        transform: DoclingTransform | None = None,
    ) -> None:
        if transform is None:
            transform = DoclingTransform(api_url=self.api_url, options=self.options)
        _process_sample(sample, transform, self.out_dir)

    def run(self, split_iterator: Iterable[MultiPageDocumentInstance]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.num_workers <= 1:
            with DoclingTransform(
                api_url=self.api_url, options=self.options
            ) as transform:
                for sample in split_iterator:
                    _process_sample(sample, transform, self.out_dir)
            return

        with mp.Pool(
            self.num_workers,
            initializer=_init_worker,
            initargs=(self.api_url, self.options, self.out_dir),
        ) as pool:
            for _ in pool.imap_unordered(_process_sample_worker, split_iterator):
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument(
        "--split", choices=[s.value for s in DatasetSplitType], default=None
    )
    parser.add_argument(
        "--max-samples",
        type=str,
        default=None,
        help=(
            "Integer N = first N samples/decks; a name like 's0' = the exact "
            "documents listed in sample_sets/s0.json (see pipeline_v2.sampling)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    env_api_url = os.getenv("DOCLING_API_URL")
    parser.add_argument(
        "--api-url",
        type=str,
        default=env_api_url,
        required=env_api_url is None,
        help=(
            "Docling API service URL (e.g. http://serv-3334:10001). "
            "Required unless DOCLING_API_URL environment variable is set. "
            "Note: DFKI cluster node hostname and port change dynamically on each allocation."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR on the Docling API server.",
    )
    args = parser.parse_args()
    split = DatasetSplitType(args.split) if args.split is not None else None

    resolved = resolve_max_samples(args.max_samples)
    logger.info(f"Selection: {describe(resolved)}")
    dataset = cast(
        Dataset[MultiPageDocumentInstance, DatasetConfig],
        DatasetBuilder()
        .load(
            args.name,
            data_dir=args.data_dir,
            split=split,
            **dataset_load_kwargs(resolved, args.name),
        )
        .build(),
    )
    api_options = DoclingApiOptions(ocr=not args.no_ocr)
    for split_key, split_iterator in dataset.split_iterators.items():
        preprocessor = Preprocessor(
            out_dir=dataset.data_dir / "docling" / split_key.value,
            api_url=args.api_url,
            options=api_options,
            num_workers=args.num_workers,
        )
        preprocessor.run(split_iterator)


if __name__ == "__main__":
    main()
