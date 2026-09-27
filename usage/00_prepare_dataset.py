"""Small end-to-end dataset preparation example."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from atria_core.datasets import DatasetBuilder, FileStorageType, datasets
from atria_core.logger import get_logger
from atria_core.types import DatasetSplitType
from atria_core.visualizers import visualize
from tqdm import tqdm

from pipeline_v2.datasets import *

logger = get_logger(__name__)


def benchmark_dataset(dataset: Any, n: int = 10) -> None:
    """Iterate over all splits, time each sample load, and log min/max/avg."""
    for split, split_iterator in dataset.split_iterators.items():
        samples: Any = split_iterator
        count = min(len(samples), n)
        if count == 0:
            continue
        logger.info(f"[{split.value}] benchmarking {count} samples")
        times: list[float] = []
        for i in tqdm(range(count), desc=f"benchmark[{split.value}]"):
            t0 = time.perf_counter()
            samples[i].load()
            times.append(time.perf_counter() - t0)
        logger.info(
            f"[{split.value}] n={count}"
            f"  min={min(times):.3f}s"
            f"  max={max(times):.3f}s"
            f"  avg={sum(times) / len(times):.3f}s"
        )


def prepare_dataset(
    name: str,
    output_dir: str = "./dataset_visualizations",
    enable_caching: bool = False,
    storage_type: FileStorageType = FileStorageType.DELTALAKE,
    visualize_samples: bool = True,
    benchmark: bool = False,
    data_dir: str | None = None,
    max_samples: int | None = None,
    split: DatasetSplitType | None = None,
    **dataset_kwargs: Any,
) -> None:
    """Load and cache a dataset, then inspect the first sample of each split."""
    builder = DatasetBuilder().load(
        name,
        data_dir=data_dir,
        max_samples=max_samples,
        split=split,
        **dataset_kwargs,
    )
    if enable_caching:
        builder = builder.cache(storage_type, split=split)
    dataset = builder.build()

    logger.info("Cached dataset:\n%s", dataset)

    for split_key, split_iterator in dataset.split_iterators.items():
        samples: Any = split_iterator
        logger.info(f"[{split_key.value}] {len(samples)} samples")
        if not visualize_samples or len(samples) == 0:
            continue

        sample = None
        for item in samples:
            loaded = item.load()
            if len(loaded.pages) > 0 and not loaded.metadata.get("invalid_pdf"):
                sample = loaded
                break

        if sample is None or len(sample.pages) == 0:
            logger.warning(
                f"[{split_key.value}] No valid sample with pages found for visualization."
            )
            continue

        sample_dir = Path(output_dir) / name / split_key.value
        sample_dir.mkdir(parents=True, exist_ok=True)
        visualize(sample, output_dir=str(sample_dir))
        logger.info(
            f"[{split_key.value}] first sample: sample_id={sample.sample_id!r}"
            f" num_pages={len(sample.pages)}"
        )
        logger.info(f"First sample of split `{split_key}`:\n {sample}")

    if benchmark:
        benchmark_dataset(dataset)


def main() -> None:
    """Parse a registered dataset name and run its preparation pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(datasets.list()))
    parser.add_argument(
        "--split", choices=[s.value for s in DatasetSplitType], default=None
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples/decks to load and cache.",
    )
    parser.add_argument(
        "--enable-caching", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--storage-type",
        choices=[t.value for t in FileStorageType],
        default=FileStorageType.MSGPACK.value,
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="./dataset_visualizations",
        help="Path where visualizations are written.",
    )
    args = parser.parse_args()
    split = DatasetSplitType(args.split) if args.split is not None else None
    prepare_dataset(
        args.name,
        output_dir=args.output_dir,
        split=split,
        enable_caching=args.enable_caching,
        storage_type=FileStorageType(args.storage_type),
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
