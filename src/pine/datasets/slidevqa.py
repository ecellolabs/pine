from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import overload

import numpy as np
from atria_core.datasets import (
    AtriaDownloadManager,
    Dataset,
    DatasetConfig,
    UrlSpec,
    datasets,
)
from atria_core.logger import get_logger
from atria_core.types import (
    AnnotatedObject,
    BoundingBoxMode,
    DatasetMetadata,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
    ObjectDetectionAnnotation,
    SinglePageDocumentInstance,
)
from PIL.Image import Image as PILImage

_HF_REPO = "NTT-hil-insight/SlideVQA"
_HOMEPAGE = "https://github.com/nttmdlab-nlp/SlideVQA"
_LICENSE = "CC BY-NC-SA 4.0"
_UPSTREAM_BBOX_RAW = (
    "https://raw.githubusercontent.com/nttmdlab-nlp/SlideVQA/main/annotations/bbox"
)
_HF_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "val",
    DatasetSplitType.test: "test",
}
_BBOX_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "dev",
    DatasetSplitType.test: "test",
}
_MAX_PAGES = 20

logger = get_logger(__name__)


@dataclass
class _BBox:
    bbox_id: int
    label_name: str
    bbox: tuple[float, float, float, float]


_IMAGE_COLUMNS = [f"page_{page_number + 1}" for page_number in range(_MAX_PAGES)]


@dataclass
class _HFRowMeta:
    deck_name: str
    qa_id: int
    question: str
    answer: str
    arithmetic_expression: str | None
    evidence_pages: list[int]

    @classmethod
    def from_hf_row(cls, row: dict[str, object]) -> _HFRowMeta:
        arithmetic_expression = str(row["arithmetic_expression"])

        raw_evidence_pages = row["evidence_pages"]
        assert isinstance(raw_evidence_pages, list)
        evidence_pages = [int(page) - 1 for page in raw_evidence_pages]

        qa_id = row["qa_id"]
        assert isinstance(qa_id, int)

        return cls(
            deck_name=str(row["deck_name"]),
            qa_id=qa_id,
            question=str(row["question"]),
            answer=str(row["answer"]),
            arithmetic_expression=None
            if arithmetic_expression == "None"
            else arithmetic_expression,
            evidence_pages=evidence_pages,
        )


@dataclass
class _Sample:
    deck_name: str
    page_image_paths: list[Path]
    page_bboxes: dict[int, list[_BBox]]
    qa_metas: list[_HFRowMeta]


class SplitIterator(Sequence[_Sample]):
    def __init__(
        self,
        data_dir: str,
        split: DatasetSplitType,
        max_samples: int | None = None,
    ) -> None:
        from datasets import load_dataset

        split_name = _HF_SPLIT_NAMES[split]
        logger.info(f"Loading SlideVQA {split.value} split from Hugging Face")

        self._rows = load_dataset(
            _HF_REPO,
            split=split_name,
            cache_dir=data_dir,
            verification_mode="no_checks",
        )

        bbox_dir = Path(data_dir) / "slidevqa_bbox"
        bbox_dir.mkdir(parents=True, exist_ok=True)
        manager = AtriaDownloadManager(
            data_dir=Path(data_dir), download_dir=Path(data_dir) / ".download_cache"
        )
        logger.info(f"Downloading SlideVQA bounding boxes for {split.value} split")
        manager.download_and_extract(
            data_urls=[
                UrlSpec(
                    url=f"{_UPSTREAM_BBOX_RAW}/{_BBOX_SPLIT_NAMES[split]}.jsonl",
                    url_ext=".jsonl",
                    rel_output_file_path=(
                        f"slidevqa_bbox/{_BBOX_SPLIT_NAMES[split]}.jsonl"
                    ),
                )
            ],
            extract=False,
        )
        self._bbox_path = bbox_dir / f"{_BBOX_SPLIT_NAMES[split]}.jsonl"

        logger.info(f"Indexing SlideVQA bounding boxes by deck for {split.value} split")
        self._bbox_offsets_by_deck = self._index_bboxes_by_deck(self._bbox_path)

        logger.info(f"Grouping SlideVQA {split.value} split rows by deck")
        self._row_indices_by_deck = self._index_rows_by_deck()
        self._deck_names = list(self._row_indices_by_deck)
        if max_samples is not None:
            self._deck_names = self._deck_names[:max_samples]

        self._images_dir = Path(data_dir) / "images" / _HF_SPLIT_NAMES[split]
        self._images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Saving SlideVQA {split.value} split page images to {self._images_dir}"
        )
        self._save_page_images()

    def _deck_image_dir(self, deck_name: str) -> Path:
        return self._images_dir / deck_name

    def _save_deck_images(self, deck_name: str, row_indices: list[int]) -> None:
        deck_dir = self._deck_image_dir(deck_name)
        if deck_dir.exists():
            return
        # Write into a sibling temp dir and rename into place only once every
        # page has been written -- if this process is killed mid-write, the
        # partial temp dir is left behind under its own name (never at
        # `deck_dir`), so a re-run's `deck_dir.exists()` check above can't be
        # fooled by a half-written deck into skipping it forever.
        tmp_dir = deck_dir.with_name(f".{deck_dir.name}.tmp")
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        page_images = self._deck_page_images(self._rows[row_indices[0]])
        for page_number, image in enumerate(page_images):
            image.save(tmp_dir / f"{page_number}.png", format="PNG")
        tmp_dir.rename(deck_dir)

    def _save_page_images(self) -> None:
        import concurrent.futures

        import tqdm

        deck_names = self._deck_names
        row_indices_list = [self._row_indices_by_deck[name] for name in deck_names]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm.tqdm(
                    executor.map(self._save_deck_images, deck_names, row_indices_list),
                    total=len(deck_names),
                    desc="Saving SlideVQA page images",
                    unit="deck",
                )
            )

    def _index_rows_by_deck(self) -> dict[str, list[int]]:
        meta_rows = self._rows.remove_columns(_IMAGE_COLUMNS)
        row_indices_by_deck: dict[str, list[int]] = {}
        for row_index, raw_meta_row in enumerate(meta_rows):
            deck_name = str(raw_meta_row["deck_name"])
            row_indices_by_deck.setdefault(deck_name, []).append(row_index)
        return row_indices_by_deck

    @staticmethod
    def _index_bboxes_by_deck(path: Path) -> dict[str, int]:
        offsets_by_deck: dict[str, int] = {}
        with path.open("rb") as stream:
            offset = stream.tell()
            line = stream.readline()
            while line:
                if line.strip():
                    data: dict[str, object] = json.loads(line)
                    offsets_by_deck[str(data["deck_name"])] = offset
                offset = stream.tell()
                line = stream.readline()
        return offsets_by_deck

    def _load_deck_bboxes(self, deck_name: str) -> dict[int, list[_BBox]]:
        offset = self._bbox_offsets_by_deck.get(deck_name)
        if offset is None:
            return {}
        with self._bbox_path.open("rb") as stream:
            stream.seek(offset)
            line = stream.readline()
        data: dict[str, object] = json.loads(line)
        raw_pages = data["bboxes"]
        assert isinstance(raw_pages, list)
        pages: dict[int, list[_BBox]] = {}
        for raw_page in raw_pages:
            page_number, page_bboxes = raw_page
            assert isinstance(page_number, int)
            assert isinstance(page_bboxes, list)
            bboxes: list[_BBox] = []
            for item in page_bboxes:
                assert isinstance(item, dict)
                raw_bbox = item["bbox"]
                assert isinstance(raw_bbox, list)
                bboxes.append(
                    _BBox(
                        bbox_id=int(item["bbox_id"]),
                        label_name=str(item["class"]),
                        bbox=(
                            float(raw_bbox[0]),
                            float(raw_bbox[1]),
                            float(raw_bbox[2]),
                            float(raw_bbox[3]),
                        ),
                    )
                )
            pages[page_number - 1] = bboxes
        return pages

    @staticmethod
    def _deck_page_images(row: dict[str, object]) -> list[PILImage]:
        page_images: list[PILImage] = []
        for column in _IMAGE_COLUMNS:
            image = row.get(column)
            if image is None:
                break
            assert isinstance(image, PILImage)
            page_images.append(image)
        return page_images

    @overload
    def __getitem__(self, index: int) -> _Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Sample]: ...

    def __getitem__(self, index: int | slice) -> _Sample | Sequence[_Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        deck_name = self._deck_names[index]
        row_indices = self._row_indices_by_deck[deck_name]
        qa_metas = [
            _HFRowMeta.from_hf_row(self._rows[row_index]) for row_index in row_indices
        ]
        deck_dir = self._deck_image_dir(deck_name)
        page_image_paths = sorted(deck_dir.glob("*.png"), key=lambda p: int(p.stem))
        return _Sample(
            deck_name=deck_name,
            page_image_paths=page_image_paths,
            page_bboxes=self._load_deck_bboxes(deck_name),
            qa_metas=qa_metas,
        )

    def __len__(self) -> int:
        return len(self._deck_names)


class InputTransform:
    @staticmethod
    def _bbox_annotation(bboxes: list[_BBox]) -> ObjectDetectionAnnotation:
        return replace(
            ObjectDetectionAnnotation.from_objects(
                [
                    AnnotatedObject(
                        label_value=index,
                        label_name=item.label_name,
                        bbox=np.asarray(item.bbox, dtype=np.float64),
                    )
                    for index, item in enumerate(bboxes)
                ]
            ),
            bbox_mode=BoundingBoxMode.XYWH,
            normalized=False,
        )

    def __call__(self, sample: _Sample) -> MultiPageDocumentInstance:
        pages: list[SinglePageDocumentInstance] = []
        for page_number, image_path in enumerate(sample.page_image_paths):
            page = SinglePageDocumentInstance.from_image(
                image_path, sample_id=f"{sample.deck_name}#{page_number}"
            )
            bboxes = sample.page_bboxes.get(page_number, [])
            pages.append(
                page.add_annotation(annotation=self._bbox_annotation(bboxes))
                if bboxes
                else page
            )

        qa_pairs = [
            MultiPageQAPair(
                id=meta.qa_id,
                question_text=meta.question,
                answer_text=meta.answer,
                evidence_pages=meta.evidence_pages,
                arithmetic_expression=meta.arithmetic_expression,
            )
            for meta in sample.qa_metas
        ]
        return MultiPageDocumentInstance(
            sample_id=sample.deck_name,
            pages=pages,
            metadata={"deck_name": sample.deck_name},
        ).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=qa_pairs)
        )


class SlideVQAConfig(DatasetConfig):
    max_samples: int | None = None


@datasets.register
class SlideVQA(Dataset[MultiPageDocumentInstance, SlideVQAConfig]):
    """SlideVQA: question answering over multi-page slide decks, with
    evidence-page and bounding-box ground truth."""

    __module_name__ = "slidevqa"
    requires_access_token = True

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        import os

        if access_token is not None:
            os.environ.setdefault("HF_TOKEN", access_token)
        return {}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "SlideVQA: question answering over multi-page slide decks, "
                "requiring multi-hop reasoning across evidence pages."
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_HF_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(
            data_dir=data_dir, split=split, max_samples=self.config.max_samples
        )

    def _build_input_transform(self) -> Callable[[_Sample], MultiPageDocumentInstance]:
        return InputTransform()
