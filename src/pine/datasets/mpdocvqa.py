from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from atria_core.datasets import Dataset, DatasetConfig, datasets
from atria_core.logger import get_logger
from atria_core.types import (
    BoundingBoxMode,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    ElementArray,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
    SinglePageDocumentInstance,
)

from .utils import require_manual_path

_HOMEPAGE = "https://github.com/rubenpt91/MP-DocVQA-Framework"
_RRC_PORTAL = "https://rrc.cvc.uab.es/?ch=17&com=downloads"
_IMDBS_URL = "https://datasets.cvc.uab.es/rrc/DocVQA/Task4/mpdocvqa_imdbs.zip"
_SPLIT_NAMES = {
    DatasetSplitType.train: "train",
    DatasetSplitType.validation: "val",
    DatasetSplitType.test: "test",
}

logger = get_logger(__name__)


@dataclass
class _IMDBRecord:
    question_id: int
    question: str
    image_id: str
    image_name: list[str]
    answer_page_idx: int | None
    answers: list[str]
    ocr_tokens: list[list[str]]
    ocr_normalized_boxes: list[list[tuple[float, float, float, float]]]

    @classmethod
    def from_raw(cls, record: dict[str, object]) -> _IMDBRecord:
        image_name = record["image_name"]
        assert isinstance(image_name, list)

        # Test-split records have no ground truth (has_answer=False in the
        # .npy header): "answers"/"answer_page_idx" are simply absent.
        raw_answers = record.get("answers")
        assert raw_answers is None or isinstance(raw_answers, list)
        deduplicated_answers = (
            list(dict.fromkeys(str(answer) for answer in raw_answers))
            if raw_answers is not None
            else []
        )

        answer_page_idx = record.get("answer_page_idx")
        assert answer_page_idx is None or isinstance(answer_page_idx, int)

        ocr_tokens = record["ocr_tokens"]
        assert isinstance(ocr_tokens, list)

        ocr_normalized_boxes = record["ocr_normalized_boxes"]
        assert isinstance(ocr_normalized_boxes, list)

        question_id = record["question_id"]
        assert isinstance(question_id, int)

        return cls(
            question_id=question_id,
            question=str(record["question"]),
            image_id=str(record["image_id"]),
            image_name=[str(name) for name in image_name],
            answer_page_idx=answer_page_idx,
            answers=deduplicated_answers,
            ocr_tokens=[
                [str(token) for token in page_tokens] for page_tokens in ocr_tokens
            ],
            ocr_normalized_boxes=[
                [
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    for box in page_boxes
                ]
                for page_boxes in ocr_normalized_boxes
            ],
        )


@dataclass
class _Sample:
    image_id: str
    records: list[_IMDBRecord]


class SplitIterator(Sequence[_Sample]):
    def __init__(
        self,
        imdb_dir: Path,
        split: DatasetSplitType,
        max_samples: int | None = None,
    ) -> None:
        import numpy as np

        imdb_path = imdb_dir / f"imdb_{_SPLIT_NAMES[split]}.npy"
        logger.info(f"Loading MP-DocVQA {split.value} split from {imdb_path}")
        data = np.load(imdb_path, allow_pickle=True)
        records = [_IMDBRecord.from_raw(record) for record in data[1:]]

        logger.info(f"Grouping MP-DocVQA {split.value} split records by document")
        records_by_image_id: dict[str, list[_IMDBRecord]] = {}
        for record in records:
            records_by_image_id.setdefault(record.image_id, []).append(record)
        self._image_ids = list(records_by_image_id)
        if max_samples is not None:
            self._image_ids = self._image_ids[:max_samples]
        self._records_by_image_id = records_by_image_id

    @overload
    def __getitem__(self, index: int) -> _Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Sample]: ...

    def __getitem__(self, index: int | slice) -> _Sample | Sequence[_Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        image_id = self._image_ids[index]
        return _Sample(image_id=image_id, records=self._records_by_image_id[image_id])

    def __len__(self) -> int:
        return len(self._image_ids)


class InputTransform:
    def __init__(self, images_dir: Path) -> None:
        self._images_dir = images_dir

    def _build_page(
        self,
        image_id: str,
        page_number: int,
        image_name: str,
        tokens: list[str],
        boxes: list[tuple[float, float, float, float]],
    ) -> SinglePageDocumentInstance:
        content = None
        if tokens:
            content = DocumentContent(
                elements=ElementArray.from_words(
                    texts=tokens,
                    bboxes=boxes,
                    bbox_mode=BoundingBoxMode.XYXY,
                    normalized=True,
                )
            )
        return SinglePageDocumentInstance.from_image(
            self._images_dir / f"{image_name}.jpg",
            sample_id=f"{image_id}#{page_number}",
            content=content,
        )

    def __call__(self, sample: _Sample) -> MultiPageDocumentInstance:
        # All records for a document share the same pages/OCR content; only
        # the question/answer fields differ per record.
        first_record = sample.records[0]
        pages = [
            self._build_page(
                first_record.image_id,
                page_number,
                image_name,
                first_record.ocr_tokens[page_number],
                first_record.ocr_normalized_boxes[page_number],
            )
            for page_number, image_name in enumerate(first_record.image_name)
        ]

        # Test-split records carry no ground truth: leave answer fields empty
        # rather than indexing into an empty `answers` list.
        qa_pairs = [
            MultiPageQAPair(
                id=record.question_id,
                question_text=record.question,
                answer_text=record.answers[0] if record.answers else "",
                evidence_pages=(
                    [record.answer_page_idx]
                    if record.answer_page_idx is not None
                    else []
                ),
                alternative_answers=record.answers[1:],
            )
            for record in sample.records
        ]
        return MultiPageDocumentInstance(
            sample_id=sample.image_id, pages=pages
        ).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=qa_pairs)
        )


class MPDocVQAConfig(DatasetConfig):
    max_samples: int | None = None


def _download_and_extract_imdbs(imdb_dir: Path) -> None:
    """Download mpdocvqa_imdbs.zip and extract into imdb_dir.

    Falls back to disabling SSL verification if the host's CA chain is not trusted
    by Python's certifi bundle (common with academic server certificates).
    """
    import zipfile

    import requests
    import urllib3
    from tqdm import tqdm

    download_dir = imdb_dir.parent / ".download_cache"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "mpdocvqa_imdbs.zip"
    incomplete_path = download_dir / "mpdocvqa_imdbs.zip.incomplete"

    if not zip_path.exists():
        logger.info(f"Downloading MP-DocVQA IMDBs from {_IMDBS_URL}")
        try:
            response = requests.get(_IMDBS_URL, stream=True, timeout=30)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            logger.warning(
                f"SSL verification failed for {_IMDBS_URL}. Retrying with verification disabled."
            )
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(_IMDBS_URL, stream=True, verify=False, timeout=30)
            response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024

        try:
            with (
                open(incomplete_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="mpdocvqa_imdbs.zip",
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            incomplete_path.rename(zip_path)
        except (OSError, RuntimeError, requests.RequestException):
            incomplete_path.unlink(missing_ok=True)
            raise
        finally:
            response.close()

    logger.info(f"Extracting {zip_path} to {imdb_dir}")
    temp_extract_dir = imdb_dir.parent / "imdb_extract_tmp"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_extract_dir)
        if imdb_dir.exists():
            shutil.rmtree(imdb_dir)
        temp_extract_dir.rename(imdb_dir)
    except (OSError, RuntimeError, zipfile.BadZipFile):
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        raise


@datasets.register
class MPDocVQA(Dataset[MultiPageDocumentInstance, MPDocVQAConfig]):
    """MP-DocVQA: question answering over multi-page scanned documents,
    reusing SP-DocVQA's questions and answers with added document context."""

    __module_name__ = "mpdocvqa"

    def _download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        imdb_dir = Path(data_dir) / "mpdocvqa" / "imdb"
        has_imdbs = (
            imdb_dir.is_dir()
            and (imdb_dir / "imdb_train.npy").exists()
            and (imdb_dir / "imdb_val.npy").exists()
            and (imdb_dir / "imdb_test.npy").exists()
        )
        if not has_imdbs:
            if imdb_dir.exists():
                shutil.rmtree(imdb_dir)
            (Path(data_dir) / "mpdocvqa").mkdir(parents=True, exist_ok=True)
            try:
                _download_and_extract_imdbs(imdb_dir)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(
                    f"Automatic download of IMDBs failed from {_IMDBS_URL}: {e}. "
                    "Falling back to manual download instructions."
                )

        imdb_dir = require_manual_path(
            data_dir=data_dir,
            expected_path="mpdocvqa/imdb",
            homepage=_HOMEPAGE,
            instructions=(
                f"Register at {_RRC_PORTAL} (Task 4: MP-DocVQA), download the "
                "IMDBs archive, and extract it here so that imdb_train.npy, "
                "imdb_val.npy, and imdb_test.npy are directly inside."
            ),
        )
        images_dir = require_manual_path(
            data_dir=data_dir,
            expected_path="mpdocvqa/images",
            homepage=_HOMEPAGE,
            instructions=(
                f"Register at {_RRC_PORTAL} (Task 4: MP-DocVQA), download the "
                "Images archive (or run wget on cluster from Task 4 URL), and extract it "
                "here so that each page's .jpg file is directly inside."
            ),
        )
        return {"imdb": imdb_dir, "images": images_dir}

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "MP-DocVQA: question answering over multi-page scanned "
                "documents, extending SP-DocVQA with surrounding document "
                "pages for context."
            ),
            homepage=_HOMEPAGE,
            license=(
                "Data usage requires registration at the RRC portal and is "
                "subject to its terms of use."
            ),
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return list(_SPLIT_NAMES)

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(
            imdb_dir=Path(data_dir) / "mpdocvqa" / "imdb",
            split=split,
            max_samples=self.config.max_samples,
        )

    def _build_input_transform(
        self,
    ) -> Callable[[_Sample], MultiPageDocumentInstance]:
        return InputTransform(images_dir=Path(self.data_dir) / "mpdocvqa" / "images")
