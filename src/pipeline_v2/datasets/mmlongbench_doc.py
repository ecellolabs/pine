from __future__ import annotations

import ast
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

from atria_core.datasets import (
    AtriaDownloadManager,
    Dataset,
    DatasetConfig,
    UrlSpec,
    datasets,
)
from atria_core.logger import get_logger
from atria_core.types import (
    DatasetMetadata,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQAPair,
    MultiPageQuestionAnsweringAnnotation,
)

_HF_REPO = "yubo2333/MMLongBench-Doc"
_HOMEPAGE = "https://github.com/mayubo2333/MMLongBench-Doc"
_LICENSE = "CC BY-NC 4.0"

logger = get_logger(__name__)


@dataclass
class _RowMeta:
    doc_id: str
    doc_type: str
    question: str
    answer: str
    evidence_pages: list[int]
    evidence_sources: list[str]
    answer_format: str

    @classmethod
    def from_hf_row(cls, row: dict[str, object]) -> _RowMeta:
        raw_evidence_pages = json.loads(str(row["evidence_pages"]))
        assert isinstance(raw_evidence_pages, list)
        evidence_pages = [int(page) - 1 for page in raw_evidence_pages]

        raw_evidence_sources = ast.literal_eval(str(row["evidence_sources"]))
        assert isinstance(raw_evidence_sources, list)

        return cls(
            doc_id=str(row["doc_id"]),
            doc_type=str(row["doc_type"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            evidence_pages=evidence_pages,
            evidence_sources=[str(source) for source in raw_evidence_sources],
            answer_format=str(row["answer_format"]),
        )


@dataclass
class _Sample:
    doc_id: str
    pdf_path: Path
    qa_metas: list[_RowMeta]


class SplitIterator(Sequence[_Sample]):
    def __init__(
        self,
        data_dir: str,
        split: DatasetSplitType,
        max_samples: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> None:
        from datasets import load_dataset

        logger.info(f"Loading MMLongBench-Doc {split.value} split from Hugging Face")
        self._rows = load_dataset(_HF_REPO, split="train", cache_dir=data_dir)

        logger.info(f"Grouping MMLongBench-Doc {split.value} split rows by document")
        self._row_indices_by_doc = self._index_rows_by_doc()
        self._doc_ids = list(self._row_indices_by_doc)  # used for PDF downloading only
        if doc_ids:
            # Explicit selection (sample sets): keep the requested order, skip
            # unknown ids with a warning.
            unknown = [d for d in doc_ids if d not in self._row_indices_by_doc]
            if unknown:
                logger.warning(
                    f"MMLongBench-Doc: {len(unknown)} requested doc_id(s) are not in "
                    f"the dataset and were skipped: {unknown}"
                )
            self._doc_ids = [d for d in doc_ids if d in self._row_indices_by_doc]
        if max_samples is not None:
            self._doc_ids = self._doc_ids[:max_samples]

        self._pdf_dir = Path(data_dir) / "pdfs"
        self._pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Downloading MMLongBench-Doc {split.value} split documents to {self._pdf_dir}"
        )
        self._download_documents()

    def _index_rows_by_doc(self) -> dict[str, list[int]]:
        row_indices_by_doc: dict[str, list[int]] = {}
        for row_index, row in enumerate(self._rows):
            doc_id = str(row["doc_id"])
            row_indices_by_doc.setdefault(doc_id, []).append(row_index)
        return row_indices_by_doc

    def _pdf_path(self, doc_id: str) -> Path:
        return self._pdf_dir / doc_id

    def _download_documents(self) -> None:
        missing_doc_ids = [
            doc_id for doc_id in self._doc_ids if not self._pdf_path(doc_id).exists()
        ]
        if not missing_doc_ids:
            return

        manager = AtriaDownloadManager(
            data_dir=self._pdf_dir.parent,
            download_dir=self._pdf_dir.parent / ".download_cache",
        )
        manager.download_and_extract(
            data_urls=[
                UrlSpec(
                    url=(
                        f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/"
                        f"documents/{doc_id}"
                    ),
                    url_ext=".pdf",
                    rel_output_file_path=f"pdfs/{doc_id}",
                )
                for doc_id in missing_doc_ids
            ],
            extract=False,
        )

    @overload
    def __getitem__(self, index: int) -> _Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_Sample]: ...

    def __getitem__(self, index: int | slice) -> _Sample | Sequence[_Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        doc_id = self._doc_ids[index]
        row_indices = self._row_indices_by_doc[doc_id]
        qa_metas = [
            _RowMeta.from_hf_row(self._rows[row_index]) for row_index in row_indices
        ]
        return _Sample(
            doc_id=doc_id, pdf_path=self._pdf_path(doc_id), qa_metas=qa_metas
        )

    def __len__(self) -> int:
        return len(self._doc_ids)


class InputTransform:
    def __call__(self, sample: _Sample) -> MultiPageDocumentInstance:
        document = MultiPageDocumentInstance.from_pdf(
            sample.pdf_path, sample_id=sample.doc_id
        )

        qa_pairs = [
            MultiPageQAPair(
                id=qa_id,
                question_text=meta.question,
                answer_text=meta.answer,
                evidence_pages=meta.evidence_pages,
                evidence_sources=meta.evidence_sources,
                answer_format=meta.answer_format,
            )
            for qa_id, meta in enumerate(sample.qa_metas)
        ]
        metadata: dict[str, Any] = {
            "doc_id": sample.doc_id,
            "doc_type": sample.qa_metas[0].doc_type,
        }

        return replace(document, metadata=metadata).add_annotation(
            annotation=MultiPageQuestionAnsweringAnnotation(qa_pairs=qa_pairs)
        )


class MMLongBenchDocConfig(DatasetConfig):
    max_samples: int | None = None
    # Explicit list of PDF file names (doc_id) to load, in order. Used by sample
    # sets (`--max-samples s0`); see `pipeline_v2.sampling`.
    doc_ids: list[str] | None = None


@datasets.register
class MMLongBenchDoc(Dataset[MultiPageDocumentInstance, MMLongBenchDocConfig]):
    """MMLongBench-Doc: question answering over long, multi-page PDF
    documents spanning research reports, financial filings, and manuals."""

    __module_name__ = "mmlongbench_doc"

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=(
                "MMLongBench-Doc: question answering over long multi-page "
                "PDF documents, requiring long-context and cross-page "
                "reasoning."
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _available_splits(self, data_dir: str) -> list[DatasetSplitType]:
        return [DatasetSplitType.train]

    def _build_split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> SplitIterator:
        return SplitIterator(
            data_dir=data_dir,
            split=split,
            max_samples=self.config.max_samples,
            doc_ids=self.config.doc_ids,
        )

    def _build_input_transform(self) -> Callable[[_Sample], MultiPageDocumentInstance]:
        return InputTransform()
