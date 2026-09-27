from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from atria_core.datasets import datasets
from atria_core.types import (
    AnnotationType,
    BoundingBoxMode,
    DatasetSplitType,
    MultiPageDocumentInstance,
    MultiPageQuestionAnsweringAnnotation,
    ObjectDetectionAnnotation,
    SinglePageDocumentInstance,
)
from docling_core.types.doc.document import DoclingDocument

from pipeline_v2.datasets.citevqa import (
    CiteVQA,
    CiteVQAConfig,
    InputTransform,
    _EvidenceMeta,
    _RowMeta,
    _Sample,
)
import importlib.util
from pipeline_v2.parsers.docling import DoclingTransform

import sys

_preprocess_path = Path(__file__).resolve().parent.parent / "usage" / "01_preprocess.py"
_spec = importlib.util.spec_from_file_location("preprocess_mod", _preprocess_path)
assert _spec is not None and _spec.loader is not None
preprocess_mod = importlib.util.module_from_spec(_spec)
sys.modules["preprocess_mod"] = preprocess_mod
_spec.loader.exec_module(preprocess_mod)
Preprocessor = preprocess_mod.Preprocessor


def test_citevqa_registration_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "citevqa" in datasets.list()
    dataset_cls = datasets.get("citevqa")
    assert dataset_cls is CiteVQA

    monkeypatch.setattr(
        CiteVQA, "_build_split_iterator", lambda self, split, data_dir: []
    )
    dataset = CiteVQA(config=CiteVQAConfig(max_samples=5))
    metadata = dataset._metadata()
    assert "CiteVQA" in metadata.description
    assert dataset._available_splits("/dummy/dir") == [DatasetSplitType.validation]


def test_evidence_and_row_meta_parsing() -> None:
    raw_evidence = [
        {
            "type": "table",
            "content": "<table>...</table>",
            "bbox": [224, 84, 469, 927],
            "source_pdf_name": "sample_doc.pdf",
            "source_page_id": 10,
            "source_doc_index": 1,
            "necessity": "necessary",
        }
    ]
    ev = _EvidenceMeta.from_dict(raw_evidence[0])
    assert ev.type == "table"
    assert ev.bbox == [224.0, 84.0, 469.0, 927.0]
    assert ev.source_pdf_name == "sample_doc.pdf"
    assert ev.source_page_id == 10

    raw_hf_row = {
        "index": "q_123",
        "Question_Type": "Multimodal Parsing",
        "Question": "What is the OD value?",
        "Standard_Answer": "0.075",
        "Evidence": raw_evidence,
        "dataset_type": "Single-Doc",
        "description": "Drug Package Inserts",
        "language": "en",
        "PDF_Source": ["data/pdf/sample_doc.pdf"],
    }
    row_meta = _RowMeta.from_hf_row(raw_hf_row)
    assert row_meta.question_id == "q_123"
    assert row_meta.question_type == "Multimodal Parsing"
    assert row_meta.standard_answer == "0.075"
    assert row_meta.pdf_sources == ["sample_doc.pdf"]
    assert len(row_meta.evidence_list) == 1


def test_input_transform_missing_pdf() -> None:
    transform = InputTransform()
    qa_meta = _RowMeta(
        question_id="q_missing",
        question_type="Factoid",
        question="Where is it?",
        standard_answer="Unknown",
        evidence_list=[],
        dataset_type="Single-Doc",
        description="Test",
        language="en",
        pdf_sources=["missing.pdf"],
    )
    sample = _Sample(
        question_id="q_missing",
        dataset_type="Single-Doc",
        pdf_paths=[Path("/nonexistent/path/missing.pdf")],
        qa_meta=qa_meta,
    )
    res = transform(sample)
    assert res.metadata.get("invalid_pdf") is True
    assert len(res.pages) == 0


def test_input_transform_with_mocked_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Create mock PDF files
    dummy_pdf1 = tmp_path / "test_doc1.pdf"
    dummy_pdf1.write_bytes(b"%PDF-1.4 dummy content 1")
    dummy_pdf2 = tmp_path / "test_doc2.pdf"
    dummy_pdf2.write_bytes(b"%PDF-1.4 dummy content 2")

    p0 = SinglePageDocumentInstance(
        sample_id="test_doc1.pdf#0", visual=MagicMock(), metadata={}
    )
    p1 = SinglePageDocumentInstance(
        sample_id="test_doc1.pdf#1", visual=MagicMock(), metadata={}
    )
    mock_doc1 = MultiPageDocumentInstance(
        sample_id="test_doc1.pdf",
        pages=[p0, p1],
        metadata={},
    )
    p2 = SinglePageDocumentInstance(
        sample_id="test_doc2.pdf#0", visual=MagicMock(), metadata={}
    )
    mock_doc2 = MultiPageDocumentInstance(
        sample_id="test_doc2.pdf",
        pages=[p2],
        metadata={},
    )

    def mock_from_pdf(path: Path, sample_id: str) -> MultiPageDocumentInstance:
        if "test_doc1" in path.name:
            return mock_doc1
        return mock_doc2

    monkeypatch.setattr(
        MultiPageDocumentInstance,
        "from_pdf",
        mock_from_pdf,
    )

    qa_meta = _RowMeta(
        question_id="q_001",
        question_type="Factoid",
        question="What is the test result?",
        standard_answer="Positive",
        evidence_list=[
            _EvidenceMeta(
                type="text",
                content="Result is Positive",
                bbox=[100.0, 50.0, 200.0, 150.0],
                source_pdf_name="test_doc1.pdf",
                source_page_id=2,  # 1-indexed page 2 in doc1 -> global page 1
                source_doc_index=1,
                necessity="necessary",
            ),
            _EvidenceMeta(
                type="table",
                content="Evidence table in doc2",
                bbox=[300.0, 100.0, 400.0, 200.0],
                source_pdf_name="test_doc2.pdf",
                source_page_id=1,  # 1-indexed page 1 in doc2 -> global page 2
                source_doc_index=2,
                necessity="necessary",
            ),
        ],
        dataset_type="N-to-N-Gold",
        description="Medical Report",
        language="en",
        pdf_sources=["test_doc1.pdf", "test_doc2.pdf"],
    )

    sample = _Sample(
        question_id="q_001",
        dataset_type="N-to-N-Gold",
        pdf_paths=[dummy_pdf1, dummy_pdf2],
        qa_meta=qa_meta,
    )

    transform = InputTransform()
    doc_instance = transform(sample)

    assert doc_instance.sample_id == "q_001"
    assert doc_instance.metadata["question_id"] == "q_001"
    assert doc_instance.metadata["dataset_type"] == "N-to-N-Gold"
    assert doc_instance.metadata["pdf_sources"] == ["test_doc1.pdf", "test_doc2.pdf"]
    assert len(doc_instance.pages) == 3  # 2 pages from doc1 + 1 page from doc2

    # Page annotations
    assert not doc_instance.pages[0].has_annotation_type(
        AnnotationType.object_detection
    )
    assert doc_instance.pages[1].has_annotation_type(AnnotationType.object_detection)
    assert doc_instance.pages[2].has_annotation_type(AnnotationType.object_detection)

    det_ann = doc_instance.pages[1].get_annotation_by_type(
        AnnotationType.object_detection
    )
    assert isinstance(det_ann, ObjectDetectionAnnotation)
    assert det_ann.bbox_mode == BoundingBoxMode.XYXY
    np.testing.assert_allclose(det_ann.bboxes[0], [50.0, 100.0, 150.0, 200.0])

    # QA annotation
    assert doc_instance.has_annotation_type(
        AnnotationType.multi_page_question_answering
    )
    qa_ann = doc_instance.get_annotation_by_type(
        AnnotationType.multi_page_question_answering
    )
    assert isinstance(qa_ann, MultiPageQuestionAnsweringAnnotation)
    assert len(qa_ann.qa_pairs) == 1
    qa_pair = qa_ann.qa_pairs[0]
    assert qa_pair.id == 0
    assert qa_pair.question_text == "What is the test result?"
    assert qa_pair.answer_text == "Positive"
    assert qa_pair.evidence_pages == [1, 2]  # global 0-indexed page positions
    assert qa_pair.evidence_sources == ["text", "table"]


from PIL import Image


def test_citevqa_docling_preprocess_run(tmp_path: Path) -> None:
    # Verify that a CiteVQA MultiPageDocumentInstance can be preprocessed by 01_preprocess.Preprocessor
    raw_doc = DoclingDocument(name="test_page").model_dump()
    mock_payload = {
        "document": {
            "json_content": raw_doc,
            "md_content": "# Parsed content",
            "filename": "test.png",
        },
        "status": "success",
        "processing_time": 0.1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url="http://mock-docling:10001", client=client)

    mock_page = MagicMock(spec=SinglePageDocumentInstance)
    mock_page.sample_id = "cite_doc#0"
    mock_page.key = "cite_doc_0"
    img = Image.new("RGB", (50, 50), color="white")
    mock_content = MagicMock()
    mock_content.require_content.return_value = img
    mock_page.load.return_value = mock_content

    sample = MultiPageDocumentInstance(
        sample_id="cite_doc", pages=[mock_page], metadata={}
    )

    out_dir = tmp_path / "docling" / "validation"
    preprocessor = Preprocessor(
        out_dir=out_dir,
        api_url="http://mock-docling:10001",
        num_workers=1,
    )
    preprocessor._process_sample(sample, transform=transform)

    saved_file = out_dir / sample.key / f"{mock_page.key}.json"
    assert saved_file.exists()
    assert "test_page" in saved_file.read_text()
