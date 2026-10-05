from unittest.mock import MagicMock

import httpx
import pytest
from docling_core.types.doc.document import DoclingDocument
from PIL import Image

from pipeline_v2.parsers.docling import (
    ConvertDocumentApiResponse,
    DoclingApiError,
    DoclingApiOptions,
    DoclingTransform,
)

MOCK_API_URL = "http://mock-docling-server:10001"


def test_docling_api_options_defaults() -> None:
    options = DoclingApiOptions()
    assert options.to_formats == ["json"]
    assert options.ocr is True
    assert options.force_ocr is False
    assert options.image_export_mode == "placeholder"

    form_data = options.to_form_data()
    assert form_data["to_formats"] == ["json"]
    assert form_data["ocr"] == "true"
    assert form_data["force_ocr"] == "false"
    assert form_data["image_export_mode"] == "placeholder"


def test_convert_document_api_response_validation() -> None:
    raw_doc = DoclingDocument(name="sample_page").model_dump()
    payload = {
        "document": {
            "json_content": raw_doc,
            "md_content": "# Sample Title\nSample content",
            "filename": "sample.png",
        },
        "status": "success",
        "processing_time": 0.35,
    }

    response_model = ConvertDocumentApiResponse.model_validate(payload)
    assert response_model.status == "success"
    assert response_model.processing_time == 0.35
    assert response_model.document.json_content is not None
    assert isinstance(response_model.document.json_content, DoclingDocument)
    assert response_model.document.json_content.name == "sample_page"

    # Ensure export_to_dict works on the deserialized document
    exported = response_model.document.json_content.export_to_dict()
    assert isinstance(exported, dict)
    assert exported["name"] == "sample_page"


def test_docling_transform_missing_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCLING_API_URL", raising=False)
    with pytest.raises(ValueError, match="Docling API URL must be provided"):
        DoclingTransform(api_url="")


def test_docling_transform_health_check() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "docling_version": "2.120.0",
                "service": "docling-serve-dfki",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url=MOCK_API_URL, client=client)
    health = transform.health_check()
    assert health["status"] == "ok"
    assert health["service"] == "docling-serve-dfki"


def test_docling_transform_convert_bytes_success() -> None:
    raw_doc = DoclingDocument(name="test_page").model_dump()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/convert/file"
        assert request.method == "POST"
        content_type = request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

        return httpx.Response(
            200,
            json={
                "document": {
                    "json_content": raw_doc,
                    "md_content": "# Extracted Markdown",
                },
                "status": "success",
                "processing_time": 0.42,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(
        api_url=MOCK_API_URL,
        options=DoclingApiOptions(ocr=True),
        client=client,
    )

    doc = transform.convert_bytes(b"dummy image bytes", filename="test.png")
    assert isinstance(doc, DoclingDocument)
    assert doc.name == "test_page"

    exported = doc.export_to_dict()
    assert isinstance(exported, dict)
    assert exported["name"] == "test_page"


def test_docling_transform_call_with_page() -> None:
    raw_doc = DoclingDocument(name="page_1").model_dump()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "document": {
                    "json_content": raw_doc,
                },
                "status": "success",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url=MOCK_API_URL, client=client)

    # Mock SinglePageDocumentInstance
    mock_page = MagicMock()
    mock_page.key = "page_1"
    img = Image.new("RGB", (50, 50), color="white")
    mock_content = MagicMock()
    mock_content.require_content.return_value = img
    mock_page.load.return_value = mock_content

    doc = transform(mock_page)
    assert isinstance(doc, DoclingDocument)
    assert doc.name == "page_1"


def test_docling_transform_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error during OCR")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url=MOCK_API_URL, client=client)

    with pytest.raises(
        DoclingApiError, match="Docling API request failed with status 500"
    ):
        transform.convert_bytes(b"dummy bytes")


def test_docling_transform_missing_json_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "document": {
                    "md_content": "only md, no json",
                },
                "status": "success",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transform = DoclingTransform(api_url=MOCK_API_URL, client=client)

    with pytest.raises(DoclingApiError, match="returned no 'json_content'"):
        transform.convert_bytes(b"dummy bytes")
