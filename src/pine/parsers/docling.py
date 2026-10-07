"""Docling-based document parsing via external Docling API service.
Parallelism (multiprocessing, Ray, etc.) is deliberately not this module's
concern -- wrap `DoclingTransform.__call__` from the outside if you want it.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Self

import httpx
from atria_core.types import SinglePageDocumentInstance
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel, ConfigDict, Field


class DoclingApiError(RuntimeError):
    """Raised when communication with the Docling API fails or returns an error."""


class DoclingApiOptions(BaseModel):
    """Pydantic model representing options for the Docling API conversion."""

    model_config = ConfigDict(extra="ignore")

    to_formats: list[str] = Field(default_factory=lambda: ["json"])
    ocr: bool = True
    force_ocr: bool = False
    image_export_mode: str = "placeholder"
    ocr_engine: str | None = None
    pdf_backend: str | None = None
    table_mode: str | None = None
    abort_on_error: bool = False

    def to_form_data(self) -> dict[str, Any]:
        """Convert options to form-data compatible format for httpx."""
        data: dict[str, Any] = {}
        for key, value in self.model_dump(exclude_none=True).items():
            if isinstance(value, bool):
                data[key] = "true" if value else "false"
            else:
                data[key] = value
        return data


class DoclingDocumentResponse(BaseModel):
    """Parsed document contents in the API response."""

    model_config = ConfigDict(extra="ignore")

    json_content: DoclingDocument | None = None
    md_content: str | None = None
    html_content: str | None = None
    text_content: str | None = None
    filename: str | None = None


class ConvertDocumentApiResponse(BaseModel):
    """Top-level response returned by POST /v1/convert/file."""

    model_config = ConfigDict(extra="ignore")

    document: DoclingDocumentResponse
    status: str = "success"
    processing_time: float | None = None
    errors: list[Any] = Field(default_factory=list)


@dataclass
class DoclingTransform:
    """Runs docling document parsing via external Docling API service
    (layout analysis + OCR) on page images and returns a validated
    `DoclingDocument` tree."""

    api_url: str = field(default_factory=lambda: os.getenv("DOCLING_API_URL", ""))
    options: DoclingApiOptions = field(default_factory=DoclingApiOptions)
    pipeline_options: Any = None
    timeout: float = 120.0
    client: httpx.Client | None = field(default=None, repr=False)
    _owns_client: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        if not self.api_url:
            raise ValueError(
                "Docling API URL must be provided via api_url or the DOCLING_API_URL "
                "environment variable. Note that the DFKI cluster server node and port "
                "change dynamically across job allocations."
            )
        if self.pipeline_options is not None:
            do_ocr = getattr(self.pipeline_options, "do_ocr", None)
            if do_ocr is not None:
                self.options = self.options.model_copy(update={"ocr": bool(do_ocr)})
        if self.client is None:
            self._owns_client = True
            self.client = httpx.Client(timeout=self.timeout)
        else:
            self._owns_client = False

    def health_check(self) -> dict[str, Any]:
        """Queries GET /health on the Docling API server."""
        assert self.client is not None
        url = f"{self.api_url.rstrip('/')}/health"
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    def convert_bytes(
        self,
        content: bytes,
        filename: str = "document.png",
        content_type: str = "image/png",
    ) -> DoclingDocument:
        """Uploads document bytes to POST /v1/convert/file and returns validated DoclingDocument."""
        assert self.client is not None
        url = f"{self.api_url.rstrip('/')}/v1/convert/file"
        files = {
            "files": (filename, content, content_type),
        }
        data = self.options.to_form_data()

        try:
            response = self.client.post(url, files=files, data=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DoclingApiError(
                f"Docling API request failed with status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise DoclingApiError(
                f"Docling API request connection error to {url}: {exc}"
            ) from exc

        try:
            result = ConvertDocumentApiResponse.model_validate(response.json())
        except Exception as exc:
            raise DoclingApiError(
                f"Failed to parse Docling API response with Pydantic: {exc}"
            ) from exc

        if result.document.json_content is None:
            raise DoclingApiError(
                "Docling API converted successfully but returned no 'json_content' in document."
            )

        return result.document.json_content

    def __call__(self, page: SinglePageDocumentInstance) -> DoclingDocument:
        image = page.load().require_content()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.convert_bytes(buffer.getvalue(), filename=f"{page.key}.png")

    def close(self) -> None:
        """Closes the underlying HTTP client if owned."""
        if self._owns_client and self.client is not None:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
