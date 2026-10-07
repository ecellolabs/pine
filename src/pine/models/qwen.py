"""Qwen Vision-Language model interface via OpenAI-compatible API (e.g. vLLM)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import httpx
from atria_core.logger import get_logger
from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field

logger = get_logger(__name__)

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_API_URL = "http://localhost:8000/v1"


def resolve_chat_url(base_url: str) -> str:
    """Normalizes an endpoint URL to an OpenAI-compatible /chat/completions endpoint."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def resolve_models_url(base_url: str) -> str:
    """Normalizes an endpoint URL to an OpenAI-compatible /models endpoint."""
    url = base_url.rstrip("/")
    url = url.removesuffix("/chat/completions")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/models"


def pil_to_base64_data_uri(img: PILImage, format: str = "PNG") -> str:
    """Encodes a PIL Image into a base64 Data URI."""
    buffered = io.BytesIO()
    # Convert RGBA/P to RGB when saving as JPEG
    if format.upper() in ("JPEG", "JPG") and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() in ("JPEG", "JPG") else "image/png"
    return f"data:{mime};base64,{encoded}"


def extract_answer_content(content: str) -> str:
    """Cleans raw output, stripping any reasoning or thinking tokens."""
    if "</think>" in content:
        content = content.split("</think>", 1)[-1]
    return content.strip()


class QwenVLConfig(BaseModel):
    """Configuration for Qwen Vision-Language model API client."""

    model_config = ConfigDict(extra="ignore")

    api_url: str = Field(
        default_factory=lambda: (
            os.environ.get("QWEN_API_URL")
            or os.environ.get("VLM_BASE_URL")
            or DEFAULT_QWEN_API_URL
        )
    )
    model_id: str = Field(
        default_factory=lambda: (
            os.environ.get("QWEN_MODEL_ID")
            or os.environ.get("VLM_MODEL")
            or DEFAULT_QWEN_MODEL_ID
        )
    )
    api_key: str = Field(
        default_factory=lambda: (
            os.environ.get("QWEN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
    )
    temperature: float = 0.0
    max_tokens: int = 128
    timeout: float = 120.0
    system_prompt: str | None = None
    mock: bool = False

    # Backwards compatibility fields
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int | None = None

    def model_post_init(self, __context: Any, /) -> None:
        if self.max_new_tokens is not None:
            self.max_tokens = self.max_new_tokens


class QwenVLModel:
    """Client for Qwen Vision-Language model hosted on vLLM / OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: QwenVLConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or QwenVLConfig()
        self._chat_url = resolve_chat_url(self.config.api_url)
        self._models_url = resolve_models_url(self.config.api_url)
        self._client = client or httpx.Client(timeout=self.config.timeout)

    @property
    def device(self) -> str:
        return getattr(self.config, "device", "api")

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def check_connection(self) -> tuple[bool, str]:
        """Tests whether the API endpoint is reachable and reports served models."""
        if self.config.mock:
            return True, "Mock mode active (no network calls required)."

        try:
            resp = self._client.get(self._models_url, headers=self.headers, timeout=5.0)
            if resp.is_success:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                return True, f"Connected to {self.config.api_url} (models: {models})"
            return (
                False,
                f"Endpoint {self._models_url} returned HTTP {resp.status_code}: {resp.text}",
            )
        except Exception as exc:  # noqa: BLE001
            return (
                False,
                f"Cannot reach endpoint at {self.config.api_url} ({type(exc).__name__}: {exc})",
            )

    def predict(self, images: list[PILImage], question: str) -> str:
        """Runs inference over a list of slide/document images and a question prompt."""
        if self.config.mock:
            return self.predict_mock(images, question)

        content: list[dict[str, Any]] = []

        # Encode slide images in order
        for img in images:
            data_uri = pil_to_base64_data_uri(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )

        prompt_text = (
            f"Question: {question}\n"
            "Using the visual information from the slide deck above, provide a direct, "
            "factual, and concise answer to the question. Do not elaborate."
        )
        content.append({"type": "text", "text": prompt_text})

        system_text = self.config.system_prompt or (
            "You are an expert visual document reader analyzing presentation slides. "
            "Carefully inspect visual details, chart legends, axes, table cells, and diagrams. "
            "Provide a concise, direct, and factual answer to the question without unnecessary explanation."
        )

        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
        }

        try:
            resp = self._client.post(self._chat_url, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            raw_content = (
                choice.get("content")
                or choice.get("reasoning")
                or choice.get("reasoning_content")
                or ""
            )
            return extract_answer_content(str(raw_content))
        except httpx.HTTPStatusError as err:
            err_msg = f"API error from {self._chat_url} ({err.response.status_code}): {err.response.text}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from err
        except Exception as exc:
            err_msg = f"Failed to query model at {self._chat_url}: {exc}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

    def predict_mock(self, images: list[PILImage], question: str) -> str:
        """Deterministic mock response for pipeline verification without network calls."""
        return f"answer to '{question}'"
