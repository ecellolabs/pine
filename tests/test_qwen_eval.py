from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
from PIL import Image

from pipeline_v2.evaluation import evaluate_dataset
from pipeline_v2.models.qwen import (
    QwenVLConfig,
    QwenVLModel,
    extract_answer_content,
    pil_to_base64_data_uri,
    resolve_chat_url,
    resolve_models_url,
)


def test_resolve_urls() -> None:
    assert (
        resolve_chat_url("http://serv-100:10001")
        == "http://serv-100:10001/v1/chat/completions"
    )
    assert (
        resolve_chat_url("http://serv-100:10001/v1")
        == "http://serv-100:10001/v1/chat/completions"
    )
    assert (
        resolve_chat_url("http://serv-100:10001/v1/chat/completions")
        == "http://serv-100:10001/v1/chat/completions"
    )

    assert (
        resolve_models_url("http://serv-100:10001") == "http://serv-100:10001/v1/models"
    )
    assert (
        resolve_models_url("http://serv-100:10001/v1")
        == "http://serv-100:10001/v1/models"
    )
    assert (
        resolve_models_url("http://serv-100:10001/v1/chat/completions")
        == "http://serv-100:10001/v1/models"
    )


def test_qwen_config_defaults() -> None:
    config = QwenVLConfig()
    assert "Qwen" in config.model_id
    assert config.max_tokens == 128
    assert config.mock is False
    assert config.api_url.startswith("http")


def test_extract_answer_content() -> None:
    raw = "<think>Let me count the items: 1, 2, 3</think>There are 3 items."
    assert extract_answer_content(raw) == "There are 3 items."

    plain = "42 percent"
    assert extract_answer_content(plain) == "42 percent"


def test_pil_to_base64_data_uri() -> None:
    img = Image.new("RGB", (10, 10), color="blue")
    uri = pil_to_base64_data_uri(img)
    assert uri.startswith("data:image/png;base64,")


def test_qwen_mock_prediction() -> None:
    config = QwenVLConfig(mock=True)
    model = QwenVLModel(config=config)

    img = Image.new("RGB", (100, 100), color="white")
    ans = model.predict([img], "What is on the first slide?")
    assert "first slide" in ans


def test_qwen_check_connection_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = QwenVLConfig(api_url="http://mock-serv:8000/v1")
    model = QwenVLModel(config=config, client=client)

    ok, msg = model.check_connection()
    assert ok is True
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in msg


def test_qwen_check_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = QwenVLConfig(api_url="http://mock-serv:8000/v1")
    model = QwenVLModel(config=config, client=client)

    ok, msg = model.check_connection()
    assert ok is False
    assert "503" in msg


def test_qwen_api_prediction_with_mock_client() -> None:
    captured_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        assert request.url.path == "/v1/chat/completions"
        captured_payload = httpx.Response(200, content=request.content).json()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Estimated revenue is $5M.",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = QwenVLConfig(
        api_url="http://mock-serv:8000/v1",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    model = QwenVLModel(config=config, client=client)

    img = Image.new("RGB", (20, 20), color="red")
    answer = model.predict([img], "What is the estimated revenue?")

    assert answer == "Estimated revenue is $5M."
    assert captured_payload["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    messages = captured_payload["messages"]
    assert len(messages) == 2
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1]["type"] == "text"
    assert "What is the estimated revenue?" in user_content[1]["text"]


def test_evaluate_dataset_mock(tmp_path: Path) -> None:
    config = QwenVLConfig(mock=True)
    model = QwenVLModel(config=config)

    # Construct mock dataset
    mock_dataset = MagicMock()
    mock_split_type = MagicMock()
    mock_split_type.value = "validation"

    mock_sample = MagicMock()
    mock_sample.sample_id = "sample_deck_1"
    mock_sample.metadata = {"deck_name": "sample_deck_1"}

    mock_page = MagicMock()
    mock_page_content = MagicMock()
    mock_page_content.require_content.return_value = Image.new(
        "RGB", (50, 50), color="white"
    )
    mock_page.load.return_value = mock_page_content
    mock_sample.pages = [mock_page]

    mock_qa = MagicMock()
    mock_qa.id = 101
    mock_qa.question_text = "What is the title?"
    mock_qa.answer_text = "answer to 'What is the title?'"
    mock_qa.evidence_pages = [0]
    mock_qa.alternative_answers = []

    mock_annotation = MagicMock()
    mock_annotation.qa_pairs = [mock_qa]
    mock_sample.get_annotation_by_type.return_value = mock_annotation

    mock_dataset.split_iterators = {mock_split_type: [mock_sample]}

    out_file = tmp_path / "results.json"
    report = evaluate_dataset(mock_dataset, model, output_file=out_file)

    assert "validation" in report["splits"]
    summary = report["splits"]["validation"]
    assert summary["num_decks"] == 1
    assert summary["num_questions"] == 1
    assert summary["mean_anls"] == 1.0
    assert summary["mean_f1"] == 1.0
    assert summary["mean_em"] == 1.0
    assert out_file.exists()
