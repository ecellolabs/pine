"""Pydantic AI model plumbing: provider selection (OpenRouter or any
OpenAI-compatible server such as vLLM), an on-disk response cache so identical
requests are never paid for twice, and a per-call trace (``llm_calls.jsonl``)
with tokens, cost and latency.

``TracedModel`` is a ``pydantic_ai.models.wrapper.WrapperModel``: every agent
in the pipeline is built on top of it, so caching and accounting are uniform."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from pipeline_v2.agent.common import AgentSettings, CostLedger, append_jsonl

_RESPONSE_ADAPTER: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)
_PARAMS_ADAPTER: TypeAdapter[ModelRequestParameters] = TypeAdapter(
    ModelRequestParameters
)
_VOLATILE_KEYS = {"timestamp", "run_id", "conversation_id", "metadata", "state"}


def base_model(settings: AgentSettings, model_name: str) -> Model:
    """Build the provider model for ``model_name`` from the run settings."""
    if settings.model_factory is not None:
        return settings.model_factory(model_name)
    if "openrouter.ai" in settings.api_url:
        from pydantic_ai.models.openrouter import (
            OpenRouterModel,
            OpenRouterModelSettings,
        )
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(
            model_name,
            provider=OpenRouterProvider(api_key=settings.api_key),
            # ask OpenRouter to report the cost of every request
            settings=OpenRouterModelSettings(openrouter_usage={"include": True}),
        )
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=settings.api_url, api_key=settings.api_key or "dummy"
        ),
    )


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _redact(obj: Any) -> Any:
    """Replace inline binary content with a size marker for the trace."""
    if isinstance(obj, dict):
        if obj.get("kind") == "binary" and "data" in obj:
            return {
                "kind": "binary",
                "media_type": obj.get("media_type"),
                "data": f"<{len(obj['data'])} base64 chars>",
            }
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _user_content(part: UserPromptPart) -> Any:
    if isinstance(part.content, str):
        return part.content
    out: list[Any] = []
    for item in part.content:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, BinaryContent):
            out.append(f"<{item.media_type} {len(item.data)} bytes>")
        else:
            out.append(str(item)[:200])
    return out


def messages_to_chat(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Flatten Pydantic AI messages into the role/content shape used by the
    visual-samples page and the logs."""
    chat: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions:
                chat.append({"role": "system", "content": message.instructions})
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    chat.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    chat.append({"role": "user", "content": _user_content(part)})
                elif isinstance(part, ToolReturnPart):
                    chat.append(
                        {
                            "role": "tool",
                            "name": part.tool_name,
                            "content": part.model_response_str(),
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    chat.append(
                        {
                            "role": "retry",
                            "name": part.tool_name,
                            "content": part.model_response(),
                        }
                    )
        elif isinstance(message, ModelResponse):
            chat.append(response_to_chat(message))
    return chat


def response_to_chat(response: ModelResponse) -> dict[str, Any]:
    text = "\n".join(p.content for p in response.parts if isinstance(p, TextPart))
    calls = [
        {
            "id": p.tool_call_id,
            "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
        }
        for p in response.parts
        if isinstance(p, ToolCallPart)
    ]
    out: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        out["tool_calls"] = calls
    return out


def response_usage(response: ModelResponse) -> dict[str, Any]:
    usage = response.usage
    cost: Any = usage.cost
    if cost is None and response.provider_details:
        cost = response.provider_details.get("cost")
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "cost": float(cost) if cost is not None else None,
    }


class TracedModel(WrapperModel):
    """Wraps any Pydantic AI model with a sha256-keyed on-disk cache, a jsonl
    trace per step (and per run) and cost accounting.

    Set ``purpose`` before an agent run so the trace says what the call was for."""

    def __init__(
        self,
        wrapped: Model,
        *,
        settings: AgentSettings,
        step_dir: Path,
        ledger: CostLedger,
        logger: logging.Logger,
        step_name: str,
    ) -> None:
        super().__init__(wrapped)
        self.run_settings = settings
        self.step_dir = step_dir
        self.ledger = ledger
        self.logger = logger
        self.step_name = step_name
        self.purpose = ""
        self.cache_dir = settings.llm_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = step_dir / "llm_calls.jsonl"
        self.run_trace_path = ledger.run_dir / "llm_calls.jsonl"

    def _cache_key(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        params: ModelRequestParameters,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.wrapped.model_name,
            "messages": _strip_volatile(
                ModelMessagesTypeAdapter.dump_python(messages, mode="json")
            ),
            "settings": dict(model_settings) if model_settings else None,
        }
        try:
            payload["params"] = _PARAMS_ADAPTER.dump_python(params, mode="json")
        except Exception:  # noqa: BLE001  (unserialisable parameter objects)
            payload["params"] = repr(params)
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        key = self._cache_key(messages, model_settings, model_request_parameters)
        cache_file = self.cache_dir / f"{key}.json"
        from_cache = cache_file.exists()
        t0 = time.perf_counter()
        error: str | None = None
        response: ModelResponse | None = None
        exc_to_raise: BaseException | None = None
        if from_cache:
            response = _RESPONSE_ADAPTER.validate_json(cache_file.read_bytes())
        else:
            try:
                response = await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                exc_to_raise = exc
            else:
                cache_file.write_bytes(_RESPONSE_ADAPTER.dump_json(response))
        latency = time.perf_counter() - t0
        usage = response_usage(response) if response else {}
        self.ledger.add(usage, latency, from_cache)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step": self.step_name,
            "purpose": self.purpose,
            "model": self.wrapped.model_name,
            "from_cache": from_cache,
            "latency_s": round(latency, 2),
            "usage": usage,
            "error": error,
            "request": {
                "messages": messages_to_chat(messages),
                "tools": [t.name for t in model_request_parameters.function_tools]
                + [t.name for t in model_request_parameters.output_tools],
                "output_mode": model_request_parameters.output_mode,
                "settings": dict(model_settings) if model_settings else None,
            },
            "response_message": response_to_chat(response) if response else {},
            "finish_reason": response.finish_reason if response else None,
            "provider": response.provider_name if response else None,
        }
        append_jsonl(self.trace_path, record)
        append_jsonl(self.run_trace_path, record)
        self.logger.info(
            "LLM[%s] %s cache=%s tokens=%s/%s cost=$%s latency=%.1fs finish=%s%s",
            self.purpose,
            self.wrapped.model_name,
            from_cache,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("cost"),
            latency,
            record["finish_reason"],
            f" ERROR={error}" if error else "",
        )
        if exc_to_raise is not None:
            raise exc_to_raise
        assert response is not None
        return response


def traced_model(
    settings: AgentSettings,
    model_name: str,
    *,
    step_dir: Path,
    ledger: CostLedger,
    logger: logging.Logger,
    step_name: str,
) -> TracedModel:
    """Convenience: provider model for ``model_name`` wrapped in ``TracedModel``."""
    return TracedModel(
        base_model(settings, model_name),
        settings=settings,
        step_dir=step_dir,
        ledger=ledger,
        logger=logger,
        step_name=step_name,
    )
