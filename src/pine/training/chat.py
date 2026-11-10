"""Thinking-on sampling client for STaR: K answers per prompt, each with its reasoning.

The other chat helpers keep only ``message.content`` (``baseline.make_openai_chat``) or turn
thinking off (``vidore.make_chat``); STaR trains on the reasoning, so it has to come back.
Servers return it in different places:

- OpenRouter: ``message.reasoning``
- vLLM with ``--reasoning-parser qwen3``: ``message.reasoning_content``
- vLLM without a reasoning parser: inside ``content``, ended by ``</think>`` or, with the
  Qwen3.5 template, by a gap of three newlines (see ``baseline.clean_reply``)

Sampling follows ``--navigator direct`` (``agent/llm.apply_thinking_sampling``): temperature
0.6, top_p 0.95, top_k 20, up to 12000 tokens.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

THINK_MAX_TOKENS = (
    12000  # direct mode's thinking budget (agent/direct.ANSWER_MAX_TOKENS)
)


@dataclass
class Sample:
    reasoning: str
    answer: str
    finish_reason: str
    # True when the reply ran out of tokens before an answer: never counted as correct
    truncated: bool
    completion_tokens: int = 0


@dataclass
class Batch:
    """The K samples of one prompt and what they cost (prompt tokens counted once per call)."""

    samples: list[Sample]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


# sample(messages, n) -> Batch
SampleFn = Callable[[list[dict[str, Any]], int], Batch]


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def split_reply(message: Any, finish_reason: str) -> Sample:
    """Reasoning and final answer of one choice, whichever way the server returns them."""
    content = re.sub(
        r"<\|(?:im_end|endoftext)\|>", "", _field(message, "content") or ""
    )
    reasoning = _field(message, "reasoning_content") or _field(message, "reasoning")
    if reasoning is None:
        if "</think>" in content:
            reasoning, content = content.rsplit("</think>", 1)
            reasoning = reasoning.replace("<think>", "", 1)
        elif "\n\n\n" in content:
            reasoning, content = content.rsplit("\n\n\n", 1)
        elif finish_reason != "length":
            reasoning = (
                ""  # a reply with no visible thinking: the whole content is the answer
            )
        else:
            reasoning, content = content, ""
    answer = content.strip()
    truncated = finish_reason == "length" or not answer
    return Sample(
        reasoning=str(reasoning).strip(),
        answer="" if truncated else answer,
        finish_reason=finish_reason,
        truncated=truncated,
    )


def make_thinking_chat(
    api_url: str,
    api_key: str,
    model: str,
    timeout: float = 900.0,
    max_tokens: int = THINK_MAX_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> SampleFn:
    """``n`` thinking samples from one request; falls back to single requests when the
    provider returns fewer choices than asked for (some OpenRouter providers ignore ``n``)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=api_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=3
    )
    extra: dict[str, Any] = {
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": True},  # vLLM
        "reasoning": {"enabled": True},  # OpenRouter
    }
    if "openrouter.ai" in api_url:
        extra["usage"] = {"include": True}  # puts the call's cost in usage.cost

    def call(messages: list[dict[str, Any]], n: int) -> Batch:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            extra_body=extra,
        )
        samples = [split_reply(c.message, c.finish_reason or "") for c in resp.choices]
        usage = resp.usage
        out = Batch(
            samples=samples,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost=float(_field(usage, "cost") or 0.0) if usage else 0.0,
        )
        if len(samples) == 1:
            samples[0].completion_tokens = out.completion_tokens
        return out

    def sample(messages: list[dict[str, Any]], n: int) -> Batch:
        batch = call(messages, n)
        while len(batch.samples) < n:
            more = call(messages, 1)
            batch.samples += more.samples
            batch.prompt_tokens += more.prompt_tokens
            batch.completion_tokens += more.completion_tokens
            batch.cost += more.cost
        batch.samples = batch.samples[:n]
        return batch

    return sample
