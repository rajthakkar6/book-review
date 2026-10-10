"""LLM interface with an OpenAI implementation and a scripted fake for offline tests."""
import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0
    raw: dict | None = None  # provider payload, for tracing


class LLM(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAILLM:
    def __init__(self, model: str = config.MODEL_NAME, api_key: str | None = None):
        from openai import AsyncOpenAI  # imported lazily so tests need no key

        client = AsyncOpenAI(api_key=api_key)
        if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":  # optional; also traces LLM calls
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        self.model, self.client = model, client

    async def complete(self, messages, tools):
        extra = {"reasoning_effort": config.REASONING_EFFORT} if config.REASONING_EFFORT else {}
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None, **extra)
        m = r.choices[0].message
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {"_malformed": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))
        return LLMResponse(m.content, calls, r.usage.total_tokens if r.usage else 0,
                           r.model_dump())


class FakeLLM:
    """Plays a script. Each item is an LLMResponse, an Exception to raise, or a callable(messages)->LLMResponse."""

    def __init__(self, script: list):
        self.script, self.calls = list(script), []

    async def complete(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages) if callable(item) else item
