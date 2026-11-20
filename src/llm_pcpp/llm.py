"""Minimal chat-model interface and an OpenAI implementation.

The paper used ``gpt-4o`` through the OpenAI API with ``temperature=0``,
``max_tokens=1000`` and ``stop=["\\n"]`` for the LLM-A* waypoints. Any object
with a ``name`` attribute and a ``complete(prompt) -> str`` method can be used
instead (e.g. a local model, or a fake model in the tests).
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, Protocol, Sequence


class ChatModel(Protocol):
    """Anything that turns a prompt into a completion."""

    name: str

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


# Reasoning models (o1, o3, o4-mini, ...) reject temperature/stop/max_tokens.
_REASONING_MODEL = re.compile(r"^o\d")


def is_reasoning_model(name: str) -> bool:
    return bool(_REASONING_MODEL.match(name))


class OpenAIChatModel:
    """Single-turn chat completion through the OpenAI Python SDK (``openai>=1.0``).

    The API key is read from the ``OPENAI_API_KEY`` environment variable. The
    client is created on the first query, so replaying cached responses does
    not need a key.
    """

    def __init__(
        self,
        name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        stop: Optional[Sequence[str]] = ("\n",),
        client: Any = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = list(stop) if stop else None
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "LLM queries need the OpenAI Python SDK 1.0 or newer: pip install -U 'openai>=1.0'"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "Set the OPENAI_API_KEY environment variable to query the LLM."
                )
            self._client = OpenAI(timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def request_kwargs(self, prompt: str) -> dict:
        kwargs: dict = {"model": self.name, "messages": [{"role": "user", "content": prompt}]}
        if is_reasoning_model(self.name):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens
            if self.stop:
                kwargs["stop"] = self.stop
        return kwargs

    def complete(self, prompt: str) -> str:
        response = self._get_client().chat.completions.create(**self.request_kwargs(prompt))
        return response.choices[0].message.content or ""
