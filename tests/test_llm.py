import inspect

import pytest

from llm_pcpp.llm import OpenAIChatModel, is_reasoning_model


def test_request_parameters():
    # gpt-4o settings of the paper: temperature 0, max_tokens 1000, stop at the first newline.
    chat = OpenAIChatModel("gpt-4o", client=object())
    assert chat.request_kwargs("hi") == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 1000,
        "stop": ["\n"],
    }
    reasoning = OpenAIChatModel("o1", max_tokens=8000, stop=None, client=object())
    assert reasoning.request_kwargs("hi") == {
        "model": "o1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 8000,
    }
    assert is_reasoning_model("o4-mini") and not is_reasoning_model("gpt-4o")


def test_complete_calls_chat_completions():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("Message", (), {"content": "Generated Path: [[1, 2], [3, 4]]"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    chat = OpenAIChatModel("gpt-4o", client=client)
    assert chat.complete("prompt") == "Generated Path: [[1, 2], [3, 4]]"
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]


def test_request_matches_installed_openai_sdk():
    """The request parameters must be accepted by the installed SDK's chat.completions.create."""
    openai = pytest.importorskip("openai")
    if not hasattr(openai, "OpenAI"):
        pytest.skip("openai < 1.0 is installed")
    client = openai.OpenAI(api_key="test-key")
    signature = inspect.signature(client.chat.completions.create)
    for model in (OpenAIChatModel("gpt-4o", client=client), OpenAIChatModel("o1", stop=None, client=client)):
        signature.bind(**model.request_kwargs("prompt"))  # TypeError on unknown parameters


def test_missing_api_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY|OpenAI Python SDK"):
        OpenAIChatModel("gpt-4o").complete("prompt")
