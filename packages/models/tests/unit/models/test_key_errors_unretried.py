"""The provider factories' ``unretried_key_errors`` mode: a refusal that belongs to the key reaches the caller
on its first occurrence, while a 5xx is still retried as often as before.

A key pool fails over to another key on a 429 (KP-H4). The SDK must not first sleep on the 429 and
retry it on the same key. Every test here drives the real SDK over a scripted wire and counts the
requests it sent; no network. Each retried answer carries ``retry-after-ms: 1`` (``retry-after`` for
the openrouter SDK, which reads no milliseconds) so the SDK's sleep between retries stays out of the run.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import anthropic
import httpx
import openai
import openrouter.errors
import pytest

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers.anthropic import create_anthropic_chat
from threetears.models.providers.openai import create_openai_chat
from threetears.models.providers.openrouter import create_openrouter_chat

from . import key_error_fixtures as fx
from .provider_wire import ChatCompletionsWire, anthropic_model, serve_http_handler

#: The SDKs' default retries of a failed request: two, so three requests in all.
_DEFAULT_ATTEMPTS = 3

#: A wait the SDK honours instead of its own backoff, short enough to keep out of the run.
_STAINLESS_SHORT_WAIT = {"retry-after-ms": "1"}
_OPENROUTER_SHORT_WAIT = {"retry-after": "0.001"}


class _CountingWire:
    """an API that counts the requests it receives and answers the n-th with ``answer(n)``."""

    def __init__(self, answer: Callable[[int], httpx.Response]) -> None:
        self._answer = answer
        self._lock = threading.Lock()
        self.requests = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        """answer one request.

        :param request: the request the SDK sent
        :ptype request: httpx.Request
        :return: the scripted answer
        :rtype: httpx.Response
        """
        del request
        with self._lock:
            self.requests += 1
            attempt = self.requests
        return self._answer(attempt)


def _always(recorded: fx.RecordedError, extra: dict[str, str]) -> _CountingWire:
    return _CountingWire(lambda attempt: recorded.response(extra_headers=extra))


def _fails_then_answers(
    recorded: fx.RecordedError, extra: dict[str, str], *, failures: int, ok: httpx.Response
) -> _CountingWire:
    return _CountingWire(lambda attempt: recorded.response(extra_headers=extra) if attempt <= failures else ok)


# -- anthropic -----------------------------------------------------------------------------------


def _anthropic_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "scripted",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
    )


def _anthropic(base_url: str, **settings: Any) -> Any:  # noqa: ANN401 -- the factory's chat model
    return create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-api03-test", base_url=base_url, **settings)


class TestAnthropic:
    def test_a_429_reaches_the_caller_after_one_request(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=True).invoke("hi")
        assert wire.requests == 1

    async def test_a_429_reaches_an_async_caller_after_one_request(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                await _anthropic(base_url, unretried_key_errors=True).ainvoke("hi")
        assert wire.requests == 1

    def test_a_429_the_server_says_to_retry_is_still_not_retried(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, {**_STAINLESS_SHORT_WAIT, "x-should-retry": "true"})
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=True).invoke("hi")
        assert wire.requests == 1

    def test_the_429_keeps_what_the_classifier_reads(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, {})
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError) as raised:
                _anthropic(base_url, unretried_key_errors=True).invoke("hi")
        assert raised.value.response.headers["retry-after"] == "17"
        assert raised.value.type == "rate_limit_error"

    def test_a_5xx_is_still_retried_the_default_number_of_times(self) -> None:
        wire = _always(fx.ANTHROPIC_API_ERROR_500, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.InternalServerError):
                _anthropic(base_url, unretried_key_errors=True).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_5xx_then_an_answer_still_answers(self) -> None:
        wire = _fails_then_answers(fx.ANTHROPIC_OVERLOADED_529, _STAINLESS_SHORT_WAIT, failures=1, ok=_anthropic_ok())
        with serve_http_handler(wire.handle) as base_url:
            reply = _anthropic(base_url, unretried_key_errors=True).invoke("hi")
        assert reply.content == "ok"
        assert wire.requests == 2

    def test_by_default_a_429_is_retried_as_before(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_the_mode_is_refused_on_a_subscription_token(self) -> None:
        with pytest.raises(ValueError, match="subscription"):
            create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-test", unretried_key_errors=True)

    def test_a_model_in_the_mode_still_answers(self) -> None:
        wire = _CountingWire(lambda attempt: _anthropic_ok())
        with serve_http_handler(wire.handle) as base_url:
            assert anthropic_model(base_url, unretried_key_errors=True).invoke("hi").content == "ok"


# -- openai --------------------------------------------------------------------------------------


def _openai(wire: _CountingWire, **settings: Any) -> Any:  # noqa: ANN401 -- the factory's chat model
    transport = httpx.MockTransport(wire.handle)
    return create_openai_chat(
        "gpt-4o",
        "sk-test",
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **settings,
    )


class TestOpenAI:
    def test_a_429_reaches_the_caller_after_one_request(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=True).invoke("hi")
        assert wire.requests == 1

    async def test_a_429_reaches_an_async_caller_after_one_request(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            await _openai(wire, unretried_key_errors=True).ainvoke("hi")
        assert wire.requests == 1

    def test_on_the_factorys_own_http_client_too(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            model = create_openai_chat("gpt-4o", "sk-test", base_url=base_url, unretried_key_errors=True)
            with pytest.raises(openai.RateLimitError):
                model.invoke("hi")
        assert wire.requests == 1

    def test_a_5xx_is_still_retried_the_default_number_of_times(self) -> None:
        wire = _always(fx.OPENAI_SERVER_500, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.InternalServerError):
            _openai(wire, unretried_key_errors=True).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_refused_connection_is_still_retried_the_default_number_of_times(self) -> None:
        attempts: list[httpx.Request] = []

        def refuse(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            raise httpx.ConnectError("connection refused", request=request)

        transport = httpx.MockTransport(refuse)
        model = create_openai_chat(
            "gpt-4o",
            "sk-test",
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            unretried_key_errors=True,
        )
        with pytest.raises(openai.APIConnectionError):
            model.invoke("hi")
        assert len(attempts) == _DEFAULT_ATTEMPTS

    def test_by_default_a_429_is_retried_as_before(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            _openai(wire).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS


# -- openrouter ----------------------------------------------------------------------------------


def _openrouter(base_url: str, **settings: Any) -> Any:  # noqa: ANN401 -- the factory's chat model
    # through the public base_url, so the model's SDK client is the one LangChain builds from max_retries
    return create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-or-test", base_url=base_url, **settings)


def _openrouter_ok() -> httpx.Response:
    # the chat-completions answer the provider-wrapper suites already serve OpenRouter models
    request = httpx.Request("POST", "http://openrouter.test/chat/completions", json={"model": "scripted"})
    return ChatCompletionsWire().handle(request)


class TestOpenRouter:
    """the openrouter SDK retries only 5xx statuses and connection failures, so a 429 reaches the
    caller at once in either mode; the mode is accepted so a pool asks every factory alike."""

    @pytest.mark.parametrize("unretried", [True, False], ids=["unretried", "default"])
    def test_a_429_reaches_the_caller_after_one_request(self, unretried: bool) -> None:
        wire = _always(fx.OPENROUTER_RATE_LIMIT_429, _OPENROUTER_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(openrouter.errors.TooManyRequestsResponseError):
                _openrouter(base_url, unretried_key_errors=unretried).invoke("hi")
        assert wire.requests == 1

    async def test_a_429_reaches_an_async_caller_after_one_request(self) -> None:
        wire = _always(fx.OPENROUTER_RATE_LIMIT_429, _OPENROUTER_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(openrouter.errors.TooManyRequestsResponseError):
                await _openrouter(base_url, unretried_key_errors=True).ainvoke("hi")
        assert wire.requests == 1

    def test_a_5xx_is_still_retried(self) -> None:
        wire = _fails_then_answers(fx.OPENROUTER_UPSTREAM_502, _OPENROUTER_SHORT_WAIT, failures=2, ok=_openrouter_ok())
        with serve_http_handler(wire.handle) as base_url:
            reply = _openrouter(base_url, unretried_key_errors=True).invoke("hi")
        assert reply.content == "ok"
        assert wire.requests == _DEFAULT_ATTEMPTS
