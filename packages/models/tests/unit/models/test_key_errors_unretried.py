"""The provider factories' ``unretried_key_errors`` mode: a response the key pool's rule says is the key's
reaches the caller on its first occurrence, while a rate limit, a 5xx or a dropped connection is still
retried as often as before.

A rate limit means retry; a used-up quota means move to the next key (owner, 2026-11-26). Both are 429s,
told apart by the body, so the hook reads a 429's body and no other response's. Every test here drives
the real SDK over a scripted wire and counts the requests it sent; no network. The rule is the one a
pooled model's SDK gets: its key pool's ``refuses_retry``. Each retried answer carries
``retry-after-ms: 1`` (``retry-after`` for the openrouter SDK, which reads no milliseconds) so the SDK's
sleep between retries stays out of the run.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal
from typing import Any

import anthropic
import httpx
import openai
import openrouter.errors
import pytest

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.key_pool import KeyPool, KeySpec, Secret
from threetears.models.providers.anthropic import create_anthropic_chat
from threetears.models.providers.openai import create_openai_chat
from threetears.models.providers.openrouter import create_openrouter_chat

from . import key_error_fixtures as fx
from .pooled_chat_support import reply
from .provider_wire import ChatCompletionsWire, anthropic_model, serve_http_handler

#: The SDKs' default retries of a failed request: two, so three requests in all.
_DEFAULT_ATTEMPTS = 3

#: A wait the SDK honours instead of its own backoff, short enough to keep out of the run.
_STAINLESS_SHORT_WAIT = {"retry-after-ms": "1"}
_OPENROUTER_SHORT_WAIT = {"retry-after": "0.001"}

#: A 429 whose body is not the provider's JSON: a proxy's page, say.
_HTML_429 = httpx.Response(429, content=b"<html>Too Many Requests</html>", headers=_STAINLESS_SHORT_WAIT)


def _rule(provider: str, *, exhaustion_wait: timedelta = timedelta(seconds=120)) -> Callable[..., bool]:
    """the rule a pooled model hands its SDK hook: a one-key pool's ``refuses_retry``."""
    keys = [KeySpec(Secret(f"sk-{provider}-hook-test-0123456789"), "only", Decimal("1"))]
    return KeyPool(provider, "hook-test", keys, rate_limit_exhaustion_wait=exhaustion_wait).refuses_retry


class _RecordingRule:
    """a rule that records the status and body of every response it was asked about, and defers to another."""

    def __init__(self, inner: Callable[..., bool]) -> None:
        self._inner = inner
        self.asked: list[tuple[int, bytes | None]] = []

    def __call__(self, status: int, headers: Mapping[str, str], body: bytes | None) -> bool:
        self.asked.append((status, body))
        return self._inner(status, headers, body)


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


def _html_429() -> _CountingWire:
    return _CountingWire(lambda attempt: httpx.Response(429, content=_HTML_429.content, headers=_STAINLESS_SHORT_WAIT))


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


_ANTHROPIC_QUOTAS = [
    pytest.param(fx.ANTHROPIC_SPEND_CAP_429, id="enforced_spend_limit_reached"),
    pytest.param(fx.ANTHROPIC_CONTRACT_SPEND_LIMIT_429, id="contract_spend_limit_reached"),
]


class TestAnthropic:
    def test_a_rate_limit_is_retried_by_the_sdk_as_often_as_before(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_ramp_limit_is_a_rate_and_retried(self) -> None:
        wire = _always(fx.ANTHROPIC_RAMP_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_rate_limit_then_an_answer_answers(self) -> None:
        wire = _fails_then_answers(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT, failures=1, ok=_anthropic_ok())
        with serve_http_handler(wire.handle) as base_url:
            reply = _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert reply.content == "ok"
        assert wire.requests == 2

    def test_a_rate_limit_is_retried_with_the_callers_max_retries(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, max_retries=4, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == 5

    @pytest.mark.parametrize("recorded", _ANTHROPIC_QUOTAS)
    def test_a_quota_429_reaches_the_caller_after_one_request(self, recorded: fx.RecordedError) -> None:
        wire = _always(recorded, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == 1

    @pytest.mark.parametrize("recorded", _ANTHROPIC_QUOTAS)
    async def test_a_quota_429_reaches_an_async_caller_after_one_request(self, recorded: fx.RecordedError) -> None:
        wire = _always(recorded, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                await _anthropic(base_url, unretried_key_errors=_rule("anthropic")).ainvoke("hi")
        assert wire.requests == 1

    async def test_a_rate_limit_is_retried_for_an_async_caller(self) -> None:
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                await _anthropic(base_url, unretried_key_errors=_rule("anthropic")).ainvoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_quota_429_the_server_says_to_retry_is_still_not_retried(self) -> None:
        wire = _always(fx.ANTHROPIC_SPEND_CAP_429, {**_STAINLESS_SHORT_WAIT, "x-should-retry": "true"})
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == 1

    def test_a_rate_limit_asking_a_wait_past_the_threshold_is_not_slept_on(self) -> None:
        # unmarked, the SDK would sleep the 2 s twice and send three requests
        wire = _always(fx.ANTHROPIC_RATE_LIMIT_429, {"retry-after": "2"})
        rule = _rule("anthropic", exhaustion_wait=timedelta(seconds=1))
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=rule).invoke("hi")
        assert wire.requests == 1

    def test_a_429_whose_body_is_not_json_stays_retriable(self) -> None:
        wire = _html_429()
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_the_quota_429_keeps_what_the_classifier_reads(self) -> None:
        wire = _always(fx.ANTHROPIC_SPEND_CAP_429, {"retry-after": "17"})
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError) as raised:
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert raised.value.response.headers["retry-after"] == "17"
        assert raised.value.body == fx.ANTHROPIC_SPEND_CAP_429.body  # the body the hook read, kept
        assert raised.value.type == "rate_limit_error"

    @pytest.mark.parametrize("stream", [False, True], ids=["invoke", "stream"])
    def test_the_body_is_read_only_for_a_429(self, stream: bool) -> None:
        answers = [fx.ANTHROPIC_API_ERROR_500.response(extra_headers=_STAINLESS_SHORT_WAIT), None]
        rule = _RecordingRule(_rule("anthropic"))

        def answer(request: httpx.Request) -> httpx.Response:
            # an answer as JSON or as a stream, as the request asked
            return answers.pop(0) or reply()(json.loads(request.content))

        with serve_http_handler(answer) as base_url:
            model = _anthropic(base_url, unretried_key_errors=rule)
            if stream:
                assert "".join(str(chunk.content) for chunk in model.stream("hi")) == "ok"
            else:
                assert model.invoke("hi").content == "ok"
        assert rule.asked == [(500, None), (200, None)]
        wire = _always(fx.ANTHROPIC_SPEND_CAP_429, {})
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url, unretried_key_errors=rule).invoke("hi")
        assert rule.asked[-1][0] == 429
        assert rule.asked[-1][1] is not None and b"enforced_spend_limit_reached" in rule.asked[-1][1]

    def test_a_5xx_is_still_retried_the_default_number_of_times(self) -> None:
        wire = _always(fx.ANTHROPIC_API_ERROR_500, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.InternalServerError):
                _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_a_5xx_then_an_answer_still_answers(self) -> None:
        wire = _fails_then_answers(fx.ANTHROPIC_OVERLOADED_529, _STAINLESS_SHORT_WAIT, failures=1, ok=_anthropic_ok())
        with serve_http_handler(wire.handle) as base_url:
            reply = _anthropic(base_url, unretried_key_errors=_rule("anthropic")).invoke("hi")
        assert reply.content == "ok"
        assert wire.requests == 2

    def test_by_default_a_quota_429_is_retried_as_before(self) -> None:
        wire = _always(fx.ANTHROPIC_SPEND_CAP_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(anthropic.RateLimitError):
                _anthropic(base_url).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_the_mode_is_refused_on_a_subscription_token(self) -> None:
        with pytest.raises(ValueError, match="subscription"):
            create_anthropic_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-test", unretried_key_errors=_rule("anthropic"))

    def test_a_model_in_the_mode_still_answers(self) -> None:
        wire = _CountingWire(lambda attempt: _anthropic_ok())
        with serve_http_handler(wire.handle) as base_url:
            model = anthropic_model(base_url, unretried_key_errors=_rule("anthropic"))
            assert model.invoke("hi").content == "ok"


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


_OPENAI_QUOTAS = [
    pytest.param(fx.OPENAI_INSUFFICIENT_QUOTA_429, id="insufficient_quota"),
    pytest.param(fx.OPENAI_CREDIT_BALANCE_EXHAUSTED_429, id="credit_balance_exhausted"),
    pytest.param(fx.OPENAI_ORGANIZATION_SPEND_LIMIT_429, id="organization_spend_limit_exceeded"),
    pytest.param(fx.OPENAI_PROJECT_SPEND_LIMIT_429, id="project_spend_limit_exceeded"),
    pytest.param(fx.OPENAI_ORGANIZATION_USAGE_LIMIT_429, id="organization_usage_limit_exceeded"),
    pytest.param(fx.OPENAI_BILLING_HARD_LIMIT_429, id="billing_hard_limit_reached"),
]

_OPENAI_RATES = [
    pytest.param(fx.OPENAI_RATE_LIMIT_429, id="rate_limit_exceeded"),
    pytest.param(fx.OPENAI_SLOW_DOWN_429, id="slow_down"),
]


class TestOpenAI:
    @pytest.mark.parametrize("recorded", _OPENAI_RATES)
    def test_a_rate_limit_is_retried_by_the_sdk_as_often_as_before(self, recorded: fx.RecordedError) -> None:
        wire = _always(recorded, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=_rule("openai")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    async def test_a_rate_limit_is_retried_for_an_async_caller(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            await _openai(wire, unretried_key_errors=_rule("openai")).ainvoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    @pytest.mark.parametrize("recorded", _OPENAI_QUOTAS)
    def test_a_quota_429_reaches_the_caller_after_one_request(self, recorded: fx.RecordedError) -> None:
        wire = _always(recorded, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=_rule("openai")).invoke("hi")
        assert wire.requests == 1

    @pytest.mark.parametrize("recorded", _OPENAI_QUOTAS)
    async def test_a_quota_429_reaches_an_async_caller_after_one_request(self, recorded: fx.RecordedError) -> None:
        wire = _always(recorded, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.RateLimitError):
            await _openai(wire, unretried_key_errors=_rule("openai")).ainvoke("hi")
        assert wire.requests == 1

    def test_on_the_factorys_own_http_client_too(self) -> None:
        wire = _always(fx.OPENAI_INSUFFICIENT_QUOTA_429, _STAINLESS_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            model = create_openai_chat("gpt-4o", "sk-test", base_url=base_url, unretried_key_errors=_rule("openai"))
            with pytest.raises(openai.RateLimitError):
                model.invoke("hi")
        assert wire.requests == 1

    def test_a_rate_limit_asking_a_wait_past_the_threshold_is_not_slept_on(self) -> None:
        wire = _always(fx.OPENAI_RATE_LIMIT_429, {"retry-after": "2"})
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=_rule("openai", exhaustion_wait=timedelta(seconds=1))).invoke("hi")
        assert wire.requests == 1

    def test_a_429_whose_body_is_not_json_stays_retriable(self) -> None:
        wire = _html_429()
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=_rule("openai")).invoke("hi")
        assert wire.requests == _DEFAULT_ATTEMPTS

    def test_the_quota_429_keeps_what_the_classifier_reads(self) -> None:
        wire = _always(fx.OPENAI_BILLING_HARD_LIMIT_429, {})
        with pytest.raises(openai.RateLimitError) as raised:
            _openai(wire, unretried_key_errors=_rule("openai")).invoke("hi")
        assert (raised.value.code, raised.value.type) == ("billing_hard_limit_reached", "insufficient_quota")

    @pytest.mark.parametrize("stream", [False, True], ids=["invoke", "astream"])
    async def test_the_body_is_read_only_for_a_429(self, stream: bool) -> None:
        rule = _RecordingRule(_rule("openai"))
        transport = httpx.MockTransport(ChatCompletionsWire().handle)
        model = create_openai_chat(
            "gpt-4o",
            "sk-test",
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            unretried_key_errors=rule,
        )
        if stream:
            assert "".join([str(chunk.content) async for chunk in model.astream("hi")]) == "ok"
        else:
            assert (await model.ainvoke("hi")).content == "ok"
        assert rule.asked == [(200, None)]
        wire = _always(fx.OPENAI_INSUFFICIENT_QUOTA_429, {})
        with pytest.raises(openai.RateLimitError):
            _openai(wire, unretried_key_errors=rule).invoke("hi")
        assert rule.asked[-1][0] == 429
        assert rule.asked[-1][1] is not None and b"insufficient_quota" in rule.asked[-1][1]

    def test_a_shared_client_carries_one_rules_hook_once(self) -> None:
        rule = _rule("openai")
        wire = _always(fx.OPENAI_INSUFFICIENT_QUOTA_429, _STAINLESS_SHORT_WAIT)
        client = httpx.Client(transport=httpx.MockTransport(wire.handle))
        for _ in range(2):
            create_openai_chat("gpt-4o", "sk-test", http_client=client, unretried_key_errors=rule)
        assert len(client.event_hooks["response"]) == 1

    def test_a_5xx_is_still_retried_the_default_number_of_times(self) -> None:
        wire = _always(fx.OPENAI_SERVER_500, _STAINLESS_SHORT_WAIT)
        with pytest.raises(openai.InternalServerError):
            _openai(wire, unretried_key_errors=_rule("openai")).invoke("hi")
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
            unretried_key_errors=_rule("openai"),
        )
        with pytest.raises(openai.APIConnectionError):
            model.invoke("hi")
        assert len(attempts) == _DEFAULT_ATTEMPTS

    def test_by_default_a_quota_429_is_retried_as_before(self) -> None:
        wire = _always(fx.OPENAI_INSUFFICIENT_QUOTA_429, _STAINLESS_SHORT_WAIT)
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
    caller at once in either mode (the key pool retries it instead); the mode is accepted so a pool asks
    every factory alike."""

    @pytest.mark.parametrize("unretried", [True, False], ids=["unretried", "default"])
    def test_a_429_reaches_the_caller_after_one_request(self, unretried: bool) -> None:
        wire = _always(fx.OPENROUTER_RATE_LIMIT_429, _OPENROUTER_SHORT_WAIT)
        rule = _rule("openrouter") if unretried else None
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(openrouter.errors.TooManyRequestsResponseError):
                _openrouter(base_url, unretried_key_errors=rule).invoke("hi")
        assert wire.requests == 1

    async def test_a_429_reaches_an_async_caller_after_one_request(self) -> None:
        wire = _always(fx.OPENROUTER_RATE_LIMIT_429, _OPENROUTER_SHORT_WAIT)
        with serve_http_handler(wire.handle) as base_url:
            with pytest.raises(openrouter.errors.TooManyRequestsResponseError):
                await _openrouter(base_url, unretried_key_errors=_rule("openrouter")).ainvoke("hi")
        assert wire.requests == 1

    def test_a_5xx_is_still_retried(self) -> None:
        wire = _fails_then_answers(fx.OPENROUTER_UPSTREAM_502, _OPENROUTER_SHORT_WAIT, failures=2, ok=_openrouter_ok())
        with serve_http_handler(wire.handle) as base_url:
            reply = _openrouter(base_url, unretried_key_errors=_rule("openrouter")).invoke("hi")
        assert reply.content == "ok"
        assert wire.requests == _DEFAULT_ATTEMPTS
