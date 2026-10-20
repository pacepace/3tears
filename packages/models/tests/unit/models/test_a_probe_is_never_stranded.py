"""a HALF_OPEN probe always ends, so a breaker can never refuse every request forever.

metallm dev, 2026-10-20, 3tears 0.72.0: a model's rate-limit breaker went HALF_OPEN, its
probe got OpenRouter's 404 (not a rate limit), and only the provider breaker heard about it.
The rate-limit breaker kept its probe "in flight" and refused every later call with
"retry in 0s" until a restart. A request the rate-limit breaker refuses after the provider
breaker admitted it as its probe strands one too, and a cancelled call must neither strand a
probe nor be counted as the provider's verdict.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk

from threetears.models.circuit_breaker import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limited() -> Exception:
    return type("TooManyRequests", (Exception,), {"status_code": 429})()


def _not_found() -> Exception:
    return type("NotFound", (Exception,), {"status_code": 404})()


def _server_error() -> Exception:
    return type("ServerError", (Exception,), {"status_code": 502})()


def _registry(clock: _Clock, *, probe_timeout_seconds: float = 120.0) -> CircuitBreakerRegistry:
    return CircuitBreakerRegistry(
        failure_threshold=2, recovery_timeout_seconds=30.0, probe_timeout_seconds=probe_timeout_seconds, clock=clock
    )


def _open_the_provider(callback, clock: _Clock) -> None:  # noqa: ANN001 - a callback handler
    for _ in range(2):
        callback.on_llm_error(_server_error())
    clock.now += 31.0


def _callback(registry: CircuitBreakerRegistry):  # noqa: ANN202 - a callback handler
    provider = registry.get("openrouter", credential="k")
    return provider.make_callback(rate_limit=registry.get("openrouter", credential="k", model="m"))


def _open_the_rate_limit(callback, clock: _Clock) -> None:  # noqa: ANN001 - a callback handler
    for _ in range(2):
        callback.on_llm_error(_limited())
    clock.now += 31.0


class TestAProbeAnsweredWithAnotherErrorEnds:
    def test_a_404_on_a_rate_limit_probe_ends_the_probe(self) -> None:
        """The reported case: the probe is answered, just not with a rate limit, so the
        route is not limited and the next request runs."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the probe
        callback.on_llm_error(_not_found())
        callback.on_chat_model_start({}, [[]])  # admitted, not "retry in 0s"
        assert registry.model_status() == {}  # only a breaker that is not closed is listed

    def test_a_non_rate_limit_error_mid_streak_keeps_the_streak(self) -> None:
        """Only a probe is ended by another error: a stray 404 between two 429s is not a
        success, and the second 429 still opens the breaker."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        callback.on_llm_error(_limited())
        callback.on_llm_error(_not_found())
        callback.on_llm_error(_limited())
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])


class TestALostProbeIsReplaced:
    def test_a_probe_that_never_reports_back_is_replaced_after_the_probe_timeout(self) -> None:
        """A caller that vanishes reports nothing; the probe is replaced once it is older than
        the probe timeout -- its own setting, not the recovery wait."""
        clock = _Clock()
        registry = _registry(clock, probe_timeout_seconds=120.0)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the probe, and then nothing reports
        clock.now += 31.0  # past the recovery wait: a long call is not a lost one
        with pytest.raises(CircuitOpenError) as waiting:
            callback.on_chat_model_start({}, [[]])
        assert waiting.value.remaining_seconds == pytest.approx(89.0)  # an honest wait, not "retry in 0s"
        clock.now += 90.0
        callback.on_chat_model_start({}, [[]])  # the lost probe is replaced
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])  # and the new one is still the only one

    def test_a_probe_still_within_the_timeout_keeps_the_others_out(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])
        clock.now += 119.0
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])

    def test_the_default_probe_timeout_outlasts_a_model_calls_deadline(self) -> None:
        """OpenRouter's whole-call deadline defaults to 120 s; a probe must be able to outlive it."""
        clock = _Clock()
        breaker = CircuitBreaker("openrouter", failure_threshold=1, recovery_timeout_seconds=30.0, clock=clock)
        breaker.record_failure()
        clock.now += 31.0
        breaker.check()  # the probe
        clock.now += 150.0
        with pytest.raises(CircuitOpenError):
            breaker.check()
        assert DEFAULT_PROBE_TIMEOUT_SECONDS > 150.0

    def test_a_restored_breaker_takes_its_probe_timeout(self) -> None:
        clock = _Clock()
        breaker = CircuitBreaker.restore(
            "openrouter", state=CircuitState.OPEN, failure_count=5, probe_timeout_seconds=10.0, clock=clock
        )
        breaker.check()  # the window has elapsed: the probe
        clock.now += 11.0
        breaker.check()  # lost after 10 s, replaced

    def test_a_probe_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker("openrouter", probe_timeout_seconds=0)


class TestARateLimitAnswerEndsTheProvidersProbe:
    def test_a_429_on_a_provider_probe_closes_the_provider(self) -> None:
        """A rate-limited answer is still an answer: the provider is up. Without this, a
        provider probe answered with a 429 would hold the provider HALF_OPEN for every model."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(_server_error())
        clock.now += 31.0
        callback.on_chat_model_start({}, [[]])  # the provider's probe
        callback.on_llm_error(_limited())
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        sibling = registry.get("openrouter", credential="k").make_callback(
            rate_limit=registry.get("openrouter", credential="k", model="other")
        )
        sibling.on_chat_model_start({}, [[]])  # another model runs

    def test_a_probe_both_breakers_share_answered_with_a_429_closes_the_provider_and_reopens_the_model(
        self,
    ) -> None:
        """Both breakers are due and admit one request as their probe; the 429 is the
        provider's answer (it is up) and the model's verdict (still limited)."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(_limited())  # the model's rate limit opens
        clock.now += 1.0
        for _ in range(2):
            callback.on_llm_error(_server_error())  # provider opens
        clock.now += 30.5  # both breakers are due a probe
        callback.on_chat_model_start({}, [[]])  # both admit this request as their probe
        callback.on_llm_error(_limited())
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])  # the model's own breaker refuses


class TestARateLimitRefusalHandsBackTheProvidersProbe:
    def test_a_request_the_model_breaker_refuses_does_not_hold_the_providers_probe(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(_server_error())
        for _ in range(2):
            callback.on_llm_error(_limited())  # the model opens 0s later than the provider
        clock.now += 30.0  # the provider is due a probe
        registry.get("openrouter", credential="k", model="m").record_failure()  # the model's breaker: not due
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])  # provider admitted it as a probe, model refused it
        sibling = registry.get("openrouter", credential="k").make_callback(
            rate_limit=registry.get("openrouter", credential="k", model="other")
        )
        sibling.on_chat_model_start({}, [[]])  # the provider's probe was handed back: another model probes


class TestACancelledCallIsNoVerdict:
    """langchain's streaming paths report a cancellation to on_llm_error: it is the caller
    stopping, not the provider answering, so it counts against neither breaker."""

    def test_a_cancelled_provider_probe_is_handed_back_not_failed(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_provider(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the provider's probe
        callback.on_llm_error(asyncio.CancelledError())
        assert registry.status() == {"openrouter": CircuitState.HALF_OPEN}  # not re-opened
        callback.on_chat_model_start({}, [[]])  # the next request is the probe, at once

    def test_a_cancelled_rate_limit_probe_does_not_close_the_rate_limit(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the rate limit's probe
        callback.on_llm_error(asyncio.CancelledError())
        assert registry.model_status() == {"openrouter/m": CircuitState.HALF_OPEN}  # no verdict
        assert registry.get("openrouter", credential="k").failure_count == 0  # nor a provider failure
        callback.on_chat_model_start({}, [[]])  # handed back: the next request probes at once

    @pytest.mark.parametrize("cancel", [asyncio.CancelledError, KeyboardInterrupt, GeneratorExit])
    def test_cancellations_in_closed_never_open_a_breaker(self, cancel: type[BaseException]) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(10):
            callback.on_chat_model_start({}, [[]])
            callback.on_llm_error(cancel())
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        assert registry.get("openrouter", credential="k").failure_count == 0
        callback.on_chat_model_start({}, [[]])

    def test_a_timeout_with_no_answer_still_counts_against_the_provider_but_not_the_rate_limit(self) -> None:
        """A provider never heard from is the outage the breaker exists for; it says nothing
        about one model's rate limit, so that probe is handed back, not closed."""
        from threetears.models.errors import ModelCallTimeout

        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the rate limit's probe
        callback.on_llm_error(ModelCallTimeout("no chunk within 120 s"))
        assert registry.model_status() == {"openrouter/m": CircuitState.HALF_OPEN}
        assert registry.get("openrouter", credential="k").failure_count == 1

    async def test_a_real_streaming_model_cancelled_mid_stream_releases_the_probe(self) -> None:
        """Through langchain itself: a stream cancelled while the provider is still sending is
        reported to on_llm_error as the CancelledError, and hands the probe back."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_provider(callback, clock)
        started = asyncio.Event()

        async def consume() -> None:
            async for _chunk in _SlowStream(messages=iter([AIMessage(content="one two")])).astream(
                "hi", config={"callbacks": [callback]}
            ):
                started.set()

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.status() == {"openrouter": CircuitState.HALF_OPEN}
        assert registry.get("openrouter", credential="k").failure_count == 2  # the cancel counted nothing
        callback.on_chat_model_start({}, [[]])  # the probe was handed back

    async def test_a_stream_its_consumer_closes_early_releases_the_probe(self) -> None:
        """A consumer that stops reading and closes the stream: langchain reports the
        GeneratorExit to on_llm_error, which is no verdict either."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_provider(callback, clock)
        stream = GenericFakeChatModel(messages=iter([AIMessage(content="one two three")])).astream(
            "hi", config={"callbacks": [callback]}
        )
        async with aclosing(stream):
            async for _chunk in stream:
                break
        assert registry.status() == {"openrouter": CircuitState.HALF_OPEN}
        callback.on_chat_model_start({}, [[]])  # the probe was handed back


class _SlowStream(GenericFakeChatModel):
    """A fake chat model whose provider is still sending after its first chunk."""

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        first = True
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
            if first:
                first = False
                await asyncio.sleep(10)
