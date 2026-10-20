"""a HALF_OPEN probe always ends, so a breaker can never refuse every request forever.

metallm dev, 2026-10-20, 3tears 0.72.0: a model's rate-limit breaker went HALF_OPEN, its
probe got OpenRouter's 404 (not a rate limit), and only the provider breaker heard about it.
The rate-limit breaker kept its probe "in flight" and refused every later call with
"retry in 0s" until a restart. A cancelled call strands a probe the same way, and so does a
request the rate-limit breaker refuses after the provider breaker admitted it as its probe.
"""

from __future__ import annotations

import pytest

from threetears.models.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError, CircuitState


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limited() -> Exception:
    return type("TooManyRequests", (Exception,), {"status_code": 429})()


def _not_found() -> Exception:
    return type("NotFound", (Exception,), {"status_code": 404})()


def _registry(clock: _Clock) -> CircuitBreakerRegistry:
    return CircuitBreakerRegistry(failure_threshold=2, recovery_timeout_seconds=30.0, clock=clock)


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
    def test_a_probe_that_never_reports_back_is_replaced_after_the_recovery_timeout(self) -> None:
        """A cancelled call fires neither on_llm_end nor on_llm_error."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])  # the probe, then cancelled: nothing reports
        with pytest.raises(CircuitOpenError) as waiting:
            callback.on_chat_model_start({}, [[]])
        assert waiting.value.remaining_seconds > 0  # an honest wait, not "retry in 0s"
        clock.now += 31.0
        callback.on_chat_model_start({}, [[]])  # the lost probe is replaced
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])  # and the new one is still the only one

    def test_a_probe_still_within_the_timeout_keeps_the_others_out(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        _open_the_rate_limit(callback, clock)
        callback.on_chat_model_start({}, [[]])
        clock.now += 29.0
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])


class TestARefusedRequestHandsBackTheProvidersProbe:
    def test_the_provider_probe_is_released_when_the_rate_limit_refuses_the_request(self) -> None:
        """The provider breaker admits a request as its probe, then the model's rate-limit
        breaker refuses it: the request never goes out, so the provider's probe is handed
        back and the provider's next request (any model) becomes its probe."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(_limited())  # the model's rate limit opens
        clock.now += 1.0
        for _ in range(2):
            callback.on_llm_error(type("ServerError", (Exception,), {"status_code": 502})())  # provider opens
        clock.now += 30.5  # both breakers are due a probe
        callback.on_chat_model_start({}, [[]])  # both admit this request as their probe
        callback.on_llm_error(_limited())  # still limited: model re-opens; the provider got an answer
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [[]])  # the model's own breaker refuses


class TestARateLimitAnswerEndsTheProvidersProbe:
    def test_a_429_on_a_provider_probe_closes_the_provider(self) -> None:
        """A rate-limited answer is still an answer: the provider is up. Without this, a
        provider probe answered with a 429 would hold the provider HALF_OPEN for every model."""
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(type("ServerError", (Exception,), {"status_code": 502})())
        clock.now += 31.0
        callback.on_chat_model_start({}, [[]])  # the provider's probe
        callback.on_llm_error(_limited())
        assert registry.status() == {"openrouter": CircuitState.CLOSED}
        sibling = registry.get("openrouter", credential="k").make_callback(
            rate_limit=registry.get("openrouter", credential="k", model="other")
        )
        sibling.on_chat_model_start({}, [[]])  # another model runs


class TestARateLimitRefusalHandsBackTheProvidersProbe:
    def test_a_request_the_model_breaker_refuses_does_not_hold_the_providers_probe(self) -> None:
        clock = _Clock()
        registry = _registry(clock)
        callback = _callback(registry)
        for _ in range(2):
            callback.on_llm_error(type("ServerError", (Exception,), {"status_code": 502})())
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
