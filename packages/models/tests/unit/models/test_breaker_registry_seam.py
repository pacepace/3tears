"""The factory takes a breaker registry, and a breaker can be read without being moved.

The model key pool builds one model per key and reads each key's breaker state; ``check()`` admits a
probe, so it cannot be the eligibility test.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from threetears.models.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerCallback,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from threetears.models.factory import create_chat_model


def _limited() -> Exception:
    return type("TooManyRequests", (Exception,), {"status_code": 429})()


def _provider_down() -> Exception:
    return type("ServerError", (Exception,), {"status_code": 502})()


class _Clock:
    """a hand-wound monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _breaker_callback(registry: CircuitBreakerRegistry, key: str, model_id: str) -> CircuitBreakerCallback:
    model = create_chat_model(model_id=model_id, api_key=key, provider="openrouter", breaker_registry=registry)
    return next(c for c in model.config["callbacks"] if isinstance(c, CircuitBreakerCallback))


class TestTheFactoryTakesARegistry:
    def test_a_model_records_into_the_given_registrys_credential_and_model_breakers(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        key = f"k-{uuid4()}"
        callback = _breaker_callback(registry, key, "deepseek/deepseek-v4.1-flash")
        for _ in range(2):
            callback.on_llm_error(_limited())
        # the rate limit counted against the model's breaker, not the credential's
        model_breaker = registry.peek("openrouter", credential=key, model="deepseek/deepseek-v4.1-flash")
        credential_breaker = registry.peek("openrouter", credential=key)
        assert model_breaker is not None
        assert credential_breaker is not None
        assert model_breaker.state is CircuitState.OPEN
        assert credential_breaker.state is CircuitState.CLOSED

    def test_a_provider_failure_records_into_the_given_registrys_credential_breaker(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2)
        key = f"k-{uuid4()}"
        callback = _breaker_callback(registry, key, "anthropic/claude")
        for _ in range(2):
            callback.on_llm_error(_provider_down())
        credential_breaker = registry.peek("openrouter", credential=key)
        assert credential_breaker is not None
        assert credential_breaker.state is CircuitState.OPEN

    def test_two_registries_keep_two_breakers_for_one_key(self) -> None:
        one, other = CircuitBreakerRegistry(failure_threshold=1), CircuitBreakerRegistry(failure_threshold=1)
        key = f"k-{uuid4()}"
        _breaker_callback(one, key, "anthropic/claude").on_llm_error(_provider_down())
        _breaker_callback(other, key, "anthropic/claude")
        opened, untouched = one.peek("openrouter", credential=key), other.peek("openrouter", credential=key)
        assert opened is not None
        assert untouched is not None
        assert opened.state is CircuitState.OPEN
        assert untouched.state is CircuitState.CLOSED

    def test_both_breaker_and_registry_is_refused(self) -> None:
        registry = CircuitBreakerRegistry()
        with pytest.raises(ValueError, match="not both"):
            create_chat_model(
                model_id="anthropic/claude",
                api_key="k",
                provider="openrouter",
                breaker=registry.get("openrouter"),
                breaker_registry=registry,
            )


class TestAReadThatMovesNothing:
    @staticmethod
    def _opened(clock: _Clock) -> CircuitBreaker:
        registry = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout_seconds=30.0, clock=clock)
        breaker = registry.get("openrouter", credential="k")
        breaker.record_failure()
        return breaker

    def test_an_open_breaker_inside_its_window_reports_the_wait_and_stays_open(self) -> None:
        clock = _Clock()
        breaker = self._opened(clock)
        clock.now += 10.0
        assert breaker.seconds_until_admit() == pytest.approx(20.0)
        assert breaker.state is CircuitState.OPEN

    def test_an_open_breaker_past_its_window_reports_zero_and_stays_open(self) -> None:
        clock = _Clock()
        breaker = self._opened(clock)
        clock.now += 31.0
        assert breaker.seconds_until_admit() == 0.0
        assert breaker.state is CircuitState.OPEN  # check() would have made it HALF_OPEN and admitted a probe
        breaker.check()  # the probe is still there to be admitted
        assert breaker.state is CircuitState.HALF_OPEN

    def test_a_closed_breaker_reports_zero(self) -> None:
        registry = CircuitBreakerRegistry()
        assert registry.get("openrouter").seconds_until_admit() == 0.0

    def test_a_half_open_breaker_with_a_probe_out_reports_the_probe_wait_and_keeps_it(self) -> None:
        clock = _Clock()
        breaker = self._opened(clock)
        clock.now += 31.0
        breaker.check()
        assert breaker.seconds_until_admit() > 0.0
        assert breaker.state is CircuitState.HALF_OPEN
        with pytest.raises(CircuitOpenError):
            breaker.check()  # the read neither freed nor took the probe slot

    def test_reading_does_not_count_as_activity(self) -> None:
        clock = _Clock()
        breaker = self._opened(clock)
        before = breaker.last_activity
        clock.now += 5.0
        breaker.seconds_until_admit()
        assert breaker.last_activity == before


class TestPeek:
    def test_peek_never_creates(self) -> None:
        registry = CircuitBreakerRegistry()
        assert registry.peek("openrouter", credential="k", model="m") is None
        assert registry.status() == {}
        assert registry.model_status() == {}

    def test_peek_returns_what_get_made(self) -> None:
        registry = CircuitBreakerRegistry()
        made = registry.get("openrouter", credential="k", model="m")
        assert registry.peek("openrouter", credential="k", model="m") is made
        assert registry.peek("openrouter", credential="other", model="m") is None
