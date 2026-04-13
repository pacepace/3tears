"""tests for circuit breaker fault isolation."""

from __future__ import annotations

import threading
from enum import StrEnum
from unittest.mock import patch

import pytest

from threetears.models.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitState:
    """tests for CircuitState enum."""

    def test_circuit_state_is_str_enum(self) -> None:
        """CircuitState inherits from StrEnum."""
        assert issubclass(CircuitState, StrEnum)

    def test_circuit_state_values(self) -> None:
        """CircuitState contains CLOSED, OPEN, and HALF_OPEN with correct values."""
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"

    def test_circuit_state_member_count(self) -> None:
        """CircuitState has exactly three members."""
        assert len(CircuitState) == 3


class TestCircuitOpenError:
    """tests for CircuitOpenError exception."""

    def test_error_attributes(self) -> None:
        """CircuitOpenError stores provider_name and remaining_seconds."""
        err = CircuitOpenError("anthropic", 15.5)
        assert err.provider_name == "anthropic"
        assert err.remaining_seconds == 15.5

    def test_error_message(self) -> None:
        """CircuitOpenError message contains provider name and seconds."""
        err = CircuitOpenError("openai", 10.0)
        msg = str(err)
        assert "openai" in msg
        assert "10" in msg

    def test_is_exception(self) -> None:
        """CircuitOpenError is subclass of Exception."""
        assert issubclass(CircuitOpenError, Exception)


class TestCircuitBreaker:
    """tests for CircuitBreaker state machine."""

    def test_initial_state_is_closed(self) -> None:
        """new circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test-provider")
        assert cb.state == CircuitState.CLOSED

    def test_check_passes_when_closed(self) -> None:
        """check does not raise when circuit is CLOSED."""
        cb = CircuitBreaker("test-provider")
        cb.check()

    def test_check_raises_when_open(self) -> None:
        """check raises CircuitOpenError when circuit is OPEN."""
        cb = CircuitBreaker("test-provider", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.provider_name == "test-provider"

    def test_check_transitions_to_half_open_after_timeout(self) -> None:
        """check transitions OPEN to HALF_OPEN after recovery timeout elapses."""
        cb = CircuitBreaker(
            "test-provider", failure_threshold=2, recovery_timeout_seconds=10.0
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with patch("threetears.models.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0 + 11.0
            cb._last_failure_time = 1000.0
            cb.check()

        assert cb.state == CircuitState.HALF_OPEN

    def test_check_open_error_has_remaining_seconds(self) -> None:
        """CircuitOpenError includes remaining seconds until timeout."""
        cb = CircuitBreaker(
            "test-provider", failure_threshold=1, recovery_timeout_seconds=30.0
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.remaining_seconds > 0

    def test_success_resets_failure_count(self) -> None:
        """record_success resets failure count in CLOSED state."""
        cb = CircuitBreaker("test-provider", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # should not open after one more failure since count was reset
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_transitions_half_open_to_closed(self) -> None:
        """record_success transitions HALF_OPEN back to CLOSED."""
        cb = CircuitBreaker(
            "test-provider", failure_threshold=1, recovery_timeout_seconds=10.0
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with patch("threetears.models.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0 + 11.0
            cb._last_failure_time = 1000.0
            cb.check()

        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_single_failure_stays_closed(self) -> None:
        """single failure does not open circuit with default threshold."""
        cb = CircuitBreaker("test-provider", failure_threshold=5)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_threshold_failures_opens_circuit(self) -> None:
        """reaching failure threshold opens circuit."""
        cb = CircuitBreaker("test-provider", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_failure_in_half_open_reopens(self) -> None:
        """failure in HALF_OPEN state transitions immediately to OPEN."""
        cb = CircuitBreaker(
            "test-provider", failure_threshold=1, recovery_timeout_seconds=10.0
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with patch("threetears.models.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0 + 11.0
            cb._last_failure_time = 1000.0
            cb.check()

        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_full_cycle_closed_open_halfopen_closed(self) -> None:
        """circuit transitions through full lifecycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        cb = CircuitBreaker(
            "test-provider", failure_threshold=2, recovery_timeout_seconds=10.0
        )

        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with patch("threetears.models.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0 + 11.0
            cb._last_failure_time = 1000.0
            cb.check()

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reset_returns_to_closed(self) -> None:
        """reset forces circuit back to CLOSED regardless of current state."""
        cb = CircuitBreaker("test-provider", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_concurrent_failures(self) -> None:
        """multiple threads can record failures without corruption."""
        cb = CircuitBreaker("test-provider", failure_threshold=100)
        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()
            for _ in range(10):
                cb.record_failure()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb._failure_count == 100
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """tests for CircuitBreakerRegistry."""

    def test_get_creates_breaker(self) -> None:
        """first get for provider creates new breaker."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get("anthropic")
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.state == CircuitState.CLOSED

    def test_get_returns_same_breaker(self) -> None:
        """second get for same provider returns same instance."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get("anthropic")
        b2 = registry.get("anthropic")
        assert b1 is b2

    def test_different_providers(self) -> None:
        """different provider names get different breaker instances."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get("anthropic")
        b2 = registry.get("openai")
        assert b1 is not b2

    def test_reset_provider(self) -> None:
        """reset returns breaker for provider to CLOSED state."""
        registry = CircuitBreakerRegistry(failure_threshold=1)
        breaker = registry.get("anthropic")
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        registry.reset("anthropic")
        assert breaker.state == CircuitState.CLOSED

    def test_reset_unknown_provider(self) -> None:
        """reset for unknown provider does not raise."""
        registry = CircuitBreakerRegistry()
        registry.reset("nonexistent")

    def test_status_snapshot(self) -> None:
        """status returns dict of all provider states."""
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.get("anthropic")
        breaker_openai = registry.get("openai")
        breaker_openai.record_failure()

        status = registry.status()
        assert status == {
            "anthropic": CircuitState.CLOSED,
            "openai": CircuitState.OPEN,
        }
