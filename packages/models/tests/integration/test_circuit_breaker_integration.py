"""integration tests for circuit breaker full state machine cycle."""

from __future__ import annotations

import time

import pytest

from threetears.models.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreakerIntegration:
    """integration tests for circuit breaker lifecycle through registry."""

    def test_circuit_breaker_full_lifecycle(self) -> None:
        """CLOSED, OPEN, HALF_OPEN, CLOSED lifecycle via registry."""
        registry = CircuitBreakerRegistry(failure_threshold=3, recovery_timeout_seconds=0.05)
        breaker = registry.get("anthropic")

        # normal operations
        breaker.check()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

        # failures trip the breaker
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # verify fast-fail
        with pytest.raises(CircuitOpenError):
            breaker.check()

        # wait for recovery
        time.sleep(0.06)
        breaker.check()
        assert breaker.state == CircuitState.HALF_OPEN

        # probe success
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_probe_failure_reopens(self) -> None:
        """HALF_OPEN to OPEN on probe failure."""
        registry = CircuitBreakerRegistry(failure_threshold=2, recovery_timeout_seconds=0.05)
        breaker = registry.get("openai")

        # trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # wait for recovery window
        time.sleep(0.06)
        breaker.check()
        assert breaker.state == CircuitState.HALF_OPEN

        # probe failure reopens
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_registry_reset_recovers_provider(self) -> None:
        """manual reset brings provider back online."""
        registry = CircuitBreakerRegistry(failure_threshold=1)
        breaker = registry.get("anthropic")

        # trip the breaker
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            breaker.check()

        # manual reset
        registry.reset("anthropic")
        assert breaker.state == CircuitState.CLOSED

        # verify request flows again
        breaker.check()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_multiple_providers_independent_state(self) -> None:
        """breakers for different providers operate independently."""
        registry = CircuitBreakerRegistry(failure_threshold=2)
        anthropic_breaker = registry.get("anthropic")
        openai_breaker = registry.get("openai")

        # trip anthropic breaker
        anthropic_breaker.record_failure()
        anthropic_breaker.record_failure()
        assert anthropic_breaker.state == CircuitState.OPEN

        # openai breaker unaffected
        assert openai_breaker.state == CircuitState.CLOSED
        openai_breaker.check()
        openai_breaker.record_success()
        assert openai_breaker.state == CircuitState.CLOSED

    def test_status_reflects_all_provider_states(self) -> None:
        """registry status snapshot reflects current state of all breakers."""
        registry = CircuitBreakerRegistry(failure_threshold=1, recovery_timeout_seconds=0.05)

        # create breakers in different states
        registry.get("healthy")
        registry.get("healthy").record_success()

        registry.get("failing")
        registry.get("failing").record_failure()

        status = registry.status()
        assert status["healthy"] == CircuitState.CLOSED
        assert status["failing"] == CircuitState.OPEN
