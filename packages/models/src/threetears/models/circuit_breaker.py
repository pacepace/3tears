"""per-provider circuit breaker for fault isolation in model routing."""

from __future__ import annotations

import threading
import time
from enum import StrEnum

from threetears.observe import get_logger

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
]

logger = get_logger(__name__)


class CircuitState(StrEnum):
    """three-state lifecycle of circuit breaker.

    :cvar CLOSED: circuit is healthy, requests flow normally
    :cvar OPEN: circuit is tripped, requests are fast-failed
    :cvar HALF_OPEN: circuit is probing, single request allowed through
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """raised when circuit breaker is open and request should be rejected.

    :param provider_name: name of provider whose circuit is open
    :ptype provider_name: str
    :param remaining_seconds: seconds until recovery timeout expires
    :ptype remaining_seconds: float
    """

    def __init__(self, provider_name: str, remaining_seconds: float) -> None:
        self.provider_name = provider_name
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Circuit open for {provider_name}, retry in {remaining_seconds:.0f}s")


class CircuitBreaker:
    """per-provider circuit breaker with three-state fault isolation.

    tracks consecutive failures and transitions between CLOSED (normal),
    OPEN (fast-fail), and HALF_OPEN (probe) states to prevent cascading
    failures from unhealthy providers.

    :param provider_name: identifier for provider this breaker protects
    :ptype provider_name: str
    :param failure_threshold: consecutive failures before circuit opens
    :ptype failure_threshold: int
    :param recovery_timeout_seconds: seconds to wait in OPEN before probing
    :ptype recovery_timeout_seconds: float
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        self._provider_name = provider_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """returns current circuit state.

        :return: current circuit breaker state
        :rtype: CircuitState
        """
        with self._lock:
            return self._state

    def check(self) -> None:
        """verifies circuit allows request to proceed.

        transitions OPEN to HALF_OPEN when recovery timeout has elapsed.
        raises CircuitOpenError if circuit is OPEN and timeout has not elapsed.

        :raises CircuitOpenError: if circuit is open and recovery timeout not elapsed
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.HALF_OPEN:
                return

            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.warning(
                    "circuit breaker transitioning to HALF_OPEN for %s",
                    self._provider_name,
                )
                return

            remaining = self._recovery_timeout_seconds - elapsed
            raise CircuitOpenError(self._provider_name, remaining)

    def record_success(self) -> None:
        """records successful request and transitions state if needed.

        transitions HALF_OPEN back to CLOSED. resets failure count
        defensively in CLOSED state.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.warning(
                    "circuit breaker transitioning to CLOSED for %s",
                    self._provider_name,
                )
                return

            if self._state == CircuitState.CLOSED:
                self._failure_count = 0
                return

    def record_failure(self) -> None:
        """records failed request and transitions state if threshold reached.

        increments failure count and records failure timestamp. transitions
        HALF_OPEN immediately to OPEN. transitions CLOSED to OPEN when
        failure count reaches threshold.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit breaker re-opening for %s after probe failure",
                    self._provider_name,
                )
                return

            if self._state == CircuitState.CLOSED and self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit breaker opening for %s after %d failures",
                    self._provider_name,
                    self._failure_count,
                )
                return

    def reset(self) -> None:
        """forces circuit breaker back to CLOSED state.

        resets failure count and state unconditionally.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(
                "circuit breaker manually reset for %s",
                self._provider_name,
            )


class CircuitBreakerRegistry:
    """registry of per-provider circuit breakers.

    creates circuit breakers on demand for each provider. thread-safe
    access to shared breaker instances.

    :param failure_threshold: consecutive failures before circuit opens
    :ptype failure_threshold: int
    :param recovery_timeout_seconds: seconds to wait in OPEN before probing
    :ptype recovery_timeout_seconds: float
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, provider_name: str) -> CircuitBreaker:
        """returns circuit breaker for provider, creating one if needed.

        :param provider_name: identifier for provider
        :ptype provider_name: str
        :return: circuit breaker instance for provider
        :rtype: CircuitBreaker
        """
        with self._lock:
            if provider_name not in self._breakers:
                self._breakers[provider_name] = CircuitBreaker(
                    provider_name=provider_name,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout_seconds=self._recovery_timeout_seconds,
                )
            return self._breakers[provider_name]

    def reset(self, provider_name: str) -> None:
        """forces circuit breaker for provider back to CLOSED state.

        no-op if no breaker exists for provider.

        :param provider_name: identifier for provider to reset
        :ptype provider_name: str
        """
        with self._lock:
            breaker = self._breakers.get(provider_name)
        if breaker is not None:
            breaker.reset()

    def status(self) -> dict[str, CircuitState]:
        """returns snapshot of all provider circuit states.

        :return: mapping of provider name to current circuit state
        :rtype: dict[str, CircuitState]
        """
        with self._lock:
            return {name: breaker.state for name, breaker in self._breakers.items()}
