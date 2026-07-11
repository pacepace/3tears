"""per-provider circuit breaker for fault isolation in model routing.

Exposes both a thread-safe :class:`CircuitBreaker` and a LangChain
``BaseCallbackHandler`` factory (``CircuitBreaker.make_callback()``) that
fires the breaker's success/failure transitions in response to the
``on_llm_start`` / ``on_llm_end`` / ``on_llm_error`` hooks.
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from threetears.observe import get_logger

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerCallback",
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
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        # HALF_OPEN admits exactly ONE probe at a time: the request that trips
        # OPEN -> HALF_OPEN (or the first to arrive while HALF_OPEN) sets this;
        # every other concurrent request is fast-failed until the probe resolves
        # (record_success -> CLOSED / record_failure -> OPEN). without it,
        # HALF_OPEN admits unbounded concurrent probes -> a thundering herd onto
        # a provider that may still be dead.
        self._probe_in_flight = False
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
                # a probe is already testing the provider: fast-fail the rest so
                # recovery is a single request, not a thundering herd.
                if self._probe_in_flight:
                    raise CircuitOpenError(self._provider_name, 0.0)
                self._probe_in_flight = True
                return

            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self._recovery_timeout_seconds:
                # this request becomes the single recovery probe.
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
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
                self.failure_count = 0
                self._probe_in_flight = False
                logger.warning(
                    "circuit breaker transitioning to CLOSED for %s",
                    self._provider_name,
                )
                return

            if self._state == CircuitState.CLOSED:
                self.failure_count = 0
                return

    def record_failure(self) -> None:
        """records failed request and transitions state if threshold reached.

        increments failure count and records failure timestamp. transitions
        HALF_OPEN immediately to OPEN. transitions CLOSED to OPEN when
        failure count reaches threshold.
        """
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._probe_in_flight = False
                logger.warning(
                    "circuit breaker re-opening for %s after probe failure",
                    self._provider_name,
                )
                return

            if self._state == CircuitState.CLOSED and self.failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit breaker opening for %s after %d failures",
                    self._provider_name,
                    self.failure_count,
                )
                return

    def reset(self) -> None:
        """forces circuit breaker back to CLOSED state.

        resets failure count and state unconditionally.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self.failure_count = 0
            self._probe_in_flight = False
            logger.info(
                "circuit breaker manually reset for %s",
                self._provider_name,
            )

    def make_callback(self) -> BaseCallbackHandler:
        """builds a LangChain callback that drives this breaker.

        the callback short-circuits ``on_llm_start`` by calling
        :meth:`check`, records a success in ``on_llm_end``, and records a
        failure in ``on_llm_error``.

        :return: callback handler suitable for ``model.with_config(callbacks=[...])``
        :rtype: BaseCallbackHandler
        """
        return CircuitBreakerCallback(self)


class CircuitBreakerCallback(BaseCallbackHandler):
    """LangChain callback that wires a :class:`CircuitBreaker` into model events.

    fast-fails ``on_llm_start`` by raising :class:`CircuitOpenError` when
    the breaker is open. on success records via
    :meth:`CircuitBreaker.record_success`; on error via
    :meth:`CircuitBreaker.record_failure`.

    ``raise_error = True`` is REQUIRED: langchain's callback manager
    (``langchain_core.callbacks.manager.handle_event``) catches every callback
    exception and only re-raises when the handler opts in via ``raise_error``.
    without it a raised :class:`CircuitOpenError` is logged and SWALLOWED and the
    request proceeds to the known-dead provider -- the breaker delivers zero
    fault isolation. with it, the open-circuit fast-fail actually propagates and
    aborts the request.
    """

    raise_error: bool = True

    def __init__(self, breaker: CircuitBreaker) -> None:
        """initialises the callback with the breaker it should drive.

        :param breaker: backing circuit breaker
        :ptype breaker: CircuitBreaker
        """
        super().__init__()
        self._breaker = breaker

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """raises :class:`CircuitOpenError` when the breaker is open.

        :param serialized: serialized LLM definition (unused)
        :ptype serialized: dict[str, Any]
        :param prompts: prompt strings (unused)
        :ptype prompts: list[str]
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        :raises CircuitOpenError: if the breaker is currently OPEN
        """
        _ = serialized
        _ = prompts
        _ = run_id
        _ = kwargs
        self._breaker.check()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """raises :class:`CircuitOpenError` for chat models when breaker is open.

        :param serialized: serialized model definition (unused)
        :ptype serialized: dict[str, Any]
        :param messages: input messages (unused)
        :ptype messages: list[list[Any]]
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        :raises CircuitOpenError: if the breaker is currently OPEN
        """
        _ = serialized
        _ = messages
        _ = run_id
        _ = kwargs
        self._breaker.check()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """records a success on the breaker.

        :param response: LangChain LLM result (unused)
        :ptype response: Any
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = response
        _ = run_id
        _ = kwargs
        self._breaker.record_success()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """records a failure on the breaker.

        :class:`CircuitOpenError` itself does not count as a provider
        failure — it's the breaker fast-failing the request, not a real
        upstream error. swallow it here so a tripped breaker doesn't
        keep racking up its own failure count.

        :param error: raised exception
        :ptype error: BaseException
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = run_id
        _ = kwargs
        if isinstance(error, CircuitOpenError):
            return
        self._breaker.record_failure()


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
