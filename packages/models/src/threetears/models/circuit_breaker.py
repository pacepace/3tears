"""per-provider circuit breaker for fault isolation in model routing.

Exposes both a thread-safe :class:`CircuitBreaker` and a LangChain
``BaseCallbackHandler`` factory (``CircuitBreaker.make_callback()``) that
fires the breaker's success/failure transitions in response to the
``on_llm_start`` / ``on_llm_end`` / ``on_llm_error`` hooks.
"""

from __future__ import annotations

import hashlib
import secrets
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

    @classmethod
    def restore(
        cls,
        provider_name: str,
        *,
        state: CircuitState,
        failure_count: int,
        seconds_until_probe_permitted: float = 0.0,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ) -> CircuitBreaker:
        """rebuilds a breaker from state that was persisted somewhere else.

        this class holds its state in memory behind a ``threading.Lock``, so
        it is process-local by construction. a consumer that needs the SAME
        circuit honoured across pods and restarts has to keep the state in a
        durable store of its own -- but the transition rules (when a
        threshold trips, when a probe is admitted, what a probe's outcome
        does) should not be reimplemented alongside it, because a second
        copy of a state machine is a second copy that can disagree.

        ``restore`` is the seam for that: hydrate from the durable row, drive
        the transition through :meth:`check` / :meth:`record_success` /
        :meth:`record_failure`, then persist :attr:`state` and
        :attr:`failure_count` back. the rules stay here; only the storage
        moves.

        ``seconds_until_probe_permitted`` is how long remains before an OPEN
        circuit may admit its recovery probe, which is what a durable
        "blocked until" timestamp already records. it is expressed that way
        rather than as an age because :attr:`last_failure_time` is a
        ``time.monotonic()`` reading, and a monotonic clock is meaningless
        across the process boundary the caller just crossed. zero or less
        means the recovery window has already elapsed, so the next
        :meth:`check` promotes OPEN to HALF_OPEN and admits the probe.

        no probe is treated as in flight on a restored breaker: an in-flight
        probe belongs to whichever process issued it, and a different process
        cannot observe it. a caller needing cross-pod single-probe admission
        has to reach for a distributed primitive; this restores the state,
        not the other process's in-flight request.

        :param provider_name: identifier for the provider this breaker protects
        :ptype provider_name: str
        :param state: the persisted circuit state
        :ptype state: CircuitState
        :param failure_count: the persisted consecutive-failure count
        :ptype failure_count: int
        :param seconds_until_probe_permitted: seconds remaining before an OPEN
            circuit may probe; zero or less means the window has elapsed
        :ptype seconds_until_probe_permitted: float
        :param failure_threshold: consecutive failures before the circuit opens
        :ptype failure_threshold: int
        :param recovery_timeout_seconds: seconds to wait in OPEN before probing
        :ptype recovery_timeout_seconds: float
        :return: a breaker positioned at the persisted state
        :rtype: CircuitBreaker
        """
        breaker = cls(
            provider_name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
        )
        breaker._state = state
        breaker.failure_count = max(0, failure_count)
        breaker.last_failure_time = (
            time.monotonic() - recovery_timeout_seconds + max(0.0, seconds_until_probe_permitted)
        )
        return breaker

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


# how bad each state is, for reporting one state per provider: a provider with
# any credential's circuit open reports open.
_STATE_SEVERITY = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


class CircuitBreakerRegistry:
    """registry of circuit breakers, one per provider -- or per provider and credential.

    creates circuit breakers on demand. thread-safe access to shared breaker
    instances.

    a breaker is keyed by provider, never by model: a provider's failures are
    shared across its models, and that granularity is deliberate. what a
    multi-tenant caller also needs is to keep CREDENTIALS apart -- one
    customer's revoked, rate-limited or out-of-credit key failing repeatedly
    must not fast-fail every other customer on the same provider. passing
    ``credential=`` gives each credential its own breaker on that provider.

    the credential is never held. the registry keys it by a short keyed
    blake2b fingerprint whose key is random per registry, so the fingerprint
    identifies the credential only inside this registry: it is not a digest
    anyone could recompute from the key, and it is not stable across
    processes. it never reaches a log line, an error, or :meth:`status` --
    each breaker still reports and logs under the provider name alone.

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
        # (provider_name, credential fingerprint or None) -> breaker
        self._breakers: dict[tuple[str, str | None], CircuitBreaker] = {}
        self._fingerprint_key = secrets.token_bytes(32)
        self._lock = threading.Lock()

    def _fingerprint(self, credential: str | None) -> str | None:
        """return the registry-local fingerprint of ``credential``, or ``None`` for none.

        :param credential: the api key or token, or ``None``
        :ptype credential: str | None
        :return: 16 hex characters of a keyed blake2b, or ``None``
        :rtype: str | None
        """
        if credential is None:
            return None
        return hashlib.blake2b(credential.encode(), key=self._fingerprint_key, digest_size=8).hexdigest()

    def get(self, provider_name: str, *, credential: str | None = None) -> CircuitBreaker:
        """returns the circuit breaker for a provider (and credential), creating one if needed.

        without ``credential`` there is one breaker per provider, as there always
        was. with it, each distinct credential on the provider has its own.

        :param provider_name: identifier for provider
        :ptype provider_name: str
        :param credential: the api key or token the calls are made with, when
            calls on this provider are made with more than one; ``None`` for one
            breaker per provider
        :ptype credential: str | None
        :return: circuit breaker instance for the provider and credential
        :rtype: CircuitBreaker
        """
        key = (provider_name, self._fingerprint(credential))
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = CircuitBreaker(
                    provider_name=provider_name,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout_seconds=self._recovery_timeout_seconds,
                )
            return self._breakers[key]

    def reset(self, provider_name: str, *, credential: str | None = None) -> None:
        """forces circuit breakers for a provider back to CLOSED state.

        without ``credential`` every breaker on the provider is reset, whatever
        credential it was created for; with it, only that credential's. no-op
        when no such breaker exists.

        :param provider_name: identifier for provider to reset
        :ptype provider_name: str
        :param credential: the one credential to reset, or ``None`` for all
        :ptype credential: str | None
        """
        fingerprint = self._fingerprint(credential)
        with self._lock:
            breakers = [
                breaker
                for (name, key_fingerprint), breaker in self._breakers.items()
                if name == provider_name and (credential is None or key_fingerprint == fingerprint)
            ]
        for breaker in breakers:
            breaker.reset()

    def status(self) -> dict[str, CircuitState]:
        """returns snapshot of each provider's circuit state.

        keyed by provider name only, so it stays one entry per provider however
        many credentials are in use -- safe to export as metric labels. a
        provider with several credential-scoped breakers reports the worst of
        them (open, then half-open, then closed): "some caller of this provider
        is being fast-failed".

        :return: mapping of provider name to current circuit state
        :rtype: dict[str, CircuitState]
        """
        with self._lock:
            snapshot = [(name, breaker.state) for (name, _fingerprint), breaker in self._breakers.items()]
        result: dict[str, CircuitState] = {}
        for name, state in snapshot:
            current = result.get(name)
            if current is None or _STATE_SEVERITY[state] > _STATE_SEVERITY[current]:
                result[name] = state
        return result
