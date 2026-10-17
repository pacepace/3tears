"""per-provider circuit breaker for fault isolation in model routing, and per model for rate limits.

A rate limit (HTTP 429) belongs to one model's route: OpenRouter limits per
model and upstream host. Counted against the provider, one model's 429s opened
the breaker for every model on OpenRouter (metallm dev e2e, 2026-10-15: 26 OpenRouter 429s, then three models refused). A
rate limit is counted against that model's own breaker; any other failure
against the provider's.

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
from collections import OrderedDict
from collections.abc import Callable
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


def _monotonic() -> float:
    """read :func:`time.monotonic` through this module's ``time`` binding, at call time.

    the default clock. resolving ``time`` when called rather than binding
    ``time.monotonic`` as a default argument keeps the one clock every breaker
    reads replaceable in a single place.

    :return: monotonic seconds
    :rtype: float
    """
    return time.monotonic()


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
    :param credential_scoped: whether the open circuit is ONE credential's on the provider
        rather than the provider's own -- a revoked key, not an outage. the credential is
        never named
    :ptype credential_scoped: bool
    :param model_name: the one model whose circuit is open for its rate limit, ``None`` when
        the provider's is
    :ptype model_name: str | None
    """

    def __init__(
        self,
        provider_name: str,
        remaining_seconds: float,
        *,
        credential_scoped: bool = False,
        model_name: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.remaining_seconds = remaining_seconds
        self.credential_scoped = credential_scoped
        self.model_name = model_name
        scope = " (one credential on it; the provider itself may be healthy)" if credential_scoped else ""
        target = (
            f"{model_name} on {provider_name} (rate-limited; its other models are not)" if model_name else provider_name
        )
        super().__init__(f"Circuit open for {target}{scope}, retry in {remaining_seconds:.0f}s")


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
    :param clock: monotonic seconds source for the recovery window and
        :attr:`last_activity`; ``None`` reads :func:`time.monotonic`
    :ptype clock: Callable[[], float] | None
    :param credential_scoped: whether this breaker guards ONE credential on the provider
        (:meth:`CircuitBreakerRegistry.get` with ``credential=``) rather than the provider
        itself. carried into every transition's log line and :class:`CircuitOpenError`, so
        one customer's revoked key tripping its own breaker does not read as the provider
        going down; the credential itself is never logged
    :ptype credential_scoped: bool
    :param model_name: the one model this breaker guards for its rate limits
        (:meth:`CircuitBreakerRegistry.get` with ``model=``), ``None`` for the provider
    :ptype model_name: str | None
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        *,
        clock: Callable[[], float] | None = None,
        credential_scoped: bool = False,
        model_name: str | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._credential_scoped = credential_scoped
        self._model_name = model_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._clock = clock if clock is not None else _monotonic
        self._state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self._last_activity = self._clock()
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
        clock: Callable[[], float] | None = None,
        credential_scoped: bool = False,
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
        :param clock: monotonic seconds source, as for the constructor
        :ptype clock: Callable[[], float] | None
        :param credential_scoped: whether the breaker guards one credential, as for the constructor
        :ptype credential_scoped: bool
        :return: a breaker positioned at the persisted state
        :rtype: CircuitBreaker
        """
        breaker = cls(
            provider_name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            clock=clock,
            credential_scoped=credential_scoped,
        )
        breaker._state = state
        breaker.failure_count = max(0, failure_count)
        now = clock() if clock is not None else _monotonic()
        breaker.last_failure_time = now - recovery_timeout_seconds + max(0.0, seconds_until_probe_permitted)
        return breaker

    @property
    def state(self) -> CircuitState:
        """returns current circuit state.

        :return: current circuit breaker state
        :rtype: CircuitState
        """
        with self._lock:
            return self._state

    @property
    def credential_scoped(self) -> bool:
        """whether this breaker guards one credential on its provider rather than the provider.

        :return: ``True`` for a credential-scoped breaker
        :rtype: bool
        """
        return self._credential_scoped

    def _log_extra(self) -> dict[str, Any]:
        """the structured fields every transition line carries -- never the credential.

        :return: the log ``extra``
        :rtype: dict[str, Any]
        """
        return {
            "extra_data": {
                "provider": self._provider_name,
                "credential_scoped": self._credential_scoped,
                "model": self._model_name,
            }
        }

    def _subject(self) -> str:
        """what a transition line names: the provider, or one credential on it.

        :return: the subject of the log line
        :rtype: str
        """
        where = f"one credential on {self._provider_name}" if self._credential_scoped else self._provider_name
        return f"{self._model_name}'s rate limit on {where}" if self._model_name else where

    @property
    def last_activity(self) -> float:
        """the clock reading of the last check, outcome or reset -- or of creation.

        what a registry measures idleness by: a breaker that a live model keeps
        checking is in use however long ago anyone asked the registry for it.

        :return: clock reading of the breaker's last activity
        :rtype: float
        """
        with self._lock:
            return self._last_activity

    def check(self) -> None:
        """verifies circuit allows request to proceed.

        transitions OPEN to HALF_OPEN when recovery timeout has elapsed.
        raises CircuitOpenError if circuit is OPEN and timeout has not elapsed.

        :raises CircuitOpenError: if circuit is open and recovery timeout not elapsed
        """
        with self._lock:
            self._last_activity = self._clock()
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.HALF_OPEN:
                # a probe is already testing the provider: fast-fail the rest so
                # recovery is a single request, not a thundering herd.
                if self._probe_in_flight:
                    raise CircuitOpenError(
                        self._provider_name, 0.0, credential_scoped=self._credential_scoped, model_name=self._model_name
                    )
                self._probe_in_flight = True
                return

            elapsed = self._clock() - self.last_failure_time
            if elapsed >= self._recovery_timeout_seconds:
                # this request becomes the single recovery probe.
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                logger.warning(
                    "circuit breaker transitioning to HALF_OPEN for %s",
                    self._subject(),
                    extra=self._log_extra(),
                )
                return

            remaining = self._recovery_timeout_seconds - elapsed
            raise CircuitOpenError(
                self._provider_name, remaining, credential_scoped=self._credential_scoped, model_name=self._model_name
            )

    def record_success(self) -> None:
        """records successful request and transitions state if needed.

        transitions HALF_OPEN back to CLOSED. resets failure count
        defensively in CLOSED state.
        """
        with self._lock:
            self._last_activity = self._clock()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self.failure_count = 0
                self._probe_in_flight = False
                logger.warning(
                    "circuit breaker transitioning to CLOSED for %s",
                    self._subject(),
                    extra=self._log_extra(),
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
            self.last_failure_time = self._clock()
            self._last_activity = self.last_failure_time

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._probe_in_flight = False
                logger.warning(
                    "circuit breaker re-opening for %s after probe failure",
                    self._subject(),
                    extra=self._log_extra(),
                )
                return

            if self._state == CircuitState.CLOSED and self.failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit breaker opening for %s after %d failures",
                    self._subject(),
                    self.failure_count,
                    extra=self._log_extra(),
                )
                return

    def reset(self) -> None:
        """forces circuit breaker back to CLOSED state.

        resets failure count and state unconditionally.
        """
        with self._lock:
            self._last_activity = self._clock()
            self._state = CircuitState.CLOSED
            self.failure_count = 0
            self._probe_in_flight = False
            logger.info(
                "circuit breaker manually reset for %s",
                self._subject(),
                extra=self._log_extra(),
            )

    def make_callback(self, *, rate_limit: CircuitBreaker | None = None) -> BaseCallbackHandler:
        """builds a LangChain callback that drives this breaker, and the model's own for its rate limits.

        the callback short-circuits ``on_llm_start`` by calling :meth:`check`
        on both, records a success on both in ``on_llm_end``, and records a
        failure in ``on_llm_error``: a rate limit on ``rate_limit``, anything
        else on this breaker.

        :param rate_limit: the model's own breaker (:meth:`CircuitBreakerRegistry.get` with
            ``model=``); ``None`` counts a rate limit against this breaker
        :ptype rate_limit: CircuitBreaker | None
        :return: callback handler suitable for ``model.with_config(callbacks=[...])``
        :rtype: BaseCallbackHandler
        """
        return CircuitBreakerCallback(self, rate_limit=rate_limit)


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

    def __init__(self, breaker: CircuitBreaker, *, rate_limit: CircuitBreaker | None = None) -> None:
        """initialises the callback with the breakers it should drive.

        :param breaker: the provider's breaker, which every failure but a rate limit counts against
        :ptype breaker: CircuitBreaker
        :param rate_limit: the model's own breaker, which its rate limits count against
        :ptype rate_limit: CircuitBreaker | None
        """
        super().__init__()
        self._breaker = breaker
        self._rate_limit = rate_limit

    def _check(self) -> None:
        self._breaker.check()
        if self._rate_limit is not None:
            self._rate_limit.check()

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
        self._check()

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
        self._check()

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
        if self._rate_limit is not None:
            self._rate_limit.record_success()

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
        from threetears.models.errors import is_rate_limit

        if isinstance(error, CircuitOpenError):
            return
        if self._rate_limit is not None and isinstance(error, Exception) and is_rate_limit(error):
            # one model's route is limited: its own breaker, never the provider's other models
            self._rate_limit.record_failure()
            return
        self._breaker.record_failure()


# how bad each state is, for reporting one state per provider: a provider with
# any credential's circuit open reports open.
_STATE_SEVERITY = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


class CircuitBreakerRegistry:
    """registry of circuit breakers, one per provider -- or per provider and credential.

    creates circuit breakers on demand. thread-safe access to shared breaker
    instances.

    a breaker is keyed by provider: a provider's failures are shared across its
    models, and that granularity is deliberate -- except for a rate limit, which
    belongs to one model's route, so ``model=`` gives a model its own breaker for
    its rate limits (:meth:`CircuitBreaker.make_callback`). what a
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

    **credential-scoped breakers are bounded.** a process that sees many keys,
    or rotates them, would otherwise keep one breaker per key it ever saw.
    whenever a new credential's breaker is created, the registry first drops
    every CLOSED credential breaker idle for ``credential_idle_seconds`` (no
    :meth:`get` for it, and no check or outcome on it), then, while it still
    holds ``max_credential_breakers`` or more, the least recently used CLOSED
    ones. an OPEN or HALF_OPEN breaker is never dropped -- that would forget a
    tripped credential and let it be hammered again -- so the count can exceed
    the cap only by breakers that are tripped right now. a dropped breaker held
    little: a CLOSED circuit and at most a partial failure count. provider-only
    breakers (no credential) are never dropped; there is one per provider.

    :param failure_threshold: consecutive failures before circuit opens
    :ptype failure_threshold: int
    :param recovery_timeout_seconds: seconds to wait in OPEN before probing
    :ptype recovery_timeout_seconds: float
    :param credential_idle_seconds: how long a CLOSED credential breaker may
        sit unused before it is dropped
    :ptype credential_idle_seconds: float
    :param max_credential_breakers: how many credential breakers the registry
        keeps before dropping the least recently used CLOSED ones
    :ptype max_credential_breakers: int
    :param clock: monotonic seconds source, shared with every breaker created;
        ``None`` reads :func:`time.monotonic`
    :ptype clock: Callable[[], float] | None
    :raises ValueError: when ``credential_idle_seconds`` is not positive or
        ``max_credential_breakers`` is less than one
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        *,
        credential_idle_seconds: float = 3600.0,
        max_credential_breakers: int = 1024,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if credential_idle_seconds <= 0:
            raise ValueError(f"credential_idle_seconds must be positive, got {credential_idle_seconds}")
        if max_credential_breakers < 1:
            raise ValueError(f"max_credential_breakers must be at least 1, got {max_credential_breakers}")
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._credential_idle_seconds = credential_idle_seconds
        self._max_credential_breakers = max_credential_breakers
        self._clock = clock if clock is not None else _monotonic
        # (provider_name, credential fingerprint or None, model or None) -> breaker, least
        # recently asked-for first, and the clock reading of each key's last get()
        self._breakers: OrderedDict[tuple[str, str | None, str | None], CircuitBreaker] = OrderedDict()
        self._last_get: dict[tuple[str, str | None, str | None], float] = {}
        # every provider ever asked for, so status() keeps reporting one whose
        # breakers were all dropped as idle (they were CLOSED) instead of losing it
        self._providers: set[str] = set()
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

    def get(self, provider_name: str, *, credential: str | None = None, model: str | None = None) -> CircuitBreaker:
        """returns the circuit breaker for a provider (and credential), creating one if needed.

        without ``credential`` there is one breaker per provider, as there always
        was. with it, each distinct credential on the provider has its own.

        :param provider_name: identifier for provider
        :ptype provider_name: str
        :param credential: the api key or token the calls are made with, when
            calls on this provider are made with more than one; ``None`` for one
            breaker per provider
        :ptype credential: str | None
        :param model: the model whose rate limits this breaker counts, ``None`` for the provider's own
        :ptype model: str | None
        :return: circuit breaker instance for the provider, credential and model
        :rtype: CircuitBreaker
        """
        key = (provider_name, self._fingerprint(credential), model)
        now = self._clock()
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                if _droppable(key):
                    self._evict_credential_breakers_locked(now)
                breaker = CircuitBreaker(
                    provider_name=provider_name,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout_seconds=self._recovery_timeout_seconds,
                    clock=self._clock,
                    credential_scoped=key[1] is not None,
                    model_name=model,
                )
                self._breakers[key] = breaker
                self._providers.add(provider_name)
            self._breakers.move_to_end(key)
            self._last_get[key] = now
        return breaker

    def _evict_credential_breakers_locked(self, now: float) -> None:
        """drop idle, then least recently used, CLOSED credential breakers. **caller holds the lock.**

        :param now: the current clock reading
        :ptype now: float
        :return: nothing
        :rtype: None
        """
        last_used = {
            key: max(self._last_get.get(key, 0.0), breaker.last_activity)
            for key, breaker in self._breakers.items()
            if _droppable(key) and breaker.state is CircuitState.CLOSED
        }
        for key, used in last_used.items():
            if now - used >= self._credential_idle_seconds:
                self._drop_locked(key)
        credential_count = sum(1 for key in self._breakers if _droppable(key))
        for key in sorted((k for k in last_used if k in self._breakers), key=lambda k: last_used[k]):
            if credential_count < self._max_credential_breakers:
                break
            self._drop_locked(key)
            credential_count -= 1
        if credential_count >= self._max_credential_breakers:
            logger.warning(
                "circuit breaker registry is at its credential cap and every remaining credential breaker "
                "is tripped; keeping them all rather than forgetting a tripped credential",
                extra={"extra_data": {"credential_breakers": credential_count, "cap": self._max_credential_breakers}},
            )

    def _drop_locked(self, key: tuple[str, str | None, str | None]) -> None:
        """forget one breaker. **caller holds the lock.**

        :param key: ``(provider_name, fingerprint, model)``
        :ptype key: tuple[str, str | None, str | None]
        :return: nothing
        :rtype: None
        """
        self._breakers.pop(key, None)
        self._last_get.pop(key, None)

    def reset(self, provider_name: str, *, credential: str | None = None) -> None:
        """forces circuit breakers for a provider back to CLOSED state.

        without ``credential`` every breaker on the provider is reset, whatever
        credential or model it was created for; with it, only that credential's.
        no-op when no such breaker exists.

        :param provider_name: identifier for provider to reset
        :ptype provider_name: str
        :param credential: the one credential to reset, or ``None`` for all
        :ptype credential: str | None
        """
        fingerprint = self._fingerprint(credential)
        with self._lock:
            breakers = [
                breaker
                for (name, key_fingerprint, _model), breaker in self._breakers.items()
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
        is being fast-failed". a provider whose breakers were all dropped as
        idle stays in the snapshot as closed, which is what they were.

        :return: mapping of provider name to current circuit state
        :rtype: dict[str, CircuitState]
        """
        with self._lock:
            # the provider's own breakers: one model's rate limit is not the provider's state (model_status)
            snapshot = [
                (name, breaker.state)
                for (name, _fingerprint, model), breaker in self._breakers.items()
                if model is None
            ]
            providers = sorted(self._providers)
        result: dict[str, CircuitState] = dict.fromkeys(providers, CircuitState.CLOSED)
        for name, state in snapshot:
            current = result.get(name)
            if current is None or _STATE_SEVERITY[state] > _STATE_SEVERITY[current]:
                result[name] = state
        return result

    def model_status(self) -> dict[str, CircuitState]:
        """returns each rate-limited model's circuit state that is not closed, as ``provider/model``.

        a model's breaker counts only its rate limits, so a closed one says
        nothing and is left out; several credentials' breakers for one model
        report the worst of them.

        :return: mapping of ``provider/model`` to its open or half-open state
        :rtype: dict[str, CircuitState]
        """
        with self._lock:
            snapshot = [
                (f"{name}/{model}", breaker.state)
                for (name, _fingerprint, model), breaker in self._breakers.items()
                if model is not None and breaker.state is not CircuitState.CLOSED
            ]
        result: dict[str, CircuitState] = {}
        for name, state in snapshot:
            current = result.get(name)
            if current is None or _STATE_SEVERITY[state] > _STATE_SEVERITY[current]:
                result[name] = state
        return result


def _droppable(key: tuple[str, str | None, str | None]) -> bool:
    """whether a breaker may be dropped when idle: one credential's, or one model's -- never a provider's own."""
    return key[1] is not None or key[2] is not None
