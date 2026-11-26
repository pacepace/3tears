"""The key pool: 1..n credentials for one provider, each inside a budget, with sticky rotation.

A :class:`KeyPool` hands each call a :class:`KeyPoolLease` on its **active key**: the key every call
starts on, held in the shared store so every process sharing the pool starts on the same one. The
active key moves only when it can't serve -- exhausted, revoked, forbidden, tripped or low on room --
and the move is recorded once, so later calls don't rediscover a dead key (KP-D).

**A rate limit is not a key state** (owner, 2026-11-26). It never moves the active key and is never
recorded: the call is retried on the same key, by the provider's SDK where it retries one (the pool
only stops it retrying what is the key's, :meth:`KeyPool.refuses_retry`), and by the pool where it
doesn't (:meth:`KeyPool.wait_to_retry`). After those retries the rate limit reaches the caller. Only a
rate limit whose wait runs past ``rate_limit_exhaustion_wait`` is the key's: it is read as exhaustion
until the wait ends.

**The success path is two store operations** (KP-G6): the reservation an acquire makes, and the
settle (or release) that closes it. The pool's record -- the active key and every key's state -- is
cached, and read again only after ``refresh_interval`` (5 s), or when a write learns of a change; a
move, a state change and the orphan sweep are the only other operations.

**The record.** One versioned JSON document per pool, named ``key-pool/<provider>/<pool>``, written
only by compare-and-set; on a conflict the pool re-reads and re-applies its change, so a loser adopts
the winner's active key (KP-D4). Its layout::

    {
      "layout": 1,
      "period": "2026-11",          # the billing period the pool last started over in
      "active": "primary",          # the active key's label
      "keys": {
        "primary": {
          "budget": "100",          # the budget the last process to start configured (KP-A5)
          "generations": {          # one per secret the label has held, newest four kept
            "0": {
              "fingerprint": "<16 hex>",
              "revoked": false,
              "exhausted": {            # or null
                "period": "2026-11",    # the period it was exhausted in
                "until": null           # when the provider said it resets; null: the period's end
              },
              "low_room": null,         # the period its room fell below min_headroom in, or null
              "forbidden": ["*"]        # "*" for every model, else model ids
            }
          }
        }
      }
    }

An exhausted key is held until the reset the provider named (an OpenRouter budget's ``resets_at``, the
end of a long rate-limit wait), or else until its billing period ends. A breaker's ``tripped`` state is
never stored; it is read live from the pool's
:class:`~threetears.models.circuit_breaker.CircuitBreakerRegistry`, with no side effects (KP-C6).

**Fingerprints and generations** (KP-B2). Each secret's fingerprint is an HMAC-SHA256 of it, keyed by
the provider and pool name, truncated to 64 bits: the same in every process, and not reversible. The
secrets a pool holds are provider-issued keys of well over 64 bits of entropy, so a fast digest gives
no shortcut to one; what it does allow is confirming a guessed key against the store, which only a
reader of the store can try, and which is why the key is the pool's name and the fingerprint never
leaves the record -- not in a log, ``status()``, an exception, or an account name. A label holding a new
secret gets a new generation: fresh state, and, because the ledger account names the generation, zero
spend. A process still running the old secret keeps its own generation, so two processes mid-rollout
never reset each other's spend.

**Accounts.** A key's spend is kept in the ledger account ``<provider>/<pool>/<label>/<generation>/
<period>``, so a new period starts at zero with no rollover job, and a reservation settles into the
period it was made in (KP-F1, F2).

**Store failures** (KP-G5). A store call that raises anything its contract doesn't name is raised as
:class:`~threetears.models.key_pool.store.KeyPoolStoreError` with the cause chained, logged at ERROR
and counted, and the call fails closed. The contract is per operation, stated by builtin base as the
store Protocol states it: ``ValueError`` for an invalid argument or a reused reservation id (reserve,
settle, release, sweep, compare-and-set), and ``LookupError`` for a settle or release of a reservation
the account doesn't hold. A bare ``KeyError``, ``IndexError`` or ``LookupError`` is not that contract --
a store's not-found is its own subclass -- so it is a store's bug, and fails closed as a storage failure
rather than passing as "not found". A read or a snapshot has no error in its contract at all.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

from threetears.models.circuit_breaker import CircuitBreakerRegistry
from threetears.models.key_pool.classify import (
    KeyErrorClassifier,
    Verdict,
    VerdictKind,
    VerdictScope,
    classifier_for,
)
from threetears.models.key_pool.errors import NoKeyAvailableError
from threetears.models.key_pool.observe import KeyPoolCounters, KeyPoolObserver, MoveReason
from threetears.models.key_pool.period import period_bounds, period_id
from threetears.models.key_pool.secret import KeySpec, Secret
from threetears.models.key_pool.selection import KeyCandidate, SelectionPolicy, StickySelection
from threetears.models.key_pool.status import KeyPoolStatus, KeyState, KeyStatus
from threetears.models.key_pool.store import (
    AccountSnapshot,
    InMemoryKeyPoolStore,
    KeyPoolStore,
    KeyPoolStoreError,
    ReservationNotFound,
)
from threetears.models.providers.claude_cli import is_subscription_token
from threetears.observe import contained, get_logger
from uuid_utils import uuid7

__all__ = ["KeyPool", "KeyPoolLease"]

logger = get_logger(__name__)

#: The record layout this module reads and writes.
_LAYOUT: Final = 1

#: A forbid that covers the whole key, not one model.
_WHOLE_KEY: Final = "*"

#: Compare-and-set rounds a write gets before the pool gives up and fails the call closed. A round is
#: lost only to another writer's change to the same pool's record, which is a move or a state change.
_MAX_CAS_ROUNDS: Final = 16

#: The secrets a label's record remembers. A process still running a secret older than this many
#: rotations starts that secret over, from fresh state and zero spend.
_GENERATIONS_KEPT: Final = 4

#: How often the pool logs a ``NoKeyAvailableError`` (KP-I2): once per this interval while its keys'
#: states stay the same, carrying the count of raises it didn't log; a raise whose states differ from
#: the last line's is logged at once. The counter is never deduplicated.
_NO_KEY_LOG_INTERVAL: Final = timedelta(seconds=60)

#: A pool name: short, and safe as a metric label and inside a ledger account name.
_POOL_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

#: The bare lookup types: a store's own not-found error is a subclass, so one of these is a bug.
_BARE_LOOKUPS: Final = (LookupError, KeyError, IndexError)

#: What each store operation may raise as its contract, passing through the pool unchanged.
_RESERVE_CONTRACT: Final[tuple[type[Exception], ...]] = (ValueError,)
_CLOSE_CONTRACT: Final[tuple[type[Exception], ...]] = (LookupError, ValueError)
_NO_CONTRACT: Final[tuple[type[Exception], ...]] = ()

#: How many times a call is retried on the same key after a rate limit its provider's SDK doesn't retry
#: (owner, 2026-11-26: two), before the rate limit reaches the caller.
_RATE_LIMIT_RETRIES: Final = 2

#: The first wait of such a retry when the provider named none, doubled for the next: the backoff the
#: Stainless SDKs start from.
_RATE_LIMIT_BACKOFF_SECONDS: Final = 0.5

#: Moves logged at WARNING (KP-D7): a key that won't serve again this period without an operator or a
#: new secret. The rest pass on their own and are logged at INFO.
_WARNING_REASONS: Final = frozenset({MoveReason.EXHAUSTED, MoveReason.REVOKED, MoveReason.FORBIDDEN})

#: The move reason each blocking state gives.
_MOVE_REASON: Final = {
    KeyState.EXHAUSTED: MoveReason.EXHAUSTED,
    KeyState.REVOKED: MoveReason.REVOKED,
    KeyState.FORBIDDEN: MoveReason.FORBIDDEN,
    KeyState.TRIPPED: MoveReason.TRIPPED,
    KeyState.NO_ROOM: MoveReason.LOW_ROOM,
}

#: The reason, and so the log level, each key-moving verdict gives.
_VERDICT_REASON: Final = {
    VerdictKind.EXHAUSTED: MoveReason.EXHAUSTED,
    VerdictKind.REVOKED: MoveReason.REVOKED,
    VerdictKind.FORBIDDEN: MoveReason.FORBIDDEN,
}


def _utc_now() -> datetime:
    """The wall clock, in UTC."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Key:
    """A configured key, with what the pool derives from it once."""

    spec: KeySpec
    fingerprint: str

    @property
    def label(self) -> str:
        return self.spec.label


@dataclass(frozen=True, slots=True)
class _View:
    """The pool's record as this process last read or wrote it."""

    version: int
    doc: dict[str, Any]
    read_at: datetime


@dataclass(frozen=True, slots=True)
class _Block:
    """One reason a key can't serve, and when it ends, if that is known."""

    state: KeyState
    until: datetime | None


#: A change to the record. It edits the document in place and returns whether the pool starts over
#: from its most-preferred key (a period rollover) rather than keeping its active key if it can.
_Edit = Callable[[dict[str, Any]], bool]


class KeyPoolLease:
    """One call's hold on one key: the key to send the call with, and the reservation to close.

    Built by :meth:`KeyPool.acquire` and :meth:`KeyPool.reserve`, which hand it the two closures that
    settle its reservation and record a failure's verdict. Close it exactly once, with
    :meth:`settle`, :meth:`release` or :meth:`fail`. A lease never closed is settled at its full amount
    when its reservation expires, and counted as orphaned (KP-E5).

    :ivar label: the key's label
    :ivar secret: the key's credential, for the call
    :ivar account: the key's configured provider-side account, ``None`` when it is its own
    :ivar model_id: the model the call is for, ``None`` for a plain reservation
    :ivar amount: the money reserved
    :ivar reservation_id: the reservation's id, unique to this lease
    :ivar expires_at: when the reservation expires
    :ivar period: the billing period the reservation is in, and settles into
    """

    def __init__(
        self,
        *,
        label: str,
        secret: Secret,
        account: str | None,
        model_id: str | None,
        amount: Decimal,
        reservation_id: str,
        expires_at: datetime,
        period: str,
        ledger_account: str,
        close: Callable[[Decimal | None], Awaitable[None]],
        record: Callable[[Verdict], Awaitable[None]],
    ) -> None:
        self._ledger_account = ledger_account
        self._close = close
        self._record = record
        self._closed = False
        self.label = label
        self.secret = secret
        self.account = account
        self.model_id = model_id
        self.amount = amount
        self.reservation_id = reservation_id
        self.expires_at = expires_at
        self.period = period

    def __repr__(self) -> str:
        """The lease by its label and reservation, never its secret."""
        return f"KeyPoolLease(label={self.label!r}, model_id={self.model_id!r}, amount={self.amount}, period={self.period!r})"

    @property
    def closed(self) -> bool:
        """Whether the lease was settled, released or failed.

        :return: ``True`` once closed
        :rtype: bool
        """
        return self._closed

    async def settle(self, cost: Decimal) -> None:
        """Close the lease with the call's actual cost, which replaces the reservation as spend.

        ``cost`` may be above or below the reservation: spend records the truth.

        :param cost: what the call cost; a finite ``Decimal``, 0 or more
        :ptype cost: Decimal
        :raises ValueError: when ``cost`` is invalid
        :raises ReservationNotFound: when the lease is already closed, or the store no longer holds its
            reservation
        :raises KeyPoolStoreError: when the store failed
        """
        _check_amount("cost", cost)
        self._check_open()
        await self._close(cost)
        self._closed = True

    async def release(self) -> None:
        """Close the lease with no spend, for a call the provider can't have billed.

        :raises ReservationNotFound: when the lease is already closed, or the store no longer holds its
            reservation
        :raises KeyPoolStoreError: when the store failed
        """
        self._check_open()
        await self._close(None)
        self._closed = True

    async def fail(self, verdict: Verdict) -> None:
        """Close the lease for a failed call, by what the classifier said about it.

        A key-moving verdict (exhausted, revoked, forbidden) is recorded first: the key's state (its
        account's every key, for an account-scoped one), and a move of the active key if it can no
        longer serve. A rate limit, an oversized request and a failure that isn't the key's record
        nothing. Then the reservation is settled at its full amount when the provider may have billed
        the attempt, and released when it can't have (KP-E4).

        :param verdict: the classifier's verdict on the failure
        :ptype verdict: Verdict
        :raises ReservationNotFound: when the lease is already closed
        :raises KeyPoolStoreError: when the store failed
        """
        self._check_open()
        if verdict.moves_key:
            await self._record(verdict)
        await self._close(self.amount if verdict.may_have_billed else None)
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise ReservationNotFound(self._ledger_account, self.reservation_id)


class KeyPool:
    """1..n credentials for one provider, each inside a budget, with sticky rotation (KP-A to KP-K).

    One key is a normal pool: nothing branches on how many keys there are (KP-A2).

    :param provider: the provider every key is for (``"anthropic"``, ``"openai"``, ``"openrouter"``)
    :ptype provider: str
    :param name: the pool's name: 1 to 64 letters, digits, ``_``, ``.`` or ``-``. Every process that
        builds a pool with this provider and name on one store shares its spend, state and active key
    :ptype name: str
    :param keys: the keys, in configuration order, which breaks ties between equal priorities
    :ptype keys: Sequence[KeySpec]
    :param store: the shared store. Defaults to a new :class:`InMemoryKeyPoolStore`, the store for a
        single process; a pool shared by several processes must pass the one store they share, or each
        process budgets alone
    :ptype store: KeyPoolStore | None
    :param anchor_day: the day of the month each billing period starts on, 1 to 28
    :ptype anchor_day: int
    :param min_headroom: the fraction of a key's budget below which its room counts as too low, so the
        pool moves off it; 0 or more and below 1. A reservation that doesn't fit on a key with more
        room than this skips the key for that call only (KP-D3)
    :ptype min_headroom: Decimal
    :param rate_limit_exhaustion_wait: the longest wait a rate limit may ask for and still be one: a
        rate limit whose ``retry-after`` (or reset) is further off is read as exhaustion until then, so
        the call moves to the next key rather than wait (owner, 2026-11-26). Positive
    :ptype rate_limit_exhaustion_wait: timedelta
    :param refresh_interval: how stale the cached record may get before a call reads it again (KP-D1)
    :ptype refresh_interval: timedelta
    :param reservation_margin: how long past a call's deadline its reservation is kept before the sweep
        may settle it as orphaned; a call with no deadline gets this from when it started (KP-E5)
    :ptype reservation_margin: timedelta
    :param sweep_interval: how often, at most, the pool sweeps its accounts for orphaned reservations
    :ptype sweep_interval: timedelta
    :param classifier: reads a failed call's exception; defaults to the provider's own
    :ptype classifier: KeyErrorClassifier | None
    :param breaker_registry: the registry the per-key models' breakers live in and the pool reads them
        from; defaults to a new one this pool owns
    :ptype breaker_registry: CircuitBreakerRegistry | None
    :param selection: the selection policy; defaults to :class:`StickySelection`
    :ptype selection: SelectionPolicy | None
    :param observer: receives the pool's events, for metrics; the pool always counts them in
        :attr:`counters` as well
    :ptype observer: KeyPoolObserver | None
    :param clock: the current time, timezone-aware; defaults to the UTC wall clock
    :ptype clock: Callable[[], datetime] | None
    :param sleep: waits a number of seconds, for :meth:`wait_to_retry`; defaults to ``asyncio.sleep``
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    :raises ValueError: for zero keys, duplicate labels, duplicate secrets, a Claude subscription token,
        a provider with no classifier, an invalid name, or an invalid setting
    """

    def __init__(
        self,
        provider: str,
        name: str,
        keys: Sequence[KeySpec],
        store: KeyPoolStore | None = None,
        *,
        anchor_day: int = 1,
        min_headroom: Decimal = Decimal("0.01"),
        rate_limit_exhaustion_wait: timedelta = timedelta(seconds=120),
        refresh_interval: timedelta = timedelta(seconds=5),
        reservation_margin: timedelta = timedelta(minutes=10),
        sweep_interval: timedelta = timedelta(minutes=1),
        classifier: KeyErrorClassifier | None = None,
        breaker_registry: CircuitBreakerRegistry | None = None,
        selection: SelectionPolicy | None = None,
        observer: KeyPoolObserver | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"provider must be a provider name, got {provider!r}")
        if not isinstance(name, str) or _POOL_NAME.fullmatch(name) is None:
            raise ValueError(f"pool name {name!r} must be 1 to 64 letters, digits, '_', '.' or '-'")
        self._classifier = classifier if classifier is not None else classifier_for(provider)
        self._provider = provider
        self._name = name
        self._clock = clock if clock is not None else _utc_now
        period_id(self._clock_now(), anchor_day)  # refuses an anchor day outside 1 to 28
        self._anchor_day = anchor_day
        _check_settings(
            min_headroom=min_headroom,
            exhaustion_wait=rate_limit_exhaustion_wait,
            refresh=refresh_interval,
            margin=reservation_margin,
            sweep=sweep_interval,
        )
        self._min_headroom = min_headroom
        self._exhaustion_wait = rate_limit_exhaustion_wait
        self._refresh = refresh_interval
        self._margin = reservation_margin
        self._sweep_interval = sweep_interval
        self._specs = _checked_keys(keys)
        configured = [_Key(spec, self._fingerprint(spec.secret)) for spec in self._specs]
        ordered = sorted(enumerate(configured), key=lambda item: (_rank(item[1].spec.priority), item[0]))
        self._keys: tuple[_Key, ...] = tuple(key for _, key in ordered)
        self._by_label = {key.label: key for key in self._keys}
        self._store: KeyPoolStore = store if store is not None else InMemoryKeyPoolStore()
        self._breakers = breaker_registry if breaker_registry is not None else CircuitBreakerRegistry()
        self._policy: SelectionPolicy = selection if selection is not None else StickySelection()
        self._counters = KeyPoolCounters()
        self._observer = observer
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._record_name = f"key-pool/{provider}/{name}"
        self._view: _View | None = None
        self._generation: dict[str, int] = {}
        self._last_sweep: datetime | None = None
        self._no_key_lock = threading.Lock()
        self._no_key_logged_at: datetime | None = None
        self._no_key_suppressed = 0
        self._no_key_logged_states: tuple[tuple[str, KeyState], ...] = ()

    def __repr__(self) -> str:
        """The pool by provider, name and labels, never a secret."""
        return f"KeyPool(provider={self._provider!r}, name={self._name!r}, labels={self.labels!r})"

    # -- what the pool is ------------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """The provider every key is for.

        :return: the provider's name
        :rtype: str
        """
        return self._provider

    @property
    def name(self) -> str:
        """The pool's name.

        :return: the name
        :rtype: str
        """
        return self._name

    @property
    def keys(self) -> tuple[KeySpec, ...]:
        """The keys, in configuration order.

        :return: the key specs
        :rtype: tuple[KeySpec, ...]
        """
        return self._specs

    @property
    def labels(self) -> tuple[str, ...]:
        """The keys' labels, in preference order: by ``(priority, configuration order)``.

        :return: the labels
        :rtype: tuple[str, ...]
        """
        return tuple(key.label for key in self._keys)

    @property
    def breaker_registry(self) -> CircuitBreakerRegistry:
        """The registry the per-key models' breakers live in, which the pool reads key health from.

        :return: the registry
        :rtype: CircuitBreakerRegistry
        """
        return self._breakers

    @property
    def counters(self) -> KeyPoolCounters:
        """The pool's event counts: moves, failovers, refusals for room, orphans, store failures and
        ``NoKeyAvailableError`` raises.

        :return: the counters
        :rtype: KeyPoolCounters
        """
        return self._counters

    def classify(self, exc: BaseException, *, stream_started: bool = False) -> Verdict:
        """What a failed attempt says about the key it ran on, by the pool's classifier and clock.

        :param exc: what the attempt raised
        :ptype exc: BaseException
        :param stream_started: whether the attempt had begun receiving the provider's answer
        :ptype stream_started: bool
        :return: the verdict, for :meth:`KeyPoolLease.fail`
        :rtype: Verdict
        """
        return self._classifier.classify(
            exc, now=self._clock_now(), stream_started=stream_started, exhaustion_wait=self._exhaustion_wait
        )

    def refuses_retry(self, status: int, headers: Mapping[str, str], body: bytes | None) -> bool:
        """Whether a provider's SDK must raise a response at once instead of retrying it on the same key.

        The rule a key's model hands its SDK hook (``unretried_key_errors``): a response that is the
        key's -- revoked, out of credit, forbidden, a quota a 429 reports, a rate limit whose wait runs
        past ``rate_limit_exhaustion_wait`` -- reaches the pool on its first occurrence, so the call
        moves to the next key. A rate limit with a short wait stays the SDK's to retry. It reads a
        response as :meth:`classify` reads the error raised from it.

        :param status: the response's HTTP status
        :ptype status: int
        :param headers: the response's headers
        :ptype headers: Mapping[str, str]
        :param body: the response's body for a 429, ``None`` for any other status
        :ptype body: bytes | None
        :return: ``True`` to raise the response at once
        :rtype: bool
        """
        return self._classifier.refuses_retry(
            status, headers, body, now=self._clock_now(), exhaustion_wait=self._exhaustion_wait
        )

    async def wait_to_retry(self, verdict: Verdict, retries: int) -> bool:
        """Wait out a rate limit the provider's SDK doesn't retry; whether to retry the call on the same key.

        A rate limit means retry (owner, 2026-11-26). Where the provider's SDK retries one itself, it
        already has, so this returns ``False`` at once. Otherwise the call is retried on the same key up
        to two times, each after the provider's ``retry-after`` (never more than
        ``rate_limit_exhaustion_wait``), or a short backoff when it named none; then this returns
        ``False`` and the rate limit reaches the caller. A retry is not a failover, and nothing about
        the key changes.

        :param verdict: the verdict on the failed attempt; anything but a rate limit returns ``False``
        :ptype verdict: Verdict
        :param retries: how many times the call has already been retried this way
        :ptype retries: int
        :return: ``True``, after the wait, to run the call again on the same key
        :rtype: bool
        :raises ValueError: when ``retries`` is negative
        """
        if retries < 0:
            raise ValueError(f"retries must not be negative, got {retries}")
        if (
            verdict.kind is not VerdictKind.RATE_LIMITED
            or self._classifier.sdk_retries_rate_limits
            or retries >= _RATE_LIMIT_RETRIES
        ):
            return False
        if verdict.until is not None:
            pause = (verdict.until - self._clock_now()).total_seconds()
        else:
            pause = _RATE_LIMIT_BACKOFF_SECONDS * 2**retries
        await self._sleep(min(max(pause, 0.0), self._exhaustion_wait.total_seconds()))
        return True

    def record_failover(self, reason: MoveReason) -> None:
        """Count one call that a key refused running its next attempt on another key (KP-K1).

        The pooled chat model reports each of its failovers here; a caller that closes leases itself
        reports its own. The failover is counted in :attr:`counters` and reported to the observer.

        :param reason: why the key refused the call: a key-moving verdict's kind, or ``tripped`` for a
            breaker that refused it after the pool chose the key
        :ptype reason: MoveReason
        :raises TypeError: when ``reason`` is not a :class:`MoveReason`
        """
        if not isinstance(reason, MoveReason):
            raise TypeError(f"a failover's reason is a MoveReason, got {type(reason).__name__}")
        self._notify(lambda observer: observer.on_failover(self._name, reason))

    # -- calls -----------------------------------------------------------------------------------

    async def acquire(
        self,
        model_id: str | None,
        amount: Decimal,
        *,
        tried: Set[str] = frozenset(),
        deadline: datetime | None = None,
    ) -> KeyPoolLease:
        """Reserve ``amount`` for one call on the key it should run on.

        The call starts on the active key. A key the call has tried, or whose room the reservation
        doesn't fit, is skipped for this call only; a key that can't serve at all moves the active key
        first (KP-D2, D3). When no key can serve, it raises, before any network call.

        :param model_id: the model the call is for; its per-model states (forbidden, the per-model
            breaker) count. ``None`` counts only the key's own
        :ptype model_id: str | None
        :param amount: the money to reserve: the call's estimate (``reservation_estimate``)
        :ptype amount: Decimal
        :param tried: keys this call has already tried, which it doesn't try again (KP-H2)
        :ptype tried: Set[str]
        :param deadline: when the call gives up, timezone-aware; the reservation expires a margin after
        :ptype deadline: datetime | None
        :return: the lease, to close with its result
        :rtype: KeyPoolLease
        :raises NoKeyAvailableError: when no key can serve the call
        :raises KeyPoolStoreError: when the store failed; the call fails closed
        :raises ValueError: when ``amount``, ``model_id`` or ``deadline`` is invalid
        """
        _check_amount("amount", amount)
        if model_id is not None and (not isinstance(model_id, str) or not model_id):
            raise ValueError(f"model_id must be a model id or None, got {model_id!r}")
        if deadline is not None and (deadline.tzinfo is None or deadline.utcoffset() is None):
            raise ValueError(f"deadline must be timezone-aware, got the naive {deadline.isoformat()}")
        now = self._clock_now()
        await self._ready(now)
        await self._sweep_if_due(now)
        skipped: set[str] = set()
        while True:
            now = self._clock_now()
            view = await self._ready(now)
            view = await self._roll_over_if_due(view, now)
            view = await self._keep_active_eligible(view, model_id, now)
            blocks = self._all_blocks(view.doc, model_id, now)
            label = self._policy.serving_key(self._candidates(blocks), _active(view.doc), set(tried) | skipped)
            if label is None:
                raise await self._no_key_available(view, model_id, blocks, tried, skipped, now)
            lease = await self._reserve_on(self._by_label[label], model_id, amount, now, deadline)
            if lease is not None:
                return lease
            skipped.add(label)
            await self._refused_for_room(self._by_label[label], model_id, now)

    async def reserve(self, amount: Decimal, *, deadline: datetime | None = None) -> KeyPoolLease:
        """Reserve ``amount`` on the active key, for a caller that isn't a chat model (KP-E6).

        A batch client, or any caller with its own transport, reserves its estimate here, sends with
        the lease's secret, and closes the lease with :meth:`KeyPoolLease.settle`,
        :meth:`KeyPoolLease.release` or :meth:`KeyPoolLease.fail` (classified by :meth:`classify`).
        Only the keys' own states count; no model's.

        :param amount: the money to reserve
        :ptype amount: Decimal
        :param deadline: when the work gives up, timezone-aware; the reservation expires a margin after
        :ptype deadline: datetime | None
        :return: the lease
        :rtype: KeyPoolLease
        :raises NoKeyAvailableError: when no key can take the reservation
        :raises KeyPoolStoreError: when the store failed
        :raises ValueError: when ``amount`` or ``deadline`` is invalid
        """
        return await self.acquire(None, amount, deadline=deadline)

    # -- operators ------------------------------------------------------------------------------

    async def status(self) -> KeyPoolStatus:
        """Every key's state, until when, and money, and the active key (KP-K2); no secrets.

        Reads the store afresh, so a health endpoint sees what every process sees.

        :return: the snapshot
        :rtype: KeyPoolStatus
        :raises KeyPoolStoreError: when the store failed
        """
        now = self._clock_now()
        await self._ready(now)
        view = await self._read(now)
        snapshots = await self._snapshots(now)
        period = period_id(now, self._anchor_day)
        _, rollover = period_bounds(now, self._anchor_day)
        active = _active(view.doc)
        keys: list[KeyStatus] = []
        for key in self._keys:
            blocks = self._blocks(view.doc, key, None, now)
            spent, reserved, remaining = _money_of(snapshots[self._account(key, period)], key.spec.budget)
            state, until = (blocks[0].state, blocks[0].until) if blocks else (KeyState.OK, None)
            if not blocks and remaining < self._headroom(key):
                state, until = KeyState.NO_ROOM, rollover
            generation = self._generation_of(view.doc, key) or {}
            keys.append(
                KeyStatus(
                    label=key.label,
                    state=state,
                    until=until,
                    spent=spent,
                    reserved=reserved,
                    remaining=remaining,
                    budget=key.spec.budget,
                    active=key.label == active,
                    forbidden_models=tuple(m for m in generation.get("forbidden", []) if m != _WHOLE_KEY),
                )
            )
        return KeyPoolStatus(pool=self._name, provider=self._provider, period=period, active=active, keys=tuple(keys))

    async def reset(self, label: str) -> None:
        """Clear a key's stored state, as an operator does once its cause is fixed.

        Clears exhausted, revoked, forbidden and low room for every secret the label has held, and resets the key's breakers in this process. Spend is untouched: it is the ledger's.
        With distinct priorities, a reset key the pool prefers takes over again (KP-D5), as does a reset
        key when the active key can't serve; that move's reason is ``operator_reset``, logged at INFO,
        not the active key's state, so an operator's reset never reads as a new exhaustion.

        :param label: the key to reset
        :ptype label: str
        :raises ValueError: when the pool has no key with that label
        :raises KeyPoolStoreError: when the store failed
        """
        key = self._by_label.get(label)
        if key is None:
            raise ValueError(f"key pool {self._name!r} has no key labelled {label!r}; its keys: {self.labels}")
        now = self._clock_now()
        await self._ready(now)
        self._breakers.reset(self._provider, credential=key.spec.secret.reveal())

        def clear(doc: dict[str, Any]) -> bool:
            entry = doc.get("keys", {}).get(label, {})
            for generation in entry.get("generations", {}).values():
                generation.update(_cleared_state())
            return False

        await self._commit(now, None, clear, reset=label)
        logger.info(
            "key pool %r: an operator reset key %r",
            self._name,
            label,
            extra={"extra_data": {"pool": self._name, "label": label}},
        )

    # -- the lease's half -----------------------------------------------------------------------

    async def _close(self, account: str, reservation: str, cost: Decimal | None) -> None:
        """Settle a lease's reservation at ``cost``, or release it when ``cost`` is ``None``."""
        if cost is None:
            await self._call("release", _CLOSE_CONTRACT, lambda: self._store.release(account, reservation))
        else:
            await self._call("settle", _CLOSE_CONTRACT, lambda: self._store.settle(account, reservation, cost))

    async def _record_verdict(self, key: _Key, model_id: str | None, verdict: Verdict) -> None:
        """Record a key-moving verdict on its keys, and move the active key if it can't serve.

        A key this makes exhausted, revoked or forbidden -- the states that need an operator -- is
        logged at WARNING when no move off it follows (the last key that could serve, or a key that
        wasn't active), so such a change is never left unlogged for want of a move (KP-D7).
        """
        now = self._clock_now()
        active_before = _active((await self._ready(now)).doc)
        if verdict.scope is VerdictScope.ACCOUNT:
            targets = [other for other in self._keys if other.spec.account_tag == key.spec.account_tag]
        else:
            targets = [key]
        scope = model_id if verdict.scope is VerdictScope.KEY_AND_MODEL and model_id is not None else _WHOLE_KEY
        period = period_id(now, self._anchor_day)
        until = verdict.until.isoformat() if verdict.until is not None else None
        changed: list[str] = []

        def record(doc: dict[str, Any]) -> bool:
            changed.clear()
            for target in targets:
                generation = self._writable_generation(doc, target)
                previous = copy.deepcopy(generation)
                if verdict.kind is VerdictKind.EXHAUSTED:
                    generation["exhausted"] = {"period": period, "until": until}
                elif verdict.kind is VerdictKind.REVOKED:
                    generation["revoked"] = True
                elif verdict.kind is VerdictKind.FORBIDDEN:
                    generation["forbidden"] = sorted({*generation["forbidden"], scope})
                if generation != previous:
                    changed.append(target.label)
            return False

        view = await self._commit(now, model_id, record)
        reason = _VERDICT_REASON[verdict.kind]
        if reason not in _WARNING_REASONS:
            return
        active = _active(view.doc)
        for label in changed:
            if label == active_before and active != active_before:
                continue  # the move off it was logged, with this reason
            logger.warning(
                "key pool %r: key %r is now %s; the active key is %r",
                self._name,
                label,
                reason,
                active,
                extra={"extra_data": {"pool": self._name, "label": label, "state": str(reason), "active": active}},
            )

    # -- selection -------------------------------------------------------------------------------

    async def _roll_over_if_due(self, view: _View, now: datetime) -> _View:
        """Start the pool over on its most-preferred key when the billing period has rolled over."""
        period = period_id(now, self._anchor_day)
        if view.doc.get("period") == period:
            return view

        def roll_over(doc: dict[str, Any]) -> bool:
            if doc.get("period") == period:
                return False
            doc["period"] = period
            return True

        return await self._commit(now, None, roll_over)

    async def _keep_active_eligible(self, view: _View, model_id: str | None, now: datetime) -> _View:
        """Move the active key when the policy says to, from the cached record: no store call when it stays."""
        current = _active(view.doc)
        target, _ = self._target(view.doc, model_id, now, restart=False)
        if target is None or target == current:
            return view
        return await self._commit(now, model_id, _no_change)

    def _target(
        self, doc: dict[str, Any], model_id: str | None, now: datetime, *, restart: bool
    ) -> tuple[str | None, MoveReason]:
        """The key the policy would hold active in ``doc``, and why it would move there."""
        blocks = self._all_blocks(doc, model_id, now)
        active = _active(doc)
        current = None if restart or active not in self._by_label else active
        target = self._policy.active_key(self._candidates(blocks), current)
        if restart:
            reason = MoveReason.PERIOD_ROLLOVER
        elif active is not None and active not in self._by_label:
            reason = MoveReason.NOT_CONFIGURED
        elif active is not None and blocks[active]:
            reason = _MOVE_REASON[blocks[active][0].state]
        else:
            reason = MoveReason.PULL_BACK
        return target, reason

    def _candidates(self, blocks: dict[str, list[_Block]]) -> list[KeyCandidate]:
        return [KeyCandidate(key.label, key.spec.priority, eligible=not blocks[key.label]) for key in self._keys]

    def _all_blocks(self, doc: dict[str, Any], model_id: str | None, now: datetime) -> dict[str, list[_Block]]:
        return {key.label: self._blocks(doc, key, model_id, now) for key in self._keys}

    def _blocks(self, doc: dict[str, Any], key: _Key, model_id: str | None, now: datetime) -> list[_Block]:
        """Every reason ``key`` can't serve ``model_id`` now, worst first; empty when it can."""
        generation = self._generation_of(doc, key) or {}
        period = period_id(now, self._anchor_day)
        _, rollover = period_bounds(now, self._anchor_day)
        scopes = (_WHOLE_KEY, model_id) if model_id is not None else (_WHOLE_KEY,)
        blocks: list[_Block] = []
        if generation.get("revoked"):
            blocks.append(_Block(KeyState.REVOKED, None))
        exhausted_until = _exhausted_until(generation, period, rollover)
        if exhausted_until is not None and exhausted_until > now:
            blocks.append(_Block(KeyState.EXHAUSTED, exhausted_until))
        if any(scope in generation.get("forbidden", []) for scope in scopes):
            blocks.append(_Block(KeyState.FORBIDDEN, None))
        wait = self._breaker_wait(key)
        if wait > 0:
            blocks.append(_Block(KeyState.TRIPPED, now + timedelta(seconds=wait)))
        if generation.get("low_room") == period:
            blocks.append(_Block(KeyState.NO_ROOM, rollover))
        return blocks

    def _breaker_wait(self, key: _Key) -> float:
        """Seconds until the key's credential breaker admits a call, read with no side effects (KP-C6).

        The per-model rate-limit breaker is not read: a rate limit never moves the active key, and every
        key shares the same rates, so it fast-fails a call on the key that serves it and nothing more.
        """
        breaker = self._breakers.peek(self._provider, credential=key.spec.secret.reveal())
        return breaker.seconds_until_admit() if breaker is not None else 0.0

    # -- reservations ----------------------------------------------------------------------------

    async def _reserve_on(
        self, key: _Key, model_id: str | None, amount: Decimal, now: datetime, deadline: datetime | None
    ) -> KeyPoolLease | None:
        """Reserve ``amount`` on ``key``: its lease, or ``None`` when it doesn't fit."""
        period = period_id(now, self._anchor_day)
        account = self._account(key, period)
        reservation = uuid7().hex
        expires_at = max(deadline, now) + self._margin if deadline is not None else now + self._margin
        held = await self._call(
            "reserve",
            _RESERVE_CONTRACT,
            lambda: self._store.reserve(account, reservation, amount, key.spec.budget, expires_at),
        )
        if not held:
            return None
        return KeyPoolLease(
            label=key.label,
            secret=key.spec.secret,
            account=key.spec.account,
            model_id=model_id,
            amount=amount,
            reservation_id=reservation,
            expires_at=expires_at,
            period=period,
            ledger_account=account,
            close=lambda cost: self._close(account, reservation, cost),
            record=lambda verdict: self._record_verdict(key, model_id, verdict),
        )

    async def _refused_for_room(self, key: _Key, model_id: str | None, now: datetime) -> None:
        """A reservation didn't fit on ``key``: skip it for this call, or, if its room is below
        ``min_headroom``, mark it low and move off it (KP-D3)."""
        self._notify(lambda observer: observer.on_room_refusal(self._name, key.label))
        period = period_id(now, self._anchor_day)
        account = self._account(key, period)
        snapshot = await self._call("snapshot", _NO_CONTRACT, lambda: self._store.snapshot([account]))
        _, _, remaining = _money_of(snapshot[account], key.spec.budget)
        if remaining >= self._headroom(key):
            return

        def mark_low(doc: dict[str, Any]) -> bool:
            self._writable_generation(doc, key)["low_room"] = period
            return False

        await self._commit(now, model_id, mark_low)

    async def _sweep_if_due(self, now: datetime) -> None:
        """Settle expired reservations on the pool's accounts, at most once per ``sweep_interval``."""
        if self._last_sweep is not None and now - self._last_sweep < self._sweep_interval:
            return
        self._last_sweep = now
        start, _ = period_bounds(now, self._anchor_day)
        periods = (period_id(now, self._anchor_day), period_id(start - timedelta(microseconds=1), self._anchor_day))
        accounts = {self._account(key, period): key.label for key in self._keys for period in periods}
        swept = await self._call(
            "sweep_expired", _RESERVE_CONTRACT, lambda: self._store.sweep_expired(list(accounts), now)
        )
        for account, reservations in swept.items():
            label = accounts.get(account)
            if label is None or not reservations:
                continue
            count = len(reservations)
            self._notify(lambda observer: observer.on_orphaned(self._name, label, count))
            logger.warning(
                "key pool %r: settled %d orphaned reservation(s) on key %r at their full amount",
                self._name,
                count,
                label,
                extra={"extra_data": {"pool": self._name, "label": label, "orphaned": count}},
            )

    # -- the record ------------------------------------------------------------------------------

    async def _ready(self, now: datetime) -> _View:
        """The cached record, registering this process's keys on first use and refreshing a stale one."""
        if self._view is None:
            await self._register(now)
        elif now - self._view.read_at >= self._refresh:
            await self._read(now)
        return self._require_view()

    async def _read(self, now: datetime) -> _View:
        version, doc = await self._call("read_record", _NO_CONTRACT, lambda: self._store.read_record(self._record_name))
        self._view = _View(version, _checked_doc(doc), now)
        return self._view

    async def _register(self, now: datetime) -> None:
        """Record this process's fingerprints and budgets, warning where a budget differs (KP-A5) and
        noting a label that holds a new secret (KP-B2)."""
        view = await self._read(now)
        stored_keys = view.doc.get("keys", {})
        for key in self._keys:
            entry = stored_keys.get(key.label)
            if entry is None:
                continue
            stored_budget = entry.get("budget")
            if stored_budget is not None and Decimal(stored_budget) != key.spec.budget:
                logger.warning(
                    "key pool %r: key %r was last configured with a budget of %s USD; this process configures %s",
                    self._name,
                    key.label,
                    stored_budget,
                    key.spec.budget,
                    extra={
                        "extra_data": {
                            "pool": self._name,
                            "label": key.label,
                            "stored_budget": stored_budget,
                            "configured_budget": str(key.spec.budget),
                        }
                    },
                )
            if entry.get("generations") and self._generation_of(view.doc, key) is None:
                logger.info(
                    "key pool %r: key %r holds a new secret; it starts from fresh state and zero spend this period",
                    self._name,
                    key.label,
                    extra={"extra_data": {"pool": self._name, "label": key.label}},
                )
        period = period_id(now, self._anchor_day)

        def register(doc: dict[str, Any]) -> bool:
            for key in self._keys:
                entry = doc["keys"].setdefault(key.label, {"budget": None, "generations": {}})
                entry["budget"] = str(key.spec.budget)
                self._writable_generation(doc, key)
            if doc.get("period") is None:
                doc["period"] = period
            return False

        written = await self._commit(now, None, register)
        for key in self._keys:
            self._generation[key.label] = self._generation_number(written.doc, key)

    async def _commit(self, now: datetime, model_id: str | None, edit: _Edit, *, reset: str | None = None) -> _View:
        """Apply ``edit`` to the record, let the policy settle the active key, and compare-and-set it.

        On a conflict the winner's document is adopted and the edit applied to it again, so a loser
        moving the same key from the same place finds it already moved and writes nothing (KP-D4). A
        move first reads every key's room, so it never lands on a key below ``min_headroom``; that read
        also refreshes each key's low-room mark. A move onto ``reset``, the key an operator just reset,
        is that reset's doing, and carries its reason.
        """
        view = self._require_view()
        rooms: dict[str, Decimal] | None = None
        for _ in range(_MAX_CAS_ROUNDS):
            doc, before, target, reason = self._applied(view.doc, edit, model_id, now, rooms)
            if target is not None and target != before and rooms is None:
                rooms = await self._rooms(now)
                doc, before, target, reason = self._applied(view.doc, edit, model_id, now, rooms)
            if target is not None:
                doc["active"] = target
                if target == reset:
                    reason = MoveReason.OPERATOR_RESET
            if doc == view.doc:
                return view
            written, version, current = await self._call(
                "compare_and_set_record",
                _RESERVE_CONTRACT,
                lambda: self._store.compare_and_set_record(self._record_name, view.version, doc),
            )
            view = _View(version, _checked_doc(current), now)
            self._view = view
            if written:
                if before is not None and target is not None and target != before:
                    self._moved(before, target, reason)
                return view
        self._notify(lambda observer: observer.on_store_failure(self._name, "compare_and_set_record"))
        logger.error(
            "key pool %r: lost %d compare-and-set rounds on its record; the call fails closed",
            self._name,
            _MAX_CAS_ROUNDS,
            extra={"extra_data": {"pool": self._name, "operation": "compare_and_set_record"}},
        )
        raise KeyPoolStoreError(
            f"key pool {self._name!r}: lost {_MAX_CAS_ROUNDS} compare-and-set rounds on its record; the call fails closed"
        )

    def _applied(
        self,
        base: dict[str, Any],
        edit: _Edit,
        model_id: str | None,
        now: datetime,
        rooms: dict[str, Decimal] | None,
    ) -> tuple[dict[str, Any], str | None, str | None, MoveReason]:
        """``base`` with ``edit`` applied (and room marks, once read), its active key before the policy,
        and the policy's target and reason."""
        doc = copy.deepcopy(base)
        restart = edit(doc)
        if rooms is not None:
            self._mark_rooms(doc, rooms, now)
        target, reason = self._target(doc, model_id, now, restart=restart)
        return doc, _active(doc), target, reason

    def _mark_rooms(self, doc: dict[str, Any], rooms: dict[str, Decimal], now: datetime) -> None:
        period = period_id(now, self._anchor_day)
        for key in self._keys:
            generation = self._writable_generation(doc, key)
            if rooms[key.label] < self._headroom(key):
                generation["low_room"] = period
            elif generation["low_room"] is not None:
                generation["low_room"] = None

    def _moved(self, before: str, target: str, reason: MoveReason) -> None:
        level = logging.WARNING if reason in _WARNING_REASONS else logging.INFO
        logger.log(
            level,
            "key pool %r moved its active key from %r to %r: %s",
            self._name,
            before,
            target,
            reason,
            extra={"extra_data": {"pool": self._name, "from": before, "to": target, "reason": str(reason)}},
        )
        self._notify(lambda observer: observer.on_move(self._name, before, target, reason))

    def _require_view(self) -> _View:
        if self._view is None:
            raise RuntimeError(f"key pool {self._name!r} used its record before reading it")
        return self._view

    def _generation_of(self, doc: dict[str, Any], key: _Key) -> dict[str, Any] | None:
        """The record's state for ``key``'s secret, ``None`` when the record holds none."""
        generations = doc.get("keys", {}).get(key.label, {}).get("generations", {})
        return next((g for g in generations.values() if g.get("fingerprint") == key.fingerprint), None)

    def _generation_number(self, doc: dict[str, Any], key: _Key) -> int:
        generations = doc.get("keys", {}).get(key.label, {}).get("generations", {})
        for number, generation in generations.items():
            if generation.get("fingerprint") == key.fingerprint:
                return int(number)
        raise KeyPoolStoreError(f"key pool {self._name!r}: its record lost key {key.label!r} as it registered it")

    def _writable_generation(self, doc: dict[str, Any], key: _Key) -> dict[str, Any]:
        """The record's state for ``key``'s secret, added as a new generation when the label has none
        for it, the oldest beyond the kept number dropped."""
        existing = self._generation_of(doc, key)
        if existing is not None:
            return existing
        entry = doc["keys"].setdefault(key.label, {"budget": str(key.spec.budget), "generations": {}})
        generations: dict[str, Any] = entry["generations"]
        number = self._generation.get(key.label)
        if number is None:
            number = max((int(n) for n in generations), default=-1) + 1
        generation = {"fingerprint": key.fingerprint, **_cleared_state()}
        generations[str(number)] = generation
        for stale in sorted((int(n) for n in generations), reverse=True)[_GENERATIONS_KEPT:]:
            if stale != number:
                del generations[str(stale)]
        return generation

    # -- money -----------------------------------------------------------------------------------

    def _account(self, key: _Key, period: str) -> str:
        generation = self._generation.get(key.label, 0)
        return f"{self._provider}/{self._name}/{key.label}/{generation}/{period}"

    def _headroom(self, key: _Key) -> Decimal:
        return key.spec.budget * self._min_headroom

    async def _snapshots(self, now: datetime) -> dict[str, AccountSnapshot]:
        period = period_id(now, self._anchor_day)
        accounts = [self._account(key, period) for key in self._keys]
        return await self._call("snapshot", _NO_CONTRACT, lambda: self._store.snapshot(accounts))

    async def _rooms(self, now: datetime) -> dict[str, Decimal]:
        period = period_id(now, self._anchor_day)
        snapshots = await self._snapshots(now)
        return {key.label: _money_of(snapshots[self._account(key, period)], key.spec.budget)[2] for key in self._keys}

    # -- failure ---------------------------------------------------------------------------------

    async def _no_key_available(
        self,
        view: _View,
        model_id: str | None,
        blocks: dict[str, list[_Block]],
        tried: Set[str],
        skipped: Set[str],
        now: datetime,
    ) -> NoKeyAvailableError:
        """The error for a call no key can serve, logged (deduplicated) and counted (KP-I1, I2)."""
        snapshots = await self._snapshots(now)
        period = period_id(now, self._anchor_day)
        _, rollover = period_bounds(now, self._anchor_day)
        active = _active(view.doc)
        statuses: list[KeyStatus] = []
        returns: list[datetime] = []
        for key in self._keys:
            reasons = blocks[key.label]
            spent, reserved, remaining = _money_of(snapshots[self._account(key, period)], key.spec.budget)
            if reasons:
                state, until = reasons[0].state, reasons[0].until
                ends = [block.until for block in reasons]
                back = max(end for end in ends if end is not None) if all(end is not None for end in ends) else None
            elif key.label in skipped:
                state, until, back = KeyState.NO_ROOM, rollover, rollover
            else:
                state, until, back = KeyState.TRIED, None, None
            if back is not None:
                returns.append(back)
            statuses.append(
                KeyStatus(
                    label=key.label,
                    state=state,
                    until=until,
                    spent=spent,
                    reserved=reserved,
                    remaining=remaining,
                    budget=key.spec.budget,
                    active=key.label == active,
                )
            )
        error = NoKeyAvailableError(
            pool=self._name,
            provider=self._provider,
            model_id=model_id,
            keys=tuple(statuses),
            earliest=min(returns, default=None),
        )
        self._report_no_key(error, now)
        return error

    def _report_no_key(self, error: NoKeyAvailableError, now: datetime) -> None:
        """Count the raise, and log it unless it repeats the keys' states the last line logged within
        the interval: a line whose states differ is always logged, so the newest line is never stale."""
        self._notify(lambda observer: observer.on_no_key_available(self._name))
        states = tuple((key.label, key.state) for key in error.keys)
        with self._no_key_lock:
            last = self._no_key_logged_at
            if last is not None and now - last < _NO_KEY_LOG_INTERVAL and states == self._no_key_logged_states:
                self._no_key_suppressed += 1
                return
            suppressed, self._no_key_suppressed = self._no_key_suppressed, 0
            self._no_key_logged_at = now
            self._no_key_logged_states = states
        logger.error(
            "key pool %r: %s (%d more not logged in the last %d s)",
            self._name,
            error.detail,
            suppressed,
            int(_NO_KEY_LOG_INTERVAL.total_seconds()),
            extra={
                "extra_data": {
                    "pool": self._name,
                    "model": error.model_id,
                    "suppressed": suppressed,
                    "states": {key.label: str(key.state) for key in error.keys},
                }
            },
        )

    async def _call[T](
        self, operation: str, contract: tuple[type[Exception], ...], call: Callable[[], Awaitable[T]]
    ) -> T:
        """Run one store call, failing closed on anything its contract doesn't name (KP-G5)."""
        try:
            return await call()
        except Exception as exc:
            if isinstance(exc, contract) and type(exc) not in _BARE_LOOKUPS:
                raise
            self._notify(lambda observer: observer.on_store_failure(self._name, operation))
            logger.error(
                "key pool %r: the store's %s failed (%s); the call fails closed",
                self._name,
                operation,
                type(exc).__name__,
                exc_info=exc,
                extra={"extra_data": {"pool": self._name, "operation": operation, "exc_type": type(exc).__name__}},
            )
            if isinstance(exc, KeyPoolStoreError):
                raise
            raise KeyPoolStoreError(
                f"key pool {self._name!r}: the store's {operation} failed ({type(exc).__name__}); the call fails closed"
            ) from exc

    def _notify(self, event: Callable[[KeyPoolObserver], None]) -> None:
        event(self._counters)
        if self._observer is not None:
            with contained("key pool observer failed", log=logger, extra={"pool": self._name}):
                event(self._observer)

    # -- helpers ---------------------------------------------------------------------------------

    def _clock_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(f"key pool {self._name!r}: its clock returned the naive {now.isoformat()}")
        return now.astimezone(UTC)

    def _fingerprint(self, secret: Secret) -> str:
        """A stable, non-reversible fingerprint of ``secret``, keyed by this pool (KP-B2)."""
        salt = f"threetears.models.key_pool/{self._provider}/{self._name}".encode()
        return hmac.new(salt, secret.reveal().encode(), hashlib.sha256).hexdigest()[:16]


def _checked_keys(keys: Sequence[KeySpec]) -> tuple[KeySpec, ...]:
    """The pool's keys, refusing what can't be pooled (KP-A1)."""
    specs = tuple(keys)
    if not specs:
        raise ValueError("a key pool needs at least one key")
    labels: set[str] = set()
    values: list[str] = []
    for spec in specs:
        if not isinstance(spec, KeySpec):
            raise ValueError(f"a key pool's keys must be KeySpec, got {type(spec).__name__}")
        if spec.label in labels:
            raise ValueError(f"key label {spec.label!r} appears twice; labels must be unique in a pool")
        labels.add(spec.label)
        value = spec.secret.reveal()
        if is_subscription_token(value):
            raise ValueError(
                f"key {spec.label!r} is a Claude subscription token; a pool budgets API keys, which have a per-call price"
            )
        if any(hmac.compare_digest(value.encode(), seen.encode()) for seen in values):
            raise ValueError(f"key {spec.label!r} holds the same secret as another key in the pool")
        values.append(value)
    return specs


def _check_settings(
    *,
    min_headroom: Decimal,
    exhaustion_wait: timedelta,
    refresh: timedelta,
    margin: timedelta,
    sweep: timedelta,
) -> None:
    if not isinstance(min_headroom, Decimal) or not min_headroom.is_finite() or not 0 <= min_headroom < 1:
        raise ValueError(
            f"min_headroom must be a Decimal fraction of the budget, 0 or more and below 1, got {min_headroom!r}"
        )
    positive = (
        ("rate_limit_exhaustion_wait", exhaustion_wait),
        ("refresh_interval", refresh),
        ("reservation_margin", margin),
    )
    for what, value in positive:
        if value <= timedelta(0):
            raise ValueError(f"{what} must be positive, got {value}")
    if sweep < timedelta(0):
        raise ValueError(f"sweep_interval must not be negative, got {sweep}")


def _check_amount(what: str, amount: Decimal) -> None:
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
        raise ValueError(f"{what} must be a finite Decimal, 0 or more, got {amount!r}")


def _rank(priority: int | None) -> float:
    return float("inf") if priority is None else float(priority)


def _active(doc: dict[str, Any]) -> str | None:
    active = doc.get("active")
    return active if isinstance(active, str) else None


def _no_change(doc: dict[str, Any]) -> bool:
    return False


def _cleared_state() -> dict[str, Any]:
    return {"revoked": False, "exhausted": None, "low_room": None, "forbidden": []}


def _exhausted_until(generation: dict[str, Any], period: str, rollover: datetime) -> datetime | None:
    """When a key's exhaustion ends: the reset the provider named, else the end of the period it was
    exhausted in; ``None`` when it isn't exhausted, or was in an earlier period with no reset named."""
    exhausted = generation.get("exhausted")
    if not isinstance(exhausted, dict):
        return None
    until = exhausted.get("until")
    if until is not None:
        return datetime.fromisoformat(until)
    return rollover if exhausted.get("period") == period else None


def _money_of(snapshot: AccountSnapshot, budget: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    spent, reserved = snapshot["spent"], snapshot["reserved"]
    return spent, reserved, max(budget - spent - reserved, Decimal(0))


def _checked_doc(doc: dict[str, Any] | None) -> dict[str, Any]:
    """The record as stored, or a new one; a record in another layout is refused, not overwritten."""
    if doc is None:
        return {"layout": _LAYOUT, "period": None, "active": None, "keys": {}}
    if doc.get("layout") != _LAYOUT or not isinstance(doc.get("keys"), dict):
        raise KeyPoolStoreError(f"a key pool record in layout {doc.get('layout')!r} can't be read by layout {_LAYOUT}")
    return doc
