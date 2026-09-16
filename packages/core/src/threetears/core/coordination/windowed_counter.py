"""generic windowed attempt counter -- a rate-limiter / throttle primitive.

Sibling to :class:`~threetears.core.coordination.replay_guard.ReplayGuard` and
:class:`~threetears.core.coordination.revocation.RevocationGuard`, but a different shape: those
answer "have I seen this exact key"; this answers "how many times has this key been attempted
inside a fixed window anchored at its first attempt" -- what a throttle needs ("no more than N
login attempts per IP per minute")::

    counter = WindowedCounter(registry, purpose="login_ip_throttle", window_seconds=60)
    attempts = await counter.record_attempt(ip_address)
    if attempts > 20:
        raise <throttled>

The window is anchored at the FIRST attempt in it and carried in the row, not refreshed per
write: a steady stream of attempts must not extend the window the caller asked for. The row
expires when its window closes, so a stale window reads as absent at every tier rather than as a
count the reader has to range-check.

**Where the count lives.** L2 (NATS KV) is the fence every replica compares against, and L3 is the
durable record behind it, written behind through the collection's write buffer: a broker wipe then
costs at most the increments since the last flush, which for a throttle is a few extra attempts
and not worth a database write per login. A deployment with no L3 (identity-edge, which holds no
database by design) still throttles across replicas on L2 alone.

**Fail-open versus fail-closed on a storage failure is the CALLER's choice**, through the
constructor's ``fail_open`` flag -- unlike ``ReplayGuard``/``RevocationGuard``, which are always
fail-closed. An edge-tier IP throttle is a best-effort first line in front of an authoritative
fail-closed check, so degrading it to "not currently throttling" during an outage is an acceptable
trade for availability. A counter with nothing behind it stays ``fail_open=False`` (the default).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.coordination.tables import (
    STORAGE_FAILURES,
    CoordinationCountersCollection,
    coordination_collection,
)
from threetears.core.exceptions import ConcurrentModificationError
from threetears.observe import get_logger

__all__ = ["WindowState", "WindowedCounter"]

log = get_logger(__name__)

#: compare-and-swap rounds this counter allows per increment. See the call site for why it is not
#: the framework default.
_MAX_CAS_ATTEMPTS: Final = 30

#: what a ``fail_open`` counter degrades on: storage being unreachable, plus losing every
#: compare-and-swap round.
#:
#: ``ConcurrentModificationError`` is here and NOT in :data:`STORAGE_FAILURES` because the counter
#: is the one primitive for which exhausted contention is an expected operating condition rather
#: than a signal. Its traffic shape is a burst against ONE key -- a credential-stuffing run against
#: one account -- which is exactly when every replica contends on that key's fence, and exactly
#: when the throttle matters most. Raising there would turn the attack into a 500 on the request
#: path of a counter whose posture promises the opposite.
#:
#: ``IdempotencyKeyStore`` deliberately reads the same exception as its own signal, so widening
#: the shared tuple would break it.
_DEGRADABLE_FAILURES: Final[tuple[type[BaseException], ...]] = (*STORAGE_FAILURES, ConcurrentModificationError)


@dataclass(frozen=True, slots=True)
class WindowState:
    """A key's live window: how many attempts, and when the window opened.

    ``window_start`` is exposed so a caller can report a truthful "retry after" -- the time
    remaining, not the whole window length.
    """

    count: int
    window_start: float


class WindowedCounter:
    """per-key attempt counter over a fixed window anchored at the first attempt.

    Generic: the caller supplies the key (already hashed if it carries anything sensitive -- this
    class hashes it again into a fixed-length form regardless, but does not otherwise interpret
    it) and decides what threshold makes a count "too many". This class only tracks the count.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        *,
        purpose: str,
        window_seconds: int,
        fail_open: bool = False,
        clock: Callable[[], float] = time.time,
        config: CoreConfig | None = None,
    ) -> None:
        """configure the counter over its registry's coordination tables.

        :param registry: the collection registry whose tiers this counter reads and writes. L2
            and L3 are both optional: with no L3 the count lives in L2 and is lost on a broker
            wipe; with no L2 it is per-process, which throttles one replica rather than the fleet
        :ptype registry: CollectionRegistry
        :param purpose: what this counter counts (``"login_ip_throttle"``), carried in the row key
            so counters over one table never collide. This is what the KV bucket name used to be
        :ptype purpose: str
        :param window_seconds: length of the fixed window in seconds, measured from the first
            attempt in it. MUST be positive: a non-positive window would mean the count never
            resets
        :ptype window_seconds: int
        :param fail_open: on a storage failure (L2 or L3) or exhausted compare-and-swap
            contention, whether to treat the key as NOT over any threshold (``True`` --
            :meth:`record_attempt` and :meth:`count` return ``0`` and a warning is logged) rather
            than propagating for the caller to deny (``False``, the default). Contention counts
            because a burst against one key is this counter's expected shape, not an anomaly --
            see :data:`_DEGRADABLE_FAILURES`
        :ptype fail_open: bool
        :param clock: the time source, injectable so a test can stand on the window boundary
        :ptype clock: Callable[[], float]
        :param config: core config forwarded when this process first builds the counters
            collection; defaults to the framework defaults, which its declared write policy
            overrides anyway
        :ptype config: CoreConfig | None
        :raises ValueError: when ``window_seconds`` is not positive, or ``purpose`` is empty
        """
        if window_seconds <= 0:
            raise ValueError(f"WindowedCounter window_seconds must be positive, got {window_seconds}")
        if not purpose.strip():
            raise ValueError("WindowedCounter purpose must be a non-empty name, e.g. 'login_ip_throttle'")
        self._purpose = purpose
        self._window = timedelta(seconds=window_seconds)
        self._fail_open = fail_open
        self._clock = clock
        self._collection = coordination_collection(registry, CoordinationCountersCollection, config)

    @property
    def purpose(self) -> str:
        """what this counter counts, carried in every row it writes.

        :return: the purpose
        :rtype: str
        """
        return self._purpose

    @property
    def clock(self) -> Callable[[], float]:
        """the time source this counter reads.

        Exposed so an adapter computing a "retry after" uses the SAME clock the window is measured
        with; two clocks in one verdict is two answers.

        :return: the configured clock
        :rtype: Callable[[], float]
        """
        return self._clock

    @property
    def fail_open(self) -> bool:
        """whether this counter fails open (``True``) or closed (``False``) on a storage failure.

        :return: the configured posture
        :rtype: bool
        """
        return self._fail_open

    async def record_attempt(self, key: str) -> int:
        """record one attempt for ``key``; return the attempt count within its live window.

        Starts a fresh window (count 1) when the previous one has closed. The increment is a
        compare-and-swap against L2, so concurrent callers on one key never lose an increment
        silently.

        :param key: the identifier to count (an IP address, an already-hashed compound key).
            Hashed into a fixed-length form before storage, so the raw identifier is never a key
        :ptype key: str
        :return: the attempt count within the live window (``>= 1``), or ``0`` when this counter
            is ``fail_open`` and storage failed or every compare-and-swap round was lost
        :rtype: int
        :raises threetears.nats.KvError: on an L2 failure, when ``fail_open=False``
        :raises threetears.core.exceptions.ConcurrentModificationError: when every
            compare-and-swap round was lost, and ``fail_open=False``
        """
        try:
            return await self._record_attempt(key)
        except _DEGRADABLE_FAILURES as exc:
            if self._fail_open:
                log.warning(
                    "windowed counter failing open; storage failed or every compare-and-swap round was lost",
                    extra={"extra_data": {"purpose": self._purpose, "error": f"{type(exc).__name__}: {exc}"}},
                )
                return 0
            raise

    async def count(self, key: str) -> int:
        """the current attempt count for ``key`` within its live window, recording nothing.

        :param key: the identifier to look up
        :ptype key: str
        :return: the live count, 0 when absent, expired, or (when ``fail_open``) storage failed
        :rtype: int
        :raises threetears.core.exceptions.DataLayerUnavailableError: on an L3 failure, when
            ``fail_open=False``. An L2 failure degrades: the read falls through to L3
        """
        state = await self.state(key)
        return 0 if state is None else state.count

    async def state(self, key: str) -> WindowState | None:
        """the live window for ``key``, or ``None`` when absent or closed.

        Unlike :meth:`count`, this exposes ``window_start`` too, so a caller reporting a
        ``Retry-After`` can say how long is actually left rather than restating the window.

        :param key: the identifier to look up
        :ptype key: str
        :return: the live window state, or ``None``
        :rtype: WindowState | None
        :raises threetears.core.exceptions.DataLayerUnavailableError: on an L3 failure, when
            ``fail_open=False``. An L2 failure degrades: the read falls through to L3
        """
        try:
            entity = await self._collection.get(self._row_id(key))
        except _DEGRADABLE_FAILURES as exc:
            if self._fail_open:
                log.warning(
                    "windowed counter failing open; storage failed or every compare-and-swap round was lost",
                    extra={"extra_data": {"purpose": self._purpose, "error": f"{type(exc).__name__}: {exc}"}},
                )
                return None
            raise
        if entity is None:
            return None
        row = entity.to_dict()
        window_start = _as_epoch(row["window_start"])
        if window_start + self._window.total_seconds() < self._clock():
            # closed on this counter's clock, whatever the row's wall-clock expiry says.
            return None
        return WindowState(count=int(row["count"]), window_start=window_start)

    async def clear(self, key: str) -> None:
        """drop ``key``'s counter entirely -- the "authentication succeeded" reset.

        Always fails open, whatever the configured posture: failing to clear leaves a counter that
        can only ever deny too much, and raising here would turn a successful login into an error.

        :param key: the identifier to reset
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        try:
            await self._collection.delete(self._row_id(key))
        except _DEGRADABLE_FAILURES as exc:
            log.warning(
                "windowed counter could not clear a counter; it expires with its window",
                extra={"extra_data": {"purpose": self._purpose, "error": f"{type(exc).__name__}: {exc}"}},
            )

    async def is_over_threshold(self, key: str, *, threshold: int) -> bool:
        """whether ``key``'s live count is at or above ``threshold``, recording nothing.

        :param key: the identifier to look up
        :ptype key: str
        :param threshold: the caller's chosen ceiling
        :ptype threshold: int
        :return: whether the key is currently over threshold
        :rtype: bool
        :raises threetears.core.exceptions.DataLayerUnavailableError: on an L3 failure, when
            ``fail_open=False``. An L2 failure degrades: the read falls through to L3
        """
        return await self.count(key) >= threshold

    async def _record_attempt(self, key: str) -> int:
        """increment the live window, or open a new one.

        :param key: the identifier to count
        :ptype key: str
        :return: the count within the live window
        :rtype: int
        """
        now = datetime.fromtimestamp(self._clock(), tz=UTC)
        row_id = self._row_id(key)

        def _increment(
            current: dict[str, Any] | None,
        ) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            # The window is judged here, from the row's own ``window_start`` and THIS counter's
            # clock. The row's ``expires_at`` says the same thing to the tiers below -- it is what
            # makes a closed window absent to a reader and sweepable -- but it is read against the
            # wall clock, so a caller that injected a clock (a test standing on the boundary, a
            # service whose time source is not `time.time`) would otherwise get two answers.
            fresh = {
                "purpose": self._purpose,
                "key": row_id[1],
                "count": 1,
                "window_start": now,
                "expires_at": now + self._window,
            }
            if current is None or _as_epoch(current["window_start"]) + self._window.total_seconds() < now.timestamp():
                return "upsert", fresh
            return "upsert", {**current, "count": int(current["count"]) + 1}

        # 30 rounds, not the default 8: the contention this counter sees is a burst against ONE
        # key -- a credential-stuffing run against one account -- and 8 raised a conflict on about
        # three quarters of a 25-connection integration run, where 30 raised none. A budget tuned
        # for the quiet path fails precisely when the counter is the control that matters.
        outcome = await self._collection.l2_cas_mutate(row_id, _increment, max_retries=_MAX_CAS_ATTEMPTS)
        return int(outcome.row["count"]) if outcome.row is not None else 1

    def _row_id(self, key: str) -> tuple[str, str]:
        """the row this counter's key addresses.

        :param key: the caller's identifier
        :ptype key: str
        :return: ``(purpose, hashed key)`` in declared column order
        :rtype: tuple[str, str]
        """
        return (self._purpose, hashlib.sha256(key.encode("utf-8")).hexdigest())


def _as_epoch(value: Any) -> float:
    """read a stored window start back as epoch seconds.

    Rows carry aware-UTC datetimes; :class:`WindowState` reports epoch seconds, because callers
    compute a "retry after" against the same clock they passed in.

    :param value: the stored value
    :ptype value: Any
    :return: epoch seconds
    :rtype: float
    """
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)
