"""generic windowed attempt counter backed by NATS JetStream KV -- a rate-limiter/throttle
primitive, sibling to :class:`~threetears.core.coordination.replay_guard.ReplayGuard` and
:class:`~threetears.core.coordination.replay_guard.RevocationGuard`, but a different shape: those
two answer "have I seen this exact key" (bare presence / timestamped presence); this answers "how
many times has this key been attempted inside a moving time window" -- the shape a throttle or
rate limiter needs (e.g. "no more than N login attempts per IP per minute").

    counter = WindowedCounter(nats_client, bucket_name="login_ip_throttle", window_seconds=60)
    attempts = await counter.record_attempt(ip_address)
    if attempts > 20:
        raise <throttled>

The window is an app-level ``window_start`` timestamp carried in the KV value, anchored at the
FIRST attempt of the current window -- NOT the KV entry's own write-time TTL. JetStream KV TTL is
refreshed on every write to a key, so relying on it directly would let a steady stream of attempts
keep extending the window indefinitely instead of the fixed window callers ask for (the same
reasoning as identity-core's ``LockoutTracker``, which this primitive generalizes). The bucket's
own TTL is still set to the window, purely as eventual garbage collection for abandoned counters.

Bucket uses ``file`` storage (not the default ``memory``): a throttle counter is a security
control, not a cache -- a NATS restart must not silently reset it back to zero, handing an
in-progress attacker a fresh window for free (the same fail-open concern documented on
``ReplayGuard``'s and ``RevocationGuard``'s buckets).

**Fail-open vs. fail-closed on a `KvError` is the CALLER's choice**, via the constructor's
``fail_open`` flag -- unlike ``ReplayGuard``/``RevocationGuard``, which are always fail-closed. A
windowed counter is not always sitting on a hard security boundary the way a replay nonce or a
revocation entry is: an edge-tier IP throttle is a best-effort, cheap first-line defense in front
of an authoritative second check (identity-core's Chunk 05 anti-automation design explicitly
layers a fail-open edge throttle in front of a fail-closed core throttle over a SEPARATE bucket)
-- degrading a first-line defense to "not currently throttling" on a transport blip is an
acceptable trade for availability, since the authoritative layer behind it is unaffected. A
counter guarding something with no such second layer behind it should be constructed
``fail_open=False`` (the default) so a transport failure denies rather than silently admits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from threetears.nats.errors import KvError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = ["WindowedCounter"]

log = get_logger(__name__)

_MAX_CAS_ATTEMPTS = 5


@dataclass(frozen=True, slots=True)
class _WindowState:
    count: int
    window_start: float


def _encode(state: _WindowState) -> bytes:
    return json.dumps({"count": state.count, "window_start": state.window_start}).encode("utf-8")


def _decode(value: bytes) -> _WindowState:
    payload = json.loads(value)
    return _WindowState(count=int(payload["count"]), window_start=float(payload["window_start"]))


class WindowedCounter:
    """per-key attempt counter over a fixed sliding window, in a NATS JetStream KV bucket.

    generic: the caller supplies the key (already hashed if it carries anything sensitive -- this
    class hashes it again into a KV-safe form regardless, but does not otherwise interpret it) and
    decides what threshold makes a count "too many". This class only tracks the count.
    """

    def __init__(
        self,
        nats_client: "NatsClient",
        *,
        bucket_name: str,
        window_seconds: int,
        fail_open: bool = False,
    ) -> None:
        """configure the counter; defer bucket binding until the first use.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the namespace. Pick a
            bucket dedicated to one throttle purpose (e.g. ``login_ip_throttle``) so unrelated
            counters never collide across surfaces, and so two independent counters (e.g. an
            edge-tier and a core-tier throttle over the "same" logical key) can be given
            deliberately SEPARATE buckets with different write access
        :ptype bucket_name: str
        :param window_seconds: length of the sliding window in seconds. MUST be positive: a
            non-positive window would mean the count never resets
        :ptype window_seconds: int
        :param fail_open: on a :class:`~threetears.nats.KvError` (KV transport failure), whether
            to treat the key as NOT over any threshold (``fail_open=True`` -- `record_attempt`
            returns ``0``, `count` returns ``0``, a warning is logged) rather than propagating the
            error for the caller to deny (``fail_open=False``, the default -- mirrors
            `ReplayGuard`/`RevocationGuard`'s always-fail-closed posture)
        :ptype fail_open: bool
        :raises ValueError: when ``window_seconds`` is not positive
        """
        if window_seconds <= 0:
            raise ValueError(f"WindowedCounter window_seconds must be positive, got {window_seconds}")
        self._client = nats_client
        self._bucket_name = bucket_name
        self._window = timedelta(seconds=window_seconds)
        self._fail_open = fail_open
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    @property
    def fail_open(self) -> bool:
        """whether this counter fails open (``True``) or closed (``False``) on `KvError`.

        :return: the configured fail-open/fail-closed posture
        :rtype: bool
        """
        return self._fail_open

    async def record_attempt(self, key: str) -> int:
        """record one attempt for ``key``; return the attempt count within the current
        (possibly just-started) window.

        Starts a fresh window (count=1) if the previous one has already expired. CAS
        create-if-absent / compare-and-swap retry loop, so concurrent callers on the same key never
        lose an increment silently.

        :param key: the identifier to key the counter on (e.g. an IP address, or an
            already-hashed compound key). Hashed into a fixed, KV-safe key before storage, so any
            key format is accepted and the raw identifier is never stored as a KV key
        :ptype key: str
        :return: the attempt count within the live window (``>= 1``), or ``0`` if this counter is
            ``fail_open`` and a `KvError` occurred
        :rtype: int
        :raises threetears.nats.KvError: on a KV transport failure, when ``fail_open=False``
        """
        try:
            return await self._record_attempt(key)
        except KvError:
            if self._fail_open:
                log.warning("WindowedCounter %s fail-open on KvError for record_attempt", self._bucket_name)
                return 0
            raise

    async def count(self, key: str) -> int:
        """read-only: the current attempt count for ``key`` within the live window, without
        recording a new attempt.

        :param key: the identifier to look up
        :ptype key: str
        :return: the live count, or ``0`` if absent, expired, or (when ``fail_open``) on `KvError`
        :rtype: int
        :raises threetears.nats.KvError: on a KV transport failure, when ``fail_open=False``
        """
        try:
            state = await self._read_live_state(key)
        except KvError:
            if self._fail_open:
                log.warning("WindowedCounter %s fail-open on KvError for count", self._bucket_name)
                return 0
            raise
        return 0 if state is None else state.count

    async def is_over_threshold(self, key: str, *, threshold: int) -> bool:
        """``True`` if ``key``'s live count is at or above ``threshold``, without recording a new
        attempt.

        :param key: the identifier to look up
        :ptype key: str
        :param threshold: the caller's chosen ceiling
        :ptype threshold: int
        :return: whether the key is currently over threshold
        :rtype: bool
        :raises threetears.nats.KvError: on a KV transport failure, when ``fail_open=False``
        """
        return await self.count(key) >= threshold

    async def _record_attempt(self, key: str) -> int:
        bucket = await self._ensure_bucket()
        kv_key = self._key(key)
        now = time.time()
        for _ in range(_MAX_CAS_ATTEMPTS):
            entry = await bucket.get_entry(key=kv_key)
            if entry is None:
                created = await bucket.create(key=kv_key, value=_encode(_WindowState(count=1, window_start=now)))
                if created is not None:
                    return 1
                continue  # lost the create race -- next loop iteration falls into the update path
            value, revision = entry
            state = _decode(value)
            if now - state.window_start > self._window.total_seconds():
                next_state = _WindowState(count=1, window_start=now)
            else:
                next_state = _WindowState(count=state.count + 1, window_start=state.window_start)
            if await bucket.update(key=kv_key, value=_encode(next_state), revision=revision) is not None:
                return next_state.count
        # CAS retries exhausted under heavy concurrent contention on the SAME key -- exceedingly
        # unlikely in practice. Read back whatever is there now rather than silently under-counting
        # to 0 or 1; the next attempt still sees accurate state either way.
        live_state = await self._read_live_state(key)
        return 1 if live_state is None else live_state.count

    async def _read_live_state(self, key: str) -> _WindowState | None:
        bucket = await self._ensure_bucket()
        value = await bucket.get(key=self._key(key))
        if value is None:
            return None
        state = _decode(value)
        if time.time() - state.window_start > self._window.total_seconds():
            return None
        return state

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the windowed KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._window,
                    # file storage: see module docstring -- a throttle counter is a security
                    # control, not a cache; a restart must not silently reset it to zero.
                    storage="file",
                    create_if_missing=True,
                    history=1,
                )
                log.info("WindowedCounter bound bucket %s", self._bucket_name)
        return self._bucket

    @staticmethod
    def _key(key: str) -> str:
        """hash the caller's key into a fixed-length, KV-safe key (also avoids storing it raw)."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
