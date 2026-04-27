"""KVLease — TTL-bounded distributed mutex over NATS JetStream KV.

generalizes the ownership-token + expiry-in-value pattern in
``aibots/hub/security/auth_strategies.py:LoginLockout`` into a reusable
primitive for cross-pod coordination (workspace bind locks, leader
election, fencing tokens).

usage::

    lease = KVLease(nats_client, bucket_name="prod14_leases")
    handle = await lease.acquire("workspace/customer-x/path")
    async with handle:
        ...

``nats_client`` is the canonical
:class:`threetears.nats.NatsClient` wrapper. the lease opens its
bucket via :meth:`NatsClient.kv_bucket` so all CAS / miss semantics
flow through :class:`NatsKvBucket`'s typed return shape (``None`` on
conflict instead of raising :class:`KeyWrongLastSequenceError`).

all KV envelope payloads flow through
:func:`threetears.core.serialization.serialize_to_json` /
:func:`deserialize_from_json` so UUID, datetime, Decimal round-trip
correctly with the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import uuid7

from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = [
    "KVLease",
    "LeaseHandle",
    "LeaseLost",
    "LeaseTimeout",
    "LeaseUnavailable",
]

log = get_logger(__name__)


class LeaseUnavailable(LookupError):
    """raised by :meth:`KVLease.acquire` when fail-fast path finds lease held.

    caller requested ``max_wait_seconds == 0`` and key is currently owned by
    live holder. subclasses :class:`LookupError` so callers may catch broadly
    or narrowly.
    """


class LeaseTimeout(TimeoutError):
    """raised by :meth:`KVLease.acquire` when wait deadline elapses.

    caller's ``max_wait_seconds`` expired before lease became free.
    subclasses :class:`TimeoutError`.
    """


class LeaseLost(RuntimeError):
    """raised by refresh or release when ownership has silently changed.

    another holder took over (entry expired, got reclaimed) or a concurrent
    CAS advanced the revision underneath us. caller's handle is no longer
    valid; caller should abort whatever work the lease protected.
    """


@dataclass
class _Envelope:
    """internal decoded KV value.

    :ivar holder: pod identifier that currently owns the lease
    :ivar date_expires: timezone-aware datetime past which entry is stale
    :ivar date_acquired: timezone-aware datetime when entry was first created
    """

    holder: str
    date_expires: datetime
    date_acquired: datetime


def _encode_envelope(holder: str, date_expires: datetime, date_acquired: datetime) -> bytes:
    """serialize holder and timestamps to JSON bytes for KV storage.

    routes through :func:`serialize_to_json` so encoding stays consistent
    with the rest of the codebase.

    :param holder: pod identifier owning the lease
    :ptype holder: str
    :param date_expires: timezone-aware datetime at which entry goes stale
    :ptype date_expires: datetime
    :param date_acquired: timezone-aware datetime lease was first acquired
    :ptype date_acquired: datetime
    :return: JSON bytes suitable for storing as KV value
    :rtype: bytes
    """
    payload: dict[str, Any] = {
        "holder": holder,
        "expires_at": date_expires.isoformat(),
        "acquired_at": date_acquired.isoformat(),
    }
    return serialize_to_json(payload)


def _decode_envelope(value: bytes) -> _Envelope:
    """deserialize stored KV value bytes back to envelope dataclass.

    routes through :func:`deserialize_from_json`; datetimes are stored as
    isoformat strings so they come back as plain ``str`` and are parsed
    here into timezone-aware ``datetime`` objects.

    :param value: bytes payload as returned from KV bucket ``get`` call
    :ptype value: bytes
    :return: envelope dataclass with holder + parsed timestamps
    :rtype: _Envelope
    :raises ValueError: if payload is malformed or timestamps unparseable
    """
    raw = deserialize_from_json(value, field_types={})
    holder = str(raw["holder"])
    date_expires = datetime.fromisoformat(str(raw["expires_at"]))
    date_acquired = datetime.fromisoformat(str(raw["acquired_at"]))
    return _Envelope(holder=holder, date_expires=date_expires, date_acquired=date_acquired)


class LeaseHandle:
    """opaque handle representing one successful lease acquisition.

    returned by :meth:`KVLease.acquire`. exposes :meth:`refresh` and
    :meth:`release` as well as async-context-manager sugar that releases
    on exit (so callers can write ``async with handle:``).

    :ivar key: KV key under which lease entry lives
    :ivar holder: pod identifier used when the lease was acquired
    :ivar revision: last-known KV revision for CAS refresh/release
    :ivar ttl_seconds: default TTL applied by :meth:`refresh` when caller
        does not supply one
    :ivar released: flag set once :meth:`release` runs to completion,
        making subsequent :meth:`release` calls a no-op
    """

    def __init__(
        self,
        lease: KVLease,
        key: str,
        holder: str,
        revision: int,
        ttl_seconds: int,
    ) -> None:
        """bind handle to parent lease and successful-acquire result state.

        :param lease: parent lease that produced this handle
        :ptype lease: KVLease
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param holder: pod identifier used during acquire
        :ptype holder: str
        :param revision: initial KV revision from create or CAS update
        :ptype revision: int
        :param ttl_seconds: default TTL for implicit refresh TTL reuse
        :ptype ttl_seconds: int
        :return: None
        :rtype: None
        """
        self._lease = lease
        self.key = key
        self.holder = holder
        self.revision = revision
        self.ttl_seconds = ttl_seconds
        self.released = False

    async def refresh(self, ttl_seconds: int | None = None) -> None:
        """extend lease expiry; raise :class:`LeaseLost` on ownership change.

        fetches current entry, verifies holder still matches, then performs
        CAS update with new expiry. on revision mismatch, holder mismatch,
        or key absence, raises :class:`LeaseLost`.

        :param ttl_seconds: override TTL for this refresh; falls back to
            TTL supplied at acquire time when None
        :ptype ttl_seconds: int | None
        :return: None
        :rtype: None
        :raises LeaseLost: if ownership has changed or CAS failed
        """
        await self._lease.refresh_handle(self, ttl_seconds=ttl_seconds)

    async def release(self) -> None:
        """delete KV entry if still owned; no-op on repeat calls.

        idempotent: once the ``released`` flag is set, subsequent calls
        return immediately. safe under contention — if another pod has
        stolen the lease, this method marks the handle released without
        touching the new holder's entry.

        :return: None
        :rtype: None
        """
        await self._lease.release_handle(self)

    async def __aenter__(self) -> LeaseHandle:
        """async-context-manager entry; returns self unchanged.

        the handle is already acquired before ``async with`` starts; the
        context manager exists only to guarantee release on exit.

        :return: this handle
        :rtype: LeaseHandle
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """async-context-manager exit; releases lease regardless of exception.

        :param exc_type: pending exception class or None
        :ptype exc_type: type[BaseException] | None
        :param exc: pending exception instance or None
        :ptype exc: BaseException | None
        :param tb: pending traceback or None
        :ptype tb: TracebackType | None
        :return: None
        :rtype: None
        """
        await self.release()


class KVLease:
    """distributed mutex factory backed by NATS JetStream KV.

    one :class:`KVLease` instance may mint many :class:`LeaseHandle`\\ s,
    each keyed on a different KV key. holder identity (``pod_id``) is
    shared across all leases minted by one factory so one pod consistently
    owns every lease it acquires.
    """

    def __init__(
        self,
        nats_client: "NatsClient",
        bucket_name: str | None = None,
        pod_id: str | None = None,
    ) -> None:
        """configure factory; defer bucket creation until first acquire.

        default ``bucket_name`` is
        ``f"{FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE}_leases"`` when the env
        var is set, or ``"leases"`` as unscoped fallback. default
        ``pod_id`` is ``f"pod-{uuid7().hex[:12]}"`` — time-ordered and
        unique per factory instance.

        :param nats_client: connected canonical
            :class:`threetears.nats.NatsClient` wrapper; the lease
            opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: explicit bucket name; None uses env-derived default
        :ptype bucket_name: str | None
        :param pod_id: explicit holder identifier; None auto-generates one
        :ptype pod_id: str | None
        :return: None
        :rtype: None
        """
        self._client = nats_client
        self._bucket_name = bucket_name if bucket_name is not None else self._default_bucket_name()
        self._pod_id = pod_id if pod_id is not None else f"pod-{uuid7().hex[:12]}"
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """return configured bucket name.

        :return: bucket name set at construction
        :rtype: str
        """
        return self._bucket_name

    @property
    def pod_id(self) -> str:
        """return holder identifier used for every acquire.

        :return: holder identifier
        :rtype: str
        """
        return self._pod_id

    @staticmethod
    def _default_bucket_name() -> str:
        """derive default bucket name from platform namespace env var.

        reads ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE`` at call time (not
        import time) so tests can monkeypatch it.

        :return: ``"{ns}_leases"`` when env set, else ``"leases"``
        :rtype: str
        """
        ns = os.environ.get("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE")
        result = f"{ns}_leases" if ns else "leases"
        return result

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open existing bucket or create it with history=1 on first call.

        lazy, async-safe: an ``asyncio.Lock`` serializes first-call setup
        so two concurrent acquires do not race to create the same
        bucket. routes through :meth:`NatsClient.kv_bucket` which
        idempotently creates-or-binds. the wrapper auto-prefixes the
        bucket name with the connected client's namespace prefix; the
        lease's ``bucket_name`` is treated as the bucket suffix the
        wrapper layers on top of that prefix.

        :return: backing wrapper-typed KV bucket
        :rtype: NatsKvBucket
        """
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    history=1,
                    create_if_missing=True,
                )
                log.info("KVLease bound bucket %s", self._bucket_name)
        return self._bucket

    async def acquire(
        self,
        key: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> LeaseHandle:
        """acquire lease on ``key``; block up to ``max_wait_seconds`` if held.

        algorithm:

        1. try ``bucket.create(key, payload)`` — wins on empty key.
        2. on key-exists, ``get`` current entry; if expiry is in the past,
           attempt CAS ``update`` to reclaim. success -> lease acquired.
        3. if still held by live holder and ``max_wait_seconds == 0``, raise
           :class:`LeaseUnavailable` immediately (no sleep).
        4. otherwise sleep ``min(1.0, remaining_time)`` and retry.
        5. on deadline elapsed, raise :class:`LeaseTimeout`.

        :param key: KV key under which lease entry lives
        :ptype key: str
        :param ttl_seconds: seconds past acquisition at which entry goes stale
        :ptype ttl_seconds: int
        :param max_wait_seconds: total seconds caller is willing to block;
            0 disables blocking and forces fail-fast on contention
        :ptype max_wait_seconds: int
        :return: handle representing successful acquisition
        :rtype: LeaseHandle
        :raises LeaseUnavailable: if ``max_wait_seconds == 0`` and key is held
        :raises LeaseTimeout: if deadline elapses before lease becomes free
        """
        bucket = await self._ensure_bucket()
        deadline = datetime.now(UTC) + timedelta(seconds=max_wait_seconds)
        handle: LeaseHandle | None = None
        timed_out = False
        while handle is None and not timed_out:
            attempt = await self._try_once(bucket, key, ttl_seconds)
            if attempt is not None:
                handle = attempt
                break
            # key was held by a live holder; decide whether to wait
            now = datetime.now(UTC)
            remaining = (deadline - now).total_seconds()
            if max_wait_seconds == 0:
                raise LeaseUnavailable(f"lease {key!r} held by another pod; fail-fast requested")
            if remaining <= 0:
                timed_out = True
                break
            await asyncio.sleep(min(1.0, remaining))
        if handle is None:
            raise LeaseTimeout(f"lease {key!r} not acquired within {max_wait_seconds}s")
        return handle

    async def _try_once(
        self,
        bucket: "NatsKvBucket",
        key: str,
        ttl_seconds: int,
    ) -> LeaseHandle | None:
        """single pass of create-or-reclaim-stale against the KV bucket.

        returns :class:`LeaseHandle` when lease is ours; returns ``None``
        when key is currently held by a live holder and caller should wait
        or fail-fast.

        :param bucket: backing wrapper KV bucket (from :meth:`_ensure_bucket`)
        :ptype bucket: NatsKvBucket
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param ttl_seconds: TTL to record in envelope when winning
        :ptype ttl_seconds: int
        :return: handle on successful acquire, None on live contention
        :rtype: LeaseHandle | None
        """
        now = datetime.now(UTC)
        date_expires = now + timedelta(seconds=ttl_seconds)
        payload = _encode_envelope(holder=self._pod_id, date_expires=date_expires, date_acquired=now)
        result: LeaseHandle | None = None
        # NatsKvBucket.create returns the new revision on success or
        # ``None`` on CAS conflict (key already exists). no exception
        # to catch for the conflict path -- the wrapper hides
        # KeyWrongLastSequenceError behind the typed ``None`` return.
        revision = await bucket.create(key=key, value=payload)
        if revision is not None:
            result = LeaseHandle(
                lease=self,
                key=key,
                holder=self._pod_id,
                revision=revision,
                ttl_seconds=ttl_seconds,
            )
        else:
            result = await self._maybe_reclaim_stale(bucket, key, ttl_seconds, now, date_expires)
        return result

    async def _maybe_reclaim_stale(
        self,
        bucket: "NatsKvBucket",
        key: str,
        ttl_seconds: int,
        now: datetime,
        date_expires: datetime,
    ) -> LeaseHandle | None:
        """attempt CAS reclaim when existing entry is expired; else return None.

        :param bucket: backing wrapper KV bucket
        :ptype bucket: NatsKvBucket
        :param key: KV key under which lease entry lives
        :ptype key: str
        :param ttl_seconds: TTL to record on successful reclaim
        :ptype ttl_seconds: int
        :param now: captured ``datetime.now(UTC)`` for this attempt
        :ptype now: datetime
        :param date_expires: pre-computed new expires_at for reclaim payload
        :ptype date_expires: datetime
        :return: handle when reclaim wins, None when entry is still live or
            CAS lost to another pod this round
        :rtype: LeaseHandle | None
        """
        result: LeaseHandle | None = None
        # NatsKvBucket.get_entry returns (value, revision) on hit or
        # ``None`` on miss; the entry could vanish between the create
        # race and this read so a None return means "caller will
        # retry" same as if it had been a KeyNotFound raise.
        entry = await bucket.get_entry(key=key)
        if entry is None:
            return None
        value, revision = entry
        envelope = _decode_envelope(value)
        if envelope.date_expires > now:
            # still held by a live holder
            return None
        payload = _encode_envelope(holder=self._pod_id, date_expires=date_expires, date_acquired=now)
        # NatsKvBucket.update returns the new revision on success or
        # ``None`` on CAS conflict; another pod may have reclaimed the
        # stale entry between our get_entry and update.
        new_revision = await bucket.update(key=key, value=payload, revision=revision)
        if new_revision is None:
            return None
        result = LeaseHandle(
            lease=self,
            key=key,
            holder=self._pod_id,
            revision=new_revision,
            ttl_seconds=ttl_seconds,
        )
        return result

    async def refresh_handle(self, handle: LeaseHandle, ttl_seconds: int | None) -> None:
        """implementation of :meth:`LeaseHandle.refresh`.

        public because :class:`LeaseHandle` lives in a sibling class and
        forwards its :meth:`refresh` here. the refresh pipeline (fetch
        entry, verify holder, CAS-update) is owned by the lease object
        that holds the KV bucket; the handle is just the revision token.

        :param handle: handle requesting refresh
        :ptype handle: LeaseHandle
        :param ttl_seconds: override TTL; None reuses handle's recorded TTL
        :ptype ttl_seconds: int | None
        :return: None
        :rtype: None
        :raises LeaseLost: if ownership changed or CAS update failed
        """
        bucket = await self._ensure_bucket()
        entry = await bucket.get_entry(key=handle.key)
        if entry is None:
            raise LeaseLost(f"lease {handle.key!r} entry missing during refresh")
        value, _revision = entry
        envelope = _decode_envelope(value)
        if envelope.holder != handle.holder:
            raise LeaseLost(
                f"lease {handle.key!r} holder changed: expected {handle.holder!r}, found {envelope.holder!r}"
            )
        effective_ttl = ttl_seconds if ttl_seconds is not None else handle.ttl_seconds
        now = datetime.now(UTC)
        date_expires = now + timedelta(seconds=effective_ttl)
        payload = _encode_envelope(holder=handle.holder, date_expires=date_expires, date_acquired=now)
        # NatsKvBucket.update returns ``None`` on revision mismatch,
        # mapping the previous KeyWrongLastSequenceError raise into a
        # typed conflict signal we surface as LeaseLost.
        new_revision = await bucket.update(key=handle.key, value=payload, revision=handle.revision)
        if new_revision is None:
            raise LeaseLost(f"lease {handle.key!r} revision advanced during refresh")
        handle.revision = new_revision
        handle.ttl_seconds = effective_ttl

    async def release_handle(self, handle: LeaseHandle) -> None:
        """implementation of :meth:`LeaseHandle.release`; idempotent.

        public for the same reason as :meth:`refresh_handle` -- the
        handle forwards its :meth:`release` to the lease object so the
        lease can verify ownership before deleting the KV entry.

        :param handle: handle requesting release
        :ptype handle: LeaseHandle
        :return: None
        :rtype: None
        """
        if handle.released:
            return
        bucket = await self._ensure_bucket()
        entry = await bucket.get_entry(key=handle.key)
        if entry is None:
            handle.released = True
            return
        value, _revision = entry
        envelope = _decode_envelope(value)
        if envelope.holder != handle.holder:
            # someone else owns it now; not our entry to delete
            handle.released = True
            return
        # CAS delete keyed on the handle's revision so a stale-after-
        # check delete cannot evict a successor's freshly reclaimed
        # entry. NatsKvBucket.delete returns False on revision mismatch
        # which is treated as "raced with another writer; entry
        # effectively gone from our view" -- diagnostic-only here.
        deleted = await bucket.delete(key=handle.key, revision=handle.revision)
        if not deleted:
            log.debug(
                "KVLease release raced on %s; marking released anyway",
                handle.key,
            )
        handle.released = True
