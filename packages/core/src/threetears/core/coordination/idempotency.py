"""IdempotencyKeyStore — claim-once-with-TTL primitive over NATS JetStream KV.

generalizes the "has this specific operation already happened" pattern
out of ``14-eng-ai-survey``'s pre-conversion ``IdempotencyData`` (a
hand-rolled ``get``-then-``save`` collection with a TOCTOU race window
between the check and the write) into a reusable, correct-by-construction
building block for any 3tears consumer needing exactly-once claim
semantics with an expiring key.

distinguished from :class:`KVLease` (this module's sibling): a lease
answers "who currently owns this right now" (renewable, releasable,
mutual-exclusion-flavored); an idempotency claim answers "has this
specific operation already happened" (permanent until TTL, stores the
operation's outcome, no renewal concept). distinguished from
:class:`ReplayGuard` (also this module's sibling): a replay guard is a
bare boolean fresh/replay signal with no result storage or
completed/failed transition; an idempotency claim tracks pending ->
completed/failed state and the stored result/error, since a caller
retrying an already-completed operation needs the ORIGINAL result back,
not just "yes this happened."

usage::

    store = IdempotencyKeyStore(nats_client, bucket_name="export_jobs_idempotency")
    outcome = await store.claim("session-123:export")
    if outcome.status == "claimed":
        try:
            result = await do_the_work()
            await store.complete(outcome.record.key, result=result)
        except Exception as exc:
            await store.fail(outcome.record.key, error=str(exc))
    else:
        # outcome.record.status is "pending"/"completed"/"failed" -- caller
        # decides whether to wait, return the stored result, or retry
        ...

all KV envelope payloads flow through
:func:`threetears.core.serialization.serialize_to_json` /
:func:`deserialize_from_json`, matching :class:`KVLease`'s convention.
"""

from __future__ import annotations

import asyncio
import base64
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, NatsKvBucket

__all__ = [
    "ClaimResult",
    "IdempotencyConflict",
    "IdempotencyKeyNotFound",
    "IdempotencyKeyStore",
    "IdempotencyRecord",
]

log = get_logger(__name__)

#: default TTL for a claimed key -- matches the pre-conversion
#: IdempotencyData collection's 24-hour window (14-eng-ai-survey,
#: shard 20's Research section).
_DEFAULT_TTL: Final[timedelta] = timedelta(hours=24)

#: bounded CAS retry budget for complete()/fail()'s read-modify-write --
#: mirrors BaseCollection.l2_cas_mutate's max_retries shape
#: (packages/core/src/threetears/core/collections/base.py), even though
#: this primitive lives in a different module: same retry-under-
#: contention concern, same bounded-retry answer.
_CAS_MAX_RETRIES: Final[int] = 8

#: full-jitter backoff bound between CAS retries, seconds.
_CAS_RETRY_BACKOFF_SECONDS: Final[float] = 0.02

_Status = Literal["pending", "completed", "failed"]


class IdempotencyKeyNotFound(LookupError):
    """raised by :meth:`IdempotencyKeyStore.complete`/:meth:`fail` when key was never claimed.

    subclasses :class:`LookupError` so callers may catch broadly or narrowly.
    """


class IdempotencyConflict(RuntimeError):
    """raised when a complete()/fail() CAS retry budget is exhausted.

    signals concurrent writers racing to transition the SAME key's
    terminal state -- should be vanishingly rare in correct usage (only
    the one caller that received ``status="claimed"`` from :meth:`claim`
    should ever call :meth:`complete`/:meth:`fail` for a given key), so
    exhausting the retry budget indicates a caller-side correctness bug
    (double-processing the same claim) rather than ordinary contention.
    """


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """current state of one idempotency key.

    :param key: the idempotency key
    :ptype key: str
    :param status: ``"pending"`` (claimed, not yet resolved), ``"completed"``, or ``"failed"``
    :ptype status: str
    :param result: stored result bytes, present only when ``status == "completed"``
    :ptype result: bytes | None
    :param error: stored error message, present only when ``status == "failed"``
    :ptype error: str | None
    :param metadata: opaque caller-supplied bytes attached at :meth:`IdempotencyKeyStore.claim`
        time (e.g. a request-body hash for detecting the same key reused
        for a different request), present for the record's whole
        lifetime once claimed; ``None`` if the claimer supplied none
    :ptype metadata: bytes | None
    :param date_claimed: timezone-aware datetime the key was first claimed
    :ptype date_claimed: datetime
    :param date_completed: timezone-aware datetime the key reached a terminal
        state, or ``None`` while still ``"pending"``
    :ptype date_completed: datetime | None
    """

    key: str
    status: _Status
    result: bytes | None
    error: str | None
    metadata: bytes | None
    date_claimed: datetime
    date_completed: datetime | None


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """outcome of :meth:`IdempotencyKeyStore.claim`.

    :param status: ``"claimed"`` when this call created the key (caller
        must do the work); ``"exists"`` when another caller already
        claimed it (caller must NOT redo the work)
    :ptype status: str
    :param record: the key's current record either way -- for
        ``"claimed"`` this is the freshly-created ``"pending"`` record;
        for ``"exists"`` this is whatever the existing claimer's record
        currently holds (possibly still ``"pending"``, or already
        ``"completed"``/``"failed"`` with a result/error to reuse)
    :ptype record: IdempotencyRecord
    """

    status: Literal["claimed", "exists"]
    record: IdempotencyRecord


def _encode_envelope(record: IdempotencyRecord) -> bytes:
    """serialize a record to JSON bytes for KV storage.

    :param record: record to encode
    :ptype record: IdempotencyRecord
    :return: JSON-encoded bytes suitable for a KV value
    :rtype: bytes
    """
    payload: dict[str, Any] = {
        "key": record.key,
        "status": record.status,
        "result": base64.b64encode(record.result).decode("ascii") if record.result is not None else None,
        "error": record.error,
        "metadata": base64.b64encode(record.metadata).decode("ascii") if record.metadata is not None else None,
        "date_claimed": record.date_claimed.isoformat(),
        "date_completed": record.date_completed.isoformat() if record.date_completed is not None else None,
    }
    return serialize_to_json(payload)


def _decode_envelope(value: bytes) -> IdempotencyRecord:
    """deserialize stored KV value bytes back to a record.

    :param value: bytes payload as returned from a KV bucket read
    :ptype value: bytes
    :return: decoded record
    :rtype: IdempotencyRecord
    :raises ValueError: if payload is malformed
    """
    raw = deserialize_from_json(value, field_types={})
    result_b64 = raw.get("result")
    metadata_b64 = raw.get("metadata")
    return IdempotencyRecord(
        key=str(raw["key"]),
        status=raw["status"],
        result=base64.b64decode(result_b64) if result_b64 is not None else None,
        error=raw.get("error"),
        metadata=base64.b64decode(metadata_b64) if metadata_b64 is not None else None,
        date_claimed=datetime.fromisoformat(str(raw["date_claimed"])),
        date_completed=datetime.fromisoformat(str(raw["date_completed"])) if raw.get("date_completed") else None,
    )


class IdempotencyKeyStore:
    """claim-once-with-TTL primitive over one shared NATS JetStream KV bucket.

    bucket binding is lazy (deferred to the first operation), matching
    :class:`ReplayGuard`'s construction style within this module.
    """

    def __init__(self, nats_client: "NatsClient", *, bucket_name: str, ttl: timedelta | None = _DEFAULT_TTL) -> None:
        """configure the store; defer bucket binding until first use.

        :param nats_client: connected canonical :class:`threetears.nats.NatsClient`;
            the store opens its KV bucket through :meth:`NatsClient.kv_bucket`
        :ptype nats_client: NatsClient
        :param bucket_name: KV bucket suffix; the wrapper prefixes it with the
            namespace. pick a bucket dedicated to one idempotency domain so
            unrelated keys never collide across surfaces
        :ptype bucket_name: str
        :param ttl: how long a claimed key is remembered; ``None`` means keys
            never expire (rarely correct -- an unbounded idempotency bucket
            grows forever). defaults to 24 hours, matching the pre-conversion
            IdempotencyData collection this primitive replaces
        :ptype ttl: timedelta | None
        :return: none
        :rtype: None
        """
        self._client = nats_client
        self._bucket_name = bucket_name
        self._ttl = ttl
        self._bucket: "NatsKvBucket | None" = None
        self._bucket_lock = asyncio.Lock()

    @property
    def bucket_name(self) -> str:
        """the configured bucket suffix.

        :return: bucket name
        :rtype: str
        """
        return self._bucket_name

    async def claim(self, key: str, *, metadata: bytes | None = None) -> ClaimResult:
        """atomically claim ``key``, or report the existing claim.

        backed by CAS create-if-absent, so the claimed/exists decision
        is atomic even under concurrent calls across replicas.

        :param key: idempotency key to claim
        :ptype key: str
        :param metadata: optional opaque bytes to attach at claim time
            (e.g. a request-body hash so a caller receiving
            ``status="exists"`` can detect the same key reused for a
            different request) -- stored for the record's whole
            lifetime, unlike ``result``/``error`` which only appear
            after a terminal transition
        :ptype metadata: bytes | None
        :return: claim outcome -- caller does the work only on ``"claimed"``
        :rtype: ClaimResult
        :raises threetears.nats.KvError: on a KV transport failure
        """
        bucket = await self._ensure_bucket()
        now = datetime.now(UTC)
        fresh_record = IdempotencyRecord(
            key=key, status="pending", result=None, error=None, metadata=metadata, date_claimed=now, date_completed=None
        )
        revision = await bucket.create(key=key, value=_encode_envelope(fresh_record))
        if revision is not None:
            result = ClaimResult(status="claimed", record=fresh_record)
        else:
            entry = await bucket.get_entry(key=key)
            if entry is None:
                # lost a race between the failed create and this read (the
                # other claimer's key expired via TTL in between) -- retry
                # once, matching the "claim is atomic" contract rather than
                # surfacing a transient inconsistency to the caller.
                return await self.claim(key, metadata=metadata)
            existing_value, _ = entry
            result = ClaimResult(status="exists", record=_decode_envelope(existing_value))
        return result

    async def complete(self, key: str, *, result: bytes) -> None:
        """mark ``key`` completed, storing its result.

        :param key: idempotency key to complete
        :ptype key: str
        :param result: opaque result bytes to store for future callers
            reusing this claim's outcome
        :ptype result: bytes
        :return: none
        :rtype: None
        :raises IdempotencyKeyNotFound: if key was never claimed
        :raises IdempotencyConflict: if the CAS retry budget is exhausted
        :raises threetears.nats.KvError: on a KV transport failure
        """
        await self._transition(key, status="completed", result=result, error=None)

    async def fail(self, key: str, *, error: str) -> None:
        """mark ``key`` failed, storing the error message.

        :param key: idempotency key to fail
        :ptype key: str
        :param error: error message to store
        :ptype error: str
        :return: none
        :rtype: None
        :raises IdempotencyKeyNotFound: if key was never claimed
        :raises IdempotencyConflict: if the CAS retry budget is exhausted
        :raises threetears.nats.KvError: on a KV transport failure
        """
        await self._transition(key, status="failed", result=None, error=error)

    async def get(self, key: str) -> IdempotencyRecord | None:
        """read ``key``'s current record.

        :param key: idempotency key to look up
        :ptype key: str
        :return: current record, or ``None`` if never claimed (or expired)
        :rtype: IdempotencyRecord | None
        :raises threetears.nats.KvError: on a KV transport failure
        """
        bucket = await self._ensure_bucket()
        value = await bucket.get(key=key)
        return _decode_envelope(value) if value is not None else None

    async def _transition(self, key: str, *, status: _Status, result: bytes | None, error: str | None) -> None:
        """CAS read-modify-write ``key`` to a terminal status, retrying under contention.

        :param key: idempotency key to transition
        :ptype key: str
        :param status: terminal status to set (``"completed"`` or ``"failed"``)
        :ptype status: str
        :param result: result bytes when completing, else ``None``
        :ptype result: bytes | None
        :param error: error message when failing, else ``None``
        :ptype error: str | None
        :return: none
        :rtype: None
        :raises IdempotencyKeyNotFound: if key was never claimed
        :raises IdempotencyConflict: if the CAS retry budget is exhausted
        """
        bucket = await self._ensure_bucket()
        for attempt in range(_CAS_MAX_RETRIES):
            entry = await bucket.get_entry(key=key)
            if entry is None:
                raise IdempotencyKeyNotFound(f"idempotency key not found: {key!r}")
            existing_value, revision = entry
            existing = _decode_envelope(existing_value)
            updated = IdempotencyRecord(
                key=key,
                status=status,
                result=result,
                error=error,
                metadata=existing.metadata,  # preserve claim-time metadata through the terminal transition
                date_claimed=existing.date_claimed,
                date_completed=datetime.now(UTC),
            )
            new_revision = await bucket.update(key=key, value=_encode_envelope(updated), revision=revision)
            if new_revision is not None:
                return
            if attempt < _CAS_MAX_RETRIES - 1:
                backoff = random.uniform(0, _CAS_RETRY_BACKOFF_SECONDS)  # noqa: S311 - jitter, not security
                await asyncio.sleep(backoff)
        raise IdempotencyConflict(f"exhausted {_CAS_MAX_RETRIES} CAS retries transitioning key {key!r} to {status!r}")

    async def _ensure_bucket(self) -> "NatsKvBucket":
        """open (or bind) the TTL'd KV bucket once; async-safe lazy init."""
        if self._bucket is not None:
            return self._bucket
        async with self._bucket_lock:
            if self._bucket is None:
                self._bucket = await self._client.kv_bucket(
                    name=self._bucket_name,
                    ttl=self._ttl,
                    storage="memory",
                    create_if_missing=True,
                    history=1,
                )
                log.info("IdempotencyKeyStore bound bucket %s", self._bucket_name)
        return self._bucket
