"""IdempotencyKeyStore -- claim-once-with-expiry over the coordination claims table.

Generalizes "has this specific operation already happened" out of hand-rolled
``get``-then-``save`` code (which has a race between the check and the write) into one
correct-by-construction primitive::

    store = IdempotencyKeyStore(registry, purpose="export_jobs")
    outcome = await store.claim("session-123:export")
    if outcome.status == "claimed":
        try:
            result = await do_the_work()
            await store.complete(outcome.record.key, result=result)
        except Exception as exc:
            await store.fail(outcome.record.key, error=str(exc))
    else:
        # outcome.record.status is "pending"/"completed"/"failed" -- the caller decides whether
        # to wait, return the stored result, or retry
        ...

Distinguished from :class:`KVLease`: a lease answers "who owns this right now" (renewable,
releasable); a claim answers "has this happened" (permanent until it expires, and it stores the
operation's outcome). Distinguished from :class:`ReplayGuard`: a replay guard is a bare
fresh/replay signal, while a caller retrying an already-completed operation needs the ORIGINAL
result back, not just "yes this happened".

**Where a claim lives.** The claim itself is a compare-and-swap against L2, which is what makes
"claimed" versus "exists" atomic across replicas, and the row is written behind to L3 so a broker
wipe does not resurrect a completed operation. The exposure that leaves is bounded and named: a
wipe inside one flush interval can lose a claim made in that interval, and the operation then runs
a second time. That is the same trade the counters take, and it is why this primitive is not the
right shape for something that must never run twice at any cost -- that wants a synchronous row.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.coordination.tables import CoordinationClaimsCollection, coordination_collection
from threetears.core.exceptions import ConcurrentModificationError
from threetears.observe import get_logger

__all__ = [
    "ClaimResult",
    "IdempotencyConflict",
    "IdempotencyKeyNotFound",
    "IdempotencyKeyStore",
    "IdempotencyRecord",
]

log = get_logger(__name__)

#: default lifetime of a claimed key -- 24 hours, the window the pre-collection store used.
_DEFAULT_TTL: Final[timedelta] = timedelta(hours=24)

_Status = Literal["pending", "completed", "failed"]


class IdempotencyKeyNotFound(LookupError):
    """raised by :meth:`IdempotencyKeyStore.complete`/:meth:`fail` when a key was never claimed.

    Subclasses :class:`LookupError` so callers may catch broadly or narrowly.
    """


class IdempotencyConflict(RuntimeError):
    """raised when a terminal transition loses its compare-and-swap budget.

    Signals concurrent writers racing to transition the SAME key: only the caller that received
    ``status="claimed"`` should ever complete or fail a key, so exhausting the budget points at a
    caller double-processing one claim rather than at ordinary contention.
    """


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """current state of one idempotency key.

    :ivar key: the idempotency key
    :ivar status: ``"pending"`` (claimed, unresolved), ``"completed"``, or ``"failed"``
    :ivar result: stored result bytes, present only when ``status == "completed"``
    :ivar error: stored error message, present only when ``status == "failed"``
    :ivar metadata: opaque caller bytes attached at claim time (a request-body hash, typically, so
        a caller told "exists" can tell the same request from a different one reusing the key);
        present for the record's whole life
    :ivar date_claimed: when the key was first claimed, timezone-aware
    :ivar date_completed: when it reached a terminal state, or ``None`` while pending
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

    :ivar status: ``"claimed"`` when this call created the key (the caller does the work);
        ``"exists"`` when another caller already claimed it (the caller must NOT redo the work)
    :ivar record: the key's record either way -- freshly pending for ``"claimed"``, and whatever
        the existing claimer left for ``"exists"``
    """

    status: Literal["claimed", "exists"]
    record: IdempotencyRecord


class IdempotencyKeyStore:
    """claim-once-with-expiry over the shared coordination claims table."""

    def __init__(
        self,
        registry: CollectionRegistry,
        *,
        purpose: str,
        ttl: timedelta | None = _DEFAULT_TTL,
        config: CoreConfig | None = None,
    ) -> None:
        """configure the store over its registry's coordination tables.

        :param registry: the collection registry this store reads and writes through. L2 makes
            the claim atomic across replicas and L3 makes it survive a broker restart; a
            deployment with no L3 keeps the claim only as long as L2 holds it
        :ptype registry: CollectionRegistry
        :param purpose: which idempotency domain these keys belong to, carried in the row key so
            unrelated keys never collide. This is what the KV bucket name used to be
        :ptype purpose: str
        :param ttl: how long a claimed key is remembered; ``None`` never expires, which is rarely
            right because the table then grows without bound. Defaults to 24 hours
        :ptype ttl: timedelta | None
        :param config: core config forwarded when this process first builds the claims collection
        :ptype config: CoreConfig | None
        :raises ValueError: when ``purpose`` is empty, or ``ttl`` is not positive
        """
        if not purpose.strip():
            raise ValueError("IdempotencyKeyStore purpose must be a non-empty name, e.g. 'export_jobs'")
        if ttl is not None and ttl <= timedelta(0):
            raise ValueError(f"IdempotencyKeyStore ttl must be positive when set, got {ttl}")
        self._purpose = purpose
        self._ttl = ttl
        self._collection = coordination_collection(registry, CoordinationClaimsCollection, config)
        # "claimed" versus "exists" is a compare-and-swap against L2. Without one it is a
        # read-modify-write, and two replicas that both read absent would both do the work.
        self._collection.require_l2_fence("IdempotencyKeyStore")

    @property
    def purpose(self) -> str:
        """which idempotency domain this store's keys belong to.

        :return: the purpose
        :rtype: str
        """
        return self._purpose

    async def claim(self, key: str, *, metadata: bytes | None = None) -> ClaimResult:
        """atomically claim ``key``, or report the existing claim.

        The claim is a create-if-absent compare-and-swap against L2, so the claimed/exists
        decision is atomic even under concurrent calls across replicas.

        :param key: idempotency key to claim
        :ptype key: str
        :param metadata: optional opaque bytes to attach at claim time, kept for the record's
            whole life unlike ``result``/``error``, which only appear at a terminal transition
        :ptype metadata: bytes | None
        :return: the claim outcome -- do the work only on ``"claimed"``
        :rtype: ClaimResult
        :raises threetears.nats.KvError: on an L2 failure
        """
        now = datetime.now(UTC)
        fresh = {
            "purpose": self._purpose,
            "key": key,
            "status": "pending",
            "result": None,
            "error": None,
            "claim_metadata": metadata,
            "date_claimed": now,
            "date_completed": None,
            "expires_at": None if self._ttl is None else now + self._ttl,
        }

        def _claim_if_absent(
            current: dict[str, Any] | None,
        ) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            # an expired claim is already absent to this callback at every tier, so a key whose
            # window has passed is claimable again rather than blocking the operation forever.
            if current is None:
                return "upsert", dict(fresh)
            return "noop", None

        outcome = await self._collection.l2_cas_mutate(self._row_id(key), _claim_if_absent)
        await self._collection.sweep_expired_if_due()
        if outcome.action == "created" and outcome.row is not None:
            return ClaimResult(status="claimed", record=_record_from_row(outcome.row))
        existing = await self.get(key)
        if existing is None:
            # the holder's claim expired between the refused create and this read: the key is
            # claimable again, and the contract is that claim() answers one or the other.
            return await self.claim(key, metadata=metadata)
        return ClaimResult(status="exists", record=existing)

    async def complete(self, key: str, *, result: bytes) -> None:
        """mark ``key`` completed, storing its result for a later retry to reuse.

        :param key: idempotency key to complete
        :ptype key: str
        :param result: opaque result bytes to store
        :ptype result: bytes
        :return: nothing
        :rtype: None
        :raises IdempotencyKeyNotFound: if the key was never claimed, or has expired
        :raises IdempotencyConflict: if the compare-and-swap budget is exhausted
        :raises threetears.nats.KvError: on an L2 failure
        """
        await self._transition(key, status="completed", result=result, error=None)

    async def fail(self, key: str, *, error: str) -> None:
        """mark ``key`` failed, storing the error message.

        :param key: idempotency key to fail
        :ptype key: str
        :param error: error message to store
        :ptype error: str
        :return: nothing
        :rtype: None
        :raises IdempotencyKeyNotFound: if the key was never claimed, or has expired
        :raises IdempotencyConflict: if the compare-and-swap budget is exhausted
        :raises threetears.nats.KvError: on an L2 failure
        """
        await self._transition(key, status="failed", result=None, error=error)

    async def get(self, key: str) -> IdempotencyRecord | None:
        """read ``key``'s current record.

        :param key: idempotency key to look up
        :ptype key: str
        :return: the current record, or ``None`` when never claimed or expired
        :rtype: IdempotencyRecord | None
        :raises threetears.nats.KvError: on an L2 failure
        """
        entity = await self._collection.get(self._row_id(key))
        return None if entity is None else _record_from_row(entity.to_dict())

    async def _transition(self, key: str, *, status: _Status, result: bytes | None, error: str | None) -> None:
        """compare-and-swap ``key`` to a terminal status.

        :param key: idempotency key to transition
        :ptype key: str
        :param status: the terminal status to set
        :ptype status: str
        :param result: result bytes when completing, else ``None``
        :ptype result: bytes | None
        :param error: error message when failing, else ``None``
        :ptype error: str | None
        :return: nothing
        :rtype: None
        :raises IdempotencyKeyNotFound: if the key was never claimed, or has expired
        :raises IdempotencyConflict: if the compare-and-swap budget is exhausted
        """
        missing = False

        def _to_terminal(
            current: dict[str, Any] | None,
        ) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            nonlocal missing
            if current is None:
                missing = True
                return "noop", None
            return "upsert", {
                **current,
                "status": status,
                "result": result,
                "error": error,
                "date_completed": datetime.now(UTC),
            }

        try:
            await self._collection.l2_cas_mutate(self._row_id(key), _to_terminal)
        except ConcurrentModificationError as exc:
            raise IdempotencyConflict(
                f"exhausted the compare-and-swap budget transitioning {key!r} to {status!r}"
            ) from exc
        if missing:
            raise IdempotencyKeyNotFound(f"idempotency key not found: {key!r}")

    def _row_id(self, key: str) -> tuple[str, str]:
        """the row this key addresses.

        The key is stored as given, not hashed: unlike a throttle key it is the caller's own
        operation id, the caller needs it back in the record, and an idempotency key is not a
        credential or an address.

        :param key: the caller's idempotency key
        :ptype key: str
        :return: ``(purpose, key)`` in declared column order
        :rtype: tuple[str, str]
        """
        return (self._purpose, key)


def _record_from_row(row: dict[str, Any]) -> IdempotencyRecord:
    """shape one stored row into the record a caller sees.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the record
    :rtype: IdempotencyRecord
    """
    return IdempotencyRecord(
        key=str(row["key"]),
        status=row["status"],
        result=_as_bytes(row.get("result")),
        error=row.get("error"),
        metadata=_as_bytes(row.get("claim_metadata")),
        date_claimed=row["date_claimed"],
        date_completed=row.get("date_completed"),
    )


def _as_bytes(value: Any) -> bytes | None:
    """read a stored bytes column back as bytes.

    A ``bytea`` round-trips as ``bytes``, but an L2 value that travelled as JSON comes back
    base64-encoded, and a backend may hand back a ``memoryview``.

    :param value: the stored value
    :ptype value: Any
    :return: the bytes, or ``None``
    :rtype: bytes | None
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview | bytearray):
        return bytes(value)
    return base64.b64decode(value)
