"""Write-buffer and flush strategy for deferred collection persistence."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Any, NamedTuple, TYPE_CHECKING

import asyncpg
from sqlalchemy import Column, Integer, MetaData, String, Table, Text

from threetears.observe import get_logger

__all__ = [
    "FlushStrategy",
    "PendingWrite",
    "WriteBuffer",
    "flush_pending",
]

if TYPE_CHECKING:
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.collections.registry import CollectionRegistry

log = get_logger(__name__)

_WRITE_BUFFER_METADATA = MetaData()

_write_buffer_table = Table(
    "write_buffer",
    _WRITE_BUFFER_METADATA,
    Column("key", String, primary_key=True),
    Column("table_name", Text, nullable=False),
    Column("entity_id", Text, nullable=False),
    Column("data", Text, nullable=False),
    Column("retries", Integer, nullable=False, default=0),
    Column("date_updated", String, nullable=True),
)

# Retry budget for general flush failures (transient DB errors,
# serialization, etc.). Once a write fails this many times it is
# dropped from the buffer and a permanent-failure event logged.
_MAX_FLUSH_RETRIES = 10

# Retry budget for foreign-key-violation failures specifically. FK
# violations almost always mean "my parent hasn't reached Postgres
# yet" -- the parent is either later in the toposort within this
# drain (already addressed) or pending in a separate drain batch.
# In the latter case the child just needs to wait for the parent
# to land before retrying. The pre-2026-05-13 behavior of capping
# FK retries at 10 (~5 minutes at the 30s default flush interval)
# was too tight: a single dropped parent message permanently
# orphaned every descendant in the conversation, producing the
# cascading "messages dropped, half conversation missing"
# fingerprint in production (conv ``019e2372-fcdd``,
# 2026-05-13 incident). 100 retries at the same interval = ~50min,
# enough headroom for any realistic transient. Beyond that the
# parent genuinely failed and the child is unreachable -- drop
# with a clear "orphan chain" log so operators can investigate.
_FK_RETRY_LIMIT = 100


def _is_fk_violation(exc: BaseException) -> bool:
    """Detect whether an exception is a Postgres foreign-key violation.

    Two signals are checked: ``isinstance`` against the asyncpg
    typed exception, AND a substring match in the exception message
    (covers cases where the violation was raised through a wrapper
    or re-raised as a different class). Either match counts.
    """
    if isinstance(exc, asyncpg.exceptions.ForeignKeyViolationError):
        return True
    return "violates foreign key constraint" in str(exc)


class FlushStrategy(StrEnum):
    ALWAYS = "ALWAYS"
    ON_CHECKPOINT = "ON_CHECKPOINT"
    ON_SCHEDULE = "ON_SCHEDULE"
    ON_SHUTDOWN = "ON_SHUTDOWN"


class PendingWrite(NamedTuple):
    table_name: str
    entity_id: Any
    data: dict[str, Any]
    retries: int = 0


class WriteBuffer:
    """Coalescing async write buffer keyed by (table_name, entity_id).

    when l1_backend is provided, pending writes are persisted to
    SQLite so they survive process crashes. dict is retained as
    fast dedup index and fallback when l1_backend is None.
    """

    def __init__(self, l1_backend: SQLiteBackend | None = None) -> None:
        """initialize write buffer with optional L1 persistence.

        :param l1_backend: optional SQLiteBackend for crash-safe buffering
        :ptype l1_backend: SQLiteBackend | None
        """
        self._buf: dict[tuple[str, Any], PendingWrite] = {}
        self._lock = asyncio.Lock()
        self._l1 = l1_backend
        if self._l1 is not None and not self._l1.is_initialized():
            self._l1.initialize(_WRITE_BUFFER_METADATA)

    async def add(self, table_name: str, entity_id: Any, data: dict[str, Any], retries: int = 0) -> None:
        """Add or replace a pending write for the given entity."""
        async with self._lock:
            key = (table_name, entity_id)
            pw = PendingWrite(table_name, entity_id, data, retries)
            self._buf[key] = pw
            if self._l1 is not None:
                from datetime import UTC, datetime

                l1_key = f"{table_name}:{entity_id}"
                self._l1.upsert(
                    "write_buffer",
                    {
                        "key": l1_key,
                        "table_name": table_name,
                        "entity_id": str(entity_id),
                        "data": json.dumps(data, default=str),
                        "retries": retries,
                        "date_updated": datetime.now(UTC).isoformat(),
                    },
                    primary_key="key",
                )

    async def drain(self) -> list[PendingWrite]:
        """Drain all pending writes, returning them and clearing the buffer."""
        async with self._lock:
            if self._l1 is not None:
                rows = self._l1.execute_query("SELECT * FROM write_buffer")
                items: list[PendingWrite] = []
                for row in rows:
                    raw_data = row["data"]
                    parsed_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                    items.append(
                        PendingWrite(
                            table_name=row["table_name"],
                            entity_id=row["entity_id"],
                            data=parsed_data,
                            retries=row["retries"],
                        )
                    )
                conn = self._l1.get_connection()
                conn.execute("DELETE FROM write_buffer")
                self._buf.clear()
                return items
            items = list(self._buf.values())
            self._buf.clear()
            return items

    async def remove(self, table_name: str, entity_id: Any) -> bool:
        """Remove a pending write. Returns True if it existed."""
        async with self._lock:
            existed = self._buf.pop((table_name, entity_id), None) is not None
            if self._l1 is not None:
                l1_key = f"{table_name}:{entity_id}"
                self._l1.delete_by_id("write_buffer", l1_key, primary_key="key")
            return existed

    def pending_count(self) -> int:
        """Return the number of pending writes in the buffer."""
        return len(self._buf)


def _toposort_pending(
    pending: list[PendingWrite],
    parent_key_map: dict[str, str] | None = None,
) -> list[PendingWrite]:
    """Sort pending writes so parents are flushed before children.

    parent_key_map maps table_name -> FK column pointing to parent.
    Default: {"messages": "parent_message_id"}.
    """
    if parent_key_map is None:
        parent_key_map = {"messages": "parent_message_id"}

    # Separate into tables with FK deps vs without
    no_deps: list[PendingWrite] = []
    with_deps: list[PendingWrite] = []
    for pw in pending:
        if pw.table_name in parent_key_map:
            with_deps.append(pw)
        else:
            no_deps.append(pw)

    if not with_deps:
        return no_deps

    # Kahn's algorithm for each table group
    by_id: dict[Any, PendingWrite] = {pw.entity_id: pw for pw in with_deps}
    in_degree: dict[Any, int] = {pw.entity_id: 0 for pw in with_deps}
    children_of: dict[Any, list[Any]] = {}

    for pw in with_deps:
        parent_col = parent_key_map[pw.table_name]
        parent_id = pw.data.get(parent_col)
        if parent_id is not None and parent_id in by_id:
            in_degree[pw.entity_id] = in_degree.get(pw.entity_id, 0) + 1
            children_of.setdefault(parent_id, []).append(pw.entity_id)

    queue = [eid for eid, deg in in_degree.items() if deg == 0]
    ordered: list[PendingWrite] = []
    while queue:
        eid = queue.pop(0)
        ordered.append(by_id[eid])
        for child_id in children_of.get(eid, []):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    # Handle cycles — append remaining so nothing is silently dropped
    if len(ordered) < len(with_deps):
        ordered_ids = {pw.entity_id for pw in ordered}
        for pw in with_deps:
            if pw.entity_id not in ordered_ids:
                ordered.append(pw)

    return no_deps + ordered


def _resolve_batch_backend(
    sorted_pending: list[PendingWrite],
    registry: CollectionRegistry,
) -> Any:
    """Resolve the single shared backend for an atomic-batch flush, or ``None``.

    The atomic-batch path is only taken when **every** pending collection resolves to
    the **same** backend object AND that backend exposes a usable ``transaction()``.
    Any of: an unregistered table, a missing backend, divergent backends, or a backend
    without ``transaction()`` (e.g. a git-backed ``DurableStore``) → ``None``, so the
    caller degrades to the per-entity loop.

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :return: the shared backend exposing ``transaction()``, or ``None``.
    :rtype: Any
    """
    backend: Any = None
    for pw in sorted_pending:
        collection = registry.get_collection(pw.table_name)
        if collection is None:
            return None
        b = registry.get_l3_pool(pw.table_name)
        if b is None or not callable(getattr(b, "transaction", None)):
            return None
        if backend is None:
            backend = b
        elif b is not backend:
            return None
    return backend


async def _flush_batch_atomic(
    sorted_pending: list[PendingWrite],
    registry: CollectionRegistry,
    backend: Any,
) -> int:
    """Persist the whole toposorted batch inside ONE backend transaction.

    Raises on any failure so the caller can fall back to the per-entity loop (which
    keeps the ``_is_fk_violation`` classification + re-enqueue). The transaction is
    rolled back by the backend's ``transaction()`` context manager on exception.

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :param backend: the shared backend exposing ``transaction()``.
    :ptype backend: Any
    :return: number of entities persisted (the full batch on success).
    :rtype: int
    """
    flushed = 0
    async with backend.transaction() as conn:
        for pw in sorted_pending:
            collection = registry.get_collection(pw.table_name)
            # _resolve_batch_backend already proved every table resolves to a
            # collection (and to this same backend); assert for the type-checker.
            assert collection is not None
            await collection.persist_to_store(pw.data, conn=conn)
            flushed += 1
    return flushed


async def _flush_per_entity(
    sorted_pending: list[PendingWrite],
    write_buffer: WriteBuffer,
    registry: CollectionRegistry,
) -> int:
    """Persist each pending write independently, re-enqueuing on failure.

    The original per-entity flush loop: an unregistered table is skipped, and a failed
    write is re-enqueued under the FK-aware retry policy (FK violations get the generous
    ``_FK_RETRY_LIMIT`` budget; all other failures use ``_MAX_FLUSH_RETRIES``).

    :param sorted_pending: toposorted pending writes.
    :ptype sorted_pending: list[PendingWrite]
    :param write_buffer: the write buffer (for re-enqueue).
    :ptype write_buffer: WriteBuffer
    :param registry: the collection registry.
    :ptype registry: CollectionRegistry
    :return: number of entities successfully persisted.
    :rtype: int
    """
    flushed = 0
    for pw in sorted_pending:
        collection = registry.get_collection(pw.table_name)
        if collection is None:
            log.warning(
                "No collection registered for table, skipping flush",
                extra={"extra_data": {"table": pw.table_name, "entity_id": str(pw.entity_id)}},
            )
            continue
        try:
            await collection.persist_to_store(pw.data)
            flushed += 1
        except Exception as exc:
            # FK violations are "my parent hasn't landed yet" -- treat
            # them as deferral, not failure: re-enqueue with the
            # generous _FK_RETRY_LIMIT budget so the parent has time
            # to land in a subsequent drain. All other errors use the
            # original _MAX_FLUSH_RETRIES budget.
            fk_violation = _is_fk_violation(exc)
            retry_limit = _FK_RETRY_LIMIT if fk_violation else _MAX_FLUSH_RETRIES
            next_retry = pw.retries + 1
            if next_retry >= retry_limit:
                # Permanent drop. For FK violations, this means the
                # parent will never land -- log as an "orphan chain"
                # event so operators can run the conversation repair
                # endpoint (or otherwise reset the cache).
                log.error(
                    "Orphan chain — FK violation exhausted retries, dropping"
                    if fk_violation
                    else "Flush write permanently failed after max retries, dropping",
                    extra={
                        "extra_data": {
                            "table": pw.table_name,
                            "entity_id": str(pw.entity_id),
                            "retries": next_retry,
                            "retry_limit": retry_limit,
                            "fk_violation": fk_violation,
                            "error": str(exc),
                        }
                    },
                )
            else:
                log.warning(
                    "Flush write deferred (FK parent pending), re-adding to buffer"
                    if fk_violation
                    else "Flush write failed, re-adding to buffer for retry",
                    extra={
                        "extra_data": {
                            "table": pw.table_name,
                            "entity_id": str(pw.entity_id),
                            "retry": next_retry,
                            "retry_limit": retry_limit,
                            "fk_violation": fk_violation,
                            "error": str(exc),
                        }
                    },
                )
                await write_buffer.add(pw.table_name, pw.entity_id, pw.data, retries=next_retry)
    log.debug("Flush complete", extra={"extra_data": {"flushed": flushed, "total": len(sorted_pending)}})
    return flushed


async def flush_pending(
    write_buffer: WriteBuffer,
    registry: CollectionRegistry,
    parent_key_map: dict[str, str] | None = None,
) -> int:
    """Drain the write buffer and persist all pending writes to the durable tier.

    **Retry partition (orphan isolation).** After the toposort, pending writes are
    split by their ``retries`` count. Writes with ``retries == 0`` (never failed)
    form the *fresh* set and take the atomic-batch fast path; writes with
    ``retries > 0`` (already failed at least once — e.g. an FK orphan whose parent
    row was deleted and is never coming back) route STRAIGHT to the per-entity loop.
    This keeps a previously-failing write out of the atomic transaction entirely: a
    single un-satisfiable FK among the already-failed writes can never abort the
    batch, so a co-buffered fresh write still commits instead of being dragged into
    per-entity fallback every cycle for the whole ``_FK_RETRY_LIMIT`` budget. The
    already-failed writes keep the per-entity loop's ``_is_fk_violation``
    classification + FK-aware re-enqueue, so a genuinely-transient FK still drains
    once its parent lands.

    **Fresh-set atomic batch.** When every collection in the fresh set shares ONE
    backend that exposes a usable ``transaction()``, the toposorted fresh writes are
    persisted inside a SINGLE ``async with backend.transaction() as conn`` (one DB tx
    for a SQL backend; one commit for a git backend) — each write threading ``conn``
    through ``persist_to_store``. **Graceful degrade**: on ANY exception in the batch
    path the fresh set falls back to the per-entity loop, which keeps the
    ``_is_fk_violation`` classification + re-enqueue intact. A backend without
    ``transaction()`` (e.g. a git-backed ``DurableStore``) degrades to the per-entity
    loop directly. The total returned is the sum of both paths' flushed counts.

    :param write_buffer: the coalescing write buffer to drain.
    :ptype write_buffer: WriteBuffer
    :param registry: the collection registry resolving table → collection + backend.
    :ptype registry: CollectionRegistry
    :param parent_key_map: optional table → parent-FK-column map for toposort.
    :ptype parent_key_map: dict[str, str] | None
    :return: number of entities successfully persisted (both paths summed).
    :rtype: int
    """
    pending = await write_buffer.drain()
    if not pending:
        return 0

    sorted_pending = _toposort_pending(pending, parent_key_map)

    # Partition by retry count: fresh (retries == 0) writes are eligible for the
    # atomic batch; already-failed (retries > 0) writes route straight to the
    # per-entity loop so a poisoned orphan can never abort the fresh batch.
    fresh: list[PendingWrite] = [pw for pw in sorted_pending if pw.retries == 0]
    already_failed: list[PendingWrite] = [pw for pw in sorted_pending if pw.retries > 0]

    flushed = 0

    if fresh:
        backend = _resolve_batch_backend(fresh, registry)
        if backend is not None:
            try:
                batch_flushed = await _flush_batch_atomic(fresh, registry, backend)
                log.debug(
                    "Flush complete (atomic batch)",
                    extra={"extra_data": {"flushed": batch_flushed, "total": len(fresh)}},
                )
                flushed += batch_flushed
            except Exception as exc:
                # Graceful degrade: the whole transaction rolled back, so NOTHING
                # in the fresh set was committed -- replay it through the per-entity
                # loop, which preserves the FK-aware re-enqueue policy per write.
                # The fallback is the safety net, never weakened.
                log.warning(
                    "Atomic batch flush failed, falling back to per-entity flush",
                    extra={"extra_data": {"total": len(fresh), "error": str(exc)}},
                )
                flushed += await _flush_per_entity(fresh, write_buffer, registry)
        else:
            # No single shared transaction-capable backend (e.g. git-backed
            # DurableStore): degrade the fresh set to the per-entity loop directly.
            flushed += await _flush_per_entity(fresh, write_buffer, registry)

    if already_failed:
        # Previously-failed writes are isolated in the per-entity loop so one
        # un-satisfiable FK orphan cannot abort the fresh batch above.
        flushed += await _flush_per_entity(already_failed, write_buffer, registry)

    return flushed
