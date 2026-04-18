"""Write-buffer and flush strategy for deferred collection persistence."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Any, NamedTuple, TYPE_CHECKING

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

_MAX_FLUSH_RETRIES = 10


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


async def flush_pending(
    write_buffer: WriteBuffer,
    registry: CollectionRegistry,
    parent_key_map: dict[str, str] | None = None,
) -> int:
    """Drain the write buffer and persist all pending writes to Postgres."""
    pending = await write_buffer.drain()
    if not pending:
        return 0

    sorted_pending = _toposort_pending(pending, parent_key_map)

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
            await collection.persist_to_postgres(pw.data)
            flushed += 1
        except Exception as exc:
            next_retry = pw.retries + 1
            if next_retry >= _MAX_FLUSH_RETRIES:
                log.error(
                    "Flush write permanently failed after max retries, dropping",
                    extra={
                        "extra_data": {
                            "table": pw.table_name,
                            "entity_id": str(pw.entity_id),
                            "retries": next_retry,
                            "error": str(exc),
                        }
                    },
                )
            else:
                log.warning(
                    "Flush write failed, re-adding to buffer for retry",
                    extra={
                        "extra_data": {
                            "table": pw.table_name,
                            "entity_id": str(pw.entity_id),
                            "retry": next_retry,
                            "error": str(exc),
                        }
                    },
                )
                await write_buffer.add(pw.table_name, pw.entity_id, pw.data, retries=next_retry)
    log.debug("Flush complete", extra={"extra_data": {"flushed": flushed, "total": len(pending)}})
    return flushed
