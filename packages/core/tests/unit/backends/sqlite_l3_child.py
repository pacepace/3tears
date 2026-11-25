"""a process that writes the SQLite L3 once, says so, and waits to be killed (``test_sqlite_l3_survives_kill``).

Run as ``python sqlite_l3_child.py <path> <kind> <key>``. It writes one kind of write through the collection a
deployment builds (or the store, for the kinds a collection reaches through it), prints ``acked <kind>`` once the
write has returned, and sleeps until the test kills it with SIGKILL: no ``close()``, no ``atexit``, nothing flushed
after the acknowledgement.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from threetears.core.backends import SqliteL3Backend
from threetears.core.cache import SQLiteBackend
from threetears.core.collections import NO_L2
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.l1_tables import TEST_L1_TABLE

LEDGER = TableSchema(
    name="ledger",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("state", STRING_TYPE),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    cas_column="date_updated",
    l1_eviction=TEST_L1_TABLE,
)
LedgerCollection = collection_for_schema(LEDGER)


def deployment(path: Path) -> tuple[SqliteL3Backend, object]:
    """the store, opened, and the collection over it with no L2."""
    store = SqliteL3Backend(path)
    store.open()
    registry = CollectionRegistry()
    registry.configure(l1_backend=SQLiteBackend(f"child_{uuid.uuid4().hex[:8]}"), l3_pool=store)
    return store, LedgerCollection(registry, DefaultCoreConfig(), NO_L2)


async def write(path: Path, kind: str, key: uuid.UUID) -> None:
    store, ledger = deployment(path)
    now = datetime.now(UTC)
    if kind == "insert":
        await ledger.save_entity(ledger.create({"id": key, "state": "new", "date_created": now, "date_updated": now}))
    elif kind == "update":
        entity = await ledger.get(key)
        assert entity is not None
        entity.state = "updated"
        await ledger.save_entity(entity)
    elif kind == "delete":
        await ledger.delete(key)
    elif kind == "many":
        rows = [
            {"id": uuid.uuid5(key, str(n)), "state": "many", "date_created": now, "date_updated": now}
            for n in range(100)
        ]
        await store.upsert_many("ledger", rows, max_rows=100, max_bytes=1_000_000)
    elif kind == "transaction":
        async with store.transaction() as conn:
            for n in range(10):
                row = {"id": uuid.uuid5(key, str(n)), "state": "tx", "date_created": now, "date_updated": now}
                await store.upsert("ledger", row, conn=conn)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    print(f"acked {kind}", flush=True)
    await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(write(Path(sys.argv[1]), sys.argv[2], uuid.UUID(sys.argv[3])))
