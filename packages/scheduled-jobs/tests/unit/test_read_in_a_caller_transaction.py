"""the scheduled-jobs collections read on a caller's transaction with their own SELECT and row shape."""

from __future__ import annotations

from typing import Any

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection


class _Reads:
    """a pool, or a connection, that answers every row read with one row and records the read."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        self.reads.append((query, params))
        return {"partition_key": params[0], "id": params[1]}


@pytest.mark.parametrize("cls", [ScheduledJobCollection, JobFireCollection])
async def test_the_read_in_a_transaction_is_the_store_read_on_the_caller_s_connection(cls: type[Any]) -> None:
    pool, conn = _Reads(), _Reads()
    registry = CollectionRegistry()
    l3_pool: Any = pool  # a recording stand-in for the pool's one read
    registry.configure(l3_pool=l3_pool)
    collection = cls(
        registry=registry,
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=None,
    )

    in_transaction = await collection.fetch_from_store_in_transaction(("p", "j"), conn=conn)
    assert pool.reads == [], "the read went to the pool, which cannot see the caller's transaction"
    stored = await collection.fetch_from_store(("p", "j"))

    assert conn.reads == pool.reads, "the read in a transaction is not the store read"
    assert in_transaction == stored == {"partition_key": "p", "id": "j"}
