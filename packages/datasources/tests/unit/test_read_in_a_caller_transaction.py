"""each declared datasources collection reads on a caller's transaction with its own SELECT and row shape."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from threetears.datasources.collections import (
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceSchemaDigestCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)


class _Reads:
    """a pool, or a connection, that answers every row read with one row and records the read."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        self.reads.append((query, params))
        return {"key": params[0], "tags": ["a"]}


@pytest.mark.parametrize(
    "cls",
    [
        DataSourceTableCollection,
        DataSourceColumnCollection,
        DataSourceSchemaDigestCollection,
        DataSourceRelationCollection,
        TableTemplateCollection,
    ],
)
async def test_the_read_in_a_transaction_is_the_store_read_on_the_caller_s_connection(cls: type[Any]) -> None:
    pool, conn = _Reads(), _Reads()
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = cls(registry=registry, config=config)
    key = uuid4()

    in_transaction = await collection.fetch_from_store_in_transaction(key, conn=conn)
    assert pool.reads == [], "the read went to the pool, which cannot see the caller's transaction"
    stored = await collection.fetch_from_store(key)

    assert conn.reads == pool.reads, "the read in a transaction is not the store read"
    assert in_transaction == stored == {"key": key, "tags": ["a"]}
