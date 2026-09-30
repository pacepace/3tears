"""a multi-row intention scan writes no cache tier: its entities hold their own rows.

A scan reads L3 outside the per-key fence ``get`` reads under. An entity built from it that
wrote its row into L1 left L1 holding whatever the scan read -- older than a write of the key
that landed while the scan was in flight, with nothing left to evict it. ``intention_log``'s
dedup refresh then reads the want with ``get`` and saves the stale row back over L3.

Each test lands a write of the key while the scan's query is in flight (the eviction a peer's
broadcast delivers) and checks that the scan left no row in L1, while its entity still reads
every field, addresses the composite key, and saves fenced on the version it read.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Float, MetaData, String, Table, Text
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.intention.collections import IntentionsCollection

_READ_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _l1_metadata() -> MetaData:
    meta = MetaData()
    Table(
        "intentions",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("intention_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("status", String(50)),
        Column("content", Text),
        Column("embedding", Text),
        Column("salience", Float),
        Column("last_decayed_at", DateTime),
        Column("last_surfaced_at", DateTime),
        Column("source_memory_id", String(255)),
        Column("source_conversation_id", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return meta


def _row(agent_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, Any]:
    return {
        "intention_id": uuid.uuid4(),
        "agent_id": agent_id,
        "customer_id": uuid.uuid4(),
        "user_id": user_id,
        "status": "open",
        "content": "learn the user's timezone",
        "embedding": None,
        "salience": 0.5,
        "last_decayed_at": None,
        "last_surfaced_at": None,
        "source_memory_id": None,
        "source_conversation_id": None,
        "date_created": _READ_AT,
        "date_updated": _READ_AT,
    }


# parity-with: asyncpg.Pool (fetch: the one call a scan makes)
class _WriteLandsDuringScan:
    """an L3 whose scan is overtaken by a write of the row it returns.

    While the query is in flight the row's key is evicted, exactly as a peer's broadcast of a
    newer write evicts it; the scan still answers with the row as it stood before that write.
    """

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.collection: IntentionsCollection | None = None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        del sql, args
        assert self.collection is not None
        self.collection.evict_from_cache_sync((self.row["agent_id"], self.row["intention_id"]))
        return [dict(self.row)]


def _collection(pool: _WriteLandsDuringScan) -> IntentionsCollection:
    l1 = SQLiteBackend(db_name=f"intentions_scan_{uuid.uuid4().hex[:8]}")
    l1.initialize(_l1_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)  # type: ignore[arg-type]
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = IntentionsCollection(registry, config)
    pool.collection = collection
    return collection


@pytest.mark.parametrize("scan", ["find_by_user", "find_open_for_deliberation"])
async def test_a_scan_leaves_l1_alone_and_its_entity_holds_the_row(scan: str) -> None:
    agent_id, user_id = uuid.uuid4(), uuid.uuid4()
    row = _row(agent_id, user_id)
    pool = _WriteLandsDuringScan(row)
    collection = _collection(pool)
    key = (agent_id, row["intention_id"])

    if scan == "find_by_user":
        entities = await collection.find_by_user(user_id, agent_id=agent_id)
    else:
        entities = await collection.find_open_for_deliberation(user_id, agent_id=agent_id, cooldown_cutoff=_READ_AT)

    assert collection.get_row_sync(key) is None, "the scan cached a row a write had already replaced"
    [entity] = entities
    assert entity.content == "learn the user's timezone"
    assert entity.addressing_id == key
    assert entity.original_date_updated == _READ_AT
