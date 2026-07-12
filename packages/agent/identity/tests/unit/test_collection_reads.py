"""Unit coverage for the read methods' no-L3 fail-safe branches.

The full read behaviour (resolve_active / find_versions_for_block /
find_pending against a real pgvector container, plus the lifecycle
mutations) is exercised in the T2.1b integration suite. This file pins
the cheap, container-free contract: with no L3 pool configured, every
read degrades to empty / ``None`` rather than raising.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text

from threetears.agent.identity.collections import IdentityVersionsCollection
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

pytestmark = pytest.mark.asyncio

_AGENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
_USER = uuid.UUID("00000000-0000-0000-0000-00000000000b")


def _l1_metadata() -> MetaData:
    """L1 mirror of identity_versions (key-addressed row store; text-ish types)."""
    meta = MetaData()
    Table(
        "identity_versions",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("version_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("block_key", String(50)),
        Column("content", Text),
        Column("rationale", Text),
        Column("content_hash", String(255)),
        Column("parent_version_id", String(255)),
        Column("status", String(50)),
        Column("proposer_agent_id", String(255)),
        Column("consenter_user_id", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return meta


def _collection_without_l3() -> IdentityVersionsCollection:
    l1 = SQLiteBackend(db_name=f"identity_{uuid.uuid4().hex[:8]}")
    l1.initialize(_l1_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, l2_client=AsyncMock(), l3_pool=None)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return IdentityVersionsCollection(registry=reg, config=cfg, nats_client=AsyncMock())


async def test_resolve_active_none_without_l3() -> None:
    coll = _collection_without_l3()
    result = await coll.resolve_active(
        agent_id=_AGENT, customer_id=_USER, user_id=_USER, block_key="personality"
    )
    assert result is None


async def test_find_versions_empty_without_l3() -> None:
    coll = _collection_without_l3()
    result = await coll.find_versions_for_block(
        agent_id=_AGENT, customer_id=_USER, user_id=_USER, block_key="presence"
    )
    assert result == []


async def test_find_pending_empty_without_l3() -> None:
    coll = _collection_without_l3()
    result = await coll.find_pending(agent_id=_AGENT, user_id=_USER)
    assert result == []
