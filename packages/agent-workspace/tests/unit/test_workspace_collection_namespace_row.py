"""unit tests for :meth:`WorkspaceCollection._save_to_postgres`.

three-tier-task-01 phase F retired the paired
``platform.namespaces`` insert that used to ride this method: the
workspace create tool now emits the namespace row through
:meth:`NamespaceCollection.save_entity` after the workspace tx
commits. this test file asserts the collection's own ``_save_to_postgres``
is a single-statement write against the ``workspaces`` table — no
``platform.namespaces`` insert, no transaction wrap for a second
write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text

from threetears.agent.workspace.collections import WorkspaceCollection


def _workspaces_metadata() -> MetaData:
    """build SQLite metadata describing workspaces table.

    :return: populated SQLAlchemy metadata
    :rtype: MetaData
    """
    metadata = MetaData()
    Table(
        "workspaces",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("agent_id", String(64)),
        Column("name", String(255)),
        Column("description", Text),
        Column("template_name", String(255)),
        Column("created_by", String(64)),
        Column("current_version", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
        Column("date_deleted", DateTime),
    )
    return metadata


@pytest.fixture()
def workspaces_l1() -> SQLiteBackend:
    """build SQLite L1 backend with workspaces schema.

    :yield: initialized SQLite backend
    :rtype: SQLiteBackend
    """
    backend = SQLiteBackend(db_name=f"test_ws_ns_{uuid4().hex[:8]}")
    backend.initialize(_workspaces_metadata())
    yield backend
    backend.reset()


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    """return core config that flushes writes to L3 immediately.

    :return: configured core config
    :rtype: DefaultCoreConfig
    """
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.mark.asyncio
async def test_save_issues_single_workspaces_upsert(
    workspaces_l1: SQLiteBackend,
    config_always: DefaultCoreConfig,
) -> None:
    """_save_to_postgres issues exactly one statement against workspaces.

    phase F moved the paired ``platform.namespaces`` write to the
    create tool, so the collection's own upsert is a single
    single-target write. this test drives the method directly and
    asserts nothing else lands.

    :param workspaces_l1: pod-local SQLite L1 backend fixture
    :ptype workspaces_l1: SQLiteBackend
    :param config_always: flush-every-write config fixture
    :ptype config_always: DefaultCoreConfig
    :return: nothing
    :rtype: None
    """
    pool = AsyncMock()
    pool.execute = AsyncMock(return_value="INSERT 0 1")

    registry = CollectionRegistry()
    registry.configure(l1_backend=workspaces_l1)
    collection = WorkspaceCollection(
        registry=registry,
        config=config_always,
        postgres_pool=pool,
    )

    now = datetime.now(UTC).replace(tzinfo=None)
    data: dict[str, Any] = {
        "id": uuid4(),
        "agent_id": uuid4(),
        "name": "ws-under-test",
        "description": None,
        "template_name": None,
        "created_by": uuid4(),
        "current_version": 0,
        "date_created": now,
        "date_updated": now,
        "date_deleted": None,
        # customer_id + schema_name intentionally present to prove
        # they do NOT drive a second write on this path any more
        "customer_id": uuid4(),
        "schema_name": "agent_test",
    }

    affected = await collection._save_to_postgres(data, original_timestamp=None)

    assert affected == 1
    pool.execute.assert_awaited_once()
    sql_text = pool.execute.await_args.args[0]
    assert "INSERT INTO workspaces" in sql_text
    assert "platform.namespaces" not in sql_text
