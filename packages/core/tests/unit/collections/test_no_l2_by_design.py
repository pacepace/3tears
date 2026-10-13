"""A collection that runs without L2 on purpose says so once at INFO; one that lacks a client by accident still warns."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import pytest

from threetears.core.collections import NO_L2
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

_LOGGER = "threetears.core.collections.base"


def _schema() -> TableSchema:
    # a table of its own per test: the one-shot log lines are process-wide per table
    return TableSchema(
        name=f"layer_{uuid.uuid4().hex[:10]}",
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
    )


def _collection(schema: TableSchema, nats_client: Any, registry: CollectionRegistry | None = None) -> Any:
    registry = registry or CollectionRegistry()
    registry.configure(kv_key_scope="test-principal")
    return collection_for_schema(schema)(registry, DefaultCoreConfig(), nats_client)


def _lines(caplog: pytest.LogCaptureFixture, table: str, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelno == level and table in r.getMessage()
    ]


async def test_a_collection_without_l2_by_design_never_warns(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    schema = _schema()
    collection = _collection(schema, NO_L2)

    await collection.invalidate_cache("a")
    await collection.invalidate_cache_many(["b", "c"])

    assert _lines(caplog, schema.name, logging.WARNING) == []
    assert not collection.broadcasts_invalidations


async def test_it_says_once_at_info_that_it_runs_without_l2(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    schema = _schema()
    registry = CollectionRegistry()
    first = _collection(schema, NO_L2, registry)
    _collection(schema, NO_L2, CollectionRegistry())
    await first.invalidate_cache("a")

    [line] = _lines(caplog, schema.name, logging.INFO)
    assert "without L2 by design" in line


async def test_a_missing_client_without_the_declaration_still_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    schema = _schema()
    collection = _collection(schema, None)

    await collection.invalidate_cache("a")
    await collection.invalidate_cache_many(["b", "c"])

    [line] = _lines(caplog, schema.name, logging.WARNING)
    assert "silently disabled" in line


async def test_the_declaration_ignores_an_l2_client_the_registry_offers() -> None:
    schema = _schema()
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal", l2_client=FakeNatsClient())

    collection = collection_for_schema(schema)(registry, DefaultCoreConfig(), NO_L2)

    assert not collection.broadcasts_invalidations
