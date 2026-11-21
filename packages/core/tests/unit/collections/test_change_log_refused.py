"""a write generation a pod's tenants cannot read is a wiring error, said at ERROR once; an outage stays a warning.

A collection that caches absences reads its table's write generation before every miss. When the pod's change-log
reader cannot read the table's keys (no key for its tenants, or a key the broker refuses: the pod's schema is not the
one its host grants), no retry fixes it. So the collection says so at ERROR, once per table, naming the keys, the
tenants and the fix, and asks L3 as it would in an outage. A real outage keeps its rate-limited warning.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import BaseCollection, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ChangeLogKeyRefusedError, GenerationUnavailableError
from threetears.core.tenancy import PodTenants
from threetears.core.testing.kv import FakeNatsClient
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_KEEP

from packages.core.tests.durable_support import NoL3

_SCOPE = "tool_pod-0192f0c47a1070008000000000000c0d"
_LOGGER = "threetears.core.collections.base"


def _metadata(table: str) -> MetaData:
    metadata = MetaData()
    Table(
        table,
        metadata,
        Column("id", String, primary_key=True),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
        info={L1_EVICTION_INFO: TEST_L1_KEEP},
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Absences(BaseCollection[_Row]):
    """a collection that caches absences, so every miss reads its table's write generation first."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=60)
    _table: str = "refused_test_rows"

    def __init__(self, registry: CollectionRegistry, *, table: str) -> None:
        self._table = table
        super().__init__(registry, DefaultCoreConfig())

    @property
    def table_name(self) -> str:
        return self._table

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _FailingSource:
    """a generation source whose every read fails with ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.reads = 0

    async def current(self, table_name: str) -> str:
        self.reads += 1
        raise self.error

    async def advance(self, table_name: str) -> str:
        raise AssertionError("nothing here writes")


def _collection(source: _FailingSource) -> _Absences:
    table = f"refused_{uuid.uuid4().hex[:10]}"
    l1 = SQLiteBackend(db_name=f"refused_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata(table))
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=FakeNatsClient(), l3_pool=NoL3(), kv_key_scope=_SCOPE)
    registry.set_generation_source(source)
    return _Absences(registry, table=table)


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == _LOGGER and record.levelno == level]


async def test_a_refused_change_log_is_said_at_error_once_with_the_keys_tenants_and_fix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenants = PodTenants(schema="ns_wrong")
    refusal = ChangeLogKeyRefusedError("refused", table="placeholder", keys=("s-ns_wrong.t.e",), tenants=tenants)
    source = _FailingSource(refusal)
    collection = _collection(source)
    refusal.table = collection.table_name

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await collection.get("a") is None
        assert await collection.get("b") is None

    errors = _records(caplog, logging.ERROR)
    assert len(errors) == 1, "one ERROR per table, however many reads meet it"
    said = errors[0]
    assert "PodTenants.schema" in said.getMessage(), "the line names the fix"
    extra = said.__dict__["extra_data"]
    assert extra["table"] == collection.table_name
    assert extra["keys"] == ["s-ns_wrong.t.e"]
    assert extra["tenants"] == repr(tenants)
    assert not _records(caplog, logging.WARNING), "a wiring error never reads as a passing outage"
    assert source.reads == 2, "each read still asks L3 rather than trusting an absence"


async def test_an_outage_keeps_its_warning_and_says_no_error(caplog: pytest.LogCaptureFixture) -> None:
    collection = _collection(_FailingSource(GenerationUnavailableError("the change log could not be read")))

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await collection.get("a") is None

    assert not _records(caplog, logging.ERROR)
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "write generation unavailable" in warnings[0].getMessage()
