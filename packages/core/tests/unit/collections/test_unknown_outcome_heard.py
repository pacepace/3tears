"""a write whose outcome is unknown is announced before its error is raised: every epoch heard.

A statement or ``COMMIT`` whose reply never came (a dropped connection, a timeout, a cancellation) may have landed.
The write path announces a row only after its write returns, so without this every other pod would go on serving the
row L3 no longer holds. ``BaseCollection._write_heard`` announces each row such a write may have written, then raises
the write's own error; a write the database refused with its own error did not land and announces nothing.

Asserted on the invalidation broadcast, which is what reaches every other replica.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_TABLE = "unknown_outcome_rows"


class _Row(BaseEntity):
    primary_key_field = "id"


class _Rows(BaseCollection[_Row]):
    """a collection whose store raises what it is told to, as a write whose reply never came would."""

    raises: BaseException | None = None

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        del entity_id
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        del data, original_timestamp, conn
        if self.raises is not None:
            raise self.raises
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        del entity_id
        if self.raises is not None:
            raise self.raises

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


def _rows(nats: FakeNatsClient, raises: BaseException) -> _Rows:
    registry = CollectionRegistry()
    registry.configure(l3_pool=object(), kv_key_scope="unknown-outcome")  # type: ignore[arg-type]
    rows = _Rows(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client=nats)
    rows.raises = raises
    return rows


def _heard(nats: FakeNatsClient) -> list[str]:
    return [
        row.ids[0]
        for message in nats.published
        if isinstance(message, CacheInvalidationMessage) and message.table == _TABLE
        for row in message.each_row()
    ]


class TestAnUnknownOutcomeIsHeard:
    async def test_a_delete_whose_reply_never_came_is_announced_and_its_error_raised(self) -> None:
        """Flips if a delete that raised with its outcome unknown announces nothing: it may have landed."""
        nats = FakeNatsClient()
        rows = _rows(nats, ConnectionResetError("connection lost during COMMIT"))

        with pytest.raises(ConnectionResetError, match="during COMMIT"):
            await rows.delete("a")

        assert _heard(nats) == ["a"]

    async def test_a_save_whose_reply_never_came_is_announced_and_its_error_raised(self) -> None:
        """Flips if a save that raised with its outcome unknown announces nothing: every pod keeps the old row."""
        nats = FakeNatsClient()
        rows = _rows(nats, TimeoutError("no reply from L3"))
        entity = rows.create({"id": "b", "value": 1})

        with pytest.raises(TimeoutError):
            await rows.save_entity(entity)

        assert _heard(nats) == ["b"]

    async def test_a_write_the_database_refused_announces_nothing(self) -> None:
        """Flips if a write the database answered with its own error (it did not land) is announced as one."""
        nats = FakeNatsClient()
        rows = _rows(nats, asyncpg.UniqueViolationError("duplicate key"))

        with pytest.raises(asyncpg.UniqueViolationError):
            await rows.delete("c")

        assert _heard(nats) == []
