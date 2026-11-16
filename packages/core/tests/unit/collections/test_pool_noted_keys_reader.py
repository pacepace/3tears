"""A direct pool's reader of the change notes: it is a registry's noted-keys reader, and a role that may not read the
ring leaves the registry behind rather than failing its catch-up pass."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections import CollectionRegistry
from threetears.core.collections.change_notes import NotedKeysReader, PoolNotedKeysReader


class _Refused:
    """one connection whose role may read nothing of the ring."""

    def transaction(self) -> Any:
        raise AssertionError("a page opens no transaction of its own")

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        raise asyncpg.InsufficientPrivilegeError("permission denied for table threetears_change_note_ring")

    async def execute(self, query: str, *params: Any) -> str:
        raise asyncpg.InsufficientPrivilegeError("permission denied")

    async def fetchrow(self, query: str, *params: Any) -> Any:
        raise asyncpg.InsufficientPrivilegeError("permission denied for table threetears_change_note_ring")


def test_it_is_a_noted_keys_reader_a_registry_takes() -> None:
    reader = PoolNotedKeysReader(object(), notes_schema="platform", every_tenant=True)
    assert isinstance(reader, NotedKeysReader)
    registry = CollectionRegistry()
    registry.set_noted_keys_reader(reader)


async def test_a_role_that_may_not_read_the_ring_is_answered_notes_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flips if a refused read of the ring raises out of the reader: the registry's catch-up pass would fail on it,
    where it should be behind on the table (``notes_unavailable``), said at WARNING."""
    now = datetime.now(UTC)
    reader = PoolNotedKeysReader(_Refused(), notes_schema="platform", every_tenant=True)
    page = await reader.noted_keys(
        "results", noted_between=(now - timedelta(minutes=1), now), cursor=None, limit=10, max_bytes=1_000
    )
    assert page == "notes_unavailable"
    assert any("may not read the change notes" in record.getMessage() for record in caplog.records)
