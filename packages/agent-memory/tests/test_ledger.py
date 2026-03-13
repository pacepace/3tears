"""Tests for MemoryLedger -- CRUD, dedup, eviction."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from threetears.agent.memory.ledger import MemoryLedger


def _make_pool_mock(rows: list[dict] | None = None) -> AsyncMock:
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows or [])
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    return pool


class TestLedgerAddAndIds:
    async def test_add_ref_and_ledgered_ids(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()
        item_id = str(uuid.uuid7())

        await ledger.add_ref(pool, conv_id, item_id, "memory", "some desc")

        assert item_id in ledger.ledgered_ids
        assert len(ledger) == 1

    async def test_add_multiple_refs(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()

        ids = [str(uuid.uuid7()) for _ in range(3)]
        for i, item_id in enumerate(ids):
            await ledger.add_ref(pool, conv_id, item_id, "memory", f"desc {i}")

        assert ledger.ledgered_ids == set(ids)
        assert len(ledger) == 3


class TestLedgerDedup:
    async def test_adding_same_id_is_idempotent(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()
        item_id = str(uuid.uuid7())

        await ledger.add_ref(pool, conv_id, item_id, "memory", "first desc")
        await ledger.add_ref(pool, conv_id, item_id, "memory", "second desc")

        assert len(ledger) == 1
        # Should keep the first description
        context = ledger.build_context()
        assert "first desc" in context


class TestLedgerEviction:
    async def test_evicts_oldest_at_capacity(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()

        # Fill to capacity
        ids = []
        for i in range(MemoryLedger.MAX_SIZE):
            item_id = str(uuid.uuid7())
            ids.append(item_id)
            await ledger.add_ref(pool, conv_id, item_id, "memory", f"item {i}")

        assert len(ledger) == MemoryLedger.MAX_SIZE

        # Add one more -- should evict the oldest (ids[0])
        new_id = str(uuid.uuid7())
        await ledger.add_ref(pool, conv_id, new_id, "memory", "new item")

        assert len(ledger) == MemoryLedger.MAX_SIZE
        assert ids[0] not in ledger.ledgered_ids
        assert new_id in ledger.ledgered_ids

    async def test_eviction_calls_delete(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()

        for i in range(MemoryLedger.MAX_SIZE):
            await ledger.add_ref(pool, conv_id, str(uuid.uuid7()), "memory", f"item {i}")

        pool.execute.reset_mock()
        await ledger.add_ref(pool, conv_id, str(uuid.uuid7()), "memory", "overflow")

        # Should have called execute twice: one DELETE (eviction) + one INSERT
        assert pool.execute.await_count == 2


class TestLedgerDescTruncation:
    async def test_long_desc_truncated_to_150(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()
        item_id = str(uuid.uuid7())
        long_desc = "x" * 200

        await ledger.add_ref(pool, conv_id, item_id, "memory", long_desc)

        context = ledger.build_context()
        # The stored desc should be at most 150 chars
        # Find the desc in context after the " — " separator
        for line in context.split("\n"):
            if item_id in line:
                desc_part = line.split(" — ", 1)[1] if " — " in line else ""
                assert len(desc_part) <= 150


class TestLedgerBuildContext:
    async def test_format(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()
        item_id = str(uuid.uuid7())

        await ledger.add_ref(pool, conv_id, item_id, "memory", "user likes cats")

        context = ledger.build_context()
        assert "Previously recalled" in context
        assert f"[memory:{item_id}]" in context
        assert "type: memory" in context
        assert "user likes cats" in context

    def test_empty_ledger_returns_empty_string(self) -> None:
        ledger = MemoryLedger()
        assert ledger.build_context() == ""


class TestLedgerLoad:
    async def test_load_populates_refs(self) -> None:
        item_id = uuid.uuid7()
        rows = [
            {
                "item_id": item_id,
                "item_type": "memory",
                "short_desc": "loaded desc",
                "date_added": datetime.now(timezone.utc),
            }
        ]
        pool = _make_pool_mock(rows)
        ledger = MemoryLedger()

        await ledger.load(pool, uuid.uuid7())

        assert str(item_id) in ledger.ledgered_ids
        assert len(ledger) == 1

    async def test_load_multiple_rows(self) -> None:
        rows = [
            {
                "item_id": uuid.uuid7(),
                "item_type": "memory",
                "short_desc": f"desc {i}",
                "date_added": datetime.now(timezone.utc),
            }
            for i in range(5)
        ]
        pool = _make_pool_mock(rows)
        ledger = MemoryLedger()

        await ledger.load(pool, uuid.uuid7())

        assert len(ledger) == 5


class TestLedgerLen:
    def test_empty(self) -> None:
        assert len(MemoryLedger()) == 0

    async def test_after_adds(self) -> None:
        ledger = MemoryLedger()
        pool = _make_pool_mock()
        conv_id = uuid.uuid7()

        await ledger.add_ref(pool, conv_id, str(uuid.uuid7()), "memory", "a")
        await ledger.add_ref(pool, conv_id, str(uuid.uuid7()), "chunk", "b")

        assert len(ledger) == 2
