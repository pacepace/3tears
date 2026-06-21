"""Tests for WriteBuffer, _toposort_pending, and flush_pending."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from threetears.core.collections.flush import (
    _FK_RETRY_LIMIT,
    _MAX_FLUSH_RETRIES,
    PendingWrite,
    WriteBuffer,
    _is_fk_violation,
    _toposort_pending,
    flush_pending,
)
from threetears.core.collections.registry import CollectionRegistry


class TestWriteBuffer:
    """Tests for WriteBuffer."""

    @pytest.mark.asyncio
    async def test_add_and_drain(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1", "name": "Alice"})
        await buf.add("users", "u2", {"id": "u2", "name": "Bob"})

        items = await buf.drain()

        assert len(items) == 2
        table_names = {pw.table_name for pw in items}
        assert table_names == {"users"}

    @pytest.mark.asyncio
    async def test_drain_clears_buffer(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})

        await buf.drain()
        items = await buf.drain()

        assert items == []

    @pytest.mark.asyncio
    async def test_remove(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})
        await buf.add("users", "u2", {"id": "u2"})

        removed = await buf.remove("users", "u1")

        assert removed is True
        assert buf.pending_count() == 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self) -> None:
        buf = WriteBuffer()

        removed = await buf.remove("users", "missing")

        assert removed is False

    @pytest.mark.asyncio
    async def test_pending_count(self) -> None:
        buf = WriteBuffer()
        assert buf.pending_count() == 0

        await buf.add("users", "u1", {"id": "u1"})
        assert buf.pending_count() == 1

        await buf.add("users", "u2", {"id": "u2"})
        assert buf.pending_count() == 2

    @pytest.mark.asyncio
    async def test_coalesces_duplicate_keys(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1", "name": "Alice"})
        await buf.add("users", "u1", {"id": "u1", "name": "Alice Updated"})

        assert buf.pending_count() == 1
        items = await buf.drain()
        assert len(items) == 1
        assert items[0].data["name"] == "Alice Updated"


class TestToposortPending:
    """Tests for _toposort_pending."""

    def test_no_deps_tables_come_first(self) -> None:
        pending = [
            PendingWrite("messages", "m1", {"id": "m1", "parent_message_id": None}),
            PendingWrite("users", "u1", {"id": "u1"}),
        ]

        result = _toposort_pending(pending)

        # users has no deps so it should come before messages
        assert result[0].table_name == "users"

    def test_parents_before_children(self) -> None:
        pending = [
            PendingWrite("messages", "m2", {"id": "m2", "parent_message_id": "m1"}),
            PendingWrite("messages", "m1", {"id": "m1", "parent_message_id": None}),
        ]

        result = _toposort_pending(pending)

        ids = [pw.entity_id for pw in result]
        assert ids.index("m1") < ids.index("m2")

    def test_chain_ordering(self) -> None:
        """m1 -> m2 -> m3 should preserve order."""
        pending = [
            PendingWrite("messages", "m3", {"id": "m3", "parent_message_id": "m2"}),
            PendingWrite("messages", "m1", {"id": "m1", "parent_message_id": None}),
            PendingWrite("messages", "m2", {"id": "m2", "parent_message_id": "m1"}),
        ]

        result = _toposort_pending(pending)

        ids = [pw.entity_id for pw in result]
        assert ids.index("m1") < ids.index("m2")
        assert ids.index("m2") < ids.index("m3")

    def test_custom_parent_key_map(self) -> None:
        pending = [
            PendingWrite("replies", "r2", {"id": "r2", "reply_to": "r1"}),
            PendingWrite("replies", "r1", {"id": "r1", "reply_to": None}),
        ]

        result = _toposort_pending(pending, parent_key_map={"replies": "reply_to"})

        ids = [pw.entity_id for pw in result]
        assert ids.index("r1") < ids.index("r2")

    def test_no_pending_returns_empty(self) -> None:
        result = _toposort_pending([])
        assert result == []

    def test_only_non_dep_tables(self) -> None:
        pending = [
            PendingWrite("users", "u1", {"id": "u1"}),
            PendingWrite("settings", "s1", {"id": "s1"}),
        ]

        result = _toposort_pending(pending)

        assert len(result) == 2

    def test_parent_outside_pending_treated_as_root(self) -> None:
        """If parent_message_id points to an ID not in pending, treat as root."""
        pending = [
            PendingWrite("messages", "m5", {"id": "m5", "parent_message_id": "m_external"}),
        ]

        result = _toposort_pending(pending)

        assert len(result) == 1
        assert result[0].entity_id == "m5"


class TestFlushPending:
    """Tests for flush_pending."""

    @pytest.mark.asyncio
    async def test_drains_and_persists(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1", "name": "Alice"})

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "users"
        mock_coll.persist_to_store = AsyncMock(return_value=1)
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 1
        mock_coll.persist_to_store.assert_awaited_once_with({"id": "u1", "name": "Alice"})

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "users"
        mock_coll.persist_to_store = AsyncMock(side_effect=RuntimeError("db down"))
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        # Should have been re-added to buffer for retry
        assert buf.pending_count() == 1
        items = await buf.drain()
        assert items[0].retries == 1

    @pytest.mark.asyncio
    async def test_drops_after_max_retries(self) -> None:
        buf = WriteBuffer()
        # Add with retries already at max - 1
        await buf.add("users", "u1", {"id": "u1"}, retries=9)

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "users"
        mock_coll.persist_to_store = AsyncMock(side_effect=RuntimeError("db down"))
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        # Should NOT be re-added (permanently dropped)
        assert buf.pending_count() == 0

    @pytest.mark.asyncio
    async def test_empty_buffer_returns_zero(self) -> None:
        buf = WriteBuffer()
        registry = CollectionRegistry()

        flushed = await flush_pending(buf, registry)

        assert flushed == 0

    @pytest.mark.asyncio
    async def test_unregistered_table_skipped(self) -> None:
        buf = WriteBuffer()
        await buf.add("unknown_table", "x1", {"id": "x1"})

        registry = CollectionRegistry()

        flushed = await flush_pending(buf, registry)

        assert flushed == 0

    @pytest.mark.asyncio
    async def test_multiple_tables(self) -> None:
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})
        await buf.add("settings", "s1", {"id": "s1"})

        registry = CollectionRegistry()
        user_coll = MagicMock()
        user_coll.table_name = "users"
        user_coll.persist_to_store = AsyncMock(return_value=1)
        registry.register(user_coll)

        settings_coll = MagicMock()
        settings_coll.table_name = "settings"
        settings_coll.persist_to_store = AsyncMock(return_value=1)
        registry.register(settings_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 2


class TestIsFkViolation:
    """Tests for ``_is_fk_violation`` — detection of FK errors so the
    retry policy can grant them the generous ``_FK_RETRY_LIMIT``
    budget instead of dropping after ``_MAX_FLUSH_RETRIES``.
    """

    def test_typed_asyncpg_exception_returns_true(self) -> None:
        """asyncpg.exceptions.ForeignKeyViolationError is the canonical
        match -- detected via isinstance, no string fallback needed.
        """
        exc = asyncpg.exceptions.ForeignKeyViolationError(
            "insert or update on table violates foreign key constraint",
        )
        assert _is_fk_violation(exc) is True

    def test_substring_match_in_generic_exception_returns_true(self) -> None:
        """A non-asyncpg exception whose ``str()`` contains the marker
        text still counts -- defense-in-depth for wrappers or re-raised
        exceptions that lose the typed class but preserve the message.
        """
        exc = RuntimeError(
            'insert or update on table "messages" violates foreign key constraint "messages_parent_message_id_fkey"',
        )
        assert _is_fk_violation(exc) is True

    def test_unrelated_exception_returns_false(self) -> None:
        """Generic exceptions whose message lacks the marker substring
        are NOT FK violations -- the normal retry budget applies.
        """
        assert _is_fk_violation(RuntimeError("db connection lost")) is False
        assert _is_fk_violation(ValueError("bad input")) is False

    def test_empty_message_returns_false(self) -> None:
        """An exception with no useful message string returns False (no
        substring match) — exercises the substring branch's empty case.
        """
        assert _is_fk_violation(Exception()) is False


class TestFkAwareRetryPolicy:
    """Tests for the FK-aware retry policy in ``flush_pending``: FK
    violations use the generous ``_FK_RETRY_LIMIT`` budget while all
    other failures use ``_MAX_FLUSH_RETRIES``. Critic 2026-05-13
    flagged the lack of direct coverage as block-worthy; this class
    closes the gap.
    """

    @pytest.mark.asyncio
    async def test_fk_violation_uses_extended_retry_budget(self) -> None:
        """A pending write that fails with an FK violation re-enqueues
        even when ``retries`` is at the GENERAL ``_MAX_FLUSH_RETRIES``
        boundary -- because FK errors get the larger ``_FK_RETRY_LIMIT``
        budget. Otherwise a single Anthropic-class outage would orphan
        every descendant in the conversation tree (the 2026-05-13
        metallm incident fingerprint).
        """
        buf = WriteBuffer()
        # Sit exactly at the general retry boundary -- one more failure
        # would drop under the OLD policy, but the FK-specific budget
        # should still allow re-enqueue.
        await buf.add(
            "messages",
            "m1",
            {"id": "m1"},
            retries=_MAX_FLUSH_RETRIES - 1,
        )

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "messages"
        mock_coll.persist_to_store = AsyncMock(
            side_effect=asyncpg.exceptions.ForeignKeyViolationError(
                "violates foreign key constraint",
            ),
        )
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        # Re-enqueued, NOT dropped -- the FK budget hasn't been
        # exhausted (still well below _FK_RETRY_LIMIT).
        assert buf.pending_count() == 1
        items = await buf.drain()
        assert items[0].retries == _MAX_FLUSH_RETRIES

    @pytest.mark.asyncio
    async def test_fk_violation_drops_at_fk_retry_limit(self) -> None:
        """Once the FK-specific budget is exhausted the write IS dropped
        permanently with the ``Orphan chain`` log signal so operators
        can run the metallm conversation-repair endpoint. Verifies the
        upper bound of the new policy.
        """
        buf = WriteBuffer()
        await buf.add(
            "messages",
            "m1",
            {"id": "m1"},
            retries=_FK_RETRY_LIMIT - 1,
        )

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "messages"
        mock_coll.persist_to_store = AsyncMock(
            side_effect=asyncpg.exceptions.ForeignKeyViolationError(
                "violates foreign key constraint",
            ),
        )
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        assert buf.pending_count() == 0  # permanently dropped

    @pytest.mark.asyncio
    async def test_non_fk_violation_keeps_general_retry_budget(self) -> None:
        """A non-FK exception (e.g. connection lost) drops at the
        existing ``_MAX_FLUSH_RETRIES`` boundary, NOT the extended FK
        budget. Confirms the FK branch is narrow and doesn't accidentally
        grant unbounded retries to every transient failure class.
        """
        buf = WriteBuffer()
        # One short of the general boundary -- last legitimate retry.
        await buf.add(
            "users",
            "u1",
            {"id": "u1"},
            retries=_MAX_FLUSH_RETRIES - 1,
        )

        registry = CollectionRegistry()
        mock_coll = MagicMock()
        mock_coll.table_name = "users"
        mock_coll.persist_to_store = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )
        registry.register(mock_coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        assert buf.pending_count() == 0  # dropped at _MAX_FLUSH_RETRIES


# parity-exempt: a bare transaction-connection stub (records persisted rows); it mirrors no production protocol
class _FakeConn:
    """A fake transactional connection that records the entities written through it."""

    def __init__(self) -> None:
        self.persisted: list[dict[str, object]] = []


# parity-with: threetears.core.backends.protocol.DurableStore
class _FakeTxBackend:
    """A DurableStore-conformant backend exposing a real async ``transaction()``.

    Stands in for a ``SqlL3Backend`` / git backend in the flush atomic-batch tests:
    the ``transaction()`` CM yields a connection and records commit/rollback, so the
    tests can prove the whole batch threaded ONE connection (happy path) and that an
    in-batch failure rolls back + degrades to the per-entity loop.
    """

    def __init__(self, *, fail_on_enter: bool = False) -> None:
        self._fail_on_enter = fail_on_enter
        self.conn = _FakeConn()
        self.entered = 0
        self.committed = False
        self.rolled_back = False

    # DurableStore surface (unused by these tests but needed for isinstance passthrough)
    async def fetch_one(self, table: str, pk: object, *, conn: object = None) -> dict[str, object] | None:
        return None

    async def upsert(self, table: str, row: object, **kwargs: object) -> int:
        return 1

    async def delete(self, table: str, pk: object, *, conn: object = None) -> None:
        return None

    async def scan(self, table: str, filters: object = None) -> list[dict[str, object]]:
        return []

    def transaction(self, namespace: str | None = None) -> _FakeTxBackend:
        return self

    async def __aenter__(self) -> _FakeConn:
        self.entered += 1
        if self._fail_on_enter:
            raise RuntimeError("transaction unavailable")
        return self.conn

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True
        return False  # never swallow — propagate so flush_pending can fall back


def _batch_collection(table: str, backend: _FakeTxBackend) -> MagicMock:
    """A mock collection whose ``persist_to_store`` records on the conn it is handed."""
    coll = MagicMock()
    coll.table_name = table

    async def _persist(data: dict[str, object], *, conn: object = None) -> int:
        assert conn is backend.conn  # the batch path threads the transaction connection
        backend.conn.persisted.append(data)
        return 1

    coll.persist_to_store = AsyncMock(side_effect=_persist)
    return coll


class TestFlushAtomicBatch:
    """flush_pending persists a single-backend batch in ONE transaction (L3B-04)."""

    @pytest.mark.asyncio
    async def test_happy_path_persists_whole_batch_in_one_transaction(self) -> None:
        """all pending writes share one backend → one ``transaction()`` threads every persist."""
        backend = _FakeTxBackend()
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})
        await buf.add("users", "u2", {"id": "u2"})

        registry = CollectionRegistry()
        registry.configure(l3_pool=backend)  # DurableStore → passes through un-wrapped
        registry.register(_batch_collection("users", backend))

        flushed = await flush_pending(buf, registry)

        assert flushed == 2
        assert backend.entered == 1  # exactly ONE transaction for the whole batch
        assert backend.committed is True
        assert len(backend.conn.persisted) == 2
        assert buf.pending_count() == 0  # nothing re-enqueued

    @pytest.mark.asyncio
    async def test_batch_failure_rolls_back_and_falls_back_to_per_entity_loop(self) -> None:
        """a failure INSIDE the batch transaction rolls back, then degrades to per-entity.

        The per-entity loop's FK-aware re-enqueue must remain intact: the write is
        re-added (retries incremented) rather than dropped, exactly as the existing
        per-entity path does.
        """
        backend = _FakeTxBackend()
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})

        registry = CollectionRegistry()
        registry.configure(l3_pool=backend)

        coll = MagicMock()
        coll.table_name = "users"

        async def _persist(data: dict[str, object], *, conn: object = None) -> int:
            # fail whether called inside the batch (conn set) or the fallback (conn None)
            raise RuntimeError("db down")

        coll.persist_to_store = AsyncMock(side_effect=_persist)
        registry.register(coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 0
        assert backend.entered == 1  # the batch path was attempted
        assert backend.rolled_back is True  # the in-batch failure rolled the transaction back
        # fallback per-entity loop re-enqueued under the general retry budget
        assert buf.pending_count() == 1
        items = await buf.drain()
        assert items[0].retries == 1

    @pytest.mark.asyncio
    async def test_transaction_unavailable_falls_back_to_per_entity_loop(self) -> None:
        """failure ENTERING the batch transaction also degrades to the per-entity loop."""
        backend = _FakeTxBackend(fail_on_enter=True)
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})

        registry = CollectionRegistry()
        registry.configure(l3_pool=backend)

        coll = MagicMock()
        coll.table_name = "users"
        coll.persist_to_store = AsyncMock(return_value=1)
        registry.register(coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 1
        assert backend.entered == 1  # the batch path was attempted (enter raised)
        # fallback persisted per-entity WITHOUT a conn
        coll.persist_to_store.assert_awaited_once_with({"id": "u1"})

    @pytest.mark.asyncio
    async def test_backend_without_transaction_uses_per_entity_loop(self) -> None:
        """a backend lacking ``transaction()`` (e.g. a git DurableStore) skips the batch path."""
        buf = WriteBuffer()
        await buf.add("users", "u1", {"id": "u1"})

        registry = CollectionRegistry()
        # a DurableStore with NO transaction() attribute
        no_tx_backend = MagicMock(spec=["fetch_one", "upsert", "delete", "scan"])
        registry.configure(l3_pool=no_tx_backend)

        coll = MagicMock()
        coll.table_name = "users"
        coll.persist_to_store = AsyncMock(return_value=1)
        registry.register(coll)

        flushed = await flush_pending(buf, registry)

        assert flushed == 1
        # per-entity loop persists WITHOUT a conn (no transaction to thread)
        coll.persist_to_store.assert_awaited_once_with({"id": "u1"})
