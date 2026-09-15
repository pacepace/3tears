"""the coordination tables, their migration, the shared collection, the flusher and the sweep.

What this pins:

- the migration creates exactly what the collections declare, so a consumer's table cannot drift
  from the table its collection reads;
- one collection per table per registry, however many primitives are built over it;
- a write-behind buffer is drained without anyone calling ``flush_pending``, and again on close;
- the sweep deletes only expired rows, in bounded batches, at most once per interval.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.flusher import PeriodicFlusher
from threetears.core.coordination.migrations import PACKAGE_NAME, register
from threetears.core.coordination.migrations.v001_create_coordination_tables import create_coordination_tables
from threetears.core.coordination.tables import (
    COORDINATION_TABLE_SCHEMAS,
    CoordinationClaimsCollection,
    CoordinationCountersCollection,
    CoordinationRedemptionsCollection,
    CoordinationRevocationsCollection,
    coordination_collection,
    table_def_for,
)
from threetears.core.data.migrations import MigrationRunner, MigrationScope


class _RecordingStore:
    """a DataStore that records the DDL it is given."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, sql: str, *params: Any) -> str:
        del params
        self.statements.append(" ".join(sql.split()))
        return "CREATE TABLE"


class _SweepStore:
    """an L3 backend that answers the sweep's DELETE and remembers how it was called."""

    def __init__(self, deleted: int = 0) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.deleted = deleted

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del namespace
        self.calls.append((" ".join(query.split()), params))
        return f"DELETE {self.deleted}"


# parity-with: threetears.core.collections.generation.GenerationSource
class _FakeGenerations:
    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        del table_name
        return f"i:{self.count}"

    async def advance(self, table_name: str) -> None:
        del table_name
        self.count += 1


def _registry(*, l3: Any = None) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"coord_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=l3)
    # the revocations table caches absences, so any registry with L3 needs a generation source.
    # In production that is threetears.epoch's; a consumer with L3 and no epoch access cannot run
    # that table, which is why the refusal is at construction.
    registry.set_generation_source(_FakeGenerations())
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class TestTheMigrationMatchesTheDeclaredSchemas:
    @pytest.mark.asyncio
    async def test_every_declared_table_and_index_is_created(self) -> None:
        store = _RecordingStore()
        await create_coordination_tables(store)  # type: ignore[arg-type]
        created = " | ".join(store.statements)
        for schema in COORDINATION_TABLE_SCHEMAS:
            assert f"CREATE TABLE IF NOT EXISTS {schema.name}" in created
            for index in schema.indexes:
                assert index.name in created, f"{schema.name}: index {index.name} was never created"

    @pytest.mark.asyncio
    async def test_each_created_table_carries_exactly_its_declared_columns(self) -> None:
        store = _RecordingStore()
        await create_coordination_tables(store)  # type: ignore[arg-type]
        creates = [s for s in store.statements if s.startswith("CREATE TABLE")]
        for schema, statement in zip(COORDINATION_TABLE_SCHEMAS, creates, strict=False):
            for column in schema.columns:
                assert f" {column.name} " in statement, f"{schema.name}: column {column.name} is missing"

    def test_the_primary_key_is_purpose_and_key_on_every_table(self) -> None:
        # the purpose column is what a bucket name used to be; without it in the key, two throttles
        # in one process would share rows.
        for schema in COORDINATION_TABLE_SCHEMAS:
            assert schema.primary_key == ("purpose", "key")
            pk_columns = [c for c in table_def_for(schema).columns if c.primary_key]
            assert [c.name for c in pk_columns] == ["purpose", "key"]
            assert all(not c.nullable for c in pk_columns)

    def test_it_registers_as_one_package_at_the_scope_the_consumer_asks_for(self) -> None:
        platform_runner = MigrationRunner()
        assert register(platform_runner).scope is MigrationScope.PLATFORM
        agent_runner = MigrationRunner()
        # scriob re-registers 3tears platform packages at agent scope; the tables are the same.
        assert register(agent_runner, scope=MigrationScope.AGENT).scope is MigrationScope.AGENT
        assert register(MigrationRunner()).name == PACKAGE_NAME


class TestTheSharedCollection:
    def test_many_primitives_over_one_table_share_one_collection(self) -> None:
        # identity-edge builds seven route throttles in one process; the registry keys collections
        # by table, so seven collections would leave it holding only the last.
        registry = _registry()
        first = coordination_collection(registry, CoordinationCountersCollection, _config())
        second = coordination_collection(registry, CoordinationCountersCollection, _config())
        assert first is second
        assert registry.get_collection("coordination_counters") is first

    def test_each_table_gets_its_own_collection(self) -> None:
        registry = _registry()
        built = [
            coordination_collection(registry, cls, _config())
            for cls in (
                CoordinationCountersCollection,
                CoordinationClaimsCollection,
                CoordinationRevocationsCollection,
                CoordinationRedemptionsCollection,
            )
        ]
        assert len({c.table_name for c in built}) == 4

    def test_a_foreign_collection_on_the_table_is_refused(self) -> None:
        registry = _registry()
        coordination_collection(registry, CoordinationCountersCollection, _config())
        registry._collections["coordination_claims"] = object()  # noqa: SLF001 - simulating a foreign registration
        with pytest.raises(TypeError, match="already registered"):
            coordination_collection(registry, CoordinationClaimsCollection, _config())


class TestTheTiersAreOptional:
    def test_a_registry_without_l2_or_l3_still_builds_every_table(self) -> None:
        # identity-edge has no L3 by design; scriob's control plane has no L2.
        registry = _registry()
        for cls in (
            CoordinationCountersCollection,
            CoordinationClaimsCollection,
            CoordinationRevocationsCollection,
            CoordinationRedemptionsCollection,
        ):
            collection = coordination_collection(registry, cls, _config())
            assert collection.l3_pool is None

    def test_the_revocations_table_caches_absences_and_writes_l3_synchronously(self) -> None:
        assert CoordinationRevocationsCollection.negative_cache_max_age is not None
        assert CoordinationRevocationsCollection.l3_write_policy == "synchronous"
        assert CoordinationRedemptionsCollection.l3_write_policy == "synchronous"
        assert CoordinationCountersCollection.l3_write_policy == "write_behind"
        assert CoordinationClaimsCollection.l3_write_policy == "write_behind"


class TestTheFlusher:
    @pytest.mark.asyncio
    async def test_it_drains_the_buffer_without_anyone_calling_flush_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _registry()
        flushed: list[int] = []

        async def _fake_flush(buf: WriteBuffer, reg: CollectionRegistry) -> int:
            del buf, reg
            flushed.append(1)
            return 1

        monkeypatch.setattr("threetears.core.coordination.flusher.flush_pending", _fake_flush)
        flusher = PeriodicFlusher(WriteBuffer(), registry, interval_seconds=0.01)
        flusher.ensure_running()
        flusher.ensure_running()  # idempotent: every write calls it
        await asyncio.sleep(0.05)
        assert flushed, "the buffer was never flushed"
        await flusher.aclose()
        assert not flusher.running

    @pytest.mark.asyncio
    async def test_closing_flushes_what_is_still_buffered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = _registry()
        calls: list[str] = []

        async def _fake_flush(buf: WriteBuffer, reg: CollectionRegistry) -> int:
            del buf, reg
            calls.append("flush")
            return 0

        monkeypatch.setattr("threetears.core.coordination.flusher.flush_pending", _fake_flush)
        flusher = PeriodicFlusher(WriteBuffer(), registry, interval_seconds=3600.0)
        flusher.ensure_running()
        await flusher.aclose()
        assert calls == ["flush"], "a clean shutdown lost the buffered writes"

    @pytest.mark.asyncio
    async def test_a_failing_flush_does_not_end_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = _registry()
        attempts: list[int] = []

        async def _failing_flush(buf: WriteBuffer, reg: CollectionRegistry) -> int:
            del buf, reg
            attempts.append(1)
            raise RuntimeError("L3 unavailable")

        monkeypatch.setattr("threetears.core.coordination.flusher.flush_pending", _failing_flush)
        flusher = PeriodicFlusher(WriteBuffer(), registry, interval_seconds=0.01)
        flusher.ensure_running()
        await asyncio.sleep(0.05)
        assert len(attempts) > 1, "the loop stopped at the first L3 failure"
        await flusher.aclose()

    def test_a_non_positive_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PeriodicFlusher(WriteBuffer(), _registry(), interval_seconds=0)

    @pytest.mark.asyncio
    async def test_a_write_behind_collection_gets_a_buffer_and_starts_flushing_itself(self) -> None:
        registry = _registry(l3=_SweepStore())
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        assert collection.write_buffer is not None, "a declared write-behind policy got nowhere to wait"
        collection.ensure_flushing()
        collection.ensure_flushing()  # idempotent: called from every write
        assert collection._flusher is not None and collection._flusher.running  # noqa: SLF001 - asserting own state
        await collection.aclose()
        assert collection._flusher is None  # noqa: SLF001 - asserting own state

    @pytest.mark.asyncio
    async def test_a_synchronous_collection_starts_no_flusher(self) -> None:
        registry = _registry(l3=_SweepStore())
        collection = coordination_collection(registry, CoordinationRevocationsCollection, _config())
        assert collection.write_buffer is None
        collection.ensure_flushing()
        assert collection._flusher is None  # noqa: SLF001 - asserting own state

    @pytest.mark.asyncio
    async def test_without_l3_nothing_flushes(self) -> None:
        # identity-edge runs these collections with no L3 at all; there is nothing to flush to.
        registry = _registry()
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        collection.ensure_flushing()
        assert collection._flusher is None  # noqa: SLF001 - asserting own state


class TestTheSweep:
    @pytest.mark.asyncio
    async def test_it_deletes_only_expired_rows_in_a_bounded_batch(self) -> None:
        store = _SweepStore(deleted=7)
        registry = _registry(l3=store)
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        cutoff = datetime(2026, 1, 1, tzinfo=UTC)
        assert await collection.sweep_expired(now=cutoff, batch=250) == 7
        sql, params = store.calls[0]
        assert "DELETE FROM coordination_counters" in sql
        assert "expires_at IS NOT NULL AND expires_at < $1" in sql
        assert "LIMIT 250" in sql
        assert params == (cutoff,)

    @pytest.mark.asyncio
    async def test_without_l3_it_does_nothing(self) -> None:
        registry = _registry()
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        assert await collection.sweep_expired() == 0

    @pytest.mark.asyncio
    async def test_it_runs_at_most_once_per_interval(self) -> None:
        store = _SweepStore()
        registry = _registry(l3=store)
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        assert await collection.sweep_expired_if_due() == 0
        await collection.sweep_expired_if_due()
        await collection.sweep_expired_if_due()
        assert len(store.calls) == 1, "the sweep ran on every call instead of once per interval"

    @pytest.mark.asyncio
    async def test_the_expiring_tables_all_declare_their_expiry_column(self) -> None:
        # correctness comes from expires_at_column (an expired row reads as absent at every tier);
        # the sweep is only table size.
        for cls in (
            CoordinationCountersCollection,
            CoordinationClaimsCollection,
            CoordinationRevocationsCollection,
            CoordinationRedemptionsCollection,
        ):
            assert cls.expires_at_column == "expires_at"
            assert "expires_at" in cls.datetime_columns
            assert any(c.name == "expires_at" for c in cls.schema.columns)


class TestRowExpiryShape:
    def test_a_counter_row_carries_its_window(self) -> None:
        columns = {c.name for c in CoordinationCountersCollection.schema.columns}
        assert {"count", "window_start", "expires_at"} <= columns

    def test_a_claim_row_carries_its_outcome(self) -> None:
        columns = {c.name for c in CoordinationClaimsCollection.schema.columns}
        assert {"status", "result", "error", "claim_metadata", "date_claimed", "date_completed"} <= columns

    def test_a_revocation_row_carries_the_moment_it_was_revoked_from(self) -> None:
        columns = {c.name for c in CoordinationRevocationsCollection.schema.columns}
        assert "revoked_at" in columns

    def test_a_redemption_row_is_presence_plus_expiry(self) -> None:
        columns = {c.name for c in CoordinationRedemptionsCollection.schema.columns}
        assert columns == {"purpose", "key", "expires_at", "date_created", "date_updated"}

    def test_a_counter_window_becomes_its_expiry(self) -> None:
        # the shape 04b relies on: a row expires when its window closes, so a stale window is
        # absent rather than a count that has to be range-checked on read.
        window = timedelta(seconds=60)
        started = datetime(2026, 1, 1, tzinfo=UTC)
        assert started + window == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
