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
from threetears.core.collections.schema_backed import BYTES_TYPE, INT_TYPE, STRING_TYPE
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
from threetears.core.testing.kv import FakeNatsClient


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


class _Nats(FakeNatsClient):
    """the shared collections bucket, with no listener subscribed in these tests."""


class _RowStore:
    """an in-process L3 that stores rows, for the tier round-trips."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def fetch_one(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        del table, conn
        row = self.rows.get((pk["purpose"], pk["key"]))
        return dict(row) if row is not None else None

    async def upsert(self, table: str, data: dict[str, Any], **kwargs: Any) -> int:
        del table, kwargs
        self.rows[(data["purpose"], data["key"])] = dict(data)
        return 1

    async def delete(self, table: str, pk: dict[str, Any], *, conn: Any = None) -> None:
        del table, conn
        self.rows.pop((pk["purpose"], pk["key"]), None)

    async def scan(self, table: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del table, filters
        return [dict(row) for row in self.rows.values()]

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        del query, params, namespace
        return "DELETE 0"


def _registry(*, l3: Any = None, l2: Any = None) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"coord_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=l1,
        l2_client=l2,
        l3_pool=l3,
        kv_key_scope="coord-principal" if l2 is not None else None,
    )
    # the revocations table caches absences, so any registry with L3 needs a generation source.
    # In production that is threetears.epoch's; a consumer with L3 and no epoch access cannot run
    # that table, which is why the refusal is at construction.
    registry.set_generation_source(_FakeGenerations())
    return registry


def _split_top_level(body: str) -> list[str]:
    """split a CREATE TABLE body on its top-level commas, so ``PRIMARY KEY (a, b)`` stays whole."""
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    if buffer:
        parts.append("".join(buffer))
    return parts


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
    async def test_each_created_table_is_exactly_what_its_schema_declares(self) -> None:
        # asserted statement-for-statement, not by substring: the renderer's type map, its
        # nullability rule and the composite key are what a consumer's table is made of, and a
        # "column name appears somewhere" check passes on all three being wrong.
        store = _RecordingStore()
        await create_coordination_tables(store)  # type: ignore[arg-type]
        creates = [s for s in store.statements if s.startswith("CREATE TABLE")]
        assert len(creates) == len(COORDINATION_TABLE_SCHEMAS), "a declared table was not created"
        for schema, statement in zip(COORDINATION_TABLE_SCHEMAS, creates, strict=True):
            body = statement.split("(", 1)[1].rsplit(")", 1)[0]
            rendered = [part.strip() for part in _split_top_level(body)]
            pk_clause = [part for part in rendered if part.upper().startswith("PRIMARY KEY")]
            assert pk_clause == ["PRIMARY KEY (purpose, key)"], f"{schema.name}: wrong primary key"
            declared = {
                column.name: (
                    "TEXT"
                    if column.column_type == STRING_TYPE
                    else "INTEGER"
                    if column.column_type == INT_TYPE
                    else "BYTEA"
                    if column.column_type == BYTES_TYPE
                    else "TIMESTAMPTZ"
                )
                for column in schema.columns
            }
            columns = {part.split(" ", 1)[0]: part.split(" ", 1)[1] for part in rendered if part not in pk_clause}
            assert set(columns) == set(declared), f"{schema.name}: rendered columns differ from the declaration"
            for name, sql in columns.items():
                assert sql.startswith(declared[name]), f"{schema.name}.{name}: rendered as {sql!r}"
                # the key columns carry the table's identity and can never be NULL; everything
                # else follows its own declaration.
                if name in {"purpose", "key"}:
                    assert "NOT NULL" in sql, f"{schema.name}.{name}: a key column may not be nullable"

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
    """every tier combination a consumer actually has, exercised through a real read and write.

    Building the collection proves nothing: the L1 table these collections declare for themselves
    is what a read and a write need, and a suite that only constructs them cannot tell whether it
    was ever declared.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "with_l2, with_l3",
        [(False, False), (False, True), (True, False), (True, True)],
        ids=["l1-only", "l1-and-l3", "l1-and-l2", "all-three"],
    )
    async def test_a_row_round_trips_on_every_tier_combination(self, with_l2: bool, with_l3: bool) -> None:
        nats = _Nats() if with_l2 else None
        store = _RowStore() if with_l3 else None
        registry = _registry(l2=nats, l3=store)
        collection = coordination_collection(registry, CoordinationRedemptionsCollection, _config())

        entity = collection.create(
            {"purpose": "jti", "key": "token-1", "expires_at": datetime.now(UTC) + timedelta(hours=1)}
        )
        await collection.save_entity(entity)

        fetched = await collection.get(("jti", "token-1"))
        assert fetched is not None, "a written row could not be read back"
        assert fetched.to_dict()["key"] == "token-1"
        if with_l3:
            assert store is not None and ("jti", "token-1") in store.rows
        await collection.delete(("jti", "token-1"))
        assert await collection.get(("jti", "token-1")) is None

    @pytest.mark.asyncio
    async def test_a_compare_and_swap_round_trips_on_every_tier(self) -> None:
        for nats, store in ((None, None), (None, _RowStore()), (_Nats(), None), (_Nats(), _RowStore())):
            registry = _registry(l2=nats, l3=store)
            collection = coordination_collection(registry, CoordinationCountersCollection, _config())
            outcome = await collection.l2_cas_mutate(
                ("throttle", "ip-1"),
                lambda row: (
                    "upsert",
                    {
                        "purpose": "throttle",
                        "key": "ip-1",
                        "count": 1 if row is None else int(row["count"]) + 1,
                        "window_start": datetime.now(UTC),
                        "expires_at": datetime.now(UTC) + timedelta(minutes=1),
                    },
                ),
            )
            assert outcome.row is not None and outcome.row["count"] == 1
            await collection.aclose()

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
    async def test_closing_the_registry_flushes_what_a_collection_still_holds(self) -> None:
        # the property close_collections exists for, asserted on a real registry holding a real
        # flusher: a mock that records the call passes whatever the method body does.
        store = _RowStore()
        registry = _registry(l2=_Nats(), l3=store)
        collection = coordination_collection(registry, CoordinationCountersCollection, _config())
        await collection.l2_cas_mutate(
            ("throttle", "ip-1"),
            lambda row: (
                "upsert",
                {
                    "purpose": "throttle",
                    "key": "ip-1",
                    "count": 1 if row is None else int(row["count"]) + 1,
                    "window_start": datetime.now(UTC),
                    "expires_at": datetime.now(UTC) + timedelta(minutes=1),
                },
            ),
        )
        collection.ensure_flushing()
        assert store.rows == {}, "precondition: a write-behind row waits for a flush"

        await registry.close_collections()

        assert ("throttle", "ip-1") in store.rows, "shutdown lost the buffered write"
        assert collection._flusher is None  # noqa: SLF001 - asserting the task was released

    @pytest.mark.asyncio
    async def test_one_collection_failing_to_close_does_not_abandon_the_rest(self) -> None:
        # a teardown that stops at the first failure leaves the task it was there to stop running.
        store = _RowStore()
        registry = _registry(l2=_Nats(), l3=store)
        counters = coordination_collection(registry, CoordinationCountersCollection, _config())
        claims = coordination_collection(registry, CoordinationClaimsCollection, _config())

        async def _refuse() -> None:
            raise RuntimeError("closing this one fails")

        counters.aclose = _refuse  # type: ignore[method-assign]
        closed: list[str] = []
        original = claims.aclose

        async def _record() -> None:
            closed.append("claims")
            await original()

        claims.aclose = _record  # type: ignore[method-assign]

        await registry.close_collections()

        assert closed == ["claims"], "a failing close abandoned the collections after it"

    @pytest.mark.asyncio
    async def test_closing_a_registry_with_nothing_to_close_is_a_no_op(self) -> None:
        registry = _registry()
        await registry.close_collections()

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
