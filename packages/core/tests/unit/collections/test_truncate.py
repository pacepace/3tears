"""An operator's reset: L3 emptied with TRUNCATE alone, every cache of it dropped, other pods told by its generation.

The contract this pins:

- ``reset()`` empties the table through its store's ``truncate``, outside any transaction, naming its reason;
- its epoch record names ``reason="reset"``, so a follower reads the table again as that table's cold start;
- ``truncate()`` on a table whose epochs are recorded is refused before anything is emptied (a load clears key
  prefixes instead); on one whose are not, it empties the table as before;
- this process's L1 rows go, and its L2 entries are read past until each is read through again;
- a switched-on table's write generation advances once per call, with no row broadcast, so a pod
  following the table finds an advance it heard no row of and drops the table too;
- calling it twice in a row is safe;
- a collection that cannot keep the contract is refused before anything is emptied.
"""

from __future__ import annotations

from collections.abc import Sequence

import ast
import asyncio
import inspect
import uuid
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import MetaData

import threetears.core

from threetears.core.backends.protocol import DurableStore, TruncatingStore
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.generation import GenerationVerdict
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import DataLayerUnavailableError, GenerationUnavailableError
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import set_default_namespace

SCHEMA = TableSchema(
    name="lsf_results",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

_Results = collection_for_schema(SCHEMA)


class _SwitchedOnResults(_Results):  # type: ignore[misc,valid-type]
    write_generation = WRITE_GENERATION


class _WriteBehindResults(_Results):  # type: ignore[misc,valid-type]
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("truncateprobe")


class _TableStore(DurableStore):
    """an L3 of one table held in a dict, which a TRUNCATE empties whole."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.truncated: list[str] = []
        self.reasons: list[str | None] = []
        self.fetches = 0
        #: raised after the table is emptied, as a reply lost after the broker ran the TRUNCATE is
        self.lost_reply: BaseException | None = None

    async def fetch_one(self, table: str, pk: Any, *, conn: Any = None) -> dict[str, Any] | None:
        self.fetches += 1
        row = self.rows.get(str(pk["id"]))
        return dict(row) if row is not None else None

    async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
        self.rows[str(row["id"])] = dict(row)
        return 1

    async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
        self.rows.pop(str(pk["id"]), None)

    async def scan(self, table: str, filters: Any = None) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows.values()]

    async def truncate(self, table: str, *, reason: str | None = None) -> None:
        self.truncated.append(table)
        self.reasons.append(reason)
        self.rows.clear()
        if self.lost_reply is not None:
            raise self.lost_reply


class _UntruncatableStore(_TableStore):
    """a durable store with no TRUNCATE: a git working tree, say."""

    truncate = None  # type: ignore[assignment]


class _CountingSource:
    """a generation source that counts its advances, and can be made to refuse them."""

    def __init__(self) -> None:
        self.changes: dict[tuple[str, str], EpochChange] = {}
        self.counts: dict[str, int] = {}
        self.failing = False

    async def current(self, table_name: str) -> str:
        return f"inc:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        if self.failing:
            raise GenerationUnavailableError(f"generation for {table_name!r} is unreachable")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc:{self.counts[table_name]}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        reason: str | None = None,
    ) -> str:
        token = await self.advance(table_name)
        self.changes[(table_name, token)] = EpochChange(
            table=table_name,
            token=token,
            whole_table=rows is None and not keys_unknown,
            keys_unknown=keys_unknown,
            rows=list(rows) if rows is not None else [],
            reason=reason,
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.changes.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Pod:
    """one registry with its own L1, on a bus and an L3 it shares with every other pod of the test."""

    def __init__(
        self,
        bus: FakeNatsClient,
        store: _TableStore,
        *,
        cls: type[Any] = _SwitchedOnResults,
        source: _CountingSource | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = MetaData()
        SCHEMA.to_sqlalchemy_table(metadata)
        l1 = SQLiteBackend(db_name=f"truncate_{uuid.uuid4().hex[:8]}")
        l1.initialize(metadata)
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=store, kv_key_scope="truncate-principal")  # type: ignore[arg-type]
        if source is not None:
            self.registry.set_generation_source(source)
        self.results = cls(self.registry, DefaultCoreConfig(), **kwargs)

    async def follow(self, source: _CountingSource) -> None:
        """follow the table and take the first mark, as a pod does once at startup."""
        self.registry.follow_generation(SCHEMA.name)
        await self.registry.catch_up_generation(SCHEMA.name, await source.current(SCHEMA.name), changes=source)


@pytest.fixture
def bus() -> FakeNatsClient:
    return FakeNatsClient()


@pytest.fixture
def store() -> _TableStore:
    return _TableStore()


@pytest.fixture
def source() -> _CountingSource:
    return _CountingSource()


async def _save(pod: _Pod, entity_id: str, votes: int) -> None:
    await pod.results.save_entity(pod.results.create({"id": entity_id, "votes": votes}))


def test_every_sql_backend_can_truncate() -> None:
    from threetears.core.backends.sql import SqlL3Backend

    assert isinstance(SqlL3Backend(object()), TruncatingStore)


async def test_reset_empties_the_table_through_its_store(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await _save(pod, "b", 2)
    await pod.results.reset()
    assert store.truncated == ["lsf_results"]
    assert store.rows == {}


async def test_this_process_s_l1_rows_leave(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    assert pod.results.exists_in_cache_sync("a")
    await pod.results.reset()
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_live_l2_entry_is_read_past_so_no_tier_answers_a_truncated_row(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    # the row is still in L2, which a pod cannot list to delete: after the truncate a read goes past
    # it to L3, which holds nothing
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await pod.results.reset()
    fetches = store.fetches
    assert await pod.results.ensure("a") is None
    assert store.fetches == fetches + 1


async def test_a_switched_on_table_advances_its_generation_once_and_broadcasts_no_row(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    published = len(bus.published)
    await pod.results.reset()
    assert source.counts["lsf_results"] == 2  # the save's advance, then the truncate's
    assert len(bus.published) == published


async def test_a_pod_following_the_table_drops_it_at_its_next_catch_up(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    await _save(writer, "a", 1)
    # no listener runs here, so the follower catches up on the save before it reads the row
    await follower.registry.catch_up_generation("lsf_results", await source.current("lsf_results"), changes=source)
    assert (await follower.results.ensure("a")) is not None
    assert follower.results.exists_in_cache_sync("a")
    await writer.results.reset()
    # TRUNCATE leaves no write mark and no row broadcast: the advance alone tells the follower
    verdict = await follower.registry.catch_up_generation(
        "lsf_results", await source.current("lsf_results"), changes=source
    )
    assert verdict is GenerationVerdict.MISSED
    assert not follower.results.exists_in_cache_sync("a")
    assert await follower.results.ensure("a") is None


async def test_the_writer_does_not_drop_its_own_advance_again(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await pod.follow(source)
    await pod.results.reset()
    verdict = await pod.registry.catch_up_generation("lsf_results", await source.current("lsf_results"), changes=source)
    assert verdict is GenerationVerdict.CURRENT


async def test_truncating_twice_in_a_row_is_safe(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    await pod.results.reset()
    await pod.results.reset()
    assert store.truncated == ["lsf_results", "lsf_results"]
    assert source.counts["lsf_results"] == 3
    assert await pod.results.ensure("a") is None


async def test_a_table_saved_again_after_a_truncate_reads_the_new_row(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await pod.results.reset()
    await _save(pod, "a", 7)
    row = await pod.results.ensure("a")
    assert row is not None and row["votes"] == 7


async def test_a_generation_that_cannot_advance_raises_after_the_table_is_emptied_and_dropped(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    source.failing = True
    with pytest.raises(GenerationUnavailableError):
        await pod.results.reset()
    assert store.rows == {}
    assert not pod.results.exists_in_cache_sync("a")


async def test_an_undeclared_table_resets_and_advances_nothing(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, cls=_Results, source=source)
    await _save(pod, "a", 1)
    await pod.results.reset()
    assert store.rows == {}
    assert source.counts == {}
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_collection_with_no_durable_store_is_refused(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    pod.results.l3_pool = None
    with pytest.raises(ValueError, match="no durable store"):
        await pod.results.reset()


async def test_a_store_that_cannot_truncate_is_refused(bus: FakeNatsClient) -> None:
    store = _UntruncatableStore()
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    with pytest.raises(ValueError, match="cannot truncate"):
        await pod.results.reset()
    assert pod.results.exists_in_cache_sync("a")


class _Buffer:
    """a write buffer: present, so a write-behind collection really defers."""


async def test_a_collection_deferring_its_l3_writes_is_refused_before_anything_is_emptied(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    pod = _Pod(bus, store, cls=_WriteBehindResults, write_buffer=_Buffer())
    store.rows["a"] = {"id": "a", "votes": 1}
    with pytest.raises(ValueError, match="defers its L3 writes"):
        await pod.results.reset()
    assert store.truncated == []
    assert store.rows


async def test_a_truncate_whose_reply_is_lost_still_drops_and_advances_and_raises_its_failure(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    await _save(writer, "a", 1)
    await follower.registry.catch_up_generation("lsf_results", await source.current("lsf_results"), changes=source)
    assert (await follower.results.ensure("a")) is not None
    store.lost_reply = DataLayerUnavailableError("l3.truncate: no reply")
    with pytest.raises(DataLayerUnavailableError, match="no reply"):
        await writer.results.reset()
    assert not writer.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the failed truncate's
    verdict = await follower.registry.catch_up_generation(
        "lsf_results", await source.current("lsf_results"), changes=source
    )
    assert verdict is GenerationVerdict.MISSED
    assert not follower.results.exists_in_cache_sync("a")


async def test_a_failed_truncate_s_own_failure_wins_over_a_generation_that_cannot_advance(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.lost_reply = DataLayerUnavailableError("l3.truncate: no reply")
    source.failing = True
    with pytest.raises(DataLayerUnavailableError, match="no reply"):
        await pod.results.reset()
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_truncate_cancelled_after_the_broker_ran_it_drops_and_advances_then_is_cancelled_still(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.lost_reply = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await pod.results.reset()
    assert store.rows == {}
    assert not pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the cancelled truncate's


async def test_the_one_way_to_empty_a_table_whole_is_an_operator_s_reset(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    """plain truncate() is retired (coordinator 2026-11-12): a load clears by key prefix or deletes rows by key; only an
    operator's reset() empties a table whole, and its epoch says so."""
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    assert not hasattr(pod.results, "truncate")
    await pod.results.reset()
    assert store.truncated == ["lsf_results"]
    assert store.reasons == ["reset"]


async def test_the_store_s_truncate_is_reached_only_from_an_operator_s_reset(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    """the proof of the operator_reset declarations on SqlL3Backend.truncate's TRUNCATE and on
    SchemaBackedCollection._truncate_l3's store.truncate: every write, read, delete and catch-up a pod makes leaves the
    store's truncate untouched, and reset() reaches it once, through _truncate_l3, naming the reset; nothing in the
    package source calls a store's truncate but _truncate_l3 (and SqlL3Backend.truncate handing a TRUNCATE to its
    pool)."""
    calls: list[list[str]] = []
    truncate = store.truncate

    async def recorded(table: str, *, reason: str | None = None) -> None:
        calls.append([frame.function for frame in inspect.stack()[1:6]])
        await truncate(table, reason=reason)

    store.truncate = recorded  # type: ignore[method-assign]
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    for n in range(3):
        await _save(writer, f"k{n}", n)
    await writer.results.ensure("k1")
    await writer.results.delete("k0")
    await follower.registry.catch_up_generation("lsf_results", await source.current("lsf_results"), changes=source)
    await follower.results.ensure("k2")
    assert calls == []
    await writer.results.reset()
    assert len(calls) == 1 and calls[0][:2] == ["_truncate_l3", "reset"], calls
    assert store.reasons == ["reset"]

    callers = []
    source_root = Path(threetears.core.__file__).parent
    for path in sorted(source_root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "truncate"
                    ):
                        callers.append(f"{path.name}:{node.name}")
    assert sorted(set(callers)) == ["schema_backed.py:_truncate_l3", "sql.py:truncate"], callers


async def test_a_resets_epoch_names_its_reason_and_the_store_is_told_it(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    await pod.results.reset()
    assert store.reasons == ["reset"]
    record = source.changes[("lsf_results", "inc:2")]
    assert record.whole_table and record.reason == "reset" and record.prefixes is None


async def test_a_follower_hears_a_reset_as_a_reset_never_as_an_older_writers_whole_record(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    heard: list[str] = []
    follower.registry.register_derived_cache(
        "lsf_results",
        on_row=lambda message: None,
        on_table_dropped=lambda: heard.append("dropped"),
        on_table_reset=lambda: heard.append("reset"),
    )
    await writer.results.reset()
    await follower.registry.catch_up_generation("lsf_results", await source.current("lsf_results"), changes=source)
    assert heard == ["reset"]
