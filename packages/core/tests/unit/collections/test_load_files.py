"""A table loaded from files: the store loads, every cache of it is dropped, other pods told by its generation.

The contract this pins, the same as ``truncate()``'s after the store has run:

- ``load_files()`` hands the files, columns, constants and bounds to its store's ``load_files``,
  outside any transaction, and answers the rows the store wrote;
- this process's L1 rows go, and its L2 entries are read past until each is read through again;
- a switched-on table's write generation advances once per load, with no row broadcast, so a pod
  following the table drops it at its next catch-up;
- a collection that cannot keep the contract is refused before anything is loaded.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import MetaData

from threetears.core.backends.protocol import DurableStore, FileLoad, LoadFile, LoadResult
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
from threetears.core.config import DefaultCoreConfig
from threetears.core.exceptions import GenerationUnavailableError
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

_WHEN = datetime(2026, 10, 22, 12, tzinfo=UTC)
_FILES = [LoadFile(url="https://bucket/0000.parquet", rows=2), LoadFile(url="https://bucket/0001.parquet", rows=1)]


class _SwitchedOnResults(_Results):  # type: ignore[misc,valid-type]
    write_generation = WRITE_GENERATION


class _WriteBehindResults(_Results):  # type: ignore[misc,valid-type]
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("loadprobe")


class _TableStore(DurableStore):
    """an L3 of one table held in a dict, which a load from files fills with the rows its files name."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.loads: list[dict[str, Any]] = []
        self.fetches = 0
        #: the rows each file holds, by url, which the load writes
        self.contents: dict[str, list[dict[str, Any]]] = {}

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

    async def load_files(
        self,
        table: str,
        files: Sequence[LoadFile],
        *,
        columns: Sequence[str],
        constants: Mapping[str, Any],
        rows_per_transaction: int,
        timeout_ms: int,
    ) -> LoadResult:
        self.loads.append(
            {
                "table": table,
                "files": list(files),
                "columns": list(columns),
                "constants": dict(constants),
                "rows_per_transaction": rows_per_transaction,
                "timeout_ms": timeout_ms,
            }
        )
        loaded = []
        for index, file in enumerate(files):
            for row in self.contents.get(file.url, []):
                self.rows[str(row["id"])] = {**row, **constants}
            loaded.append(FileLoad(index=index, rows_loaded=file.rows))
        return LoadResult(row_count=sum(file.rows for file in files), files=tuple(loaded))


class _UnloadableStore(_TableStore):
    """a durable store that cannot load files: a git working tree, say."""

    load_files = None  # type: ignore[assignment]


class _CountingSource:
    """a generation source that counts its advances, and can be made to refuse them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failing = False

    async def current(self, table_name: str) -> str:
        return f"inc:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        if self.failing:
            raise GenerationUnavailableError(f"generation for {table_name!r} is unreachable")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc:{self.counts[table_name]}"


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
        l1 = SQLiteBackend(db_name=f"load_{uuid.uuid4().hex[:8]}")
        l1.initialize(metadata)
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=store, kv_key_scope="load-principal")  # type: ignore[arg-type]
        if source is not None:
            self.registry.set_generation_source(source)
        self.results = cls(self.registry, DefaultCoreConfig(), **kwargs)

    async def follow(self, source: _CountingSource) -> None:
        """follow the table and take the first mark, as a pod does once at startup."""
        self.registry.follow_generation(SCHEMA.name)
        self.registry.settle_generation(SCHEMA.name, await source.current(SCHEMA.name))


@pytest.fixture
def bus() -> FakeNatsClient:
    return FakeNatsClient()


@pytest.fixture
def store() -> _TableStore:
    held = _TableStore()
    held.contents = {
        _FILES[0].url: [{"id": "a", "votes": 10}, {"id": "b", "votes": 20}],
        _FILES[1].url: [{"id": "c", "votes": 30}],
    }
    return held


@pytest.fixture
def source() -> _CountingSource:
    return _CountingSource()


async def _save(pod: _Pod, entity_id: str, votes: int) -> None:
    await pod.results.save_entity(pod.results.create({"id": entity_id, "votes": votes}))


async def _load(pod: _Pod, **kwargs: Any) -> int:
    count: int = await pod.results.load_files(
        _FILES, columns=["id", "votes"], constants={"date_created": _WHEN, "date_updated": _WHEN}, **kwargs
    )
    return count


async def test_a_load_hands_its_files_columns_constants_and_bounds_to_its_store(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    pod = _Pod(bus, store)
    assert await _load(pod, rows_per_transaction=500, timeout_ms=9000) == 3
    assert store.loads == [
        {
            "table": "lsf_results",
            "files": _FILES,
            "columns": ["id", "votes"],
            "constants": {"date_created": _WHEN, "date_updated": _WHEN},
            "rows_per_transaction": 500,
            "timeout_ms": 9000,
        }
    ]


async def test_a_load_s_bounds_default_to_the_collection_s_own(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await pod.results.load_files(_FILES, columns=["id", "votes"])
    [load] = store.loads
    assert load["constants"] == {}
    assert load["rows_per_transaction"] == pod.results.LOAD_ROWS_PER_TRANSACTION
    assert load["timeout_ms"] == pod.results.LOAD_TIMEOUT_MS


async def test_this_process_s_l1_rows_leave(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    assert pod.results.exists_in_cache_sync("a")
    await _load(pod)
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_live_l2_entry_is_read_past_so_the_loaded_row_is_read(bus: FakeNatsClient, store: _TableStore) -> None:
    # the old row is still in L2, which a pod cannot list to delete: after the load a read goes past
    # it to L3, which holds the loaded row
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await _load(pod)
    fetches = store.fetches
    row = await pod.results.ensure("a")
    assert row is not None and row["votes"] == 10
    assert store.fetches == fetches + 1


async def test_a_switched_on_table_advances_its_generation_once_and_broadcasts_no_row(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "z", 1)
    published = len(bus.published)
    await _load(pod)
    assert source.counts["lsf_results"] == 2  # the save's advance, then the load's
    assert len(bus.published) == published


async def test_a_pod_following_the_table_drops_it_at_its_next_catch_up(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    await _save(writer, "a", 1)
    follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert (await follower.results.ensure("a")) is not None
    assert follower.results.exists_in_cache_sync("a")
    await _load(writer)
    # the load leaves no write mark and no row broadcast: the advance alone tells the follower
    verdict = follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert verdict is GenerationVerdict.MISSED
    assert not follower.results.exists_in_cache_sync("a")
    row = await follower.results.ensure("a")
    assert row is not None and row["votes"] == 10


async def test_the_writer_does_not_drop_its_own_advance_again(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await pod.follow(source)
    await _load(pod)
    verdict = pod.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert verdict is GenerationVerdict.CURRENT


async def test_a_generation_that_cannot_advance_raises_after_the_rows_are_loaded_and_dropped(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    source.failing = True
    with pytest.raises(GenerationUnavailableError):
        await _load(pod)
    assert store.rows["a"]["votes"] == 10
    assert not pod.results.exists_in_cache_sync("a")


async def test_an_undeclared_table_loads_and_advances_nothing(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, cls=_Results, source=source)
    await _save(pod, "a", 1)
    await _load(pod)
    assert source.counts == {}
    assert not pod.results.exists_in_cache_sync("a")


async def test_no_files_load_nothing_and_drop_nothing(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    assert await pod.results.load_files([], columns=["id"]) == 0
    assert store.loads == []
    assert source.counts["lsf_results"] == 1
    assert pod.results.exists_in_cache_sync("a")


async def test_a_collection_with_no_durable_store_is_refused(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    pod.results.l3_pool = None
    with pytest.raises(ValueError, match="no durable store"):
        await _load(pod)


async def test_a_store_that_cannot_load_files_is_refused(bus: FakeNatsClient) -> None:
    store = _UnloadableStore()
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    with pytest.raises(ValueError, match="cannot load files"):
        await _load(pod)
    assert pod.results.exists_in_cache_sync("a")


class _Buffer:
    """a write buffer: present, so a write-behind collection really defers."""


async def test_a_collection_deferring_its_l3_writes_is_refused_before_anything_is_loaded(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    pod = _Pod(bus, store, cls=_WriteBehindResults, write_buffer=_Buffer())
    with pytest.raises(ValueError, match="defers its L3 writes"):
        await _load(pod)
    assert store.loads == []
