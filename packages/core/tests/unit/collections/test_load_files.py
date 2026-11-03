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

import asyncio
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
        #: the index of a file that fails the load, the files before it staying loaded
        self.failing_file: int | None = None
        #: what that file raises; a broker's refusal when None
        self.failure: BaseException | None = None
        #: the index of a load call that fails before writing anything, the calls before it landed
        self.failing_call: int | None = None

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
        if len(self.loads) - 1 == self.failing_call:
            raise DataLayerUnavailableError(f"load {self.failing_call} failed; the loads before it stay committed")
        loaded = []
        for index, file in enumerate(files):
            if index == self.failing_file:
                raise self.failure or DataLayerUnavailableError(
                    f"file {index} failed; the files before it stay committed"
                )
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


async def test_a_load_failing_part_way_drops_and_advances_then_raises_its_failure(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    await _save(writer, "a", 1)
    follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert (await follower.results.ensure("a")) is not None
    store.failing_file = 1
    with pytest.raises(DataLayerUnavailableError, match="file 1 failed"):
        await _load(writer)
    # the first file's rows are committed: neither pod may keep answering the old row
    assert store.rows["a"]["votes"] == 10
    assert not writer.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the failed load's
    verdict = follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert verdict is GenerationVerdict.MISSED
    row = await follower.results.ensure("a")
    assert row is not None and row["votes"] == 10


async def test_a_failed_load_s_own_failure_wins_over_a_generation_that_cannot_advance(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.failing_file = 1
    source.failing = True
    with pytest.raises(DataLayerUnavailableError, match="file 1 failed"):
        await _load(pod)
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_load_cancelled_part_way_drops_and_advances_then_is_cancelled_still(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.failing_file = 1
    store.failure = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await _load(pod)
    assert store.rows["a"]["votes"] == 10
    assert not pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the cancelled load's


def _files(*rows: int) -> list[LoadFile]:
    return [LoadFile(url=f"https://bucket/part/{index:04d}.parquet", rows=count) for index, count in enumerate(rows)]


async def test_a_load_too_long_for_one_reply_is_sent_as_several_requests_of_whole_files_in_order(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    # a second a row and a transaction a row: three files' rows fit one reply's wait of 3600 s at
    # 1 000 000 ms a transaction (three transactions, 3002 s)
    pod = _Pod(bus, store, source=source)
    files = _files(2, 1, 2, 1)
    written = await pod.results.load_files(files, columns=["id"], rows_per_transaction=1, timeout_ms=1_000_000)
    assert written == 6
    assert [load["files"] for load in store.loads] == [files[:2], files[2:]]
    assert {(load["rows_per_transaction"], load["timeout_ms"]) for load in store.loads} == {(1, 1_000_000)}
    assert source.counts["lsf_results"] == 1  # one advance for the whole load


async def test_a_state_s_many_files_load_in_one_call_with_the_defaults(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    # one call per state per table, as a caller makes it: 40 files of 120 000 rows are three
    # transactions each at the defaults, so 19 files fit one reply's wait (57 of 59 transactions)
    pod = _Pod(bus, store, source=source)
    files = _files(*([120_000] * 40))
    assert await pod.results.load_files(files, columns=["id"]) == 40 * 120_000
    assert [len(load["files"]) for load in store.loads] == [19, 19, 2]
    assert [file for load in store.loads for file in load["files"]] == files
    assert source.counts["lsf_results"] == 1


async def test_a_failure_in_a_later_request_drops_and_advances_then_raises_it(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.failing_call = 1
    with pytest.raises(DataLayerUnavailableError, match="load 1 failed"):
        await pod.results.load_files(_files(2, 1, 2, 1), columns=["id"], rows_per_transaction=1, timeout_ms=1_000_000)
    assert len(store.loads) == 2
    assert not pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the failed load's


async def test_one_file_too_long_for_one_reply_is_refused_before_anything_is_sent(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    with pytest.raises(ValueError, match="file 1 of the load holds 3000000 rows"):
        await pod.results.load_files(_files(10, 3_000_000), columns=["id"])
    assert store.loads == []
    assert pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 1


class _InFlightStore(_TableStore):
    """a store that answers each load call after a yield, counting how many are in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.most_in_flight = 0

    async def load_files(self, table: str, files: Sequence[LoadFile], **kwargs: Any) -> LoadResult:
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await super().load_files(table, files, **kwargs)
        finally:
            self.in_flight -= 1


async def test_a_load_over_several_connections_sends_its_requests_side_by_side(
    bus: FakeNatsClient, source: _CountingSource
) -> None:
    # one request's wait holds every file, but the broker copies a request's files one after another on one
    # connection: asked for three connections, the load is cut into three requests of whole files, sent at once
    store = _InFlightStore()
    pod = _Pod(bus, store, source=source)
    files = _files(*([10_000] * 12))
    assert await pod.results.load_files(files, columns=["id"], connections=3) == 120_000
    assert store.most_in_flight == 3
    assert [len(load["files"]) for load in store.loads] == [4, 4, 4]
    assert sorted((file.url for load in store.loads for file in load["files"])) == [file.url for file in files]
    assert source.counts["lsf_results"] == 1  # one advance for the whole load


async def test_a_load_over_several_connections_never_cuts_a_file_nor_makes_an_empty_request(
    bus: FakeNatsClient, source: _CountingSource
) -> None:
    store = _InFlightStore()
    pod = _Pod(bus, store, source=source)
    files = _files(50_000, 10, 10)
    assert await pod.results.load_files(files, columns=["id"], connections=8) == 50_020
    # a large file is a run of its own, and the small ones share one: no file cut, no run empty
    assert [load["files"] for load in store.loads] == [[files[0]], [files[1], files[2]]]


async def test_a_failure_in_one_of_the_side_by_side_requests_drops_and_advances_once_then_raises_it(
    bus: FakeNatsClient, source: _CountingSource
) -> None:
    store = _InFlightStore()
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.failing_call = 1
    with pytest.raises(DataLayerUnavailableError, match="load 1 failed"):
        await pod.results.load_files(_files(*([10_000] * 4)), columns=["id"], connections=4)
    assert store.in_flight == 0, "a request was left running once the load raised"
    assert not pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the failed load's


async def test_one_connection_is_the_load_as_before(bus: FakeNatsClient, source: _CountingSource) -> None:
    store = _InFlightStore()
    pod = _Pod(bus, store, source=source)
    files = _files(*([10_000] * 4))
    await pod.results.load_files(files, columns=["id"])
    assert [load["files"] for load in store.loads] == [files] and store.most_in_flight == 1


async def test_a_load_over_no_connection_is_refused(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    with pytest.raises(ValueError, match="connections"):
        await pod.results.load_files(_FILES, columns=["id"], connections=0)
    assert store.loads == []


def test_the_default_transaction_bound_is_a_minute() -> None:
    assert _Results.LOAD_TIMEOUT_MS == 60_000
    assert _Results.LOAD_ROWS_PER_TRANSACTION == 50_000
