"""A table emptied with TRUNCATE: L3 emptied alone, every cache of it dropped, other pods told by its generation.

The contract this pins:

- ``truncate()`` empties the table through its store's ``truncate``, outside any transaction;
- this process's L1 rows go, and its L2 entries are read past until each is read through again;
- a switched-on table's write generation advances once per call, with no row broadcast, so a pod
  following the table finds an advance it heard no row of and drops the table too;
- calling it twice in a row is safe;
- a collection that cannot keep the contract is refused before anything is emptied.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import MetaData

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

    async def truncate(self, table: str) -> None:
        self.truncated.append(table)
        self.rows.clear()
        if self.lost_reply is not None:
            raise self.lost_reply


class _UntruncatableStore(_TableStore):
    """a durable store with no TRUNCATE: a git working tree, say."""

    truncate = None  # type: ignore[assignment]


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
        self.registry.settle_generation(SCHEMA.name, await source.current(SCHEMA.name))


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


async def test_truncate_empties_the_table_through_its_store(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await _save(pod, "b", 2)
    await pod.results.truncate()
    assert store.truncated == ["lsf_results"]
    assert store.rows == {}


async def test_this_process_s_l1_rows_leave(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    assert pod.results.exists_in_cache_sync("a")
    await pod.results.truncate()
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_live_l2_entry_is_read_past_so_no_tier_answers_a_truncated_row(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    # the row is still in L2, which a pod cannot list to delete: after the truncate a read goes past
    # it to L3, which holds nothing
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await pod.results.truncate()
    fetches = store.fetches
    assert await pod.results.ensure("a") is None
    assert store.fetches == fetches + 1


async def test_a_switched_on_table_advances_its_generation_once_and_broadcasts_no_row(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    published = len(bus.published)
    await pod.results.truncate()
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
    follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert (await follower.results.ensure("a")) is not None
    assert follower.results.exists_in_cache_sync("a")
    await writer.results.truncate()
    # TRUNCATE leaves no write mark and no row broadcast: the advance alone tells the follower
    verdict = follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert verdict is GenerationVerdict.MISSED
    assert not follower.results.exists_in_cache_sync("a")
    assert await follower.results.ensure("a") is None


async def test_the_writer_does_not_drop_its_own_advance_again(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await pod.follow(source)
    await pod.results.truncate()
    verdict = pod.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert verdict is GenerationVerdict.CURRENT


async def test_truncating_twice_in_a_row_is_safe(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    await pod.results.truncate()
    await pod.results.truncate()
    assert store.truncated == ["lsf_results", "lsf_results"]
    assert source.counts["lsf_results"] == 3
    assert await pod.results.ensure("a") is None


async def test_a_table_saved_again_after_a_truncate_reads_the_new_row(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    await pod.results.truncate()
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
        await pod.results.truncate()
    assert store.rows == {}
    assert not pod.results.exists_in_cache_sync("a")


async def test_an_undeclared_table_truncates_and_advances_nothing(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, cls=_Results, source=source)
    await _save(pod, "a", 1)
    await pod.results.truncate()
    assert store.rows == {}
    assert source.counts == {}
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_collection_with_no_durable_store_is_refused(bus: FakeNatsClient, store: _TableStore) -> None:
    pod = _Pod(bus, store)
    pod.results.l3_pool = None
    with pytest.raises(ValueError, match="no durable store"):
        await pod.results.truncate()


async def test_a_store_that_cannot_truncate_is_refused(bus: FakeNatsClient) -> None:
    store = _UntruncatableStore()
    pod = _Pod(bus, store)
    await _save(pod, "a", 1)
    with pytest.raises(ValueError, match="cannot truncate"):
        await pod.results.truncate()
    assert pod.results.exists_in_cache_sync("a")


class _Buffer:
    """a write buffer: present, so a write-behind collection really defers."""


async def test_a_collection_deferring_its_l3_writes_is_refused_before_anything_is_emptied(
    bus: FakeNatsClient, store: _TableStore
) -> None:
    pod = _Pod(bus, store, cls=_WriteBehindResults, write_buffer=_Buffer())
    store.rows["a"] = {"id": "a", "votes": 1}
    with pytest.raises(ValueError, match="defers its L3 writes"):
        await pod.results.truncate()
    assert store.truncated == []
    assert store.rows


async def test_a_truncate_whose_reply_is_lost_still_drops_and_advances_and_raises_its_failure(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    writer = _Pod(bus, store, source=source)
    follower = _Pod(bus, store, source=source)
    await follower.follow(source)
    await _save(writer, "a", 1)
    follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
    assert (await follower.results.ensure("a")) is not None
    store.lost_reply = DataLayerUnavailableError("l3.truncate: no reply")
    with pytest.raises(DataLayerUnavailableError, match="no reply"):
        await writer.results.truncate()
    assert not writer.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the failed truncate's
    verdict = follower.registry.settle_generation("lsf_results", await source.current("lsf_results"))
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
        await pod.results.truncate()
    assert not pod.results.exists_in_cache_sync("a")


async def test_a_truncate_cancelled_after_the_broker_ran_it_drops_and_advances_then_is_cancelled_still(
    bus: FakeNatsClient, store: _TableStore, source: _CountingSource
) -> None:
    pod = _Pod(bus, store, source=source)
    await _save(pod, "a", 1)
    store.lost_reply = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await pod.results.truncate()
    assert store.rows == {}
    assert not pod.results.exists_in_cache_sync("a")
    assert source.counts["lsf_results"] == 2  # the save's advance, then the cancelled truncate's
