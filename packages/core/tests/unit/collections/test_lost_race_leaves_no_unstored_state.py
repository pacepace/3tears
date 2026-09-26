"""a write that does not reach L3 must not leave its state in L1 (or L2).

an entity is a proxy onto its collection's L1 row: constructing one writes its data into
L1, and every attribute set writes through. that is how a handle reads its own fields, and
it means the working copy of an entity that has not been saved yet is already in the pod's
cache. :meth:`BaseCollection.save_entity` runs the L3 compare-and-set afterwards. when that
write does not land -- the fence refuses it, the insert finds the row taken, the store
raises -- the working copy it was built from is state that was never stored.

left in L1 it is served as if it were: a writer that lost the race retries, reads its own
never-stored write back through ``ensure()``, concludes the change is already present, and
stops. the survey engine measured the result on a real database: 19 of 20 concurrent
members served from cache, 2 stored.

every interleaving here is driven, not slept: the store holds the first writer's commit
behind an event until the second writer has read, so both are genuinely in flight.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError

_TABLE = "race_rows"
_SCOPE = "race-principal"
_KEY = "row-1"
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _metadata() -> MetaData:
    """
    the one table the racing collection caches.

    :return: sqlalchemy metadata holding it
    :rtype: MetaData
    """
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(64), primary_key=True),
        Column("members", String(255)),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    return metadata


class RaceRow(BaseEntity):
    """an entity whose row holds a member list, the shape the survey's index raced on."""

    primary_key_field = "id"


class RacingStoreCollection(BaseCollection[RaceRow]):
    """a real :class:`BaseCollection` over an in-memory L3 with insert and CAS semantics.

    the store is the only stand-in: ``save_to_store`` refuses an insert over an existing row
    (``ON CONFLICT DO NOTHING``) and a fenced update whose fence no longer matches, exactly as
    the schema-backed store does, and it can hold a write behind an event or raise on demand.
    everything above it -- entity construction, ``save_entity``, L1, L2, invalidation -- is
    the production code path.
    """

    def __init__(
        self,
        registry: CollectionRegistry,
        nats_client: AsyncMock | None = None,
        rows: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.rows: dict[str, dict[str, Any]] = rows if rows is not None else {}
        self.hold: asyncio.Event | None = None
        self.arrived: asyncio.Event = asyncio.Event()
        self.fail_with: Exception | None = None
        super().__init__(
            registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client
        )

    @property
    def table_name(self) -> str:
        """
        :return: the racing table
        :rtype: str
        """
        return _TABLE

    @property
    def entity_class(self) -> type[RaceRow]:
        """
        :return: the racing entity
        :rtype: type[RaceRow]
        """
        return RaceRow

    @property
    def emits_cas_fence(self) -> bool:
        """
        :return: ``True``: a 0 rowcount from this store always means a lost race
        :rtype: bool
        """
        return True

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """
        :param entity_id: the row key
        :ptype entity_id: Any
        :return: a copy of the stored row, or ``None``
        :rtype: dict[str, Any] | None
        """
        row = self.rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """
        insert-or-fenced-update, optionally held or failed.

        :param data: the row to write
        :ptype data: dict[str, Any]
        :param original_timestamp: the fence: the ``date_updated`` the writer read
        :ptype original_timestamp: datetime | None
        :param conn: unused
        :ptype conn: Any
        :return: 1 when written, 0 when the insert or the fence was refused
        :rtype: int
        :raises Exception: ``fail_with``, when set
        """
        del conn
        self.arrived.set()
        if self.hold is not None:
            await self.hold.wait()
        if self.fail_with is not None:
            raise self.fail_with
        key = str(data["id"])
        existing = self.rows.get(key)
        refused = existing is not None and existing.get("date_updated") != original_timestamp
        if not refused:
            self.rows[key] = dict(data)
        return 0 if refused else 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """
        :param entity_id: the row key
        :ptype entity_id: Any
        """
        self.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        """
        :param data: a row
        :ptype data: dict[str, Any]
        :return: its JSON bytes
        :rtype: bytes
        """
        return json.dumps(data, default=lambda value: value.isoformat()).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """
        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        for column in ("date_created", "date_updated"):
            if row.get(column) is not None:
                row[column] = datetime.fromisoformat(row[column])
        return row


def _nats() -> AsyncMock:
    """
    a NATS client whose KV bucket is an in-memory dict, with publishes recorded.

    :return: the client mock
    :rtype: AsyncMock
    """
    store: dict[str, bytes] = {}

    async def _get(*, key: str) -> bytes | None:
        return store.get(key)

    async def _put(*, key: str, value: bytes) -> int:
        store[key] = value
        return len(store)

    async def _delete(*, key: str, revision: int | None = None) -> bool:
        del revision
        return store.pop(key, None) is not None

    bucket = AsyncMock()
    bucket.get = AsyncMock(side_effect=_get)
    bucket.put = AsyncMock(side_effect=_put)
    bucket.delete = AsyncMock(side_effect=_delete)
    nats = AsyncMock()
    nats.kv_bucket = AsyncMock(return_value=bucket)
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.store = store
    return nats


def _pod_registry() -> tuple[CollectionRegistry, SQLiteBackend]:
    """
    one pod's registry over its own fresh SQLite L1.

    :return: the registry and its L1, for teardown
    :rtype: tuple[CollectionRegistry, SQLiteBackend]
    """
    l1 = SQLiteBackend(db_name=f"race_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1, kv_key_scope=_SCOPE)
    return reg, l1


@pytest.fixture()
def registry() -> Iterator[CollectionRegistry]:
    """
    this pod's registry.

    :return: the registry
    :rtype: Iterator[CollectionRegistry]
    """
    reg, l1 = _pod_registry()
    yield reg
    l1.reset()


@pytest.fixture()
def peer_registry() -> Iterator[CollectionRegistry]:
    """
    a second pod's registry: its own L1, sharing L2 and L3 with this pod through the test.

    :return: the registry
    :rtype: Iterator[CollectionRegistry]
    """
    reg, l1 = _pod_registry()
    yield reg
    l1.reset()


def _stored(members: list[str]) -> dict[str, Any]:
    """
    a stored row at the base timestamp.

    :param members: its member list
    :ptype members: list[str]
    :return: the row, members joined into the one text column
    :rtype: dict[str, Any]
    """
    return {"id": _KEY, "members": ",".join(members), "date_created": _T0, "date_updated": _T0}


def _members(row: dict[str, Any] | None) -> list[str]:
    """
    the member list a row holds.

    :param row: a stored or cached row
    :ptype row: dict[str, Any] | None
    :return: its members, sorted
    :rtype: list[str]
    """
    assert row is not None
    return sorted(member for member in str(row["members"] or "").split(",") if member)


def _with(row: dict[str, Any], member: str) -> dict[str, Any]:
    """
    a copy of a row with one member added.

    :param row: the row read
    :ptype row: dict[str, Any]
    :param member: the member to add
    :ptype member: str
    :return: the new row
    :rtype: dict[str, Any]
    """
    return {**row, "members": ",".join([*_members(row), member])}


async def _race_one_winner_one_loser(coll: RacingStoreCollection) -> ConcurrentModificationError:
    """
    two writers read the same row; A commits while B holds its read; B then writes and loses.

    :param coll: the collection, seeded with a row at the base timestamp
    :ptype coll: RacingStoreCollection
    :return: the loser's error
    :rtype: ConcurrentModificationError
    """
    coll.hold = asyncio.Event()
    read_a = await coll.fetch_from_store(_KEY)
    assert read_a is not None
    winner = RaceRow(_with(read_a, "a"), is_new=False, collection=coll)
    a_saving = asyncio.create_task(winner.save())
    await coll.arrived.wait()
    # B reads while A's write is in flight, so it holds the pre-commit fence.
    read_b = await coll.fetch_from_store(_KEY)
    assert read_b is not None
    coll.hold.set()
    await a_saving
    coll.hold = None
    loser = RaceRow(_with(read_b, "b"), is_new=False, collection=coll)
    with pytest.raises(ConcurrentModificationError) as lost:
        await loser.save()
    return lost.value


class TestALostCasRace:
    """the fence refuses the second writer; nothing it wrote may still answer from cache."""

    async def test_the_loser_leaves_no_unstored_state_in_l1(self, registry: CollectionRegistry) -> None:
        """after the loss, a read-through returns the stored row, not the loser's working copy."""
        coll = RacingStoreCollection(registry, _nats())
        coll.rows[_KEY] = _stored([])

        await _race_one_winner_one_loser(coll)

        assert _members(coll.rows[_KEY]) == ["a"]
        assert _members(await coll.ensure(_KEY)) == ["a"]

    async def test_the_losers_retry_through_ensure_reaches_l3(self, registry: CollectionRegistry) -> None:
        """the survey's original loop, read through the cache: its retry must write, not no-op.

        the multi-hop property: the step after the loss re-reads, recomputes and saves, and
        both members end up stored.
        """
        coll = RacingStoreCollection(registry, _nats())
        coll.rows[_KEY] = _stored([])
        await _race_one_winner_one_loser(coll)

        current = await coll.ensure(_KEY)
        assert current is not None
        if "b" not in _members(current):
            retry = RaceRow(_with(current, "b"), is_new=False, collection=coll)
            await retry.save()

        assert _members(coll.rows[_KEY]) == ["a", "b"]
        assert _members(await coll.ensure(_KEY)) == ["a", "b"]

    async def test_a_field_set_on_a_loaded_entity_does_not_outlive_its_lost_save(
        self, registry: CollectionRegistry, peer_registry: CollectionRegistry
    ) -> None:
        """the personal-link shape: load the row, a peer pod commits, set one field, save, lose."""
        nats = _nats()
        coll = RacingStoreCollection(registry, nats)
        coll.rows[_KEY] = _stored([])
        peer = RacingStoreCollection(peer_registry, nats, rows=coll.rows)
        stale = await coll.get(_KEY)
        assert stale is not None
        peer_view = await peer.get(_KEY)
        assert peer_view is not None
        peer_view.members = "a"
        await peer_view.save()

        stale.members = "b"
        with pytest.raises(ConcurrentModificationError):
            await stale.save()

        assert _members(await coll.ensure(_KEY)) == ["a"]

    async def test_the_losing_handle_keeps_its_working_state(self, registry: CollectionRegistry) -> None:
        """evicting the cache row must not strip the caller's handle of what it was saving."""
        coll = RacingStoreCollection(registry, _nats())
        coll.rows[_KEY] = {**_stored(["a"]), "date_updated": datetime(2026, 2, 1, tzinfo=UTC)}
        loser = RaceRow(_stored(["b"]), is_new=False, collection=coll)

        with pytest.raises(ConcurrentModificationError):
            await loser.save()

        assert loser.members == "b"
        assert not loser.is_new
        assert _members(loser.to_dict()) == ["b"]


class TestALostInsertRace:
    """two first writers of one derived id; the insert finds the row already taken."""

    async def test_the_second_insert_leaves_no_unstored_state_in_l1(self, registry: CollectionRegistry) -> None:
        """the new entity's working copy does not survive a refused insert."""
        coll = RacingStoreCollection(registry, _nats())
        first = RaceRow(_stored(["a"]) | {"date_updated": None}, is_new=True, collection=coll)
        await first.save()
        second = RaceRow(_stored(["b"]) | {"date_updated": None}, is_new=True, collection=coll)

        with pytest.raises(RuntimeError):
            await second.save()

        assert _members(await coll.ensure(_KEY)) == ["a"]


class TestAFailedStoreWrite:
    """the store raises; the outcome is unknown, so the cache must defer to L3."""

    async def test_an_exception_from_the_store_leaves_no_unstored_state_in_l1(
        self, registry: CollectionRegistry
    ) -> None:
        """a raising write propagates, and L1 answers from L3 afterwards."""
        coll = RacingStoreCollection(registry, _nats())
        coll.rows[_KEY] = _stored(["a"])
        coll.fail_with = ConnectionError("store unreachable")
        entity = RaceRow(_stored(["a", "b"]), is_new=False, collection=coll)

        with pytest.raises(ConnectionError):
            await entity.save()

        coll.fail_with = None
        assert _members(await coll.ensure(_KEY)) == ["a"]


class TestTheSubscriptWritePath:
    """``collection[id] = row`` writes L1 and L2 before L3; a refused L3 write must withdraw both."""

    async def test_a_refused_background_write_is_withdrawn_from_every_tier(self, registry: CollectionRegistry) -> None:
        """after the fire-and-forget write loses, neither L1 nor L2 still holds its row."""
        nats = _nats()
        coll = RacingStoreCollection(registry, nats)
        coll.rows[_KEY] = {**_stored(["a"]), "date_updated": datetime(2026, 2, 1, tzinfo=UTC)}
        coll.fail_with = ConnectionError("store unreachable")

        coll[_KEY] = _stored(["b"])
        # the propagation runs as a task on this loop; wait for it rather than sleeping.
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        await asyncio.gather(*pending)

        coll.fail_with = None
        assert coll.get_row_sync(_KEY) is None
        assert f"{_SCOPE}.{_TABLE}.{_KEY}" not in nats.store
        assert _members(await coll.ensure(_KEY)) == ["a"]
