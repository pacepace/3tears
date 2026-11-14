"""StandingRows: a table's standing rows read cold once, then kept by each changed row read by its whole key.

And follow-since: a cold read taken before the follow is followed from the generation read just before it, so a row
written between the two is read by key, not missed until a restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import (
    EpochGenerationReader,
    EpochGenerationSource,
    RowFollower,
    Standing,
    StandingRows,
    cold_read_generations,
    follow_since,
)

_TABLE = "agents"


# parity-exempt: answers the keyed read's fetch(sql, *params) over an in-memory agents table, no pool protocol
class _FakeAgentsDb:
    """the agents table, answering a keyed read (``t."customer_id", t."id"`` bound as text arrays) by its keys."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {(row["customer_id"], row["id"]): dict(row) for row in rows}
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_next = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, params))
        if self.fail_next:
            self.fail_next -= 1
            raise OSError("database unreachable")
        assert f"FROM {_TABLE} AS t" in sql or f'FROM "{_TABLE}" AS t' in sql, sql
        named = {(UUID(customer), UUID(agent)) for customer, agent in zip(params[0], params[1], strict=True)}
        return [dict(row) for key, row in self.rows.items() if key in named]


def _agent(customer: UUID | None = None, *, deleted: bool = False) -> dict[str, Any]:
    return {"customer_id": customer or uuid4(), "id": uuid4(), "is_deleted": deleted, "name": "a"}


def _standing() -> Standing:
    return Standing(
        table=_TABLE,
        key=("customer_id", "id"),
        key_types=("uuid", "uuid"),
        held=("is_deleted",),
        member=lambda row: not row["is_deleted"],
    )


class _Cold:
    """the caller's cold read: every row of the table, counted."""

    def __init__(self, db: _FakeAgentsDb) -> None:
        self.db = db
        self.calls = 0

    async def __call__(self) -> list[dict[str, Any]]:
        self.calls += 1
        return [{"customer_id": c, "id": i, "is_deleted": r["is_deleted"]} for (c, i), r in self.db.rows.items()]


def _set(db: _FakeAgentsDb, cold: _Cold, bus: FakeNatsClient | None = None) -> StandingRows:
    return StandingRows(
        db, _standing(), cold=cold, generations=EpochGenerationReader(bus or FakeNatsClient()), name="test-agents"
    )


def _changed(row: Mapping[str, Any]) -> CacheInvalidationMessage:
    return CacheInvalidationMessage(table=_TABLE, ids=[f"{row['customer_id']}", f"{row['id']}"])


async def test_the_set_is_read_whole_only_cold_and_each_change_by_its_whole_key() -> None:
    """Flips if a question after the first reads the table whole again, a changed row is read by less than its whole
    key (its customer's every row), or a row that stopped standing is still held."""
    standing, gone = _agent(), _agent()
    db = _FakeAgentsDb([standing, gone, _agent(deleted=True)])
    cold = _Cold(db)
    rows = _set(db, cold)

    assert await rows.count() == 2
    added = _agent(standing["customer_id"])
    db.rows[(added["customer_id"], added["id"])] = added
    db.rows[(gone["customer_id"], gone["id"])]["is_deleted"] = True
    rows.rows.on_row(_TABLE, _changed(added))
    rows.rows.on_row(_TABLE, _changed(gone))
    await rows.settled()

    assert await rows.keys() == sorted([(standing["customer_id"], standing["id"]), (added["customer_id"], added["id"])])
    assert cold.calls == 1
    assert len(db.statements) == 2
    for _sql, params in db.statements:
        assert len(params[0]) == 1 and len(params[1]) == 1, "one changed row, named by its whole key"


async def test_a_change_heard_before_the_cold_read_is_not_read() -> None:
    """Flips if a changed row heard before the set holds anything is read by key (the cold read will see it)."""
    db = _FakeAgentsDb([_agent()])
    rows = _set(db, _Cold(db))
    rows.rows.on_row(_TABLE, _changed(next(iter(db.rows.values()))))
    await rows.settled()
    assert db.statements == []


async def test_a_failed_change_read_is_read_again_before_the_next_answer() -> None:
    """Flips if a changed row whose read failed is answered as it stood (a set known to have missed a row), or if the
    second failure is swallowed rather than reaching the caller."""
    row = _agent()
    db = _FakeAgentsDb([row])
    rows = _set(db, _Cold(db))
    assert await rows.count() == 1

    db.rows[(row["customer_id"], row["id"])]["is_deleted"] = True
    db.fail_next = 1
    rows.rows.on_row(_TABLE, _changed(row))
    await rows.settled()
    assert await rows.count() == 0, "the missed row is read again first"

    db.rows[(row["customer_id"], row["id"])]["is_deleted"] = False
    db.fail_next = 2
    rows.rows.on_row(_TABLE, _changed(row))
    await rows.settled()
    with pytest.raises(OSError, match="unreachable"):
        await rows.count()


async def test_a_reset_reads_cold_at_the_next_question() -> None:
    """Flips if a table dropped whole (an operator's reset) keeps answering what was held, or reads whole at once."""
    db = _FakeAgentsDb([_agent()])
    cold = _Cold(db)
    rows = _set(db, cold)
    assert await rows.count() == 1
    db.rows.clear()
    rows.on_table_dropped()
    assert cold.calls == 1
    assert await rows.count() == 0
    assert cold.calls == 2


async def test_the_rows_are_read_by_key_and_judged_again() -> None:
    """Flips if the rows are read other than by the held keys, a page's worth a statement, or a row that stopped
    standing after it was held is answered."""
    customer = uuid4()
    held = [_agent(customer) for _ in range(5)]
    db = _FakeAgentsDb(held)
    rows = _set(db, _Cold(db))
    await rows.count()
    db.rows[(held[0]["customer_id"], held[0]["id"])]["is_deleted"] = True

    found = await rows.read(("name",), page=2)

    assert [(r["customer_id"], r["id"]) for r in found] == sorted((r["customer_id"], r["id"]) for r in held[1:])
    assert all(r["name"] == "a" for r in found)
    assert [len(params[0]) for _sql, params in db.statements] == [2, 2, 1]
    assert await _set(_FakeAgentsDb([]), _Cold(_FakeAgentsDb([]))).read(("name",)) == []


def test_a_standing_declaration_is_checked() -> None:
    def member(row: Mapping[str, Any]) -> bool:
        return True

    with pytest.raises(ValueError, match="one SQL type per key column"):
        Standing(table="t", key=("id",), key_types=(), held=(), member=member)
    with pytest.raises(ValueError, match="not among"):
        Standing(table="t", key=("id",), key_types=("jsonb",), held=(), member=member)
    with pytest.raises(ValueError, match="held already"):
        Standing(table="t", key=("id",), key_types=("uuid",), held=("id",), member=member)
    with pytest.raises(ValueError, match="named 1 key values"):
        _standing().parse(["x"])


async def _wait_for(done: Any) -> None:
    # a follow's first push is judged after the broadcast grace (2 s)
    for _ in range(800):
        if done():
            return
        await asyncio.sleep(0.01)


async def test_a_row_written_between_the_cold_read_and_the_follow_is_read() -> None:
    """Flips if a row written after the set's cold read and before its follow (a process's start sweeps long before
    it follows) is missed until a restart: the follow's first sight would be taken as its mark. Also flips if a row
    written before the cold read is read again by key (the cold read already holds it)."""
    bus = FakeNatsClient()
    writes = EpochGenerationSource(bus)
    before = _agent()
    db = _FakeAgentsDb([before])
    await writes.advance_with_change(_TABLE, [ChangedRow(ids=[f"{before['customer_id']}", f"{before['id']}"])])
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="test")
    await registry.start_invalidation_listener(bus)
    rows = _set(db, _Cold(db), bus)
    assert await rows.count() == 1
    # written by another replica between the cold read and the follow; its broadcast went before this set was
    # registered for the rows, so only the change log holds it
    between = _agent()
    db.rows[(between["customer_id"], between["id"])] = between
    await writes.advance_with_change(_TABLE, [ChangedRow(ids=[f"{between['customer_id']}", f"{between['id']}"])])

    rows.follow(registry, EpochGenerationReader(bus))
    try:
        await _wait_for(lambda: db.statements)
        await rows.settled()
    finally:
        await rows.stop()
        await registry.stop_invalidation_listener()

    assert await rows.count() == 2
    assert [params[:2] for _sql, params in db.statements] == [
        ([f"{between['customer_id']}"], [f"{between['id']}"]),
    ], "only the row written between is read by key"


async def test_a_set_read_cold_after_its_follow_takes_no_mark_of_its_own() -> None:
    """Flips if a set that starts following before its first question moves the table's mark to its own later cold
    read (a change heard between the follow and that read would then count as never written)."""
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l2_client=bus, kv_key_scope="test")
    await registry.start_invalidation_listener(bus)
    db = _FakeAgentsDb([_agent()])
    reads: list[str] = []

    class _Reader(EpochGenerationReader):
        async def read(self, table_name: str) -> str | None:
            reads.append(table_name)
            return await super().read(table_name)

    rows = StandingRows(db, _standing(), cold=_Cold(db), generations=_Reader(bus), name="test-agents")
    rows.follow(registry, EpochGenerationReader(bus))
    try:
        assert await rows.count() == 1
    finally:
        await rows.stop()
        await registry.stop_invalidation_listener()
    assert reads == []


async def test_follow_since_moves_a_later_mark_back_and_keeps_an_earlier_one() -> None:
    """Flips if a mark past the cold read's generation is kept (the gap after it is never walked), or an earlier
    mark of another follower is moved forward (that follower's own gap would be skipped)."""
    bus = FakeNatsClient()
    writes = EpochGenerationSource(bus)
    reader = EpochGenerationReader(bus)
    await writes.advance_with_change("a", [ChangedRow(ids=["1"])])
    await writes.advance_with_change("b", [ChangedRow(ids=["1"])])
    cold = await cold_read_generations(reader, ("a", "b"))
    later = await writes.advance_with_change("a", [ChangedRow(ids=["2"])])
    registry = CollectionRegistry()
    marks = registry.generation_marks
    for table in ("a", "b"):
        marks.follow(table)
    marks.adopt_generation("a", later)
    marks.adopt_generation("b", None)

    follow_since(registry, cold)

    assert f"{marks.judge('a', later)}" == "missed", "the mark is back at the cold read's generation"
    assert f"{marks.judge('b', cold['b'])}" == "missed", "a mark taken before the log held any generation is kept"


async def test_a_follow_since_naming_a_table_it_does_not_follow_is_refused() -> None:
    async def read(key: tuple[str, ...]) -> None:
        return None

    follower = RowFollower({}, lambda: None, name="test", key_reads={"a": read})
    with pytest.raises(ValueError, match="does not follow"):
        follower.follow(CollectionRegistry(), EpochGenerationReader(FakeNatsClient()), since={"b": None})


async def test_each_followed_change_to_what_stands_is_told_once_after_it_is_held() -> None:
    """Flips if a row that starts or stops standing is not told to ``on_change`` (with what was held before and after),
    a row re-written unchanged is told, the cold read is told, a missed row's second read is not told, or the change is
    told before the set holds it (the callback asks the set and must see the change)."""
    kept, leaving = _agent(), _agent()
    db = _FakeAgentsDb([kept, leaving])
    told: list[tuple[tuple[Any, ...], Any, Any, int]] = []
    rows: StandingRows

    async def on_change(
        key: tuple[Any, ...], before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
    ) -> None:
        told.append((key, before, after, await rows.count()))

    rows = StandingRows(
        db,
        _standing(),
        cold=_Cold(db),
        generations=EpochGenerationReader(FakeNatsClient()),
        name="test-agents",
        on_change=on_change,
    )
    assert await rows.count() == 2
    assert told == [], "the cold read is no change"

    joining = _agent()
    db.rows[(joining["customer_id"], joining["id"])] = joining
    db.rows[(leaving["customer_id"], leaving["id"])]["is_deleted"] = True
    rows.rows.on_row(_TABLE, _changed(joining))
    rows.rows.on_row(_TABLE, _changed(leaving))
    rows.rows.on_row(_TABLE, _changed(kept))
    await rows.settled()

    joining_key = (joining["customer_id"], joining["id"])
    leaving_key = (leaving["customer_id"], leaving["id"])
    assert sorted(told, key=lambda t: t[0]) == sorted(
        [
            (joining_key, None, {"customer_id": joining["customer_id"], "id": joining["id"], "is_deleted": False}, 2),
            (leaving_key, {"customer_id": leaving["customer_id"], "id": leaving["id"], "is_deleted": False}, None, 2),
        ],
        key=lambda t: t[0],
    )

    told.clear()
    db.rows[joining_key]["is_deleted"] = True
    db.fail_next = 1
    rows.rows.on_row(_TABLE, _changed(joining))
    await rows.settled()
    assert told == []
    assert await rows.count() == 1, "the missed row is read again first"
    await rows.settled()
    assert [(key, after) for key, _before, after, _n in told] == [(joining_key, None)]


async def test_a_failing_on_change_is_contained_and_the_set_keeps_following() -> None:
    """Flips if a raising ``on_change`` stops the set holding later changes or escapes into the follower's task."""
    db = _FakeAgentsDb([])
    calls: list[tuple[Any, ...]] = []

    async def on_change(
        key: tuple[Any, ...], before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
    ) -> None:
        calls.append(key)
        raise RuntimeError("the consumer's write failed")

    rows = StandingRows(
        db,
        _standing(),
        cold=_Cold(db),
        generations=EpochGenerationReader(FakeNatsClient()),
        name="test-agents",
        on_change=on_change,
    )
    assert await rows.count() == 0
    first, second = _agent(), _agent()
    for row in (first, second):
        db.rows[(row["customer_id"], row["id"])] = row
        rows.rows.on_row(_TABLE, _changed(row))
        await rows.settled()
    assert await rows.count() == 2
    assert len(calls) == 2
