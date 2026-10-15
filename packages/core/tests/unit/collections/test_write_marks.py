"""the write-mark reconciler advances each marked table once and deletes exactly the marks it read."""

from __future__ import annotations

import re
from typing import Any

from threetears.core.collections.write_marks import (
    WRITE_MARK_TRIGGER,
    WriteMarkReconciler,
    install_write_mark_triggers,
    write_mark_trigger_sql,
    write_marks_ddl,
)


class _Queue:
    """the marks table and the catalog, as the reconciler's SQL sees them.

    :ivar marks: id to table name, in commit order
    :ivar missing: ``(schema, table)`` with no trigger yet
    :ivar created: the trigger statements run
    :ivar on_read: called after each page is read, to commit a mark between the read and the delete
    """

    def __init__(self, marks: list[str], missing: list[tuple[str, str]] | None = None) -> None:
        self.marks: dict[int, str] = dict(enumerate(marks, start=1))
        self.missing = list(missing or [])
        self.created: list[str] = []
        self.on_read: Any = None

    def add(self, table: str) -> None:
        self.marks[max(self.marks, default=0) + 1] = table

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "pg_trigger" in query:
            return [{"table_schema": schema, "table_name": table} for schema, table in self.missing]
        if "count(*)" in query:
            return [{"n": len(self.marks)}]
        limit = int(args[0])
        page = [{"id": key, "table_name": value} for key, value in sorted(self.marks.items())[:limit]]
        if self.on_read is not None:
            self.on_read()
        return page

    async def execute(self, query: str, *args: Any) -> str:
        if query.startswith("CREATE TRIGGER"):
            self.created.append(query)
            match = re.search(r'ON "([^"]+)"\."([^"]+)"', query)
            assert match is not None
            self.missing.remove((match.group(1), match.group(2)))
            return "CREATE TRIGGER"
        assert query.startswith("DELETE")
        for key in args[0]:
            self.marks.pop(key, None)
        return f"DELETE {len(args[0])}"


class _Generations:
    """advances generations, failing for the tables named."""

    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.advanced: list[str] = []
        self._failing = failing

    async def advance(self, table_name: str) -> str:
        if table_name in self._failing:
            raise RuntimeError("epoch bucket unavailable")
        self.advanced.append(table_name)
        return f"inc:{len(self.advanced)}"

    async def current(self, table_name: str) -> str:
        return "inc:0"


def _reconciler(queue: _Queue, source: _Generations, **kwargs: Any) -> WriteMarkReconciler:
    return WriteMarkReconciler(
        queue,  # type: ignore[arg-type]
        generation_source=source,  # type: ignore[arg-type]
        marks_schema="platform",
        tables=lambda: ("groups", "revocations"),
        **kwargs,
    )


async def test_each_marked_table_is_advanced_once_and_its_marks_deleted() -> None:
    queue = _Queue(["groups", "revocations", "groups", "groups"])
    source = _Generations()

    done = await _reconciler(queue, source).run_pass()

    assert sorted(source.advanced) == ["groups", "revocations"]
    assert queue.marks == {}
    assert (done.applied, done.behind) == (4, False)


async def test_a_table_that_cannot_be_advanced_keeps_its_marks_for_the_next_pass() -> None:
    queue = _Queue(["groups", "revocations"])
    source = _Generations(failing=frozenset({"revocations"}))

    done = await _reconciler(queue, source).run_pass()

    assert list(queue.marks.values()) == ["revocations"]
    assert done.failed == ("revocations",)
    assert done.behind


async def test_a_mark_committed_after_the_page_was_read_is_not_deleted() -> None:
    queue = _Queue(["groups"])
    queue.on_read = lambda: queue.add("groups") if len(queue.marks) == 1 else None
    source = _Generations()

    await _reconciler(queue, source, max_pages=1).run_pass()

    # the second mark committed after the read; its write is not covered by that advance
    assert list(queue.marks.values()) == ["groups"]


async def test_more_marks_than_one_pass_drains_is_reported_behind_and_none_is_lost() -> None:
    queue = _Queue(["groups"] * 7)
    source = _Generations()

    done = await _reconciler(queue, source, page_size=2, max_pages=2).run_pass()

    assert done.behind
    assert (done.applied, len(queue.marks)) == (4, 3)
    await _reconciler(queue, source, page_size=2, max_pages=5).run_pass()
    assert queue.marks == {}


async def test_two_reconcilers_on_the_same_page_lose_nothing() -> None:
    queue = _Queue(["groups", "revocations"])
    first, second = _Generations(), _Generations()

    await _reconciler(queue, first).run_pass()
    await _reconciler(queue, second).run_pass()

    assert queue.marks == {}
    assert sorted(first.advanced) == ["groups", "revocations"]


async def test_only_the_tables_without_the_trigger_get_one() -> None:
    queue = _Queue([], missing=[("agent_a", "groups"), ("platform", "revocations")])

    created = await install_write_mark_triggers(queue, marks_schema="platform", tables=("groups", "revocations"))  # type: ignore[arg-type]

    assert created == ["agent_a.groups", "platform.revocations"]
    assert all(WRITE_MARK_TRIGGER in statement for statement in queue.created)
    assert queue.missing == []


def test_the_trigger_fires_per_statement_on_every_write_into_the_one_queue() -> None:
    sql = write_mark_trigger_sql(marks_schema="platform", table_schema="agent_a", table="groups")
    assert "AFTER INSERT OR UPDATE OR DELETE" in sql
    assert "FOR EACH STATEMENT" in sql
    assert '"platform"."threetears_mark_collection_write"()' in sql
    table, function = write_marks_ddl("platform")
    assert '"platform"."collection_write_marks"' in table
    assert 'INSERT INTO "platform"."collection_write_marks"' in function


class _Lease:
    """a lease another replica holds, or a free one that records its release."""

    def __init__(self, *, held: bool) -> None:
        self._held = held
        self.released = False

    async def acquire(self, key: str, ttl_seconds: int = 30, max_wait_seconds: int = 60) -> Any:
        from threetears.core.coordination.lease import LeaseUnavailable

        if self._held:
            raise LeaseUnavailable(key)
        lease = self

        class _Handle:
            async def release(self) -> None:
                lease.released = True

        return _Handle()


async def test_a_replica_that_finds_the_lease_held_skips_its_pass() -> None:
    queue = _Queue(["groups"])
    source = _Generations()

    done = await _reconciler(queue, source, lease=_Lease(held=True)).run_pass()

    assert done.skipped
    assert source.advanced == []
    assert list(queue.marks.values()) == ["groups"]


async def test_the_replica_holding_the_lease_drains_and_lets_it_go() -> None:
    queue = _Queue(["groups"])
    lease = _Lease(held=False)

    done = await _reconciler(queue, _Generations(), lease=lease).run_pass()

    assert (done.skipped, done.applied) == (False, 1)
    assert lease.released
