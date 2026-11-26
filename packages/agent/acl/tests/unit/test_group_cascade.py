"""a group delete marks its memberships and assignments gone before itself, in one transaction, and announces them.

A group is stamped: its delete is a gone mark (an ``UPDATE``), so the database's ``ON DELETE CASCADE`` from
``group_members`` and ``role_assignments`` never fires. :class:`GroupCollection` reads those rows before the delete
(``delete_cascaded``); the framework's delete marks them gone, then the group, in one transaction, and announces them
once it commits through the registry's collections of those tables, one advance per table, each broadcast naming what
the row reaches.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from threetears.agent.acl import GroupCollection, GroupMemberCollection, RoleAssignmentCollection
from threetears.core.collections import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

_CONFIG = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class _Pool:
    """answers the two reads a group delete makes before it deletes."""

    def __init__(self, members: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> None:
        self.members = members
        self.assignments = assignments
        #: each write and transaction boundary, in order: ``BEGIN``, ``UPDATE <table>``, ``COMMIT`` / ``ROLLBACK``
        self.sent: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if sql.startswith(
            'SELECT t."member_type", t."member_id", t."id" FROM "group_members" AS t WHERE t."group_id" = $1'
        ):
            # the group's own key range, a page at a time in id order: one page holds every member here
            assert 'ORDER BY t."id" ASC LIMIT' in sql, sql
            return list(self.members)
        # the group's assignments by group, through the read the grant load shares
        assert sql.startswith('SELECT t."row_scope", t."assignment_id"') and 'FROM "role_assignments"' in sql, sql
        assert sql.endswith('WHERE t."group_id" = ANY($1::uuid[]) AND (NOT t."gone")'), sql
        (group_ids,) = params
        assert len(group_ids) == 1, params
        return list(self.assignments)

    async def execute(self, sql: str, *params: Any) -> str:
        """the gone marks a stamped group's delete leaves on its memberships and grants (ROW_STAMPED: the database's
        cascade never runs for a row marked gone in place), kept by the table each updates."""
        del params
        self.sent = [*self.sent, " ".join(sql.split()[:2])]
        return "UPDATE 1"

    def acquire(self) -> _Held:
        """the pool as its own connection, as a stamped parent's delete acquires one for its transaction."""
        return _Held(self, transaction=False)

    def transaction(self, **_options: Any) -> _Held:
        """a transaction, its boundaries kept with the writes."""
        return _Held(self, transaction=True)


class _Held:
    """an ``async with`` answering the pool: its connection, or a transaction whose boundaries it keeps."""

    def __init__(self, pool: _Pool, *, transaction: bool) -> None:
        self._pool = pool
        self._transaction = transaction

    async def __aenter__(self) -> _Pool:
        if self._transaction:
            self._pool.sent = [*self._pool.sent, "BEGIN"]
        return self._pool

    async def __aexit__(self, exc_type: type[BaseException] | None, *exc: object) -> None:
        if self._transaction:
            self._pool.sent = [*self._pool.sent, "COMMIT" if exc_type is None else "ROLLBACK"]


class _Source:
    def __init__(self) -> None:
        self.advances: dict[str, int] = {}

    async def current(self, table_name: str) -> str:
        return f"inc:{self.advances.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        self.advances[table_name] = self.advances.get(table_name, 0) + 1
        return f"inc:{self.advances[table_name]}"


def _rows(bus: FakeNatsClient, table: str) -> list[CacheInvalidationMessage]:
    """each row the messages published of ``table`` name, as its own row message (a write of many rows is one message
    naming them all)."""
    return [
        row
        for m in bus.published
        if isinstance(m, CacheInvalidationMessage) and m.table == table
        for row in m.each_row()
    ]


async def test_the_cascaded_rows_are_announced_under_one_advance_per_table() -> None:
    group_id = uuid4()
    members = [
        {"member_type": "user", "member_id": uuid4(), "id": uuid4()},
        {"member_type": "group", "member_id": uuid4(), "id": uuid4()},
    ]
    assignments = [{"row_scope": "customer", "assignment_id": uuid4()}]
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool(members, assignments), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    source = _Source()
    registry.set_generation_source(source)
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)
    RoleAssignmentCollection(registry, _CONFIG, nats_client=bus)
    groups.delete_from_store = AsyncMock()  # type: ignore[method-assign]

    await groups.delete(("customer", group_id))

    assert source.advances == {"groups": 1, "group_members": 1, "role_assignments": 1}
    assert [(m.ids, m.columns, m.bump_rows) for m in _rows(bus, "group_members")] == [
        ([f"{group_id}", f"{m['id']}"], {"member_type": m["member_type"], "member_id": f"{m['member_id']}"}, 2)
        for m in members
    ]
    (assignment,) = _rows(bus, "role_assignments")
    assert (assignment.ids, assignment.columns) == (
        ["customer", f"{assignments[0]['assignment_id']}"],
        {"group_id": f"{group_id}"},
    )


async def test_a_group_with_nothing_to_cascade_advances_only_itself() -> None:
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool([], []), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    source = _Source()
    registry.set_generation_source(source)
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)
    RoleAssignmentCollection(registry, _CONFIG, nats_client=bus)
    groups.delete_from_store = AsyncMock()  # type: ignore[method-assign]

    await groups.delete(("platform", uuid4()))

    assert source.advances == {"groups": 1}


class _FailingSource:
    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        from threetears.core.exceptions import GenerationUnavailableError

        raise GenerationUnavailableError(f"epoch bucket unreachable for {table_name}")


async def test_a_failed_groups_advance_still_announces_the_cascade() -> None:
    import pytest
    from threetears.core.exceptions import GenerationUnavailableError

    group_id = uuid4()
    members = [{"member_type": "user", "member_id": uuid4(), "id": uuid4()}]
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool(members, []), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_FailingSource())
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)
    groups.delete_from_store = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(GenerationUnavailableError, match="groups"):
        await groups.delete(("customer", group_id))

    assert [m.ids for m in _rows(bus, "group_members")] == [[f"{group_id}", f"{members[0]['id']}"]]


class _FailsFor:
    """a source whose advance fails for one table only."""

    def __init__(self, table: str) -> None:
        self.table = table

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        from threetears.core.exceptions import GenerationUnavailableError

        if table_name == self.table:
            raise GenerationUnavailableError(f"epoch bucket unreachable for {table_name}")
        return "inc:1"


async def test_a_failed_membership_advance_still_announces_the_assignments() -> None:
    import pytest
    from threetears.core.exceptions import GenerationUnavailableError

    group_id = uuid4()
    members = [{"member_type": "user", "member_id": uuid4(), "id": uuid4()}]
    assignments = [{"row_scope": "customer", "assignment_id": uuid4()}]
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool(members, assignments), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_FailsFor("group_members"))
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)
    RoleAssignmentCollection(registry, _CONFIG, nats_client=bus)
    groups.delete_from_store = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(GenerationUnavailableError, match="group_members"):
        await groups.delete(("customer", group_id))

    (assignment,) = _rows(bus, "role_assignments")
    assert (assignment.generation, assignment.columns) == ("inc:1", {"group_id": f"{group_id}"})


class _RaisingPool(_Pool):
    """answers the cascade's reads; its gone mark of the memberships raises after the statement was sent."""

    async def execute(self, sql: str, *params: Any) -> str:
        if '"group_members"' in sql:
            raise ConnectionError("lost after the statement was sent")
        return await super().execute(sql, *params)


async def test_a_membership_mark_that_raised_leaves_the_group_live() -> None:
    """Flips if a group's gone mark is sent outside its memberships' transaction, or after one of their marks failed:
    the transaction rolls back whole, never a group gone with its members still holding its roles. Nothing committed,
    so nothing is announced."""
    group_id = uuid4()
    members = [{"member_type": "user", "member_id": uuid4(), "id": uuid4()}]
    pool = _RaisingPool(members, [])
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_Source())
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)

    with pytest.raises(ConnectionError):
        await groups.delete(("customer", group_id))

    assert pool.sent == ["BEGIN", "ROLLBACK"]
    assert _rows(bus, "group_members") == []


async def test_the_memberships_and_grants_go_before_the_group_in_one_transaction() -> None:
    """Flips if the group is marked gone before its memberships and grants, or outside their transaction: every
    child's stamp must be at most its parent's (quiet cleanup reaches the children first), and a failure must leave
    neither half (Pace: children gone before the parent, one transaction)."""
    group_id = uuid4()
    members = [{"member_type": "user", "member_id": uuid4(), "id": uuid4()}]
    assignments = [{"row_scope": "customer", "assignment_id": uuid4()}]
    pool = _Pool(members, assignments)
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_Source())
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    GroupMemberCollection(registry, _CONFIG, nats_client=bus)
    RoleAssignmentCollection(registry, _CONFIG, nats_client=bus)

    await groups.delete(("customer", group_id))

    assert pool.sent[0] == "BEGIN" and pool.sent[-1] == "COMMIT", pool.sent
    marks = pool.sent[1:-1]
    assert sorted(marks[:-1]) == ['UPDATE "group_members"', 'UPDATE "role_assignments"'], pool.sent
    assert marks[-1] == 'UPDATE "groups"', pool.sent
    # each announced once the transaction committed, naming what it reaches
    assert [m.columns for m in _rows(bus, "group_members")] == [
        {"member_type": "user", "member_id": f"{members[0]['member_id']}"}
    ]
    assert [m.columns for m in _rows(bus, "role_assignments")] == [{"group_id": f"{group_id}"}]


async def test_a_cascade_no_registered_collection_can_announce_is_refused_before_the_delete() -> None:
    """Flips if a group delete marks grants gone that no collection announces (no pod would hear the revoke), or
    refuses only after the group itself was marked gone."""
    assignments = [{"row_scope": "customer", "assignment_id": uuid4()}]
    bus = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l3_pool=_Pool([], assignments), l2_client=bus, kv_key_scope="hub")  # type: ignore[arg-type]
    registry.set_generation_source(_Source())
    groups = GroupCollection(registry, _CONFIG, nats_client=bus)
    groups.delete_from_store = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="role_assignments"):
        await groups.delete(("customer", uuid4()))

    groups.delete_from_store.assert_not_awaited()
