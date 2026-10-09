"""a group delete announces the memberships and assignments its database cascade removed.

``ON DELETE CASCADE`` takes a group's ``group_members`` and ``role_assignments`` rows where no
collection sees them, so no write generation moves for those tables. The framework's
:class:`GroupCollection` reads the rows before the delete and announces them afterwards through the
registry's collections of those tables, one advance per table, each broadcast naming what the row
reaches.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

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

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if sql.startswith("SELECT member_type, member_id, id FROM group_members WHERE group_id = $1"):
            return list(self.members)
        assert sql.startswith("SELECT row_scope, assignment_id FROM role_assignments WHERE group_id = $1"), sql
        return list(self.assignments)


class _Source:
    def __init__(self) -> None:
        self.advances: dict[str, int] = {}

    async def current(self, table_name: str) -> str:
        return f"inc:{self.advances.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        self.advances[table_name] = self.advances.get(table_name, 0) + 1
        return f"inc:{self.advances[table_name]}"


def _rows(bus: FakeNatsClient, table: str) -> list[CacheInvalidationMessage]:
    return [m for m in bus.published if isinstance(m, CacheInvalidationMessage) and m.table == table]


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
