"""on YugabyteDB each keyed read of ``role_assignments`` names the index that serves it.

``RoleAssignmentCollection``'s schema declares every index the acl migrations create on the table, so
:func:`~threetears.interfaces.keyed.keys.serving_index` chooses among all of them: a grant's natural-key reads go to its
scope's unique index (one row), a group's reads to ``idx_role_assignments_group_scope``, a role's to
``role_assignments_page_by_role_scope_type``, a namespace's (or a set of namespaces') to
``idx_role_assignments_scope_namespace``. Declared only in part, a read
was hinted to a worse index that was declared (a grant's natural-key read to the group's every grant), or to none.

Each read runs through a reader that says it is YugabyteDB and records what it is sent; the hint is the statement's
first words (``/*+ IndexOnlyScan(<alias> <index>) */``).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

import pytest
from threetears.agent.acl import RoleAssignmentCollection, RoleCollection
from threetears.agent.acl.types import ScopeType
from threetears.core.collections.delete_cascade import refuse_restricted_delete
from threetears.core.keyed import AnyOf, Fragment, read_by, serving_index

from .collection_support import make_collection

#: the gone flag a stamped table's live read holds false
_LIVE = frozenset({"gone"})


# parity-with: asyncpg.Pool (fetch / fetchrow / fetchval / execute -- the seams a collection method drives)
class _Yugabyte:
    """a YugabyteDB L3 as a collection's reads see it: it says its database, records each statement, and answers
    ``rows`` to every read."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.sent: list[str] = []
        self._rows = rows or []

    async def answers_from_yugabyte(self) -> bool:
        return True

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.sent.append(sql)
        return self._rows

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.sent.append(sql)
        return None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.sent.append(sql)
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        self.sent.append(sql)
        return "UPDATE 0"


def _hint(alias: str, index: str) -> str:
    return f"/*+ IndexOnlyScan({alias} {index}) */ "


def _assignments(reader: _Yugabyte) -> Any:
    return make_collection(RoleAssignmentCollection, l3_pool=reader)


@pytest.mark.parametrize(
    ("scope_type", "scope_id", "index"),
    [
        ("namespace", uuid7(), "uq_role_assignments_namespace_grant"),
        ("all", None, "uq_role_assignments_all_grant"),
    ],
)
async def test_an_ensure_reads_the_grant_by_its_natural_key_index(
    scope_type: str, scope_id: UUID | None, index: str
) -> None:
    """Flips if the ensure's read of a held grant is not hinted to its scope's natural-key unique index: ordered by
    id it was served by the primary key's (row_scope, assignment_id), the partition's every grant walked in id
    order, and with only some indexes declared by (group_id, scope_type), the group's every grant of the scope."""
    reader = _Yugabyte([{"assignment_id": uuid7()}])
    _, created = await _assignments(reader).ensure_group_role_assignment(
        group_id=uuid7(), role_id=uuid7(), scope_type=scope_type, scope_id=scope_id
    )
    assert created is False
    (read,) = reader.sent
    assert read.startswith(_hint("t", index)), read


@pytest.mark.parametrize(
    ("scope_type", "scope_id", "index"),
    [
        ("namespace", uuid7(), "uq_role_assignments_namespace_grant"),
        ("all", None, "uq_role_assignments_all_grant"),
    ],
)
async def test_the_winner_read_takes_the_natural_key_index(scope_type: str, scope_id: UUID | None, index: str) -> None:
    """Flips if the read of the grant a concurrent ensure won is not hinted to the unique index that absorbed the
    loser: no declared index holds its order (date_granted, assignment_id), and it binds the whole unique key, so the
    index reads the one row the grant holds."""
    reader = _Yugabyte()
    await _assignments(reader).grant_winner(uuid7(), uuid7(), scope_type, scope_id, "manual")
    (read,) = reader.sent
    assert read.startswith(_hint("t", index)), read


@pytest.mark.parametrize(
    ("scope_type", "scope_id", "index"),
    [
        ("namespace", uuid7(), "uq_role_assignments_namespace_grant"),
        ("all", None, "uq_role_assignments_all_grant"),
    ],
)
def test_a_revived_grant_is_found_by_its_natural_key_index(scope_type: str, scope_id: UUID | None, index: str) -> None:
    """Flips if the pins the revive of a gone grant writes by (the winner read's, its whole natural key) are not
    served by the scope's unique index, live or as stored. A keyed UPDATE carries no hint; its plan is
    test_index_plans.py::test_revive_gone_grant's."""
    pins = {
        "row_scope": AnyOf(("platform", "customer"), cast=None),
        "group_id": uuid7(),
        "role_id": uuid7(),
        "scope_type": scope_type,
        "scope_namespace_id": scope_id,
        "managed_by": "manual",
    }
    for live in (_LIVE, frozenset()):
        served = serving_index("role_assignments", pins, [], live=live)
        assert served is not None and served.name == index, (live, served)


@pytest.mark.parametrize(("scope_type", "scope_id"), [("namespace", uuid7()), ("all", None)])
async def test_a_revoke_reads_the_group_s_grants_of_the_scope(scope_type: str, scope_id: UUID | None) -> None:
    """Flips if a revoke's read of a group's grants on one scope (no role named) is not hinted to
    idx_role_assignments_group_scope, which binds both its leading columns."""
    reader = _Yugabyte()
    await _assignments(reader).delete_by_group_and_scope(group_id=uuid7(), scope_type=scope_type, scope_id=scope_id)
    (read,) = reader.sent
    assert read.startswith(_hint("t", "idx_role_assignments_group_scope")), read


async def test_a_group_s_grants_read_the_group_scope_index() -> None:
    """Flips if a group's grants, listed or counted, are not hinted to idx_role_assignments_group_scope."""
    reader = _Yugabyte()
    assignments = _assignments(reader)
    await assignments.list_by_group(uuid7())
    await assignments.count_by_group(uuid7())
    listed, counted = reader.sent
    assert listed.startswith(_hint("t", "idx_role_assignments_group_scope")), listed
    assert counted.startswith(_hint("r", "idx_role_assignments_group_scope")), counted


async def test_a_role_s_grants_read_the_role_index() -> None:
    """Flips if a role's grants, listed or counted, are not hinted to role_assignments_page_by_role_scope_type (the
    hub's v163 index, acl v009; v004's idx_role_assignments_role is on no hub schema)."""
    reader = _Yugabyte()
    assignments = _assignments(reader)
    await assignments.list_by_role(uuid7())
    await assignments.count_by_role(uuid7())
    listed, counted = reader.sent
    assert listed.startswith(_hint("t", "role_assignments_page_by_role_scope_type")), listed
    assert counted.startswith(_hint("r", "role_assignments_page_by_role_scope_type")), counted


async def test_a_role_s_delete_counts_its_live_grants_through_the_role_index() -> None:
    """Flips if the restricted delete's count of a role's live grants (role_assignments.role_id ON DELETE RESTRICT,
    run by 3tears for a gone mark) is not hinted to role_assignments_page_by_role_scope_type."""
    reader = _Yugabyte()
    await refuse_restricted_delete(make_collection(RoleCollection, l3_pool=reader), [uuid7()])
    (counted,) = reader.sent
    assert counted.startswith(_hint("r", "role_assignments_page_by_role_scope_type")), counted


async def test_a_namespace_s_grants_read_its_namespace_index() -> None:
    """Flips if a namespace's grants pinned by its id are not hinted to idx_role_assignments_scope_namespace
    (WHERE scope_namespace_id IS NOT NULL, which the read states as its guard): declared but not stated, the read was
    hinted to (scope_type, assignment_id), every namespace grant of every namespace read and filtered."""
    reader = _Yugabyte()
    await _assignments(reader).list_scoped_to_namespace(uuid7())
    (read,) = reader.sent
    assert read.startswith(_hint("t", "idx_role_assignments_scope_namespace")), read


async def test_a_set_of_namespaces_grants_read_the_namespace_index() -> None:
    """Flips if the read of a set of namespaces' grants in both partitions, its guard stated (the hub's outside grants:
    ``scope_namespace_id = ANY(...)``, ``scope_type = 'namespace'``, ``scope_namespace_id IS NOT NULL``), is not
    hinted to idx_role_assignments_scope_namespace: binding no set, it was hinted to the scope-type index, every
    namespace grant read and filtered, and the hub read once a namespace to reach the namespace index."""
    reader = _Yugabyte()
    await read_by(
        reader,
        "role_assignments",
        by={
            "scope_namespace_id": AnyOf([uuid7(), uuid7(), uuid7()], cast="uuid"),
            "scope_type": ScopeType.NAMESPACE.value,
            "row_scope": AnyOf(("platform", "customer"), cast=None),
        },
        where=[Fragment('t."scope_namespace_id" IS NOT NULL')],
        columns=("row_scope", "assignment_id", "scope_namespace_id", "group_id", "role_id"),
    )
    (read,) = reader.sent
    assert read.startswith(_hint("t", "idx_role_assignments_scope_namespace")), read


async def test_a_customer_s_type_grants_read_their_index() -> None:
    """Flips if a customer's type_customer grants are not hinted to idx_role_assignments_scope_type_customer."""
    reader = _Yugabyte()
    await _assignments(reader).list_of_type_customer(uuid7(), "agent")
    (read,) = reader.sent
    assert read.startswith(_hint("t", "idx_role_assignments_scope_type_customer")), read


async def test_a_scope_type_s_grants_read_the_scope_type_index() -> None:
    """Flips if every grant of one scope type is not hinted to role_assignments_page_by_scope_type (the hub's v163
    index, range-sharded on YugabyteDB, acl v009; v004's idx_role_assignments_scope_type is on no hub schema)."""
    reader = _Yugabyte()
    await _assignments(reader).list_by_scope_type(ScopeType.ALL)
    (read,) = reader.sent
    assert read.startswith(_hint("t", "role_assignments_page_by_scope_type")), read


def test_no_grant_index_holds_live_rows_alone() -> None:
    """Flips if a role_assignments index becomes live-only (``WHERE NOT gone``) while a read of the table as stored
    (a restore, quiet cleanup: no live guard) is still served by it: such a read would miss every gone row. v008 left
    the grant indexes whole, so each read is served by the same index with or without the live guard."""
    group, role, namespace = uuid7(), uuid7(), uuid7()
    reads: list[dict[str, Any]] = [
        {"group_id": group, "role_id": role, "scope_type": "namespace", "scope_namespace_id": namespace},
        {"group_id": group, "role_id": role, "scope_type": "all", "managed_by": "manual"},
        {"group_id": group},
        {"role_id": role},
        {"scope_namespace_id": namespace, "scope_type": "namespace"},
        {"scope_type": "all"},
    ]
    for pins in reads:
        live = serving_index("role_assignments", pins, [], live=_LIVE)
        stored = serving_index("role_assignments", pins, [])
        assert live is not None and live == stored, (pins, live, stored)
        assert not live.live, (pins, live)
