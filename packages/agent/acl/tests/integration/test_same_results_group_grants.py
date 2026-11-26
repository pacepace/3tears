"""a group's grants on one scope, and the grant a concurrent ensure won, read and deleted through the keyed helpers give
the rows, and leave the table, the hand-written statements did.

``RoleAssignmentCollection`` once sent three statements of its own (``GRANT_WINNER_OF_GROUP_SQL``,
``GRANTS_OF_GROUP_SCOPE_SQL``, ``DELETE_GRANTS_OF_GROUP_SCOPE_SQL``): ``scope_namespace_id IS NOT DISTINCT FROM $4`` and
``($5::text IS NULL OR managed_by = $5)``. They are ``read_by`` / ``delete_keyed`` now, a None namespace pinned
``IS NULL`` and a None provenance pinning nothing. Each is compared with its frozen statement through
:func:`~threetears.core.testing.same_results.assert_same_results` (empty, one row, many, NULLs, ties, a tenant
boundary, a page boundary and one past it), in both scopes, with a provenance and without; the delete by what it
reports and the table it leaves.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from threetears.agent.acl.collections import RoleAssignmentCollection
from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.core.collections.scan_cache import ScanCache
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_NS = uuid.UUID("5d2a7c1e-0b44-4a3f-9e61-3c8b7f2d1a05")
#: tenant 0's group (the one every read names) and tenant 1's
_GROUPS = (uuid.uuid5(_NS, "group-0"), uuid.uuid5(_NS, "group-1"))
#: a role per seeded row: the natural-key unique indexes allow one grant of a role per group, scope and provenance
_ROLES = tuple(uuid.uuid5(_NS, f"role-{n}") for n in range(32))
#: the namespace the namespace grants name, and another one
_NAMESPACES = (uuid.uuid5(_NS, "namespace-0"), uuid.uuid5(_NS, "namespace-1"))
_T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
_PROVENANCES = ("manual", "auto:agent-yaml")
_COLUMNS = ("row_scope", "assignment_id", "role_id", "group_id", "scope_type", "scope_namespace_id", "managed_by")
#: the table as its readers see it: its live rows (a stamped table's delete marks its row gone in place, acl v006)
_STATE = f"SELECT {', '.join(_COLUMNS)} FROM role_assignments WHERE NOT gone"

#: the frozen statements
_WINNER_SQL = (
    "SELECT row_scope, assignment_id, date_granted FROM role_assignments WHERE group_id = $1"
    " AND row_scope IN ('platform', 'customer') AND role_id = $2 AND scope_type = $3"
    " AND scope_namespace_id IS NOT DISTINCT FROM $4 AND managed_by = $5"
    " ORDER BY date_granted ASC, assignment_id ASC LIMIT 1"
)
_DELETE_SQL = (
    "WITH deleted AS (DELETE FROM role_assignments WHERE row_scope = $1 AND group_id = $2 AND scope_type = $3"
    " AND scope_namespace_id IS NOT DISTINCT FROM $4 AND ($5::text IS NULL OR managed_by = $5)"
    " RETURNING assignment_id) SELECT count(*)::int AS deleted FROM deleted"
)
#: (scope_type, its namespace, its row scope), as delete_by_group_and_scope derives the row scope
_SCOPES = {"all": ("all", None, "platform"), "namespace": ("namespace", _NAMESPACES[0], "customer")}


class _Store:
    """the migration runner's store over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a connection on a fresh schema the acl package built, holding the grants' parents: both groups, the roles,
    both namespaces."""
    schema = f"aclgrants_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_acl(runner)
        register_tools_platform(runner, depends_on=("agent_acl",))
        await runner.apply_for_platform_schema(_Store(connection))
        await connection.executemany(
            "INSERT INTO groups (row_scope, group_id, customer_id, name, epoch) VALUES ('platform', $1, NULL, $2, $3)",
            [(group, f"g{n}", next_epoch()) for n, group in enumerate(_GROUPS)],
        )
        await connection.executemany(
            "INSERT INTO roles (role_id, customer_id, name, description, permissions, epoch)"
            " VALUES ($1, NULL, $2, 'r', '{\"*\": [\"read\"]}'::jsonb, $3)",
            [(role, f"r{n}", next_epoch()) for n, role in enumerate(_ROLES)],
        )
        await connection.executemany(
            "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type, customer_id, date_created,"
            " date_updated, epoch) VALUES ('customer', $1, $2, 'tool', gen_random_uuid(), now(), now(), $3)",
            [(namespace, f"tool.n{n}", next_epoch()) for n, namespace in enumerate(_NAMESPACES)],
        )
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _grant(seed: Seeded) -> Mapping[str, Any]:
    """a grant of its tenant's group, its own role; every other one on 'all' (platform row scope), the rest on a
    namespace (customer row scope; every fourth row's namespace the other one); every third row's provenance not
    manual. ``seed.null`` leaves nothing NULL that a grant may not: a namespace grant on the other namespace stands in
    for a row the namespace pin must drop."""
    on_all = seed.index % 2 == 0
    return {
        "row_scope": "platform" if on_all else "customer",
        "assignment_id": uuid.uuid5(_NS, f"assignment-{seed.index}"),
        "role_id": _ROLES[seed.index],
        "group_id": _GROUPS[seed.tenant],
        "scope_type": "all" if on_all else "namespace",
        "scope_namespace_id": None if on_all else _NAMESPACES[1 if seed.index % 4 == 3 or seed.null else 0],
        "date_granted": _T0 + timedelta(minutes=seed.tie),
        "managed_by": _PROVENANCES[1 if seed.index % 3 == 2 else 0],
    }


def _collection(conn: Any) -> RoleAssignmentCollection:
    """the collection over a stub registry whose L3 pool is ``conn``: no L1, nothing promoted anywhere else."""
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = conn
    registry.register.return_value = None
    registry.scan_cache = ScanCache(None)
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = RoleAssignmentCollection(registry=registry, config=config)
    # the rows it announces are evicted from caches this stub has none of
    collection.invalidate_cache_many = AsyncMock()  # type: ignore[method-assign]
    return collection


@pytest.mark.parametrize("role", [0, 1, 3], ids=["role-on-all", "role-on-namespace", "role-on-other-namespace"])
@pytest.mark.parametrize("scope", list(_SCOPES))
async def test_the_winner_read_gives_the_hand_written_row(conn: asyncpg.Connection, scope: str, role: int) -> None:
    """Flips if the winner read (read_by over both row scopes, scope_id None as IS NULL, the earliest first) answers
    another grant than the statement did, or one where it found none."""
    scope_type, namespace, _ = _SCOPES[scope]
    args = (_GROUPS[0], _ROLES[role], scope_type, namespace, "manual")

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        # reached in ensure_group_role_assignment only on a lost insert race, so called directly
        found = await _collection(c).grant_winner(*args)
        return [] if found is None else [found]

    await assert_same_results(
        conn,
        table="role_assignments",
        row=_grant,
        old_sql=_WINNER_SQL,
        params=list(args),
        helper=helper,
        columns=("row_scope", "assignment_id", "date_granted"),
        order_by=["date_granted"],
    )


@pytest.mark.parametrize("managed_by", [None, "manual"], ids=["any-provenance", "manual"])
@pytest.mark.parametrize("scope", list(_SCOPES))
async def test_a_group_s_scope_delete_reports_and_leaves_what_the_statements_did(
    conn: asyncpg.Connection, scope: str, managed_by: str | None
) -> None:
    """Flips if delete_by_group_and_scope (read_by then delete_keyed, scope_id None as IS NULL, a None provenance
    pinning none) deletes another set of grants than the hand-written statements did, or reports another count (its
    read first: one that missed the grants would skip the delete, and leave them)."""
    scope_type, namespace, row_scope = _SCOPES[scope]

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        deleted = await _collection(c).delete_by_group_and_scope(
            group_id=_GROUPS[0], scope_type=scope_type, scope_id=namespace, managed_by=managed_by
        )
        return [{"deleted": deleted}]

    await assert_same_results(
        conn,
        table="role_assignments",
        row=_grant,
        old_sql=_DELETE_SQL,
        params=[row_scope, _GROUPS[0], scope_type, namespace, managed_by],
        helper=helper,
        state_sql=_STATE,
    )
