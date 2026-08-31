"""three-tier collections for the canonical rbac tables.

every 3tears app shares the five rbac tables: ``groups``,
``group_members``, ``roles``, ``role_assignments``, and
``namespaces``. the schemas are universal (column shapes, constraints,
partition discriminators) so the Collections live here and
deploying apps subclass them only when admin-specific query shapes
need to ride alongside the canonical CRUD + evaluator-loader paths.

scope of the canonical Collections (kept generic; nothing app-specific
leaks in):

- table-level CRUD via :class:`SchemaBackedCollection` and the
  declarative :class:`TableSchema`
- evaluator-loader query methods that the canonical
  :class:`threetears.agent.acl.MembershipLoader` and
  :class:`threetears.agent.acl.GrantLoader` Protocols call into
  (``load_for_user`` / ``load_for_agent`` / ``load_for_groups`` /
  ``get_many``)
- bulk fetch by id list (every app needs this for the introspection /
  audit / grant-resolver paths)
- a small set of universally-useful list / find queries

scope explicitly out of bounds for the canonical classes (lives on
deploying-app subclasses):

- admin-endpoint dynamic ``list_by_filter`` shapes
- per-cardinality counts driving deploy-specific audit envelopes
- discovery JOINs that span an app-specific multi-table query

table names use the canonical RBAC vocabulary (``groups`` etc.)
without any deploy-specific schema prefix; the prefix (``platform.``
in the 3tears hub deployment) is applied at the L3 pool's
``search_path``, not in the schema name on the Collection.
"""

from __future__ import annotations

import json as _json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid7

from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.namespaces import namespace_contains
from threetears.observe import get_logger

from threetears.agent.acl.entities import (
    GroupEntity,
    GroupMemberEntity,
    ImpersonationGateEntity,
    NamespaceEntity,
    RoleAssignmentEntity,
    RoleEntity,
    row_scope_for_customer,
)
from threetears.agent.acl.types import (
    MAX_GROUP_MEMBERSHIP_DEPTH,
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)

log = get_logger(__name__)

__all__ = [
    "GroupCollection",
    "GroupMemberCollection",
    "ImpersonationGateCollection",
    "ImpersonationGateStatus",
    "NamespaceCollection",
    "NamespaceRescope",
    "NamespaceRescopeRefused",
    "RoleAssignmentCollection",
    "RoleCollection",
]


def _coerce_uuid(raw: Any) -> UUID | None:
    """coerce a database cell value to a :class:`UUID` (or ``None``).

    asyncpg returns ``UUID`` columns as native :class:`UUID` instances
    on a direct asyncpg pool; the agent-side
    :class:`NatsProxyL3Backend` pool round-trips rows through JSON
    which collapses UUIDs to their string representation. Collections
    whose method return types carry UUID fields (e.g.
    :class:`GroupMembership.member_id`) must normalize both shapes so
    callers get a stable Python type regardless of which pool answered
    the query. ``None`` passes through; any other type is passed to
    :class:`UUID`'s constructor via ``str()`` conversion.

    :param raw: value pulled directly from a row dictionary
    :ptype raw: Any
    :return: UUID instance, or ``None`` when the input is ``None``
    :rtype: UUID | None
    """
    result: UUID | None = None
    if raw is not None:
        if isinstance(raw, UUID):
            result = raw
        else:
            result = UUID(str(raw))
    return result


def _coerce_role_permissions(raw: Any) -> dict[str, frozenset[str]]:
    """coerce a JSONB ``permissions`` payload into ``{resource: frozenset(action)}``.

    asyncpg returns JSONB columns either as a parsed ``dict`` (when
    the connection has the JSONB codec registered) or as the raw
    ``str``; both shapes land here and normalize to the dataclass-
    friendly mapping that :class:`Role` expects.

    :param raw: raw JSONB column value as returned by asyncpg
    :ptype raw: Any
    :return: normalized permissions mapping
    :rtype: dict[str, frozenset[str]]
    """
    parsed: dict[str, Any] = {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw:
        loaded = _json.loads(raw)
        if isinstance(loaded, dict):
            parsed = loaded
    result: dict[str, frozenset[str]] = {}
    for resource_type, actions in parsed.items():
        if isinstance(actions, list):
            result[resource_type] = frozenset(str(a) for a in actions)
    return result


# ---------------------------------------------------------------------------
# GroupCollection
# ---------------------------------------------------------------------------


class GroupCollection(SchemaBackedCollection[GroupEntity]):
    """three-tier collection for ``groups`` rows.

    groups use hard-delete. cascading FKs on ``group_members`` and
    ``role_assignments`` (``ON DELETE CASCADE``) clean up member +
    assignment rows in the same transaction, so the collection only
    needs to delete the group row itself. CRUD comes from the
    declarative :class:`TableSchema`; the evaluator-loader / introspection
    helpers (``list_by_customer`` / ``list_all`` / ``get_many`` /
    ``get_by_name``) stay on the canonical class because every
    rbac-consuming app needs them.

    a group is addressable two ways, and both are exact: by ``group_id``,
    and by ``name`` within its scope. the platform DDL makes ``name``
    unique per customer, and unique across platform-scoped rows, so
    :meth:`get_by_name` returns at most one row.

    this class previously carried a nullable ``managed_key`` column and a
    ``get_by_managed_key``, on the stated grounds that ``name`` was "a
    non-unique human label" and so unusable as a handle. that was never
    true: the uniqueness indexes have existed since the platform's first
    migration, and they are exactly the uniqueness ``managed_key``
    re-declared. the duplicate handle cost more than it bought -- its name
    read as something an operator had to mint, and a customer-scoped key
    resolved relative to the ASKER, so it could not express "that specific
    customer's group" at all. removed in groups-task-01; ``name`` and
    ``group_id`` are the handles.

    WARNING for deploying apps that author their own ``groups`` DDL: the
    ``name``-is-unique premise above holds for the PLATFORM DDL, not
    necessarily for yours. an app that deliberately made ``name``
    non-unique (so admins can type "Editors" twice) and hung privilege
    derivation on the removed handle must NOT follow this note into
    :meth:`get_by_name` -- resolving a privilege tier through a mutable,
    non-unique label is a privilege-escalation shape. keep an app-owned
    immutable handle column and resolve through that instead; one
    consumer already does.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "group_id")
    partition_exempt_methods = frozenset(
        {
            "list_by_customer",
            "list_all",
            "get_many",
            "get_by_name",
            "delete_from_store",
            "save_entity",
            "create",
            "find_by_id",
        }
    )
    # v0.8.0 hygiene enrichment: date_created/date_updated carry the
    # platform DDL's ``DEFAULT NOW()`` server default (test fixture in
    # ``packages/agent/workspace/tests/integration/test_cross_agent_workspace.py``
    # at line 152-159 is the canonical reference). Note: this table
    # is platform-managed -- the DDL lives outside 3tears and 3tears
    # has no migration for it; ``row_scope`` is a 3tears-side
    # partition column that the platform DDL does NOT carry. v0.8.0
    # cannot reconcile that divergence without a platform-side
    # change. Documented for future cleanup.
    # v0.8.0 shard 04.6: bare-``id`` PK column renamed to
    # ``group_id`` to standardize on ``<entity>_id`` across all entity
    # tables. The rename happens at the platform DDL side (outside
    # 3tears); the schema declaration here reflects the post-rename
    # column.
    schema = TableSchema(
        name="groups",
        primary_key=("row_scope", "group_id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("group_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            # unique per scope: the platform DDL owns a partial unique index
            # per customer and another across platform-scoped rows, which is
            # what makes get_by_name an exact lookup rather than a search.
            Column("name", STRING_TYPE),
            Column("description", STRING_TYPE, nullable=True),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            Column("date_updated", DATETIMETZ_TYPE, server_default="now()"),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"groups"``
        :rtype: str
        """
        return "groups"

    @property
    def entity_class(self) -> type[GroupEntity]:
        """return entity class for this collection.

        :return: :class:`GroupEntity`
        :rtype: type[GroupEntity]
        """
        return GroupEntity

    def create(self, data: dict[str, Any]) -> GroupEntity:
        """construct new group entity, auto-deriving ``row_scope``.

        every group row carries ``customer_id`` (nullable; platform-
        scoped groups have ``customer_id IS NULL``); the partition
        column ``row_scope`` is the defensive discriminator
        (``platform`` / ``customer``) and the database CHECK constraint
        pins ``row_scope='platform' <-> customer_id IS NULL`` at the
        row level. this override sets row_scope from customer_id so
        callers keep their pre-partition shape.

        :param data: row payload; may omit ``row_scope`` (override
            sets it) or include it (override leaves explicit values
            untouched)
        :ptype data: dict[str, Any]
        :return: newly constructed (not-yet-persisted) group entity
        :rtype: GroupEntity
        """
        if "row_scope" not in data:
            data = {
                **data,
                "row_scope": row_scope_for_customer(data.get("customer_id")),
            }
        return super().create(data)

    async def find_by_id(
        self,
        group_id: UUID,
    ) -> GroupEntity | None:
        """resolve group by ``group_id`` alone via the ``UNIQUE (group_id)`` constraint.

        every endpoint that takes ``{group_id}`` in the URL knows the
        row's id but not the partition column ``row_scope``. uniqueness
        is preserved by the table-level ``UNIQUE (group_id)`` constraint
        (v0.8.0 shard 04.6 renamed from bare ``id``) so a single-column
        fetch is unambiguous.

        :param group_id: group UUID
        :ptype group_id: UUID
        :return: group entity or ``None`` when no row exists
        :rtype: GroupEntity | None
        """
        result: GroupEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM groups WHERE group_id = $1",
                group_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_name(
        self,
        name: str,
        customer_id: UUID | None,
    ) -> GroupEntity | None:
        """resolve the group named ``name`` within ``customer_id``'s scope.

        ``name`` is unique per scope -- the platform DDL owns a partial
        unique index per customer and another across platform-scoped
        rows -- so this returns at most one row. the resolved row is
        promoted into L1/L2 caches.

        the ``customer_id`` predicate uses ``IS NOT DISTINCT FROM`` so a
        ``None`` scope matches the platform partition's NULL exactly
        (mirrors :meth:`get_by_owner_and_customer`).

        :param name: the group name to resolve
        :ptype name: str
        :param customer_id: owning customer UUID, or ``None`` for a
            platform-scoped group
        :ptype customer_id: UUID | None
        :return: group entity or ``None`` when no group matches
        :rtype: GroupEntity | None
        """
        result: GroupEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                """
                SELECT * FROM groups
                 WHERE name = $1
                   AND customer_id IS NOT DISTINCT FROM $2
                """,
                name,
                customer_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def list_by_customer(
        self,
        customer_id: UUID,
    ) -> list[GroupEntity]:
        """list every ``groups`` row owned by ``customer_id``.

        rows are promoted into L1/L2 caches for subsequent ``get(id)``
        lookups. returns an empty list (never ``None``) when the
        customer has no groups so callers can iterate unconditionally.

        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: list of group entities ordered by ``date_created``
            ascending
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM groups
                 WHERE row_scope = 'customer'
                   AND customer_id = $1
                 ORDER BY date_created ASC
                """,
                customer_id,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def get_many(
        self,
        group_ids: Sequence[UUID],
    ) -> list[GroupEntity]:
        """fetch every group row whose id is in ``group_ids``.

        used by the introspection / grant-loader paths that need a
        bulk group lookup keyed by the assignment rows' ``group_id``.
        row order is database-determined (no ``ORDER BY`` clause)
        since the evaluator consumes the result as an unordered map;
        callers that need a specific order should sort the returned
        list themselves.

        empty input short-circuits without a SQL round-trip and
        returns an empty list. promotes resolved rows into L1/L2
        caches.

        :param group_ids: sequence of group UUIDs to resolve
        :ptype group_ids: Sequence[UUID]
        :return: list of group entities (subset of ``group_ids`` that
            exist in L3); order is database-determined
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None and len(group_ids) > 0:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM groups WHERE group_id = ANY($1::uuid[])",
                list(group_ids),
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                # the NATS proxy pool round-trips UUID columns through
                # JSON which collapses them to strings; the schema's
                # _coerce_row handles UUID columns it knows about, but
                # belt-and-suspenders for the two pk-adjacent columns.
                if "group_id" in data:
                    data["group_id"] = _coerce_uuid(data["group_id"])
                if "customer_id" in data:
                    data["customer_id"] = _coerce_uuid(data["customer_id"])
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def list_all(
        self,
        customer_id: UUID | None = None,
    ) -> list[GroupEntity]:
        """list every ``groups`` row, optionally filtered by customer.

        platform admins may list every group (``customer_id=None``) or
        scope to a specific customer; customer admins pass their own
        ``customer_id``. rows are ordered by ``date_created`` ascending
        and promoted into L1/L2 caches.

        :param customer_id: optional customer scope; ``None`` returns
            every row
        :ptype customer_id: UUID | None
        :return: list of group entities ordered by ``date_created``
            ascending
        :rtype: list[GroupEntity]
        """
        result: list[GroupEntity] = []
        if self.l3_pool is not None:
            if customer_id is None:
                rows = await self.l3_pool.fetch(
                    """
                    SELECT * FROM groups
                     WHERE row_scope IN ('platform', 'customer')
                     ORDER BY date_created ASC
                    """,
                )
            else:
                rows = await self.l3_pool.fetch(
                    """
                    SELECT * FROM groups
                     WHERE row_scope = 'customer'
                       AND customer_id = $1
                     ORDER BY date_created ASC
                    """,
                    customer_id,
                )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result


# ---------------------------------------------------------------------------
# GroupMemberCollection
# ---------------------------------------------------------------------------


class GroupMemberCollection(SchemaBackedCollection[GroupMemberEntity]):
    """three-tier collection for ``group_members`` rows.

    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``load_for_user`` / ``load_for_agent`` /
    ``list_by_group`` / ``find_by_group_and_id``) stay on the canonical
    class.
    """

    primary_key_column: tuple[str, ...] = ("group_id", "id")
    partition_exempt_methods = frozenset(
        {
            "load_for_user",
            "load_for_agent",
            "delete_from_store",
            "save_entity",
        }
    )
    # v0.8.0 hygiene enrichment: ``date_added`` server default matches
    # the platform DDL (test fixture line 171). Platform-managed --
    # 3tears has no migration for this table; the FK to ``groups`` and
    # the ``member_type`` CHECK constraint live in the platform DDL.
    schema = TableSchema(
        name="group_members",
        primary_key=("group_id", "id"),
        columns=[
            Column("id", UUID_TYPE),
            Column("group_id", UUID_TYPE, partition=True),
            Column("member_type", STRING_TYPE, immutable=True),
            Column("member_id", UUID_TYPE, immutable=True),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column(
                "date_added",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
        ],
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"group_members"``
        :rtype: str
        """
        return "group_members"

    @property
    def entity_class(self) -> type[GroupMemberEntity]:
        """return entity class for this collection.

        :return: :class:`GroupMemberEntity`
        :rtype: type[GroupMemberEntity]
        """
        return GroupMemberEntity

    async def load_for_user(
        self,
        user_id: UUID,
    ) -> list[GroupMembership]:
        """resolve ``user_id`` to its :class:`GroupMembership` rows.

        returns the protocol-shape :class:`GroupMembership` dataclass
        instances (NOT :class:`GroupMemberEntity`) because the unified
        evaluator's :class:`MembershipLoader` Protocol speaks in the
        ACL types. rows are NOT promoted into L1/L2 because the ACL
        flow does not re-read by primary key; the evaluator consumes
        the membership list, not per-row entities.

        empty result is an empty list (never ``None``).

        :param user_id: user UUID to resolve
        :ptype user_id: UUID
        :return: list of memberships naming ``user_id`` as a user
            member
        :rtype: list[GroupMembership]
        """
        result: list[GroupMembership] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM group_members
                 WHERE member_type = 'user'
                   AND member_id = $1
                """,
                user_id,
            )
            result = [
                GroupMembership(
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    member_type=MemberType(row["member_type"]),
                    member_id=_coerce_uuid(row["member_id"]),  # type: ignore[arg-type]
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> list[GroupMembership]:
        """resolve ``agent_id`` to its :class:`GroupMembership` rows.

        symmetric counterpart of :meth:`load_for_user` for the agent
        side of an intersection evaluation.

        :param agent_id: agent UUID to resolve
        :ptype agent_id: UUID
        :return: list of memberships naming ``agent_id`` as an agent
            member
        :rtype: list[GroupMembership]
        """
        result: list[GroupMembership] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM group_members
                 WHERE member_type = 'agent'
                   AND member_id = $1
                """,
                agent_id,
            )
            result = [
                GroupMembership(
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    member_type=MemberType(row["member_type"]),
                    member_id=_coerce_uuid(row["member_id"]),  # type: ignore[arg-type]
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def load_for_group(
        self,
        group_id: UUID,
    ) -> list[GroupMembership]:
        """resolve ``group_id`` to the memberships naming it as a GROUP member.

        each returned row's own ``group_id`` is a PARENT group the child
        belongs to -- the edges the evaluator's depth-capped walk
        follows.

        :param group_id: child group UUID to resolve
        :ptype group_id: UUID
        :return: list of memberships naming ``group_id`` as a group
            member
        :rtype: list[GroupMembership]
        """
        result: list[GroupMembership] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT group_id, member_type, member_id, customer_id
                  FROM group_members
                 WHERE member_type = 'group'
                   AND member_id = $1
                """,
                group_id,
            )
            result = [
                GroupMembership(
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    member_type=MemberType(row["member_type"]),
                    member_id=_coerce_uuid(row["member_id"]),  # type: ignore[arg-type]
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def membership_would_cycle(
        self,
        *,
        group_id: UUID,
        member_group_id: UUID,
    ) -> bool:
        """true when adding ``member_group_id`` into ``group_id`` closes a cycle.

        a group inside itself is refused outright; beyond that, the
        check asks whether ``group_id`` is already reachable from
        ``member_group_id`` by walking group edges DOWNWARD (children of
        children), out to :data:`MAX_GROUP_MEMBERSHIP_DEPTH` levels.
        cheap by construction: the cap bounds the walk, and writers are
        expected to call this BEFORE inserting a ``member_type='group'``
        row -- an admin surface that skips it can still write a cycle,
        which resolution then tolerates (the read-time walk is
        depth-capped so a cycle cannot loop it), but the row is
        nonsense and this is the guard that keeps it out.

        :param group_id: parent group receiving the new member
        :ptype group_id: UUID
        :param member_group_id: child group being added as a member
        :ptype member_group_id: UUID
        :return: whether the insert would create a membership cycle
        :rtype: bool
        """
        cycles = group_id == member_group_id
        if not cycles and self.l3_pool is not None:
            frontier: set[UUID] = {member_group_id}
            for _ in range(MAX_GROUP_MEMBERSHIP_DEPTH):
                if not frontier:
                    break
                rows = await self.l3_pool.fetch(
                    """
                    SELECT member_id
                      FROM group_members
                     WHERE member_type = 'group'
                       AND group_id = ANY($1::uuid[])
                    """,
                    list(frontier),
                )
                children = {child for row in rows if (child := _coerce_uuid(row["member_id"])) is not None}
                if group_id in children:
                    cycles = True
                    break
                frontier = children
        return cycles

    async def list_by_group(
        self,
        group_id: UUID,
    ) -> list[GroupMemberEntity]:
        """list every membership row for ``group_id`` ordered by ``date_added``.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :param group_id: owning group UUID
        :ptype group_id: UUID
        :return: list of membership entities ordered by ``date_added``
            ascending
        :rtype: list[GroupMemberEntity]
        """
        result: list[GroupMemberEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM group_members
                 WHERE group_id = $1
                 ORDER BY date_added ASC
                """,
                group_id,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def find_by_group_and_id(
        self,
        group_id: UUID,
        member_row_id: UUID,
    ) -> GroupMemberEntity | None:
        """fetch a membership by PK and assert it belongs to ``group_id``.

        used where the URL carries ``(group_id, member_row_id)`` and
        the caller wants to fail closed (404) when the membership row
        exists but lives in a different group. the method is a
        :meth:`BaseCollection.get` followed by an in-Python
        ``group_id`` predicate check so the three-tier cache path
        (L1 -> L2 -> L3) answers the fetch.

        :param group_id: expected owning group UUID
        :ptype group_id: UUID
        :param member_row_id: ``group_members.id`` UUID
        :ptype member_row_id: UUID
        :return: membership entity, or ``None`` when the row is absent
            OR exists but belongs to a different group
        :rtype: GroupMemberEntity | None
        """
        entity = await self.get((group_id, member_row_id))
        result: GroupMemberEntity | None = None
        if entity is not None:
            data = entity.to_dict()
            if data.get("group_id") == group_id:
                result = entity
        return result


# ---------------------------------------------------------------------------
# RoleCollection
# ---------------------------------------------------------------------------


class RoleCollection(SchemaBackedCollection[RoleEntity]):
    """three-tier collection for ``roles`` rows.

    roles use hard-delete; admin endpoints typically guard against
    deleting builtins or any role that is referenced by an assignment
    (the ``role_assignments.role_id`` FK is ``ON DELETE RESTRICT``).
    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``list_all`` / ``list_builtin`` /
    ``get_many``) stay on the canonical class.
    """

    # v0.8.0 hygiene enrichment: ``is_builtin`` server default + date
    # defaults match the platform DDL (test fixture lines 177-185).
    # Platform-managed table -- 3tears has no migration; the name
    # uniqueness constraints live in the platform DDL. Those are now
    # ownership-partitioned rather than global: unique across
    # ``customer_id IS NULL`` rows, and unique per ``(customer_id,
    # name)``, so two customers may both author a "Field Manager".
    # v0.8.0 shard 04.6: bare-``id`` PK renamed to ``role_id`` to
    # standardize on ``<entity>_id`` across all entity tables.
    primary_key_column: str = "role_id"
    schema = TableSchema(
        name="roles",
        primary_key="role_id",
        columns=[
            Column("role_id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("description", STRING_TYPE),
            Column("permissions", JSONB_TYPE),
            # owning customer for a customer-authored role; NULL for
            # platform-owned rows (every built-in, plus any role a
            # platform admin authors). immutable: re-owning a role
            # would silently move every assignment that references it,
            # so the write path creates a new role instead.
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column(
                "is_builtin",
                BOOL_TYPE,
                immutable=True,
                server_default="false",
            ),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            Column("date_updated", DATETIMETZ_TYPE, server_default="now()"),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"roles"``
        :rtype: str
        """
        return "roles"

    @property
    def entity_class(self) -> type[RoleEntity]:
        """return entity class for this collection.

        :return: :class:`RoleEntity`
        :rtype: type[RoleEntity]
        """
        return RoleEntity

    async def list_all(self) -> list[RoleEntity]:
        """list every role row ordered by ``date_created`` ascending.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :return: list of role entities ordered by ``date_created``
            ascending
        :rtype: list[RoleEntity]
        """
        result: list[RoleEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM roles ORDER BY date_created ASC",
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def list_builtin(self) -> list[RoleEntity]:
        """list every platform-shipped builtin role.

        rows are promoted into L1/L2 caches so subsequent ``get(id)``
        calls hit L1.

        :return: list of builtin role entities ordered by ``name``
            ascending
        :rtype: list[RoleEntity]
        """
        result: list[RoleEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM roles
                 WHERE is_builtin = TRUE
                 ORDER BY name ASC
                """,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result

    async def get_many(
        self,
        role_ids: Sequence[UUID],
    ) -> list[Role]:
        """resolve ``role_ids`` to :class:`Role` rows.

        returns the protocol-shape :class:`Role` dataclass instances
        (NOT :class:`RoleEntity`) because the unified evaluator's
        :class:`GrantLoader` Protocol speaks in the ACL types.

        empty input short-circuits without a SQL round-trip and
        returns an empty list. order is database-determined; the
        evaluator consumes the result as an unordered collection.

        :param role_ids: sequence of role UUIDs to resolve
        :ptype role_ids: Sequence[UUID]
        :return: list of :class:`Role` instances (subset of
            ``role_ids`` that exist in L3)
        :rtype: list[Role]
        """
        result: list[Role] = []
        if self.l3_pool is not None and len(role_ids) > 0:
            rows = await self.l3_pool.fetch(
                """
                SELECT role_id, name, permissions, is_builtin, customer_id
                  FROM roles
                 WHERE role_id = ANY($1::uuid[])
                """,
                list(role_ids),
            )
            result = [
                Role(
                    id=_coerce_uuid(row["role_id"]),  # type: ignore[arg-type]
                    name=row["name"],
                    permissions=_coerce_role_permissions(row["permissions"]),
                    is_built_in=bool(row["is_builtin"]),
                    customer_id=_coerce_uuid(row["customer_id"]),
                )
                for row in rows
            ]
        return result

    async def list_visible_to_customer(
        self,
        customer_id: UUID | None,
    ) -> list[RoleEntity]:
        """list the roles a caller in ``customer_id`` may see.

        that is every platform-owned role (``customer_id IS NULL`` --
        the built-ins plus anything a platform admin authored) plus the
        roles ``customer_id`` itself authored. a role authored by
        ANOTHER customer is absent from the result, so a customer cannot
        enumerate a neighbour's role names through the role menu.

        passing ``customer_id=None`` narrows to platform-owned rows
        only; a platform admin wanting every row across every customer
        uses :meth:`list_all`.

        :param customer_id: caller's owning customer UUID, or ``None``
            to see only the platform-owned roles
        :ptype customer_id: UUID | None
        :return: list of role entities ordered by ``date_created``
            ascending
        :rtype: list[RoleEntity]
        """
        result: list[RoleEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                """
                SELECT * FROM roles
                 WHERE customer_id IS NULL
                    OR customer_id = $1
                 ORDER BY date_created ASC
                """,
                customer_id,
            )
            for row in rows:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result.append(
                    self.entity_class(data, is_new=False, collection=self),
                )
        return result


# ---------------------------------------------------------------------------
# RoleAssignmentCollection
# ---------------------------------------------------------------------------


class RoleAssignmentCollection(SchemaBackedCollection[RoleAssignmentEntity]):
    """three-tier collection for ``role_assignments`` rows.

    CRUD comes from the declarative :class:`TableSchema`;
    evaluator-loader queries (``load_for_groups`` /
    ``ensure_group_role_assignment`` / ``delete_by_group_and_scope``)
    stay on the canonical class. apps that need admin-specific filter /
    listing shapes (``list_by_filter`` / ``list_by_namespace`` /
    ``count_by_*``) subclass and add their own methods.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "assignment_id")
    partition_exempt_methods = frozenset(
        {
            "load_for_groups",
            "ensure_group_role_assignment",
            "delete_by_group_and_scope",
            "delete_from_store",
            "save_entity",
            "create",
            "find_by_id",
        }
    )
    # v0.8.0 hygiene enrichment: ``date_granted`` server default
    # matches the platform DDL (test fixture line 200). Platform-
    # managed table -- 3tears has no migration; FKs to ``roles`` /
    # ``groups`` / ``namespaces`` and the ``scope_type`` CHECK live
    # in the platform DDL.
    # v0.8.0 shard 04.6: bare-``id`` PK renamed to ``assignment_id``
    # to standardize on ``<entity>_id`` across all entity tables. The
    # rename happens at the platform DDL side (outside 3tears); the
    # schema declaration here reflects the post-rename column.
    schema = TableSchema(
        name="role_assignments",
        primary_key=("row_scope", "assignment_id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("assignment_id", UUID_TYPE),
            Column("role_id", UUID_TYPE, immutable=True),
            Column("group_id", UUID_TYPE, immutable=True),
            Column("scope_type", STRING_TYPE, immutable=True),
            Column("scope_namespace_id", UUID_TYPE, nullable=True, immutable=True),
            Column("scope_namespace_type", STRING_TYPE, nullable=True, immutable=True),
            Column("scope_customer_id", UUID_TYPE, nullable=True, immutable=True),
            # the root of a ``subtree`` scope, as a NAME. containment is
            # a statement about the name space -- namespace ids are
            # minted per row and carry no hierarchy -- and a subtree
            # root need not be a materialized namespace row at all,
            # which is why this is not a second id column.
            Column("scope_namespace_name", STRING_TYPE, nullable=True, immutable=True),
            Column("granted_by", UUID_TYPE, nullable=True, immutable=True),
            Column(
                "date_granted",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            # ``managed_by`` declares assignment provenance. default
            # ``'manual'`` covers admin-authored rows; agent-side
            # automation passes ``'auto:agent-yaml'``.
            Column("managed_by", STRING_TYPE),
        ],
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"role_assignments"``
        :rtype: str
        """
        return "role_assignments"

    @property
    def entity_class(self) -> type[RoleAssignmentEntity]:
        """return entity class for this collection.

        :return: :class:`RoleAssignmentEntity`
        :rtype: type[RoleAssignmentEntity]
        """
        return RoleAssignmentEntity

    def create(
        self,
        data: dict[str, Any],
    ) -> RoleAssignmentEntity:
        """construct new assignment entity, auto-deriving ``row_scope``.

        role_assignments has no ``customer_id`` column on the row
        itself; the row's effective customer flows from the scope
        triple (``scope_type`` / ``scope_namespace_id`` /
        ``scope_customer_id``). ``row_scope`` stores the discriminator
        explicitly (``platform`` for super_admin ``scope_type='all'``
        rows or ``scope_type='type_customer'`` rows whose
        ``scope_customer_id IS NULL``; ``customer`` for every other
        shape) so the partition primitive can guard the row uniformly.

        :param data: row payload; may include ``row_scope`` directly
            (override leaves it alone) or rely on derivation from the
            scope triple
        :ptype data: dict[str, Any]
        :return: newly constructed assignment entity
        :rtype: RoleAssignmentEntity
        """
        if "row_scope" not in data:
            scope_type = data.get("scope_type")
            scope_customer_id = data.get("scope_customer_id")
            if scope_type == "all":
                row_scope = "platform"
            elif scope_type == "subtree":
                # a subtree scope names a node in the namespace-NAME
                # space and no customer at all, so by this rule -- the
                # row's effective customer flows from the scope -- it
                # partitions with the other customerless shapes. the
                # caller's customer comes from the GROUP, which is a
                # different column on a different table.
                row_scope = "platform"
            elif scope_type == "type_customer" and scope_customer_id is None:
                row_scope = "platform"
            else:
                row_scope = "customer"
            data = {**data, "row_scope": row_scope}
        return super().create(data)

    async def find_by_id(
        self,
        assignment_id: UUID,
    ) -> RoleAssignmentEntity | None:
        """resolve assignment by ``assignment_id`` alone (v0.8.0 shard 04.6).

        admin endpoints take ``{assignment_id}`` in the URL but not the
        partition column ``row_scope``. uniqueness across the whole
        table is preserved by the table-level
        ``UNIQUE (assignment_id)`` constraint (renamed from bare
        ``id`` in v0.8.0 shard 04.6).

        :param assignment_id: assignment UUID
        :ptype assignment_id: UUID
        :return: assignment entity or ``None`` when no row exists
        :rtype: RoleAssignmentEntity | None
        """
        result: RoleAssignmentEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM role_assignments WHERE assignment_id = $1",
                assignment_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def load_for_groups(
        self,
        group_ids: Sequence[UUID],
    ) -> list[RoleAssignment]:
        """resolve ``group_ids`` to every assignment they hold.

        returns the protocol-shape :class:`RoleAssignment` dataclass
        instances (NOT :class:`RoleAssignmentEntity`) because the
        unified evaluator's :class:`GrantLoader` Protocol speaks in
        the ACL types.

        this method does NOT accept a ``namespace`` filter; it returns
        every assignment every group in the input set holds. callers
        run :meth:`RoleAssignment.covers` themselves to scope the
        result to a specific namespace (the evaluator does this on the
        hot path; the canonical :class:`CollectionGrantLoader` filters
        on its way out so the Protocol contract is preserved).

        empty input short-circuits without a SQL round-trip.

        :param group_ids: sequence of group UUIDs to resolve
        :ptype group_ids: Sequence[UUID]
        :return: list of assignments held by any group in
            ``group_ids``
        :rtype: list[RoleAssignment]
        """
        result: list[RoleAssignment] = []
        if self.l3_pool is not None and len(group_ids) > 0:
            # row_scope spans both 'platform' (scope_type='all' / NULL
            # scope_customer_id) and 'customer' grants; the unified
            # evaluator's hot path needs every assignment regardless
            # of scope.
            rows = await self.l3_pool.fetch(
                """
                SELECT assignment_id, role_id, group_id, scope_type,
                       scope_namespace_id, scope_namespace_type,
                       scope_customer_id, scope_namespace_name
                  FROM role_assignments
                 WHERE row_scope IN ('platform', 'customer')
                   AND group_id = ANY($1::uuid[])
                """,
                list(group_ids),
            )
            result = [
                RoleAssignment(
                    id=_coerce_uuid(row["assignment_id"]),  # type: ignore[arg-type]
                    role_id=_coerce_uuid(row["role_id"]),  # type: ignore[arg-type]
                    group_id=_coerce_uuid(row["group_id"]),  # type: ignore[arg-type]
                    scope_type=ScopeType(row["scope_type"]),
                    scope_namespace_id=_coerce_uuid(row["scope_namespace_id"]),
                    scope_namespace_type=row["scope_namespace_type"],
                    scope_customer_id=_coerce_uuid(row["scope_customer_id"]),
                    scope_namespace_name=row["scope_namespace_name"],
                )
                for row in rows
            ]
        return result

    async def ensure_group_role_assignment(
        self,
        *,
        group_id: UUID,
        role_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str = "manual",
    ) -> tuple[UUID, bool]:
        """idempotent insert of ``(group, role, scope)`` assignment row.

        returns ``(assignment_id, created)``: the assignment's UUID — an
        existing row's id when the tuple already exists, or a freshly minted
        ``uuid7`` for a newly inserted row — paired with ``created``, ``True``
        only when a new row was inserted. Callers that re-run this on a periodic
        self-heal net use ``created`` to stay quiet (skip logging / side effects)
        when the grant was already present, instead of treating every re-check as
        a fresh materialization.

        the underlying ``role_assignments`` table does NOT carry a
        unique constraint over the lookup tuple (only ``id`` is PK)
        so this method does a SELECT-then-INSERT under a single
        round-trip pattern rather than ``ON CONFLICT (...) DO
        UPDATE``. a concurrent inserter can race; the worst case is
        two physical rows for the same logical grant, which the
        evaluator treats as a no-op duplicate. callers serialize
        admin-path writes themselves so the race is theoretical.

        the ``scope_type`` argument maps to :class:`ScopeType`:

        - ``"namespace"`` — ``scope_id`` is the namespace UUID
        - ``"all"`` — ``scope_id`` must be ``None``
        - ``"type_customer"`` — not supported via this method (raises
          ``ValueError``)

        the ``managed_by`` argument stamps provenance onto freshly-
        inserted rows. when the row already exists the existing
        ``managed_by`` stays untouched so a manual row accidentally
        re-discovered by a translator is never silently re-classed.

        :param group_id: group UUID to bind
        :ptype group_id: UUID
        :param role_id: role UUID to grant
        :ptype role_id: UUID
        :param scope_type: scope discriminator (``"namespace"`` or
            ``"all"``)
        :ptype scope_type: str
        :param scope_id: namespace UUID for ``"namespace"`` scope;
            must be ``None`` for ``"all"`` scope
        :ptype scope_id: UUID | None
        :param managed_by: provenance marker (``"manual"`` |
            ``"auto:agent-yaml"``); applied only on insert
        :ptype managed_by: str
        :return: ``(assignment_id, created)`` -- the assignment UUID (existing or
            newly inserted) and whether a NEW row was inserted this call
        :rtype: tuple[UUID, bool]
        :raises ValueError: if ``scope_type`` is unsupported or
            ``scope_id`` shape mismatches the scope
        :raises RuntimeError: if no L3 pool is bound
        """
        if scope_type not in ("namespace", "all"):
            raise ValueError(
                f"unsupported scope_type for idempotent ensure: {scope_type}; use save_entity for type_customer scope",
            )
        if scope_type == "namespace" and scope_id is None:
            raise ValueError(
                "scope_type='namespace' requires a non-None scope_id",
            )
        if scope_type == "all" and scope_id is not None:
            raise ValueError(
                "scope_type='all' requires scope_id=None",
            )
        if self.l3_pool is None:
            raise RuntimeError(
                "RoleAssignmentCollection.ensure_group_role_assignment requires an L3 pool",
            )

        row_scope = "platform" if scope_type == "all" else "customer"
        existing_row = await self.l3_pool.fetchrow(
            """
            SELECT assignment_id FROM role_assignments
             WHERE row_scope = $1
               AND group_id = $2
               AND role_id = $3
               AND scope_type = $4
               AND scope_namespace_id IS NOT DISTINCT FROM $5
             ORDER BY assignment_id ASC
             LIMIT 1
            """,
            row_scope,
            group_id,
            role_id,
            scope_type,
            scope_id,
        )
        result: UUID
        created: bool
        if existing_row is not None:
            result = existing_row["assignment_id"]
            created = False
        else:
            new_id = uuid7()
            now = datetime.now(UTC)
            await self.l3_pool.execute(
                """
                INSERT INTO role_assignments (
                    row_scope, assignment_id, role_id, group_id, scope_type,
                    scope_namespace_id, scope_namespace_type,
                    scope_customer_id, granted_by, date_granted,
                    managed_by
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, NULL, NULL, NULL, $7, $8
                )
                """,
                row_scope,
                new_id,
                role_id,
                group_id,
                scope_type,
                scope_id,
                now,
                managed_by,
            )
            result = new_id
            created = True
        return result, created

    async def delete_by_group_and_scope(
        self,
        *,
        group_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str | None = None,
    ) -> int:
        """delete every assignment matching ``(group, scope)`` predicate.

        symmetric counterpart to :meth:`ensure_group_role_assignment`.
        used when revoking a previously-granted scope; returns the
        number of rows the DB confirms it deleted so callers can
        detect a no-op (zero) vs an actual revocation.

        the ``managed_by`` filter restricts the delete to
        provenance-matched rows; ``None`` (the default) means "no
        filter on managed_by" so every row matching the
        ``(group, scope)`` tuple is removed.

        :param group_id: group UUID whose assignments should be
            removed
        :ptype group_id: UUID
        :param scope_type: scope discriminator (``"namespace"`` or
            ``"all"``)
        :ptype scope_type: str
        :param scope_id: namespace UUID for ``"namespace"`` scope;
            must be ``None`` for ``"all"`` scope
        :ptype scope_id: UUID | None
        :param managed_by: optional provenance filter (``"manual"`` |
            ``"auto:agent-yaml"``); ``None`` means no filter
        :ptype managed_by: str | None
        :return: number of rows deleted (zero when nothing matched)
        :rtype: int
        :raises ValueError: if ``scope_type`` / ``scope_id`` shape is
            invalid
        """
        if scope_type not in ("namespace", "all"):
            raise ValueError(
                f"unsupported scope_type for delete_by_group_and_scope: "
                f"{scope_type}; use the per-id delete for "
                "type_customer scope",
            )
        if scope_type == "namespace" and scope_id is None:
            raise ValueError(
                "scope_type='namespace' requires a non-None scope_id",
            )
        if scope_type == "all" and scope_id is not None:
            raise ValueError(
                "scope_type='all' requires scope_id=None",
            )
        row_scope = "platform" if scope_type == "all" else "customer"
        result: int = 0
        if self.l3_pool is not None:
            if managed_by is None:
                status = await self.l3_pool.execute(
                    """
                    DELETE FROM role_assignments
                     WHERE row_scope = $1
                       AND group_id = $2
                       AND scope_type = $3
                       AND scope_namespace_id IS NOT DISTINCT FROM $4
                    """,
                    row_scope,
                    group_id,
                    scope_type,
                    scope_id,
                )
            else:
                status = await self.l3_pool.execute(
                    """
                    DELETE FROM role_assignments
                     WHERE row_scope = $1
                       AND group_id = $2
                       AND scope_type = $3
                       AND scope_namespace_id IS NOT DISTINCT FROM $4
                       AND managed_by = $5
                    """,
                    row_scope,
                    group_id,
                    scope_type,
                    scope_id,
                    managed_by,
                )
            # asyncpg returns "DELETE <count>" status string
            parts = status.split()
            if len(parts) >= 2 and parts[0].upper() == "DELETE":
                result = int(parts[1])
        return result


# ---------------------------------------------------------------------------
# NamespaceCollection
# ---------------------------------------------------------------------------


class NamespaceRescopeRefused(RuntimeError):
    """raised when a re-scope would change WHICH customer owns a row.

    :meth:`NamespaceCollection.rescope` exists for one transition:
    ``customer_id`` going from absent to present, or present to absent, on a
    row whose id and name do not move. Carrying a row from one customer to
    another is a different act -- a re-tenanting -- and it is refused here
    rather than expressed, because every grant, every derived schema name and
    every audit record already written against the row names the customer it
    is being moved away from.
    """


@dataclass(frozen=True)
class NamespaceRescope:
    """what a :meth:`NamespaceCollection.rescope` call decided about one row.

    :ivar namespace_id: the row the call addressed
    :ivar moved: whether THIS call was the one that wrote the move. ``False``
        covers three different situations, all of them fine: no row exists
        yet, the row is already in the target scope, or another replica moved
        it between this call's read and its write
    :ivar previous_row_scope: partition the row was in before, or ``None``
        when no row existed
    :ivar previous_customer_id: customer the row carried before, or ``None``
        for a platform-scoped row and for a row that did not exist
    :ivar row_scope: partition the row is in after the call
    :ivar customer_id: customer the row carries after the call
    """

    namespace_id: UUID
    moved: bool
    previous_row_scope: str | None
    previous_customer_id: UUID | None
    row_scope: str
    customer_id: UUID | None


class NamespaceCollection(SchemaBackedCollection[NamespaceEntity]):
    """three-tier collection for ``namespaces`` rows.

    CRUD comes from the declarative :class:`TableSchema`; canonical
    lookup helpers (``find_by_id`` / ``get_by_name`` /
    ``find_by_type_and_customer`` / ``get_by_owner_and_customer`` /
    ``list_ids_by_customer_and_type`` / ``list_all_ids``) stay here
    because every rbac-consuming app needs to resolve namespace rows
    by these shapes during evaluator hydration / audit-snapshot
    composition. discovery JOINs that span app-specific tables live on
    deploying-app subclasses.
    """

    primary_key_column: tuple[str, ...] = ("row_scope", "namespace_id")
    partition_exempt_methods = frozenset(
        {
            "delete_from_store",
            "save_entity",
            "create",
            "find_by_type_and_customer",
            "list_ids_by_customer_and_type",
            "list_all_ids",
            "list_ids_under_name",
            "get_by_name",
            "get_by_agent_id",
            "get_by_owner_and_customer",
            "find_by_id",
            "list_tool_namespaces_for_actor",
            "list_skill_eligible_tool_namespaces",
            # ``rescope`` is cross-partition BY CONSTRUCTION rather than by
            # accident: it reads a row in one partition and writes it into the
            # other. It is addressed by ``namespace_id`` -- the column the
            # separate ``UNIQUE (namespace_id)`` index makes unambiguous
            # platform-wide -- so there is no ``row_scope`` argument for the
            # guard to find, and adding one would let a caller name a partition
            # the row is not in.
            "rescope",
        }
    )
    # v0.8.0 hygiene enrichment: ``metadata`` carries the test
    # fixture's ``DEFAULT '{}'::jsonb`` server default (line 144).
    # Most columns on this table are platform-managed (the deploying
    # app owns the canonical DDL); the ``row_scope`` partition column
    # is 3tears-side only.
    # v0.8.0 shard 04.6: bare-``id`` PK renamed to ``namespace_id``
    # to standardize on ``<entity>_id`` across all entity tables. The
    # rename happens at the platform DDL side (outside 3tears); the
    # schema declaration here reflects the post-rename column.
    # agent-tools-eligibility shard 01: ``tool_eligible`` /
    # ``skill_eligible`` columns added by the 3tears-side
    # ``agent_tools_platform`` migration package (v001 ``ALTER TABLE
    # namespaces ADD COLUMN IF NOT EXISTS ...``). NOT NULL with
    # DB-side defaults so existing rows keep their pre-shard
    # visibility shape without a backfill. The columns apply to every
    # namespace type but are only meaningful on ``tool``-type rows;
    # other types (``workspace``, ``agent`` ...) inherit the defaults
    # and the eligibility query helpers below additionally filter on
    # ``namespace_type='tool'`` so the flags never widen visibility
    # for non-tool namespaces.
    schema = TableSchema(
        name="namespaces",
        primary_key=("row_scope", "namespace_id"),
        columns=[
            Column("row_scope", STRING_TYPE, partition=True),
            Column("namespace_id", UUID_TYPE),
            Column("name", STRING_TYPE),
            Column("namespace_type", STRING_TYPE, immutable=True),
            Column("owner_agent_id", UUID_TYPE, nullable=True, immutable=True),
            # canonical NAME of the namespace row that OWNS this one,
            # self-referential. WITHOUT this entry the Collection
            # neither reads nor writes the column, so the evaluator
            # would see ``None`` on every row and no agent would be
            # recognised as an owner of anything.
            #
            # NOT immutable at this layer: an operator re-homing a
            # namespace under a different owner is a legitimate write.
            # The write that must never happen -- an AGENT binding it
            # through the query broker -- is refused there, where the
            # caller is known to be an agent; declaring it immutable
            # here would break the operator path instead.
            Column("owner_namespace", STRING_TYPE, nullable=True),
            Column("customer_id", UUID_TYPE, nullable=True, immutable=True),
            Column("schema_name", STRING_TYPE, nullable=True, immutable=True),
            Column(
                "metadata",
                JSONB_TYPE,
                nullable=True,
                server_default="'{}'::jsonb",
            ),
            Column("tool_eligible", BOOL_TYPE, server_default="true"),
            Column("skill_eligible", BOOL_TYPE, server_default="false"),
            # gu-task-02b: ``face_*`` columns added by the 3tears-side
            # ``agent_tools_platform`` migration package (v002 ``ALTER
            # TABLE namespaces ADD COLUMN IF NOT EXISTS ...``). They
            # persist the authored FACE flags (``TearsTool`` cvar ->
            # ``ToolManifestEntry`` field, gu-task-02) that the hub
            # ``ToolNamespaceEmitter`` stamps onto each ``tool``-type
            # row. NOT NULL with DB-side defaults so pre-face rows read
            # as "platform-tool only" without a backfill:
            # ``face_platform_tool`` DEFAULT TRUE (the pre-face shape),
            # ``face_api`` / ``face_mcp`` DEFAULT FALSE (explicit
            # opt-in). Consumed by the API namespace stamp (gu-task-24),
            # MCP export (gu-task-25, reads ``face_mcp``) and the
            # face-flip CRUD re-stamp (gu-task-26).
            Column("face_api", BOOL_TYPE, server_default="false"),
            Column("face_mcp", BOOL_TYPE, server_default="false"),
            Column("face_platform_tool", BOOL_TYPE, server_default="true"),
            # the FOURTH reach face, added by the same platform package's
            # v003 (``ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS
            # face_rest BOOLEAN NOT NULL DEFAULT FALSE`` plus
            # ``face_rest_declaration JSONB``). Declared here for the same
            # reason the three above are: a column the migration adds but
            # the schema never names is a column the generated INSERT never
            # writes and the generated SELECT never returns, so the flag
            # would read FALSE forever no matter what a tool authored.
            #
            # REST needs the pair rather than a boolean because its address
            # is authored rather than derived. The boolean stays for the
            # cheap ``WHERE face_rest`` read and is derived by the writer
            # from ``face_rest_declaration IS NOT NULL``; the JSONB carries
            # the :class:`~threetears.agent.tools.http_operation.RestAffordance`
            # a serving face needs to match a URL and bind its arguments.
            Column("face_rest", BOOL_TYPE, server_default="false"),
            # ``server_default`` on a nullable column looks redundant -- ``DEFAULT
            # NULL`` is what a nullable column already does -- and it is not. A
            # column declared WITHOUT one is written on every INSERT whether or
            # not the caller mentioned it (``build_insert``'s omission rule reads
            # ``col.server_default is not None and col.name not in data``), so
            # declaring it plain would put ``face_rest_declaration`` into the
            # INSERT that every OTHER namespace writer issues -- the knowledge
            # emitter, the workspace emitter, the datasource path -- none of
            # which knows the column exists. Every ``face_*`` column above is
            # declared the same way for the same reason.
            Column("face_rest_declaration", JSONB_TYPE, nullable=True, server_default="NULL"),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"namespaces"``
        :rtype: str
        """
        return "namespaces"

    @property
    def entity_class(self) -> type[NamespaceEntity]:
        """return entity class for this collection.

        :return: :class:`NamespaceEntity`
        :rtype: type[NamespaceEntity]
        """
        return NamespaceEntity

    def create(self, data: dict[str, Any]) -> NamespaceEntity:
        """construct new namespace entity, auto-deriving ``row_scope``.

        every namespace row carries ``customer_id`` (nullable for
        platform-scoped rows); ``row_scope`` is the defensive
        discriminator (``platform`` / ``customer``) the partition
        primitive enforces. this override pins ``row_scope`` to the
        value implied by ``customer_id`` so callers continue to pass
        the customer-bearing shape.

        :param data: row payload; may omit ``row_scope`` (override
            sets it) or include it (override leaves explicit values
            untouched)
        :ptype data: dict[str, Any]
        :return: newly constructed (not-yet-persisted) namespace entity
        :rtype: NamespaceEntity
        """
        if "row_scope" not in data:
            customer_id = data.get("customer_id")
            data = {
                **data,
                "row_scope": "platform" if customer_id is None else "customer",
            }
        return super().create(data)

    async def rescope(
        self,
        namespace_id: UUID,
        *,
        customer_id: UUID | None,
    ) -> NamespaceRescope:
        """move one namespace row between the ``platform`` and ``customer`` partitions.

        **Why this cannot be the ordinary upsert.** ``row_scope`` is DERIVED
        from ``customer_id`` (``platform`` <-> ``customer_id IS NULL``) and it
        is half the PRIMARY KEY, so a row that gains a customer changes its own
        key. :meth:`save_entity`'s generated statement nominates
        ``(row_scope, namespace_id)`` as its ``ON CONFLICT`` arbiter, which no
        longer matches the stored row -- while the separate
        ``UNIQUE (namespace_id)`` index the deploying app declares DOES. The
        write is therefore not a conflict the upsert resolves; it is an
        unretryable ``UniqueViolationError``, and re-issuing it never converges.
        ``customer_id`` is additionally declared ``immutable=True``, so even a
        matching arbiter would leave the column where it was.

        **It is an UPDATE, never a delete-and-reinsert, and that is the whole
        safety argument.** Grants reach a namespace by reference:
        ``role_assignments.scope_namespace_id`` carries a foreign key to
        ``namespaces(namespace_id)`` with ``ON DELETE CASCADE``. Removing the
        row to write it again under the new scope would destroy every
        assignment naming it -- including the operator-authored grants nothing
        rebuilds -- with no write anything observes. An in-place UPDATE leaves
        ``namespace_id`` and ``name`` untouched, so every reference survives
        the move unchanged, and no grant is added or removed by it.

        **What is refused.** A row already carrying a DIFFERENT customer raises
        :class:`NamespaceRescopeRefused`. This method's transition is a
        customer appearing or disappearing on a row that stays the same row;
        carrying one company's namespace to another is a re-tenanting, and
        every grant, schema name and audit record already written against the
        row names the customer it would be moved away from.

        **Concurrency.** The UPDATE fences on the partition this call read, so
        two replicas converging on the same destination produce one write and
        one no-op rather than two. The loser reports ``moved=False``, which is
        the same answer it gets for "already in place" -- deliberately, because
        both mean "this call changed nothing and the row is where it should
        be". Nothing here retries: the caller re-attempts its own write, and
        the row is already correct.

        The pre-move composite key is evicted from L1 and L2 and announced to
        other pods; leaving it would let a read addressed at the OLD partition
        keep answering with the pre-move row for the life of the process.

        :param namespace_id: the row to move, addressed by the column the
            secondary unique index makes unique platform-wide
        :ptype namespace_id: UUID
        :param customer_id: the customer the row should belong to, or ``None``
            to return it to the platform partition
        :ptype customer_id: UUID | None
        :return: what was decided, including whether this call wrote the move
        :rtype: NamespaceRescope
        :raises NamespaceRescopeRefused: when the row already belongs to a
            different customer
        """
        target_scope = row_scope_for_customer(customer_id)
        result = NamespaceRescope(
            namespace_id=namespace_id,
            moved=False,
            previous_row_scope=None,
            previous_customer_id=None,
            row_scope=target_scope,
            customer_id=customer_id,
        )
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT row_scope, customer_id FROM namespaces WHERE namespace_id = $1",
                namespace_id,
            )
            if row is not None:
                result = await self._rescope_existing(
                    namespace_id=namespace_id,
                    customer_id=customer_id,
                    target_scope=target_scope,
                    previous_scope=str(row["row_scope"]),
                    previous_customer=_coerce_uuid(row["customer_id"]),
                )
        return result

    async def _rescope_existing(
        self,
        *,
        namespace_id: UUID,
        customer_id: UUID | None,
        target_scope: str,
        previous_scope: str,
        previous_customer: UUID | None,
    ) -> NamespaceRescope:
        """decide and write the move for a row that is known to exist.

        split from :meth:`rescope` so each half keeps one exit; the reasoning
        for every branch is on that method.

        :param namespace_id: the row being moved
        :ptype namespace_id: UUID
        :param customer_id: the customer the row should belong to
        :ptype customer_id: UUID | None
        :param target_scope: partition implied by ``customer_id``
        :ptype target_scope: str
        :param previous_scope: partition the row is in now
        :ptype previous_scope: str
        :param previous_customer: customer the row carries now
        :ptype previous_customer: UUID | None
        :return: what was decided
        :rtype: NamespaceRescope
        :raises NamespaceRescopeRefused: on a customer-to-customer move
        """
        if previous_customer is not None and customer_id is not None and previous_customer != customer_id:
            raise NamespaceRescopeRefused(
                f"namespaces.{namespace_id}: refusing to re-tenant a row from customer "
                f"{previous_customer} to customer {customer_id}. rescope moves a row "
                f"between the platform and customer partitions; it does not carry a "
                f"namespace, its grants or its derived schema name between customers",
            )
        moved = False
        if previous_scope != target_scope or previous_customer != customer_id:
            # ``required_l3_pool`` rather than ``l3_pool``: the caller only
            # reaches this method having READ a row through the pool, so a
            # ``None`` here would be a wiring change that broke the read too,
            # and it names that rather than raising an attribute error.
            written = await self.required_l3_pool.fetchrow(
                "UPDATE namespaces"
                "   SET row_scope = $1, customer_id = $2, date_updated = $3"
                " WHERE namespace_id = $4 AND row_scope = $5"
                " RETURNING namespace_id",
                target_scope,
                customer_id,
                datetime.now(UTC),
                namespace_id,
                previous_scope,
            )
            moved = written is not None
            if moved:
                await self.invalidate_cache((previous_scope, namespace_id))
        return NamespaceRescope(
            namespace_id=namespace_id,
            moved=moved,
            previous_row_scope=previous_scope,
            previous_customer_id=previous_customer,
            row_scope=target_scope,
            customer_id=customer_id,
        )

    async def find_by_id(
        self,
        namespace_id: UUID,
    ) -> NamespaceEntity | None:
        """resolve namespace by ``namespace_id`` alone (v0.8.0 shard 04.6).

        callers know the namespace's id (often computed
        deterministically from the owning agent_id / customer_id) but
        not the partition column ``row_scope``. uniqueness is
        preserved by the ``UNIQUE (namespace_id)`` constraint
        (renamed from bare ``id`` in v0.8.0 shard 04.6) so a single-
        column fetch is unambiguous.

        :param namespace_id: namespace UUID
        :ptype namespace_id: UUID
        :return: namespace entity or ``None`` when no row exists
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces WHERE namespace_id = $1",
                namespace_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_name(self, name: str) -> NamespaceEntity | None:
        """look up namespace by unique name.

        searches L3 for namespace matching name. promotes found
        namespace into L1/L2 caches.

        :param name: unique namespace name
        :ptype name: str
        :return: namespace entity or ``None`` if not found
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces WHERE name = $1",
                name,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_agent_id(
        self,
        agent_id: UUID,
        namespace_type: str = "agent",
    ) -> NamespaceEntity | None:
        """look up agent-private namespace by owning agent.

        searches L3 for namespace where ``owner_agent_id`` matches and
        ``namespace_type`` equals the supplied value (default
        ``"agent"`` for the per-agent private namespace shape every
        rbac-consuming app shares). promotes found namespace into
        L1/L2 caches.

        :param agent_id: agent UUID to look up namespace for
        :ptype agent_id: UUID
        :param namespace_type: namespace type discriminator; defaults
            to ``"agent"``
        :ptype namespace_type: str
        :return: namespace entity or ``None`` if not found
        :rtype: NamespaceEntity | None
        """
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            # private agent namespaces always live in the customer
            # partition: every agent belongs to one customer and the
            # namespace inherits that customer's row_scope.
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM namespaces "
                "WHERE row_scope = 'customer' "
                "  AND owner_agent_id = $1 AND namespace_type = $2",
                agent_id,
                namespace_type,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> NamespaceEntity | None:
        """look up namespace by ``(namespace_type, owner_agent_id, customer_id)``.

        natural lookup key for per-agent / per-customer typed
        namespaces (one row per ``(agent, customer)`` pair).

        the underlying ``namespaces`` table does NOT carry a unique
        constraint over the triple (only ``id`` is PK and ``name`` /
        ``schema_name`` are unique). callers create one row per triple
        by convention; this method orders by ``id`` ASC and returns
        the first row to guarantee deterministic resolution if
        duplicates ever land.

        promotes the resolved row into L1/L2 caches.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID, or ``None`` for
            agent-agnostic namespaces (e.g. shared rows)
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID, or ``None`` for
            platform-scoped rows
        :ptype customer_id: UUID | None
        :return: namespace entity or ``None`` if no row matches the
            triple
        :rtype: NamespaceEntity | None
        """
        row_scope = row_scope_for_customer(customer_id)
        result: NamespaceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                """
                SELECT * FROM namespaces
                 WHERE row_scope = $1
                   AND namespace_type = $2
                   AND owner_agent_id IS NOT DISTINCT FROM $3
                   AND customer_id IS NOT DISTINCT FROM $4
                 ORDER BY namespace_id ASC
                 LIMIT 1
                """,
                row_scope,
                namespace_type,
                owner_agent_id,
                customer_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def find_by_type_and_customer(
        self,
        *,
        namespace_type: str,
        customer_id: UUID,
    ) -> list[NamespaceEntity]:
        """return every namespace entity for ``(namespace_type, customer_id)``.

        used where the caller needs the full entity surface (not just
        ids) so it can evaluate per-row authorization against each
        candidate namespace via the unified evaluator and then extract
        the authorized ids from the surviving namespaces.

        rows are NOT promoted into L1/L2 here. typical call shape is a
        scan over a per-customer slice (typically 0-10 rows per
        customer) so the per-row promotion overhead outweighs the L1
        hit ratio. callers that want a single namespace by id chase
        the warmer ``get(id)`` path which promotes naturally.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: list of namespace entities matching both filters
        :rtype: list[NamespaceEntity]
        """
        result: list[NamespaceEntity] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM namespaces WHERE row_scope = 'customer'   AND namespace_type = $1 AND customer_id = $2",
                namespace_type,
                customer_id,
            )
            result = [
                self.entity_class(
                    self._coerce_row(dict(row)),
                    is_new=False,
                    collection=self,
                )
                for row in rows
            ]
        return result

    async def list_ids_by_customer_and_type(
        self,
        customer_id: UUID,
        namespace_type: str,
    ) -> list[UUID]:
        """return every namespace id for ``(customer_id, namespace_type)``.

        used by audit-snapshot paths that need to enumerate the
        namespace set a ``type_customer`` scoped assignment covers.
        returning only ids (not full entities) keeps the call cheap.

        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :return: list of namespace UUIDs matching both filters
        :rtype: list[UUID]
        """
        result: list[UUID] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT namespace_id FROM namespaces WHERE row_scope = 'customer'   AND customer_id = $1 AND namespace_type = $2",
                customer_id,
                namespace_type,
            )
            result = [row["namespace_id"] for row in rows if row["namespace_id"] is not None]
        return result

    async def list_all_ids(self) -> list[UUID]:
        """return every namespace id in the table.

        used by audit-snapshot paths for ``scope='all'`` assignments.
        returning ids only mirrors :meth:`list_ids_by_customer_and_type`
        so the caller composes a single id set regardless of scope.

        :return: list of every namespace UUID
        :rtype: list[UUID]
        """
        result: list[UUID] = []
        if self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT namespace_id FROM namespaces WHERE row_scope IN ('platform', 'customer')",
            )
            result = [row["namespace_id"] for row in rows if row["namespace_id"] is not None]
        return result

    async def list_ids_under_name(self, node: str) -> list[UUID]:
        """return every namespace id at or beneath the name ``node``.

        the expansion of a :attr:`~threetears.agent.acl.types.ScopeType.SUBTREE`
        assignment into the concrete id set an audit snapshot needs,
        mirroring :meth:`list_all_ids` and
        :meth:`list_ids_by_customer_and_type` so the caller composes one
        id set regardless of scope shape.

        membership is decided in python by
        :func:`threetears.core.namespaces.namespace_contains` rather
        than by a SQL ``LIKE`` pattern. two reasons, and both matter: a
        second containment rule expressed in SQL is a second place the
        segment boundary can be got wrong, and a namespace name legally
        carries ``_`` (``sanitize_segment`` maps only ``.``), which is a
        ``LIKE`` wildcard -- so the pattern would need escaping that
        nothing else in this codebase performs.

        :param node: subtree root name; an empty node expands to
            nothing, never to everything
        :ptype node: str
        :return: list of namespace UUIDs at or beneath ``node``
        :rtype: list[UUID]
        """
        result: list[UUID] = []
        if node and self.l3_pool is not None:
            rows = await self.l3_pool.fetch(
                "SELECT namespace_id, name FROM namespaces WHERE row_scope IN ('platform', 'customer')",
            )
            result = [
                row["namespace_id"]
                for row in rows
                if row["namespace_id"] is not None and row["name"] is not None and namespace_contains(node, row["name"])
            ]
        return result

    async def list_tool_namespaces_for_actor(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        cache: Any,
    ) -> list[NamespaceEntity]:
        """list every ``tool``-type namespace this actor may see by default.

        agent-tools-eligibility shard 01 (TE-06). Returns the
        ACL-permitted subset of tool namespaces with
        ``tool_eligible=TRUE`` -- the set the consuming graph build
        step uses as the agent's default tool surface for a turn. The
        ``tool_eligible`` filter is applied INSIDE this method
        (database side) and the ACL filter is applied AFTER fetch (in
        Python via :func:`~threetears.agent.acl.evaluate_decision`)
        because the unified evaluator already has the in-process
        cache to answer per-namespace decisions cheaply.

        Skills cannot widen the default surface via this path; tools
        with ``tool_eligible=FALSE`` are excluded here even when the
        actor would otherwise have ACL grant. Skill-mode visibility
        flows through :meth:`list_skill_eligible_tool_namespaces`
        instead.

        :param actor_user_id: invoking user UUID
        :ptype actor_user_id: UUID
        :param actor_agent_id: invoking agent UUID
        :ptype actor_agent_id: UUID
        :param cache: shared :class:`AclCache` instance used by
            :func:`evaluate_decision` to resolve per-namespace
            ``tool.call`` decisions. Typed ``Any`` because importing
            :class:`AclCache` at module scope would create a circular
            import (the cache module imports the value types this
            module re-uses).
        :ptype cache: Any
        :return: list of namespace entities the actor can see in the
            default tool surface; empty list (never ``None``) when no
            tool is permitted
        :rtype: list[NamespaceEntity]
        """
        return await self._list_tool_namespaces_filtered(
            actor_user_id=actor_user_id,
            actor_agent_id=actor_agent_id,
            cache=cache,
            filter_column="tool_eligible",
        )

    async def list_skill_eligible_tool_namespaces(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        cache: Any,
    ) -> list[NamespaceEntity]:
        """list every ACL-permitted tool whose skills-catalog flag is set.

        agent-tools-eligibility shard 01 (TE-05). Returns the
        ACL-permitted subset of tool namespaces with
        ``skill_eligible=TRUE`` -- the set the
        ``3tears-agent-skills`` ``skill_list`` tool UNIONs with prose-
        skill rows so the LLM sees one unified catalog. The
        ``skill_eligible`` filter is applied INSIDE this method
        (database side) and the ACL filter is applied AFTER fetch via
        :func:`~threetears.agent.acl.evaluate_decision`.

        Excluding ACL-denied tools here keeps the catalog truthful:
        the LLM never sees a skill it cannot actually invoke. Skills
        compose visibility within the ACL-permitted set; they cannot
        bypass ACL.

        :param actor_user_id: invoking user UUID
        :ptype actor_user_id: UUID
        :param actor_agent_id: invoking agent UUID
        :ptype actor_agent_id: UUID
        :param cache: shared :class:`AclCache` instance
        :ptype cache: Any
        :return: list of namespace entities the actor may discover
            via the skills catalog; empty list when none qualify
        :rtype: list[NamespaceEntity]
        """
        return await self._list_tool_namespaces_filtered(
            actor_user_id=actor_user_id,
            actor_agent_id=actor_agent_id,
            cache=cache,
            filter_column="skill_eligible",
        )

    async def _list_tool_namespaces_filtered(
        self,
        *,
        actor_user_id: UUID,
        actor_agent_id: UUID,
        cache: Any,
        filter_column: str,
    ) -> list[NamespaceEntity]:
        """shared body for the tool-eligibility / skill-eligibility queries.

        the two public methods differ only in which boolean column
        they filter against; everything else (canonical query shape,
        ACL evaluation, list construction) is identical. one private
        helper keeps the two paths bit-identical so a future tweak
        (additional filter, ordering, caching) lands once.

        :param actor_user_id: invoking user UUID
        :ptype actor_user_id: UUID
        :param actor_agent_id: invoking agent UUID
        :ptype actor_agent_id: UUID
        :param cache: shared :class:`AclCache` instance
        :ptype cache: Any
        :param filter_column: boolean column name to filter against
            (``"tool_eligible"`` or ``"skill_eligible"``); whitelisted
            to those two values so the f-string interpolation is
            safe against SQL injection regardless of caller input
        :ptype filter_column: str
        :return: list of namespace entities matching the filter AND
            permitted by the unified ACL evaluator
        :rtype: list[NamespaceEntity]
        :raises ValueError: if ``filter_column`` is not one of the
            whitelisted column names
        """
        # whitelist the column name -- the canonical evaluator path
        # would also reject an injected predicate but explicit
        # validation here makes the intent obvious to a reader.
        if filter_column not in ("tool_eligible", "skill_eligible"):
            raise ValueError(
                f"_list_tool_namespaces_filtered: filter_column must "
                f"be 'tool_eligible' or 'skill_eligible'; got {filter_column!r}",
            )
        result: list[NamespaceEntity] = []
        if self.l3_pool is None:
            return result
        rows = await self.l3_pool.fetch(
            "SELECT * FROM namespaces "
            "WHERE row_scope IN ('platform', 'customer') "
            "AND namespace_type = 'tool' "
            f"AND {filter_column} = TRUE",
        )
        candidates: list[NamespaceEntity] = [
            self.entity_class(
                self._coerce_row(dict(row)),
                is_new=False,
                collection=self,
            )
            for row in rows
        ]
        # local imports keep this collection module free of a
        # circular dep on the evaluator stack: types -> cache ->
        # collections -> evaluator -> types would close the loop if
        # we imported at module scope. the per-call cost is one
        # attribute lookup, dwarfed by the fetch above.
        from threetears.agent.acl.evaluator import evaluate_decision  # noqa: PLC0415
        from threetears.agent.acl.types import (  # noqa: PLC0415
            EvaluationContext,
            Namespace as AclNamespace,
        )

        for entity in candidates:
            data = entity.to_dict()
            ns = AclNamespace(
                id=_coerce_uuid(data.get("namespace_id")),  # type: ignore[arg-type]
                customer_id=_coerce_uuid(data.get("customer_id")),
                namespace_type=str(data.get("namespace_type") or "tool"),
                owner_agent_id=_coerce_uuid(data.get("owner_agent_id")),
                name=data.get("name"),
                owner_namespace=data.get("owner_namespace"),
            )
            ctx = EvaluationContext(
                namespace=ns,
                action="tool.call",
                user_id=actor_user_id,
                agent_id=actor_agent_id,
            )
            permitted = await evaluate_decision(ctx, cache=cache)
            if permitted:
                result.append(entity)
        return result


class ImpersonationGateStatus(StrEnum):
    """``impersonation_gates.status`` -- build-plan.md Chunk 13 (identity-core
    repo), security-model.md's Impersonation paragraph: "the gate (per-tenant
    on/off + optional TTL, request/grant audit trail) lives in agent-acl".

    ``DISABLED -> REQUESTED`` (a customer admin's request) ``-> ENABLED`` (a
    platform-admin grant, optionally TTL'd). An ``ENABLED`` gate whose TTL has
    elapsed reads back as ``DISABLED`` on the next read
    (:meth:`ImpersonationGateCollection.get_effective_status`) without a
    separate write -- test-specifications.md's "Flow: Admin impersonation"
    Edge Case: "gate self-reverts after the TTL elapses".
    """

    DISABLED = "disabled"
    REQUESTED = "requested"
    ENABLED = "enabled"


_IMPERSONATION_GATE_STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in ImpersonationGateStatus)

_IMPERSONATION_GATE_COLUMNS = (
    "customer_id, status, requested_at, requested_by, granted_at, granted_by, "
    "ttl_seconds, expires_at, date_created, date_updated"
)


class ImpersonationGateCollection(SchemaBackedCollection[ImpersonationGateEntity]):
    """three-tier collection for ``impersonation_gates`` rows -- one row per
    tenant, the per-tenant admin act-as gate (security-model.md's
    Impersonation paragraph; entities.py's :class:`ImpersonationGateEntity`
    docstring explains the single-PK shape).

    Unlike :class:`GroupCollection`/:class:`RoleAssignmentCollection`, the
    domain methods below (:meth:`get_effective_status`, :meth:`request_enable`,
    :meth:`grant_enable`, :meth:`disable`) talk to :attr:`l3_pool` directly
    with parameterized SQL rather than routing through the generic
    :meth:`~threetears.core.collections.base.BaseCollection.save_entity`
    three-tier CAS path -- mirrors :meth:`GroupCollection.find_by_id`'s /
    :meth:`GroupCollection.get_by_name`'s existing style (direct
    ``l3_pool.fetchrow`` + :meth:`_coerce_row`) rather than inventing a new
    persistence pattern. A security gate this is deliberately read live on
    every check (no L1/L2 caching layer for gate reads) -- identity-core's
    OWN `TenantAuthPolicyService` (the sibling per-tenant policy read in the
    identity-core repo) makes the identical choice for the same reason: a
    stale cached read of a security gate is a worse failure mode than one
    extra round trip per check.
    """

    primary_key_column: str = "customer_id"
    schema = TableSchema(
        name="impersonation_gates",
        primary_key="customer_id",
        columns=[
            Column("customer_id", UUID_TYPE),
            Column(
                "status",
                ENUM_TYPE,
                enum_type=_IMPERSONATION_GATE_STATUS_VALUES,
                enum_name="impersonation_gate_status",
                server_default=f"'{ImpersonationGateStatus.DISABLED.value}'",
            ),
            Column("requested_at", DATETIMETZ_TYPE, nullable=True),
            Column("requested_by", UUID_TYPE, nullable=True),
            Column("granted_at", DATETIMETZ_TYPE, nullable=True),
            Column("granted_by", UUID_TYPE, nullable=True),
            # NULL ttl_seconds/expires_at = an enabled gate with no TTL (stays
            # enabled until explicitly disabled) -- security-model.md: "per-
            # tenant on/off + OPTIONAL TTL".
            Column("ttl_seconds", INT_TYPE, nullable=True),
            Column("expires_at", DATETIMETZ_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True, server_default="now()"),
            Column("date_updated", DATETIMETZ_TYPE, server_default="now()"),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: ``"impersonation_gates"``
        :rtype: str
        """
        return "impersonation_gates"

    @property
    def entity_class(self) -> type[ImpersonationGateEntity]:
        """return entity class for this collection.

        :return: :class:`ImpersonationGateEntity`
        :rtype: type[ImpersonationGateEntity]
        """
        return ImpersonationGateEntity

    async def get_effective_status(self, customer_id: UUID, *, now: datetime | None = None) -> ImpersonationGateStatus:
        """Read the gate's current status, applying TTL self-revert.

        No row on file reads as ``DISABLED`` (a tenant that has never
        requested impersonation is gated off by default -- same "absent row
        = the safe default" convention `identity_core/auth/policy.py`'s
        `TenantAuthPolicyService.get` uses for `mfa_enforcement`). An
        ``ENABLED`` row whose ``expires_at`` has passed reads back as
        ``DISABLED`` without a separate write -- the write-back (so a
        subsequent admin listing shows ``disabled`` rather than a stale
        ``enabled`` row) is Hub-repo sweep-job work, out of this package's
        scope; the read-time guarantee alone is what test-specifications.md's
        Edge Case requires and is what every caller of this method actually
        observes.

        :param customer_id: tenant to check
        :ptype customer_id: UUID
        :param now: injectable clock for deterministic TTL-expiry tests;
            defaults to the real current time
        :ptype now: datetime | None
        :return: the effective status
        :rtype: ImpersonationGateStatus
        """
        if self.l3_pool is None:
            return ImpersonationGateStatus.DISABLED
        row = await self.l3_pool.fetchrow(
            f"SELECT {_IMPERSONATION_GATE_COLUMNS} FROM impersonation_gates WHERE customer_id = $1",
            customer_id,
        )
        if row is None:
            return ImpersonationGateStatus.DISABLED
        data = self._coerce_row(dict(row))
        status = ImpersonationGateStatus(data["status"])
        if status is ImpersonationGateStatus.ENABLED and data.get("expires_at") is not None:
            current = now if now is not None else datetime.now(UTC)
            if current >= data["expires_at"]:
                return ImpersonationGateStatus.DISABLED
        return status

    async def request_enable(
        self, customer_id: UUID, *, requested_by: UUID, now: datetime | None = None
    ) -> ImpersonationGateEntity:
        """Transition ``disabled -> requested`` -- a customer admin's request
        for impersonation to be enabled on their tenant. Idempotent re-request
        (already ``requested`` or ``enabled``) simply re-stamps
        ``requested_at``/``requested_by`` rather than rejecting -- there is no
        "double request" error state described anywhere in the artifacts.

        :param customer_id: requesting tenant
        :ptype customer_id: UUID
        :param requested_by: the customer admin principal making the request
        :ptype requested_by: UUID
        :param now: injectable clock for deterministic tests
        :ptype now: datetime | None
        :return: the persisted gate row
        :rtype: ImpersonationGateEntity
        """
        current = now if now is not None else datetime.now(UTC)
        assert self.l3_pool is not None, "ImpersonationGateCollection requires an l3_pool"
        row = await self.l3_pool.fetchrow(
            f"""
            INSERT INTO impersonation_gates (customer_id, status, requested_at, requested_by)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (customer_id) DO UPDATE
                SET status = EXCLUDED.status,
                    requested_at = EXCLUDED.requested_at,
                    requested_by = EXCLUDED.requested_by,
                    date_updated = now()
            RETURNING {_IMPERSONATION_GATE_COLUMNS}
            """,
            customer_id,
            ImpersonationGateStatus.REQUESTED.value,
            current,
            requested_by,
        )
        if row is None:
            raise RuntimeError("impersonation_gates upsert (request_enable) returned no row")
        data = self._coerce_row(dict(row))
        return self.entity_class(data, is_new=False, collection=self)

    async def grant_enable(
        self,
        customer_id: UUID,
        *,
        granted_by: UUID,
        ttl_seconds: int | None,
        now: datetime | None = None,
    ) -> ImpersonationGateEntity:
        """Transition ``requested -> enabled`` -- a platform admin's grant,
        optionally TTL'd. Callable even absent a prior ``request_enable`` row
        (a platform admin can pre-emptively enable a tenant without waiting
        on a customer request) -- upserts rather than requiring a pre-existing
        ``requested`` row, since no artifact describes "grant without a
        pending request" as an error case.

        :param customer_id: tenant to enable
        :ptype customer_id: UUID
        :param granted_by: the platform admin principal granting it
        :ptype granted_by: UUID
        :param ttl_seconds: gate auto-reverts to ``disabled`` this many
            seconds after ``granted_at``; ``None`` = no auto-revert
        :ptype ttl_seconds: int | None
        :param now: injectable clock for deterministic tests
        :ptype now: datetime | None
        :return: the persisted gate row
        :rtype: ImpersonationGateEntity
        """
        current = now if now is not None else datetime.now(UTC)
        expires_at = current + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        assert self.l3_pool is not None, "ImpersonationGateCollection requires an l3_pool"
        row = await self.l3_pool.fetchrow(
            f"""
            INSERT INTO impersonation_gates (customer_id, status, granted_at, granted_by, ttl_seconds, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (customer_id) DO UPDATE
                SET status = EXCLUDED.status,
                    granted_at = EXCLUDED.granted_at,
                    granted_by = EXCLUDED.granted_by,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    expires_at = EXCLUDED.expires_at,
                    date_updated = now()
            RETURNING {_IMPERSONATION_GATE_COLUMNS}
            """,
            customer_id,
            ImpersonationGateStatus.ENABLED.value,
            current,
            granted_by,
            ttl_seconds,
            expires_at,
        )
        if row is None:
            raise RuntimeError("impersonation_gates upsert (grant_enable) returned no row")
        data = self._coerce_row(dict(row))
        return self.entity_class(data, is_new=False, collection=self)

    async def disable(self, customer_id: UUID, *, now: datetime | None = None) -> ImpersonationGateEntity | None:
        """Explicitly revoke -- ``* -> disabled``, from any prior status.

        The mid-session revocation path test-specifications.md's Error Case
        needs ("gate revoked mid-session stops the next refresh") calls this;
        the caller (identity-core's `impersonation/session.py`) re-checks
        :meth:`get_effective_status` on every refresh, so a revoke here is
        visible on the very next refresh attempt, not just the next TTL
        sweep.

        :param customer_id: tenant to disable
        :ptype customer_id: UUID
        :param now: injectable clock for deterministic tests
        :ptype now: datetime | None
        :return: the updated row, or ``None`` if no gate row existed yet (a
            no-op disable on a tenant that never had a gate row is not an
            error -- the effective status was already ``disabled``)
        :rtype: ImpersonationGateEntity | None
        """
        current = now if now is not None else datetime.now(UTC)
        assert self.l3_pool is not None, "ImpersonationGateCollection requires an l3_pool"
        row = await self.l3_pool.fetchrow(
            f"""
            UPDATE impersonation_gates
               SET status = $2, date_updated = $3
             WHERE customer_id = $1
            RETURNING {_IMPERSONATION_GATE_COLUMNS}
            """,
            customer_id,
            ImpersonationGateStatus.DISABLED.value,
            current,
        )
        if row is None:
            return None
        data = self._coerce_row(dict(row))
        return self.entity_class(data, is_new=False, collection=self)
