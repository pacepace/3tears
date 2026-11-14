"""``mcp_tool_grants`` collection -- per-tool RBAC grants for MCP servers.

shape one row per ``(principal, tool, permission)`` triple. principals
are users / groups / roles; permissions are platform-format strings
(``"myapp.conversations.read"``, ``"myapp.audit.read"``, etc.). default-
deny: a tool's :attr:`McpTool.required_permission` runs only when an
active grant matches the calling identity.

mutation discipline:

- :meth:`add_grant` writes the row, then the **caller** is responsible
  for bumping :func:`Subjects.mcp_rbac_epoch` (this collection does not
  hold an :class:`EpochClient` reference -- the bump call site is the
  REST endpoint that mutated the row, so the bump lives there for
  symmetry with task-02 Chunks B and C).
- :meth:`remove_grant` same shape.
- the table is switched on (``write_generation=WRITE_GENERATION``): every
  committed write advances its generation and records the grant it changed.
  :class:`~threetears.mcp.auth.LocalGrantAuthorizer` is a cache derived from
  the table and follows it, so each changed grant -- heard as a row broadcast,
  or found by the generation walk when the broadcast was lost -- is read by
  key (:meth:`McpToolGrantCollection.read_grant`).

the whole table is read only to fill an empty cache at start, or after a
write that changed the whole table (:meth:`McpToolGrantCollection.load_all_grants`).
the ``mcp.rbac`` epoch bump the callers still publish is no longer what keeps
the authorizer current.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import MetaData, Table
from threetears.core.collections.generation import WRITE_GENERATION
from threetears.core.declared import whole_read
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import read_row
from threetears.core.tenancy import no_tenant
from threetears.observe import get_logger
from uuid_utils import uuid7

__all__ = [
    "McpToolGrantCollection",
    "McpToolGrantEntity",
    "mcp_tool_grants_table",
]

log = get_logger(__name__)


def mcp_tool_grants_table(metadata: MetaData) -> Table:
    """register the ``mcp_tool_grants`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`McpToolGrantCollection.schema.to_sqlalchemy_table`. Call
    this before ``SQLiteBackend.initialize(metadata)`` so the L1 cache
    gets the correct schema.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``mcp_tool_grants`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, McpToolGrantCollection.schema.to_sqlalchemy_table(metadata))


class McpToolGrantEntity(BaseEntity):
    """Entity representing one MCP tool grant row.

    every grant is one ``(principal_type, principal_id, tool_name,
    permission)`` quadruple. ``grant_id`` is the surrogate PK and the
    addressable identity for revocation.

    field access flows through :class:`BaseEntity.__getattr__`.
    """

    primary_key_field: str = "grant_id"


class McpToolGrantCollection(SchemaBackedCollection[McpToolGrantEntity]):
    """three-tier collection for ``mcp_tool_grants``.

    CRUD comes from :class:`SchemaBackedCollection`; the domain
    methods are :meth:`add_grant`, :meth:`remove_grant`, :meth:`read_grant`
    (the keyed read the authorizer applies a bump with) and
    :meth:`load_all_grants` (its cold-start fill).

    instances do NOT publish epoch bumps themselves -- callers (REST
    admin endpoints) own the bump-after-commit pattern so the row
    write and the broadcast happen in the same scope. see task-02
    Chunks B + C for the same discipline.
    """

    tenancy = no_tenant("tool grants are keyed by principal, platform-wide, with no customer column")

    primary_key_column: str | tuple[str, ...] = "grant_id"
    # v0.8.0 enrichment: ``date_created`` carries ``server_default="now()"``
    # to match prod and the v001 migration's ``TIMESTAMPTZ NOT NULL
    # DEFAULT now()`` declaration. The two lookup indexes mirror v001
    # so the parity gate stays clean against the registered
    # ``mcp_tool_grants_table``.
    schema = TableSchema(
        name="mcp_tool_grants",
        primary_key=("grant_id",),
        columns=[
            Column("grant_id", UUID_TYPE),
            Column("principal_type", STRING_TYPE, immutable=True),
            Column("principal_id", UUID_TYPE, immutable=True),
            Column("tool_name", STRING_TYPE, immutable=True),
            Column("permission", STRING_TYPE, immutable=True),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
        ],
        # every committed grant write advances the table's generation and records the grant it changed,
        # so LocalGrantAuthorizer follows the table and reads each changed grant by key -- a lost
        # broadcast included -- never every grant again
        write_generation=WRITE_GENERATION,
    )

    @property
    def table_name(self) -> str:
        """return the collection's table name.

        :return: ``"mcp_tool_grants"``
        :rtype: str
        """
        return "mcp_tool_grants"

    @property
    def entity_class(self) -> type[McpToolGrantEntity]:
        """return the entity class for this collection.

        :return: :class:`McpToolGrantEntity`
        :rtype: type[McpToolGrantEntity]
        """
        return McpToolGrantEntity

    async def add_grant(
        self,
        *,
        principal_type: str,
        principal_id: UUID,
        tool_name: str,
        permission: str,
    ) -> McpToolGrantEntity:
        """insert a new grant row and return the created entity.

        the caller is responsible for bumping the rbac epoch after
        the row commits (see module docstring for rationale).

        :param principal_type: ``"user"`` / ``"group"`` / ``"role"``
        :ptype principal_type: str
        :param principal_id: principal UUID
        :ptype principal_id: UUID
        :param tool_name: target tool name (matches :attr:`McpTool.name`)
        :ptype tool_name: str
        :param permission: permission string the grant authorizes
        :ptype permission: str
        :return: created grant entity
        :rtype: McpToolGrantEntity
        """
        # uuid_utils.uuid7() returns a uuid_utils.UUID; cast to stdlib
        # UUID so downstream callers / type checks see the canonical
        # type (rest of 3tears typing is built on stdlib UUID).
        grant_id = UUID(str(uuid7()))
        entity = self.entity_class(
            {
                "grant_id": grant_id,
                "principal_type": principal_type,
                "principal_id": principal_id,
                "tool_name": tool_name,
                "permission": permission,
                "date_created": datetime.now(UTC),
            },
            is_new=True,
            collection=self,
        )
        await self.save_entity(entity)
        return entity

    async def remove_grant(self, grant_id: UUID) -> bool:
        """delete a grant by id.

        the caller is responsible for bumping the rbac epoch after
        the row commits. uses the public :meth:`BaseCollection.get`
        + :meth:`BaseCollection.delete` extension seams (the
        ``find_by_id`` / ``delete_entity`` shorthand the original
        prototype reached for does not exist on the canonical
        BaseCollection surface; this method shapes around the real
        contract).

        :param grant_id: target grant UUID
        :ptype grant_id: UUID
        :return: True when the row existed and was removed; False when
            no row matched
        :rtype: bool
        """
        existing = await self.get(grant_id)
        if existing is None:
            return False
        return await self.delete(grant_id)

    async def read_grant(self, grant_id: UUID) -> dict[str, Any] | None:
        """one grant row by its key, straight from L3: what an rbac bump naming it changed.

        :param grant_id: the grant the bump named
        :ptype grant_id: UUID
        :return: the row with keys ``(grant_id, principal_type, principal_id, tool_name, permission,
            date_created)``, or ``None`` when no grant has that key (it was revoked)
        :rtype: dict[str, Any] | None
        :raises RuntimeError: when the L3 pool is not configured on this collection's registry
        """
        if self.l3_pool is None:
            raise RuntimeError(
                "McpToolGrantCollection L3 pool is not configured; "
                "wire CollectionRegistry.configure(l3_pool=...) before "
                "calling read_grant",
            )
        row = await read_row(
            self.l3_pool,
            "mcp_tool_grants",
            by={"grant_id": grant_id},
            columns=("grant_id", "principal_type", "principal_id", "tool_name", "permission", "date_created"),
            records=True,
        )
        return None if row is None else dict(row)

    async def list_grants_page(self, *, after: UUID | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """one page of the grant rows by ``grant_id``, after ``after``: an operator's listing, never the whole table.

        :param after: the ``grant_id`` the page before ended at; ``None`` for the first page
        :ptype after: UUID | None
        :param limit: the most rows the page holds
        :ptype limit: int
        :return: dict rows with keys ``(grant_id, principal_type, principal_id, tool_name, permission,
            date_created)``, by ``grant_id``
        :rtype: list[dict[str, Any]]
        :raises RuntimeError: when the L3 pool is not configured on this collection's registry
        """
        if self.l3_pool is None:
            raise RuntimeError(
                "McpToolGrantCollection L3 pool is not configured; "
                "wire CollectionRegistry.configure(l3_pool=...) before "
                "calling list_grants_page",
            )
        rows = await self.l3_pool.fetch(
            "SELECT grant_id, principal_type, principal_id, tool_name, permission, date_created FROM mcp_tool_grants "
            "WHERE grant_id > $1 ORDER BY grant_id LIMIT $2",
            after or UUID(int=0),
            limit,
        )
        return [dict(row) for row in rows]

    async def load_all_grants(self) -> list[dict[str, Any]]:
        """return every grant row: the cold-start fill of an empty grant cache.

        the framework's :class:`~threetears.mcp.auth.LocalGrantAuthorizer`
        calls this only to fill its empty cache at start, and after a write
        that changed the whole table (a truncate, a load: the operator's
        reset). a changed grant never reaches it: the follow names it and
        :meth:`read_grant` reads that one. rows go
        straight from L3 (no caching here -- the authorizer owns the
        in-memory cache).

        uses the public :attr:`l3_pool` extension seam (matches every
        other peer collection) and fails closed when the registry
        has not been configured.

        :return: list of dict rows with keys
            ``(grant_id, principal_type, principal_id, tool_name,
            permission, date_created)``
        :rtype: list[dict[str, Any]]
        :raises RuntimeError: when the L3 pool is not configured
            on this collection's registry
        """
        if self.l3_pool is None:
            raise RuntimeError(
                "McpToolGrantCollection L3 pool is not configured; "
                "wire CollectionRegistry.configure(l3_pool=...) before "
                "calling load_all_grants",
            )
        rows = await whole_read(
            self.l3_pool.fetch(
                "SELECT grant_id, principal_type, principal_id, tool_name, permission, date_created FROM mcp_tool_grants",
            ),
            kind="cold_start",
            reason="LocalGrantAuthorizer fills its empty grant cache once at start, and again only when a write changed "
            "the whole table (a truncate or a load, the operator's reset); a changed grant is read by key",
            ruling="Pace 2026-11-12: cold start fills read whole; coordinator 2026-11-12: a whole-table change is one",
            proof="packages/mcp/tests/integration/test_multi_pod_rbac.py::"
            "test_load_all_grants_is_reached_only_at_the_cold_start_and_a_whole_table_change",
            speed="packages/mcp/tests/integration/test_multi_pod_rbac.py::test_load_all_grants_speed_at_real_size",
        )
        return [dict(row) for row in rows]
