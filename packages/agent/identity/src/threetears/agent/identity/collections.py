"""Identity-versions collection -- three-tier CRUD for versioned identity blocks.

:class:`IdentityVersionsCollection` is the single entry point for
``identity_versions``-table SQL. CRUD is generated from :attr:`schema`
and goes through :meth:`get` / :meth:`save_entity` / :meth:`delete` so the
L1 / L2 / L3 tiers stay coherent; ``date_updated`` is the CAS fence.

A version is an immutable snapshot in a **linear** parent-pointer chain
(``parent_version_id``); exactly one ``active`` version exists per
``(agent_id, customer_id, user_id, block_key)`` (a partial unique index
enforces it). User isolation is the ``user_id`` WHERE clause, NOT RBAC:
every metallm user shares one ``agent_id`` (the partition), so ``user_id``
is the sole boundary -- the reads below take it as a required parameter.

This chunk (T2.1a) ships the schema + table factory + the read paths
(:meth:`resolve_active`, :meth:`find_versions_for_block`,
:meth:`find_pending`). The lifecycle mutation ops (propose / consent /
reject / rollback) + events + owner RBAC land in T2.1b.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import MetaData, Table

from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index as SchemaIndex,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.observe import get_logger

from threetears.agent.identity.entities import IdentityVersionEntity
from threetears.agent.identity.types import (
    IDENTITY_BLOCK_KEY_VALUES,
    IDENTITY_STATUS_VALUES,
    IdentityVersionStatus,
)

__all__ = [
    "IdentityVersionsCollection",
    "identity_versions_table",
]

log = get_logger(__name__)


# Explicit column list for multi-row reads (single entry point + parity
# with the migration DDL). No embedding column -- identity blocks are not
# semantically searched.
_IDENTITY_SELECT_COLUMNS = (
    "version_id, agent_id, customer_id, user_id, block_key, content, "
    "rationale, content_hash, parent_version_id, status, proposer_agent_id, "
    "consenter_user_id, date_created, date_updated"
)


def identity_versions_table(metadata: MetaData) -> Table:
    """Register the ``identity_versions`` table on the given SA metadata.

    Thin idempotency wrapper around
    :meth:`IdentityVersionsCollection.schema.to_sqlalchemy_table`. Call
    before ``SQLiteBackend.initialize(metadata)`` so the L1 cache builds
    with the full schema, and before Alembic ``target_metadata``
    reflection so auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``identity_versions`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, IdentityVersionsCollection.schema.to_sqlalchemy_table(metadata))


class IdentityVersionsCollection(SchemaBackedCollection[IdentityVersionEntity]):
    """Collection for versioned identity-block entities with three-tier caching.

    CRUD is generated from :attr:`schema`: the version-snapshot columns
    (``content`` / ``rationale`` / ``content_hash`` / ``parent_version_id``
    / ``block_key`` / ``proposer_agent_id``) + the scope columns are marked
    immutable so the ``DO UPDATE SET`` clause narrows to the mutable
    lifecycle fields (``status`` / ``consenter_user_id``). ``date_updated``
    is the CAS fence.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "version_id")
    schema = TableSchema(
        name="identity_versions",
        primary_key=("agent_id", "version_id"),
        columns=[
            Column("version_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            # customer_id / user_id are nullable scope grains (like memory /
            # intention). metallm sets both + filters reads on user_id; a
            # null here is an agent-internal / global block. Both immutable
            # (write-once): a version never moves across scopes.
            Column("customer_id", UUID_TYPE, immutable=True, nullable=True),
            Column("user_id", UUID_TYPE, immutable=True, nullable=True),
            # which identity block this version belongs to; immutable (a new
            # block content is a new version, and the block never changes).
            Column(
                "block_key",
                ENUM_TYPE,
                enum_type=IDENTITY_BLOCK_KEY_VALUES,
                enum_name="identity_block_key",
                immutable=True,
            ),
            # the immutable snapshot fields: a version's content + audit are
            # fixed at creation; a change produces a NEW version.
            Column("content", STRING_TYPE, immutable=True),
            Column("rationale", STRING_TYPE, nullable=True, immutable=True),
            Column("content_hash", STRING_TYPE, immutable=True),
            # linear lineage: the version this supersedes (null = root).
            Column("parent_version_id", UUID_TYPE, nullable=True, immutable=True),
            # the mutable lifecycle: proposed -> active -> superseded /
            # rejected. Default 'proposed' on insert; seeds/imports pass
            # 'active' explicitly.
            Column(
                "status",
                ENUM_TYPE,
                enum_type=IDENTITY_STATUS_VALUES,
                enum_name="identity_version_status",
                server_default=f"'{IdentityVersionStatus.PROPOSED.value}'",
            ),
            # who proposed (the agent); immutable. Null = user-authored / seed.
            Column("proposer_agent_id", UUID_TYPE, nullable=True, immutable=True),
            # who consented (the user); set at apply -- mutable (once).
            Column("consenter_user_id", UUID_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        indexes=(
            # exactly one active version per (agent, customer, user, block).
            # Partial UNIQUE on status='active' -- the linear-chain single-
            # winner invariant. (Column set + WHERE here; the raw v001 DDL
            # carries the CREATE UNIQUE INDEX form for parity + L1 parity.)
            SchemaIndex(
                "uq_identity_active_per_block",
                "agent_id",
                "customer_id",
                "user_id",
                "block_key",
                where=f"status = '{IdentityVersionStatus.ACTIVE.value}'",
                unique=True,
            ),
            # a block's version history, most-recent-first (DESC lives in DDL).
            SchemaIndex(
                "idx_identity_block_history",
                "agent_id",
                "user_id",
                "block_key",
                "date_created",
            ),
            # the pending consent / veto queue: proposed versions per user.
            SchemaIndex(
                "idx_identity_pending",
                "agent_id",
                "user_id",
                where=f"status = '{IdentityVersionStatus.PROPOSED.value}'",
            ),
        ),
    )

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "identity_versions"

    @property
    def entity_class(self) -> type[IdentityVersionEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[IdentityVersionEntity]
        """
        return IdentityVersionEntity

    def _row_to_entity(self, row: Any) -> IdentityVersionEntity:
        """Build a save-able entity from an L3 row (warms L1 via ``collection=self``)."""
        data = dict(row)
        entity = self.entity_class(data, is_new=False, collection=self)
        entity.original_date_updated = data.get("date_updated")
        return entity

    async def resolve_active(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID | None,
        user_id: UUID | None,
        block_key: str,
    ) -> IdentityVersionEntity | None:
        """Return the one ``active`` version for a scope+block, or ``None``.

        This is the consumer's prompt-assembly read: the content the next
        turn injects for ``block_key``. The partial unique index guarantees
        at most one active row. Null-safe on the scope grains.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param customer_id: scope grain (nullable)
        :ptype customer_id: UUID | None
        :param user_id: owning user (isolation boundary; nullable grain)
        :ptype user_id: UUID | None
        :param block_key: the identity block
        :ptype block_key: str
        :return: the active version entity, or ``None``
        :rtype: IdentityVersionEntity | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: the scan by (agent, customer, user, block, status)
        # is not primary-key addressable, so it stays on the Collection
        # (single entry point). IS NOT DISTINCT FROM is null-safe on the
        # nullable scope grains.
        row = await self.l3_pool.fetchrow(
            f"SELECT {_IDENTITY_SELECT_COLUMNS} FROM identity_versions "
            "WHERE agent_id = $1 "
            "AND customer_id IS NOT DISTINCT FROM $2 "
            "AND user_id IS NOT DISTINCT FROM $3 "
            f"AND block_key = $4 AND status = '{IdentityVersionStatus.ACTIVE.value}'",
            agent_id,
            customer_id,
            user_id,
            block_key,
        )
        return self._row_to_entity(row) if row is not None else None

    async def find_versions_for_block(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID | None,
        user_id: UUID | None,
        block_key: str,
    ) -> list[IdentityVersionEntity]:
        """Return a block's full version history, most-recent-first.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param customer_id: scope grain (nullable)
        :ptype customer_id: UUID | None
        :param user_id: owning user (isolation boundary)
        :ptype user_id: UUID | None
        :param block_key: the identity block
        :ptype block_key: str
        :return: version entities, newest first
        :rtype: list[IdentityVersionEntity]
        """
        if self.l3_pool is None:
            return []
        rows = await self.l3_pool.fetch(
            f"SELECT {_IDENTITY_SELECT_COLUMNS} FROM identity_versions "
            "WHERE agent_id = $1 "
            "AND customer_id IS NOT DISTINCT FROM $2 "
            "AND user_id IS NOT DISTINCT FROM $3 "
            "AND block_key = $4 "
            "ORDER BY date_created DESC",
            agent_id,
            customer_id,
            user_id,
            block_key,
        )
        return [self._row_to_entity(row) for row in rows]

    async def find_pending(
        self,
        *,
        agent_id: UUID,
        user_id: UUID | None,
    ) -> list[IdentityVersionEntity]:
        """Return the user's ``proposed`` versions -- the consent / veto queue.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param user_id: owning user (isolation boundary)
        :ptype user_id: UUID | None
        :return: proposed version entities, oldest first (queue order)
        :rtype: list[IdentityVersionEntity]
        """
        if self.l3_pool is None:
            return []
        rows = await self.l3_pool.fetch(
            f"SELECT {_IDENTITY_SELECT_COLUMNS} FROM identity_versions "
            "WHERE agent_id = $1 AND user_id IS NOT DISTINCT FROM $2 "
            f"AND status = '{IdentityVersionStatus.PROPOSED.value}' "
            "ORDER BY date_created ASC",
            agent_id,
            user_id,
        )
        return [self._row_to_entity(row) for row in rows]
