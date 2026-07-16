"""Agent-skills collections -- three-tier CRUD for skills + invocations.

The package subclasses :class:`BaseCollection` directly rather than
:class:`SchemaBackedCollection` because three of the ``agent_skills``
columns (``tool_additions``, ``tool_restrictions``, ``tags``) are
Postgres ``TEXT[]`` arrays, and the framework's schema-driven CRUD
generator has no built-in ``TEXT[]`` type tag. Adding one is a
framework change; this shard scope is the skills package only. Hand-
rolled SQL keeps the contract local and lets asyncpg's native
``list[str] <-> TEXT[]`` codec round-trip cleanly without any custom
normalisation.

Both Collections declare ``partition_column = "agent_id"`` as a class
attribute so consumers (and the workspace partition-column
enforcement walker) can confirm the partition contract by
introspection.

Method contracts mirror ``docs/agent-skills/shard-01-schema-and-
collection.md`` "Public API" section verbatim. Hybrid queries
(``list_for_user`` with FTS ranking) carry ``# cache-bypass:``
annotations because they are not primary-key addressable and would
not benefit from L1 row caching -- the row cache still serves
``get(agent_id, skill_id)`` calls uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from threetears.agent.skills.entities import (
    AgentSkillEntity,
    AgentSkillInvocationEntity,
)
from threetears.agent.skills.types import (
    OutcomeSource,
    SkillOutcome,
)
from threetears.core.backends.protocol import parse_rowcount
from threetears.core.collections.base import BaseCollection
from threetears.core.serialization import (
    deserialize_from_json,
    serialize_to_json,
)
from threetears.observe import get_logger

__all__ = [
    "AgentSkillCollection",
    "AgentSkillInvocationCollection",
]


log = get_logger(__name__)


# Field-type hints used when L2 cache rounds a row through JSON. The
# helpers in ``threetears.core.serialization.deserialize_from_json``
# dispatch on these to rehydrate UUID / datetime / list values back to
# their native Python types after a NATS KV pull.
_SKILL_FIELD_TYPES: dict[str, Any] = {
    "skill_id": UUID,
    "agent_id": UUID,
    "user_id": UUID,
    "name": str,
    "summary": str,
    "body": str | None,
    "prompt_mode": str,
    "tool_additions": list[str],
    "tool_restrictions": list[str],
    "trigger_keywords": str,
    "tags": list[str],
    "source": str,
    "enabled": bool,
    "use_count": int,
    "last_used_at": datetime | None,
    "success_count": int,
    "failure_count": int,
    "last_failure_at": datetime | None,
    "date_created": datetime,
    "date_updated": datetime,
}


_INVOCATION_FIELD_TYPES: dict[str, Any] = {
    "invocation_id": UUID,
    "agent_id": UUID,
    "skill_id": UUID,
    "user_id": UUID,
    "conversation_id": UUID,
    "message_id": UUID | None,
    "invocation_source": str,
    "invoked_at": datetime,
    "outcome": str | None,
    "outcome_source": str | None,
    "notes": str | None,
}


# columns the Collection emits on INSERT. ``search_vector`` is
# trigger-maintained server-side -- emitting it from the application
# would race with the trigger and clobber the weighted ranking.
# ``date_created`` / ``date_updated`` carry server defaults
# (``now()``); omitting them on INSERT lets Postgres stamp them, but
# the framework's ``save_entity`` path writes them explicitly so we
# include them too so the value round-trips cleanly through L1 / L2.
_SKILL_INSERT_COLUMNS: tuple[str, ...] = (
    "agent_id",
    "skill_id",
    "user_id",
    "name",
    "summary",
    "body",
    "prompt_mode",
    "tool_additions",
    "tool_restrictions",
    "trigger_keywords",
    "tags",
    "source",
    "enabled",
    "use_count",
    "last_used_at",
    "success_count",
    "failure_count",
    "last_failure_at",
    "date_created",
    "date_updated",
)


# columns that get updated on ON CONFLICT. partition column + pk are
# excluded (the row is uniquely identified by them); ``date_created``
# is immutable; ``search_vector`` is trigger-maintained.
_SKILL_UPDATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in _SKILL_INSERT_COLUMNS if c not in {"agent_id", "skill_id", "date_created"}
)


_INVOCATION_INSERT_COLUMNS: tuple[str, ...] = (
    "agent_id",
    "invocation_id",
    "skill_id",
    "user_id",
    "conversation_id",
    "message_id",
    "invocation_source",
    "invoked_at",
    "outcome",
    "outcome_source",
    "notes",
)


# invocations are append-mostly with a small set of mutable fields
# (``message_id`` populated post-LLM, ``outcome`` populated by the
# marker parser). everything else is immutable once written.
_INVOCATION_UPDATE_COLUMNS: tuple[str, ...] = (
    "message_id",
    "outcome",
    "outcome_source",
    "notes",
)


def _build_upsert_sql(
    table: str,
    insert_cols: Sequence[str],
    update_cols: Sequence[str],
    pk_cols: Sequence[str],
    *,
    cas_column: str | None = None,
) -> str:
    """Build a parameterised INSERT ... ON CONFLICT DO UPDATE statement.

    Positional parameters bind to ``insert_cols`` in declared order.
    When ``cas_column`` is set the ON CONFLICT DO UPDATE carries an
    optimistic-lock fence ``WHERE {table}.{cas_column} = $N`` (with
    ``N = len(insert_cols) + 1``); the fence value binds as the single
    trailing positional parameter after the insert columns. A conflict
    whose stored ``cas_column`` no longer matches the fence updates zero
    rows, so the framework's :meth:`BaseCollection.save_entity` path
    raises :class:`ConcurrentModificationError` instead of clobbering a
    concurrent writer (the skills lost-update guard).

    :param table: table name
    :ptype table: str
    :param insert_cols: columns to write on INSERT (in positional-param
        order)
    :ptype insert_cols: Sequence[str]
    :param update_cols: columns to update on conflict
    :ptype update_cols: Sequence[str]
    :param pk_cols: conflict target columns
    :ptype pk_cols: Sequence[str]
    :param cas_column: optional optimistic-lock fence column; ``None``
        emits an unconditional DO UPDATE (insert / no-fence path)
    :ptype cas_column: str | None
    :return: SQL string ready for ``execute()``
    :rtype: str
    """
    placeholders = ", ".join(f"${i + 1}" for i in range(len(insert_cols)))
    column_list = ", ".join(insert_cols)
    pk_list = ", ".join(pk_cols)
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    sql = (
        f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_list}) DO UPDATE SET {set_clause}"
    )
    if cas_column is not None:
        sql = f"{sql} WHERE {table}.{cas_column} = ${len(insert_cols) + 1}"
    return sql


_AGENT_SKILLS_UPSERT_SQL = _build_upsert_sql(
    "agent_skills",
    _SKILL_INSERT_COLUMNS,
    _SKILL_UPDATE_COLUMNS,
    ("agent_id", "skill_id"),
)


# CAS-fenced variant of the skills upsert. Selected by
# ``save_to_store`` when the caller supplies an ``original_timestamp``
# (an edit of an already-persisted skill); binds the fence value as the
# single trailing parameter after the insert columns. Guards the
# ``skill_update`` lost-update: an edit computed against a stale read
# updates zero rows once a concurrent edit / counter bump has moved
# ``date_updated``, surfacing a ``ConcurrentModificationError`` rather
# than silently overwriting the concurrent write.
_AGENT_SKILLS_UPSERT_CAS_SQL = _build_upsert_sql(
    "agent_skills",
    _SKILL_INSERT_COLUMNS,
    _SKILL_UPDATE_COLUMNS,
    ("agent_id", "skill_id"),
    cas_column="date_updated",
)


_AGENT_SKILL_INVOCATIONS_UPSERT_SQL = _build_upsert_sql(
    "agent_skill_invocations",
    _INVOCATION_INSERT_COLUMNS,
    _INVOCATION_UPDATE_COLUMNS,
    ("agent_id", "invocation_id"),
)


_AGENT_SKILLS_FETCH_SQL = (
    "SELECT agent_id, skill_id, user_id, name, summary, body, prompt_mode, "
    "tool_additions, tool_restrictions, trigger_keywords, tags, source, "
    "enabled, use_count, last_used_at, success_count, failure_count, "
    "last_failure_at, date_created, date_updated "
    "FROM agent_skills WHERE agent_id = $1 AND skill_id = $2"
)


_AGENT_SKILLS_DELETE_SQL = "DELETE FROM agent_skills WHERE agent_id = $1 AND skill_id = $2"


_AGENT_SKILL_INVOCATIONS_FETCH_SQL = (
    "SELECT agent_id, invocation_id, skill_id, user_id, conversation_id, "
    "message_id, invocation_source, invoked_at, outcome, outcome_source, notes "
    "FROM agent_skill_invocations WHERE agent_id = $1 AND invocation_id = $2"
)


_AGENT_SKILL_INVOCATIONS_DELETE_SQL = "DELETE FROM agent_skill_invocations WHERE agent_id = $1 AND invocation_id = $2"


class AgentSkillCollection(BaseCollection[AgentSkillEntity]):
    """Three-tier collection for :class:`AgentSkillEntity`.

    Composite primary key ``(agent_id, skill_id)``. ``agent_id`` is
    the partition column; every method on this collection threads it
    through so the partition predicate is always present on the wire.

    Hybrid query helpers (``list_for_user`` with FTS ranking,
    ``count_for_user``) absorb the multi-row SQL that does not fit
    ``BaseCollection``'s primary-key-addressable shape; they carry
    explicit ``# cache-bypass:`` annotations because the L1 row cache
    cannot serve them.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "skill_id")

    # Partition column declared as a class attribute (not via
    # SchemaBackedCollection's TableSchema since we hand-roll SQL).
    # Consumers and the workspace partition-column enforcement walker
    # both read this to confirm the partition contract.
    partition_column: ClassVar[str] = "agent_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "agent_skills"

    @property
    def entity_class(self) -> type[AgentSkillEntity]:
        """Return the entity class."""
        return AgentSkillEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk.

        :param entity_id: tuple ``(agent_id, skill_id)``
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        agent_id, skill_id = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(
            _AGENT_SKILLS_FETCH_SQL,
            agent_id,
            skill_id,
        )
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a row, optionally under an optimistic-lock fence.

        When ``original_timestamp`` is supplied (the ``skill_update``
        edit path -- :meth:`BaseCollection.save_entity` threads the
        entity's ``original_date_updated`` through) the write goes
        through :data:`_AGENT_SKILLS_UPSERT_CAS_SQL`, whose ON CONFLICT
        DO UPDATE fences on ``date_updated = original_timestamp``. A
        conflict whose stored ``date_updated`` has since moved (a racing
        edit, or a ``bump_use_count`` / ``increment_outcome_counts``
        counter bump between the caller's read and this write) updates
        zero rows, so ``save_entity`` raises
        :class:`ConcurrentModificationError` rather than clobbering the
        concurrent writer. When ``original_timestamp`` is ``None`` (a
        fresh insert, e.g. ``skill_create``) the unfenced upsert runs.

        :param data: row dict keyed by column name; must carry both pk
            columns and every non-nullable column
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation ``date_updated`` fence
            value for optimistic locking; ``None`` for inserts
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection (forwarded
            so the caller can include the write in their own transaction)
        :ptype conn: Any
        :return: rows affected (1 on success, 0 on CAS-fence mismatch)
        :rtype: int
        """
        params = _skill_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        if original_timestamp is not None:
            status = await target.execute(_AGENT_SKILLS_UPSERT_CAS_SQL, *params, original_timestamp)
        else:
            status = await target.execute(_AGENT_SKILLS_UPSERT_SQL, *params)
        return parse_rowcount(status)

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a row by composite pk.

        Cascade FK on ``agent_skill_invocations.(agent_id, skill_id)``
        ensures the invocation history is removed at the same time.

        :param entity_id: tuple ``(agent_id, skill_id)``
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        agent_id, skill_id = self.normalize_pk(entity_id)
        await self.l3_pool.execute(
            _AGENT_SKILLS_DELETE_SQL,
            agent_id,
            skill_id,
        )
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _SKILL_FIELD_TYPES)

    # --- Domain queries (per shard spec "Public API") ---

    async def find_by_name_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        name: str,
    ) -> AgentSkillEntity | None:
        """Look up a skill by ``(agent_id, user_id, name)``.

        Hits the ``uq_skills_agent_user_name`` unique index; returns
        at most one row.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param user_id: owning user
        :ptype user_id: UUID
        :param name: skill name
        :ptype name: str
        :return: matching entity or ``None``
        :rtype: AgentSkillEntity | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: lookup by (user_id, name) is not primary-key
        # addressable. L1 row cache serves per-(agent_id, skill_id)
        # lookups only; keeping the query on the Collection preserves
        # the single entry point.
        row = await self.l3_pool.fetchrow(
            "SELECT agent_id, skill_id, user_id, name, summary, body, prompt_mode, "
            "tool_additions, tool_restrictions, trigger_keywords, tags, source, "
            "enabled, use_count, last_used_at, success_count, failure_count, "
            "last_failure_at, date_created, date_updated "
            "FROM agent_skills "
            "WHERE agent_id = $1 AND user_id = $2 AND name = $3",
            agent_id,
            user_id,
            name,
        )
        if row is None:
            return None
        data = dict(row)
        return AgentSkillEntity(data, is_new=False, collection=self)

    async def list_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        *,
        enabled_only: bool = True,
        tag_filter: Sequence[str] | None = None,
        query: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AgentSkillEntity]:
        """List skills for a user with optional FTS ranking.

        When ``query`` is set, the result is ranked by
        ``ts_rank_cd(search_vector, websearch_to_tsquery('english',
        query))`` and filtered to rows matching the tsquery.
        When ``query`` is not set, the result is sorted by
        ``last_used_at DESC NULLS LAST, date_created DESC`` -- recency
        first, then creation order as a tiebreaker.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param user_id: owning user
        :ptype user_id: UUID
        :param enabled_only: when ``True`` (default) hide disabled skills
        :ptype enabled_only: bool
        :param tag_filter: optional list of tags; rows are matched
            against the ``tags && filter`` operator (overlap, not
            superset -- one matching tag is enough)
        :ptype tag_filter: Sequence[str] | None
        :param query: optional FTS query string; ``None`` skips FTS ranking
        :ptype query: str | None
        :param limit: page size
        :ptype limit: int
        :param offset: page offset
        :ptype offset: int
        :return: ordered list of matching entities
        :rtype: list[AgentSkillEntity]
        """
        if self.l3_pool is None:
            return []
        conditions = ["agent_id = $1", "user_id = $2"]
        params: list[Any] = [agent_id, user_id]
        param_idx = 3
        if enabled_only:
            conditions.append("enabled = true")
        if tag_filter:
            conditions.append(f"tags && ${param_idx}")
            params.append(list(tag_filter))
            param_idx += 1
        order_clause: str
        select_extra = ""
        if query is not None and query.strip():
            conditions.append(f"search_vector @@ websearch_to_tsquery('english', ${param_idx})")
            params.append(query)
            select_extra = f", ts_rank_cd(search_vector, websearch_to_tsquery('english', ${param_idx})) AS fts_rank"
            param_idx += 1
            order_clause = "fts_rank DESC, last_used_at DESC NULLS LAST, date_created DESC"
        else:
            order_clause = "last_used_at DESC NULLS LAST, date_created DESC"
        where_clause = " AND ".join(conditions)
        limit_param = f"${param_idx}"
        offset_param = f"${param_idx + 1}"
        params.append(limit)
        params.append(offset)
        # cache-bypass: multi-row keyset / FTS-ranked scan is not
        # primary-key addressable; L1 row cache cannot serve. method
        # on the Collection preserves the single SQL entry point.
        sql = (
            "SELECT agent_id, skill_id, user_id, name, summary, body, prompt_mode, "
            "tool_additions, tool_restrictions, trigger_keywords, tags, source, "
            "enabled, use_count, last_used_at, success_count, failure_count, "
            "last_failure_at, date_created, date_updated"
            f"{select_extra} "
            f"FROM agent_skills WHERE {where_clause} "
            f"ORDER BY {order_clause} "
            f"LIMIT {limit_param} OFFSET {offset_param}"
        )
        rows = await self.l3_pool.fetch(sql, *params)
        return [AgentSkillEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def count_for_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        *,
        enabled_only: bool = True,
        tag_filter: Sequence[str] | None = None,
        query: str | None = None,
    ) -> int:
        """Return the skill count for a user, honouring the same filters as ``list_for_user``.

        Used for two purposes:

        1. The per-user prose-skill cap check (``skill_create`` /
           ``POST /skills``) -- called with no ``tag_filter`` / ``query``
           so it returns the unfiltered total the cap is defined against.
        2. ``total_count`` for a paginated ``skill_list`` /
           ``GET /skills``. When the list path applies a ``tag_filter``
           or FTS ``query``, the same predicates MUST be threaded here or
           ``total_count`` overstates the filtered result set.

        The ``tag_filter`` (``tags && filter`` overlap) and ``query``
        (``search_vector @@ websearch_to_tsquery``) predicates mirror
        :meth:`list_for_user` byte-for-byte so the count never drifts
        from what the list actually returns.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param user_id: owning user
        :ptype user_id: UUID
        :param enabled_only: when ``True`` (default) hide disabled skills
        :ptype enabled_only: bool
        :param tag_filter: optional tag-overlap filter (matches
            :meth:`list_for_user`); ``None`` counts every tag
        :ptype tag_filter: Sequence[str] | None
        :param query: optional FTS query string (matches
            :meth:`list_for_user`); ``None`` skips the FTS predicate
        :ptype query: str | None
        :return: total matching row count
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        conditions = ["agent_id = $1", "user_id = $2"]
        params: list[Any] = [agent_id, user_id]
        param_idx = 3
        if enabled_only:
            conditions.append("enabled = true")
        if tag_filter:
            conditions.append(f"tags && ${param_idx}")
            params.append(list(tag_filter))
            param_idx += 1
        if query is not None and query.strip():
            conditions.append(f"search_vector @@ websearch_to_tsquery('english', ${param_idx})")
            params.append(query)
            param_idx += 1
        # cache-bypass: aggregate COUNT(*) is not primary-key
        # addressable; L1 row cache cannot serve.
        where_clause = " AND ".join(conditions)
        sql = f"SELECT COUNT(*) FROM agent_skills WHERE {where_clause}"
        value = await self.l3_pool.fetchval(sql, *params)
        return int(value or 0)

    async def bump_use_count(
        self,
        agent_id: UUID,
        skill_ids: Sequence[UUID],
    ) -> None:
        """Atomically bump ``use_count`` + ``last_used_at`` for a batch.

        Called by the consumer's wake / invoke load path after a skill
        successfully attaches to the turn. Single UPDATE so the
        per-row write amplification is constant regardless of batch
        size.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param skill_ids: list of skill UUIDs to bump
        :ptype skill_ids: Sequence[UUID]
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not skill_ids:
            return None
        # The counter bump is a bulk UPDATE by skill_id IN (...) -- not
        # primary-key addressable, so it cannot flow through the L1 row
        # cache write path. But every bumped skill IS pk-addressable and
        # is very likely cached (the wake / invoke load path just read
        # it), so a raw UPDATE alone would leave stale ``use_count`` /
        # ``last_used_at`` in this pod's L1 and in L2 (and in every peer
        # pod's L1). Invalidate each bumped pk so the cross-tier drop +
        # cross-pod invalidation fire and the next read fetches through
        # to observe the bump.
        await self.l3_pool.execute(
            "UPDATE agent_skills "
            "SET use_count = use_count + 1, last_used_at = now(), date_updated = now() "
            "WHERE agent_id = $1 AND skill_id = ANY($2::uuid[])",
            agent_id,
            list(skill_ids),
        )
        for skill_id in skill_ids:
            await self.invalidate_cache((agent_id, skill_id))
        return None

    async def increment_outcome_counts(
        self,
        agent_id: UUID,
        skill_id: UUID,
        outcome: SkillOutcome,
    ) -> None:
        """Bump ``success_count`` or ``failure_count`` for a single skill.

        Called by the ``skill_report_outcome`` tool handler (``tools.py``)
        once the agent self-reports an outcome. ``last_failure_at`` is
        also stamped when ``outcome`` is ``'failure'`` so the catalog can
        surface "last failed N hours ago" without scanning the
        invocation history.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param skill_id: skill to update
        :ptype skill_id: UUID
        :param outcome: ``'success'`` or ``'failure'``
        :ptype outcome: SkillOutcome
        :return: nothing
        :rtype: None
        :raises ValueError: when ``outcome`` is not one of the valid values
        """
        if outcome == "success":
            sql = (
                "UPDATE agent_skills "
                "SET success_count = success_count + 1, date_updated = now() "
                "WHERE agent_id = $1 AND skill_id = $2"
            )
        elif outcome == "failure":
            sql = (
                "UPDATE agent_skills "
                "SET failure_count = failure_count + 1, "
                "    last_failure_at = now(), date_updated = now() "
                "WHERE agent_id = $1 AND skill_id = $2"
            )
        else:
            raise ValueError(
                f"increment_outcome_counts: outcome must be 'success' or 'failure'; got {outcome!r}",
            )
        if self.l3_pool is None:
            return None
        # Targeted single-row counter UPDATE. The row is pk-addressable
        # and likely cached, so a raw UPDATE alone would leave stale
        # ``success_count`` / ``failure_count`` / ``last_failure_at`` in
        # this pod's L1, in L2, and in every peer pod's L1. Invalidate
        # the pk after the write so the cross-tier drop + cross-pod
        # invalidation fire and the next read fetches through.
        await self.l3_pool.execute(sql, agent_id, skill_id)
        await self.invalidate_cache((agent_id, skill_id))
        return None


class AgentSkillInvocationCollection(BaseCollection[AgentSkillInvocationEntity]):
    """Three-tier collection for :class:`AgentSkillInvocationEntity`.

    Composite primary key ``(agent_id, invocation_id)`` -- partition
    column is ``agent_id``. The composite FK
    ``(agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id)
    ON DELETE CASCADE`` means deleting a skill discards its
    invocation history; this Collection holds no delete logic for
    that fan-out (Postgres handles it).
    """

    primary_key_column: str | tuple[str, ...] = (
        "agent_id",
        "invocation_id",
    )

    partition_column: ClassVar[str] = "agent_id"

    @property
    def table_name(self) -> str:
        """Return the L3 table name."""
        return "agent_skill_invocations"

    @property
    def entity_class(self) -> type[AgentSkillInvocationEntity]:
        """Return the entity class."""
        return AgentSkillInvocationEntity

    # --- BaseCollection contract ---

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch a row by composite pk."""
        if self.l3_pool is None:
            return None
        agent_id, invocation_id = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(
            _AGENT_SKILL_INVOCATIONS_FETCH_SQL,
            agent_id,
            invocation_id,
        )
        if row is None:
            return None
        return dict(row)

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: Any = None,
        *,
        conn: Any = None,
    ) -> int:
        """Upsert a row.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (invocations have no CAS fence)
        :ptype original_timestamp: Any
        :param conn: optional asyncpg-compatible connection
        :ptype conn: Any
        :return: rows affected
        :rtype: int
        """
        del original_timestamp
        params = _invocation_insert_params(data)
        target = conn if conn is not None else self.l3_pool
        if target is None:
            return 0
        await target.execute(_AGENT_SKILL_INVOCATIONS_UPSERT_SQL, *params)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """Delete a row by composite pk."""
        if self.l3_pool is None:
            return None
        agent_id, invocation_id = self.normalize_pk(entity_id)
        await self.l3_pool.execute(
            _AGENT_SKILL_INVOCATIONS_DELETE_SQL,
            agent_id,
            invocation_id,
        )
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Encode a row dict for L2 (NATS KV) storage."""
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Decode L2-cached bytes back to a row dict."""
        return deserialize_from_json(data, _INVOCATION_FIELD_TYPES)

    # --- Domain methods (per shard spec "Public API") ---

    async def record(
        self,
        agent_id: UUID,
        invocation: AgentSkillInvocationEntity,
    ) -> None:
        """Convenience wrapper around ``save_entity`` for invocation rows.

        ``agent_id`` is taken as an argument so the partition-column
        contract is visible on the call site (the value is already
        on the entity; the parameter is a contract reminder + a guard
        against bugs that construct an entity with a stale agent_id).

        :param agent_id: partition column expected on the entity
        :ptype agent_id: UUID
        :param invocation: invocation entity to persist
        :ptype invocation: AgentSkillInvocationEntity
        :return: nothing
        :rtype: None
        :raises ValueError: when ``invocation.agent_id`` does not match
            the caller's ``agent_id``
        """
        if invocation.agent_id != agent_id:
            raise ValueError(
                f"record(): entity agent_id {invocation.agent_id!r} does not match caller {agent_id!r}",
            )
        await self.save_entity(invocation)

    async def list_for_skill(
        self,
        agent_id: UUID,
        skill_id: UUID,
        *,
        limit: int = 20,
        offset: int = 0,
        outcome_filter: SkillOutcome | None = None,
    ) -> list[AgentSkillInvocationEntity]:
        """List invocation rows for a specific skill, newest first.

        Powers the future ``skill_history`` REST endpoint (deferred to
        v1.1 per the planning set) and per-skill admin views.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param skill_id: skill whose history to read
        :ptype skill_id: UUID
        :param limit: page size
        :ptype limit: int
        :param offset: page offset
        :ptype offset: int
        :param outcome_filter: optional outcome filter; ``None`` returns
            all (success, failure, and un-classified rows)
        :ptype outcome_filter: SkillOutcome | None
        :return: list of invocation entities ordered by ``invoked_at`` DESC
        :rtype: list[AgentSkillInvocationEntity]
        """
        if self.l3_pool is None:
            return []
        conditions = ["agent_id = $1", "skill_id = $2"]
        params: list[Any] = [agent_id, skill_id]
        param_idx = 3
        if outcome_filter is not None:
            conditions.append(f"outcome = ${param_idx}")
            params.append(outcome_filter)
            param_idx += 1
        limit_param = f"${param_idx}"
        offset_param = f"${param_idx + 1}"
        params.append(limit)
        params.append(offset)
        where_clause = " AND ".join(conditions)
        # cache-bypass: per-skill history scan is not primary-key
        # addressable; L1 row cache cannot serve. method on the
        # Collection preserves the single SQL entry point.
        sql = (
            "SELECT agent_id, invocation_id, skill_id, user_id, conversation_id, "
            "message_id, invocation_source, invoked_at, outcome, outcome_source, notes "
            "FROM agent_skill_invocations "
            f"WHERE {where_clause} "
            f"ORDER BY invoked_at DESC "
            f"LIMIT {limit_param} OFFSET {offset_param}"
        )
        rows = await self.l3_pool.fetch(sql, *params)
        return [AgentSkillInvocationEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def count_for_skill(
        self,
        agent_id: UUID,
        skill_id: UUID,
        *,
        outcome_filter: SkillOutcome | None = None,
    ) -> int:
        """Return the total invocation count for a skill, honouring ``outcome_filter``.

        Counterpart to :meth:`list_for_skill`: applies the SAME
        ``outcome_filter`` predicate so a paginated consumer can report
        an accurate ``total_count`` independent of the page size. With
        ``outcome_filter=None`` every invocation row (success, failure,
        and un-classified) is counted; a concrete ``'success'`` /
        ``'failure'`` filters to rows with that outcome.

        ``outcome_filter`` here matches :meth:`list_for_skill`'s
        ``SkillOutcome`` domain (``'success'`` | ``'failure'``).
        "Unknown" (NULL outcome) is not a ``SkillOutcome`` value, so a
        consumer that wants the NULL-row count computes it as
        ``count_for_skill(...) - <success> - <failure>`` (or post-filters
        the list); this method does not encode an "unknown" predicate so
        it stays a faithful counterpart of ``list_for_skill``.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param skill_id: skill whose invocations to count
        :ptype skill_id: UUID
        :param outcome_filter: optional outcome filter; ``None`` counts
            all rows
        :ptype outcome_filter: SkillOutcome | None
        :return: total matching invocation count
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        conditions = ["agent_id = $1", "skill_id = $2"]
        params: list[Any] = [agent_id, skill_id]
        if outcome_filter is not None:
            conditions.append("outcome = $3")
            params.append(outcome_filter)
        # cache-bypass: aggregate COUNT(*) is not primary-key
        # addressable; L1 row cache cannot serve.
        where_clause = " AND ".join(conditions)
        sql = f"SELECT COUNT(*) FROM agent_skill_invocations WHERE {where_clause}"
        value = await self.l3_pool.fetchval(sql, *params)
        return int(value or 0)

    async def list_for_conversation(
        self,
        agent_id: UUID,
        conversation_id: UUID,
        *,
        limit: int = 20,
    ) -> list[AgentSkillInvocationEntity]:
        """List invocations within a conversation, newest first.

        Supports a future "what skills loaded in this conversation"
        admin view -- handy for debugging why a turn behaved the way
        it did when a wake-attached skill was active.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param conversation_id: conversation to scan
        :ptype conversation_id: UUID
        :param limit: page size (no offset -- conversations are
            typically small enough that ``LIMIT`` is the only cap
            needed)
        :ptype limit: int
        :return: list of invocation entities ordered by ``invoked_at`` DESC
        :rtype: list[AgentSkillInvocationEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: per-conversation scan is not primary-key
        # addressable.
        rows = await self.l3_pool.fetch(
            "SELECT agent_id, invocation_id, skill_id, user_id, conversation_id, "
            "message_id, invocation_source, invoked_at, outcome, outcome_source, notes "
            "FROM agent_skill_invocations "
            "WHERE agent_id = $1 AND conversation_id = $2 "
            "ORDER BY invoked_at DESC LIMIT $3",
            agent_id,
            conversation_id,
            limit,
        )
        return [AgentSkillInvocationEntity(dict(row), is_new=False, collection=self) for row in rows]

    async def set_message_id(
        self,
        agent_id: UUID,
        invocation_ids: Sequence[UUID],
        message_id: UUID,
    ) -> None:
        """Bulk-set ``message_id`` on a batch of invocations.

        Called by the consumer's loader post-LLM once the assistant
        response message ID is known. A single UPDATE is faster than
        per-row writes and atomic with respect to readers.

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param invocation_ids: invocations to stamp with ``message_id``
        :ptype invocation_ids: Sequence[UUID]
        :param message_id: assistant-response message UUID (no FK
            constraint -- ``messages`` is consumer-owned and may be
            hard-deleted)
        :ptype message_id: UUID
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not invocation_ids:
            return None
        # cache-bypass: bulk UPDATE WHERE invocation_id IN (...) is not
        # primary-key addressable for the row cache.
        await self.l3_pool.execute(
            "UPDATE agent_skill_invocations SET message_id = $3 "
            "WHERE agent_id = $1 AND invocation_id = ANY($2::uuid[])",
            agent_id,
            list(invocation_ids),
            message_id,
        )
        return None

    async def set_outcome(
        self,
        agent_id: UUID,
        invocation_id: UUID,
        *,
        outcome: SkillOutcome,
        source: OutcomeSource,
    ) -> None:
        """Set ``outcome`` + ``outcome_source`` on a single invocation.

        Idempotent: repeated calls with the same arguments produce the
        same row state.

        Called by the ``skill_report_outcome`` tool handler (``tools.py``)
        once the agent self-reports an outcome. The shard does not pair
        this with an automatic bump to the parent skill's
        ``success_count`` / ``failure_count`` because:

        1. The caller already knows the parent ``skill_id`` from the
           invocation row (avoids the JOIN here).
        2. Decoupling lets the consumer aggregate multiple invocations
           in a single skill-counter update when a single response
           covers multiple invocations (rare but possible with
           skill-chained wakes in shard 03).

        :param agent_id: partition column
        :ptype agent_id: UUID
        :param invocation_id: row to update
        :ptype invocation_id: UUID
        :param outcome: ``'success'`` or ``'failure'``
        :ptype outcome: SkillOutcome
        :param source: provenance of the classification
        :ptype source: OutcomeSource
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: targeted single-row UPDATE on the outcome
        # columns; row cache typically holds the pre-outcome shape and
        # subsequent reads fetch-through cleanly.
        await self.l3_pool.execute(
            "UPDATE agent_skill_invocations "
            "SET outcome = $3, outcome_source = $4 "
            "WHERE agent_id = $1 AND invocation_id = $2",
            agent_id,
            invocation_id,
            outcome,
            source,
        )
        return None


# --- helpers -----------------------------------------------------------


def _skill_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the positional parameter list expected by
    :data:`_AGENT_SKILLS_UPSERT_SQL`.

    Missing values are coerced to a sensible default to match the
    NOT NULL DEFAULT semantics of the L3 schema (the upsert path
    writes every column, so the application supplies the value rather
    than relying on the server default for columns the schema permits
    to default).

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in ``_SKILL_INSERT_COLUMNS`` order
    :rtype: list[Any]
    """
    return [_skill_value_for_column(col, data) for col in _SKILL_INSERT_COLUMNS]


def _skill_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with defaults applied for columns that have
    a DB-side default; callers may omit them.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        value = data[col]
        if col in {"tool_additions", "tool_restrictions", "tags"} and value is not None:
            return list(value)
        return value
    # column-specific defaults that mirror the L3 schema
    if col == "prompt_mode":
        return "additive"
    if col in {"tool_additions", "tool_restrictions", "tags"}:
        return []
    if col == "trigger_keywords":
        return ""
    if col == "source":
        return "manual"
    if col == "enabled":
        return True
    if col in {"use_count", "success_count", "failure_count"}:
        return 0
    return None


def _invocation_insert_params(data: dict[str, Any]) -> list[Any]:
    """Project a row dict onto the positional parameter list expected by
    :data:`_AGENT_SKILL_INVOCATIONS_UPSERT_SQL`.

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: positional parameter list in
        ``_INVOCATION_INSERT_COLUMNS`` order
    :rtype: list[Any]
    """
    return [_invocation_value_for_column(col, data) for col in _INVOCATION_INSERT_COLUMNS]


def _invocation_value_for_column(col: str, data: dict[str, Any]) -> Any:
    """Return ``data[col]`` with sensible defaults for omittable columns.

    :param col: column name
    :ptype col: str
    :param data: row dict
    :ptype data: dict[str, Any]
    :return: value to bind for this column
    :rtype: Any
    """
    if col in data:
        return data[col]
    # the ``invoked_at`` column carries a server default of ``now()``;
    # passing ``None`` here would trip the NOT NULL constraint. The
    # framework's ``save_entity`` path always stamps ``date_created``
    # / ``date_updated`` but the invocation table uses ``invoked_at``
    # instead; if a caller omits it, fall back to ``now()`` here via
    # a database-side default by skipping the param entirely is not
    # an option in this hand-rolled path -- so we return ``None`` and
    # rely on the NOT NULL DEFAULT to raise loudly if the caller
    # actually forgot the value (the entity's ``invoked_at`` getter
    # type-pins it as ``datetime``).
    return None
