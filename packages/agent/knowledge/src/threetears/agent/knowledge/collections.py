"""Agent-side three-tier knowledge collections -- playbook entries + concepts.

Re-implements the hub's playbook-entry / concept collections for the agent pod:
the agent owns its OWN collections over the SAME ``platform.playbook_entries`` /
``platform.concepts`` tables (reverse-importing the hub class is banned),
constructed over the ``system.platform.rbac`` NATS proxy pool. Each schema OMITS
the ``embedding`` column from the projection so the proxy pool never decodes a
pgvector value on the hot per-turn list path; the similarity ranker fetches
vectors separately via the bounded :meth:`fetch_embeddings` read (the
``embedding::text`` cast, the proxy read-path fix).

Both collections are built entirely from 3tears primitives:
:class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`,
:func:`threetears.agent.acl.three_scope_visibility_clause` (the SAME shared
visibility clause the hub runs), and the ``threetears.knowledge`` snapshot / scope
types. The only bespoke queries are :meth:`list_visible_to_user` (an
rbac-visibility filtered list, evaluated SQL-side) and :meth:`list_own_drafts`
(the author-private draft read); both bypass the by-pk L1/L2 caches because they
are cross-table JOINs the by-pk Collection abstraction cannot express.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from threetears.agent.acl import three_scope_visibility_clause
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.knowledge import (
    ConceptSnapshot,
    EntrySnapshot,
    build_table_ref,
    derive_scope,
)

from threetears.agent.knowledge.entities import ConceptEntity, PlaybookEntryEntity
from threetears.agent.knowledge.integration import DraftView

__all__ = ["ConceptCollection", "PlaybookEntryCollection"]


def _as_uuid(value: Any) -> UUID | None:
    """Coerce a proxy-returned id back to a :class:`UUID` at the NATS border.

    The L3 proxy backend serializes every row to JSON over NATS, so UUID columns
    arrive at the agent as STRINGS. The shared knowledge merge sorts shadow chains
    on ``id.bytes`` and links ancestors by id IDENTITY -- both require real UUID
    objects (a ``str`` has no ``.bytes``; a ``str`` id never matches a ``UUID``
    ancestor key, silently breaking chain resolution). This restores the border
    rule: convert back from string the moment a value re-enters the process from
    the NATS boundary, before it reaches any internal function.

    :param value: id value as returned by the proxy (``str`` / ``UUID`` / ``None``)
    :ptype value: Any
    :return: coerced UUID, or ``None`` when the source is ``None``
    :rtype: UUID | None
    """
    if value is None or isinstance(value, UUID):
        result: UUID | None = value
    else:
        result = UUID(str(value))
    return result


def _parse_vector_text(value: Any) -> list[float] | None:
    """Parse a pgvector ``::text`` value back into ``list[float]``.

    The ``embedding::text`` cast (the proxy read-path fix) returns the bracketed
    textual form ``"[1.0,2.0,...]"`` because asyncpg has no binary codec for the
    ``vector`` type on the proxy pool. The bracketed form is plain JSON, so a
    single ``json.loads`` round-trips it; a NULL embedding (``None``) yields
    ``None`` so the caller omits the id. A list passthrough is accepted defensively
    (a pool that DID register a codec would hand back a list already).

    :param value: textual vector, list passthrough, or ``None``
    :ptype value: Any
    :return: list of floats, or ``None`` when the source is NULL / unparseable
    :rtype: list[float] | None
    """
    if value is None:
        result: list[float] | None = None
    elif isinstance(value, list):
        result = [float(v) for v in value]
    elif isinstance(value, str):
        parsed = json.loads(value)
        result = [float(v) for v in parsed] if isinstance(parsed, list) else None
    else:
        result = None
    return result


async def _fetch_embeddings(
    l3_pool: Any,
    table_name: str,
    ids: list[UUID],
    *,
    customer_scope: UUID,
) -> dict[UUID, list[float]]:
    """Read non-NULL embeddings for the given ids from one knowledge table.

    The SHARED implementation behind
    :meth:`PlaybookEntryCollection.fetch_embeddings` /
    :meth:`ConceptCollection.fetch_embeddings`: a single
    ``SELECT id, embedding::text FROM <table> WHERE id = ANY($1)`` over the
    rbac-read proxy pool. The ``::text`` cast returns the bracketed textual vector
    parsed back to ``list[float]`` by :func:`_parse_vector_text`; a NULL embedding
    casts to NULL, parses to ``None``, and is filtered out (the caller falls back
    to stable-order for it). The id column arrives as a STRING over the NATS proxy
    boundary and is coerced back to :class:`UUID` so the returned keys match the
    effective-view ids. Returns an empty map for an empty id list (no query
    issued) or when the pool is unwired.

    NO ``embedding IS NOT NULL`` predicate: the cosine index on this column is
    PARTIAL (created ``WHERE embedding IS NOT NULL``), so a query carrying that
    predicate lets the planner pick the vector index for this plain by-id read --
    then the tserver rejects the request because an HNSW scan needs a query vector
    this read does not supply. A NULL embedding instead casts to NULL, parses to
    ``None``, and is filtered below -- same result, PK-index plan.

    Cache-bypass: a targeted ``embedding::text`` vector read; the by-pk Collection
    projection deliberately omits ``embedding``.

    :param l3_pool: rbac-read proxy pool (``None`` -> empty map)
    :ptype l3_pool: Any
    :param table_name: ``playbook_entries`` or ``concepts``
    :ptype table_name: str
    :param ids: candidate ids to fetch vectors for
    :ptype ids: list[UUID]
    :param customer_scope: conversation customer the broker clamps this Class-B
        read to (broker-isolation-task-01)
    :ptype customer_scope: UUID
    :return: map of id to embedding vector (NULL-embedding ids omitted)
    :rtype: dict[UUID, list[float]]
    """
    result: dict[UUID, list[float]] = {}
    if l3_pool is not None and ids:
        rows = await l3_pool.fetch(
            f"SELECT id, embedding::text AS embedding FROM {table_name} WHERE id = ANY($1)",
            ids,
            customer_scope=customer_scope,
        )
        for row in rows:
            data = dict(row)
            vector = _parse_vector_text(data.get("embedding"))
            row_id = _as_uuid(data.get("id"))
            if vector is not None and row_id is not None:
                result[row_id] = vector
    return result


def _row_to_snapshot(row: dict[str, Any]) -> EntrySnapshot:
    """Build an :class:`EntrySnapshot` from a proxied ``playbook_entries`` row.

    The scope is DERIVED from the row's nullability via
    :func:`threetears.knowledge.derive_scope` (D1) -- never read from a stored
    column. ``tags`` arrives from the JSONB column as a list (or ``None``); it is
    normalized to a tuple for the frozen snapshot. The UUID columns (``id`` /
    ``origin_entry_id`` / ``datasource_id``) arrive as STRINGS over the NATS proxy
    boundary and are coerced back to :class:`UUID` via :func:`_as_uuid` so the
    shared merge's ``id.bytes`` sort + id-identity chain linking work.

    :param row: dict view of one ``playbook_entries`` row from the proxy
    :ptype row: dict[str, Any]
    :return: immutable snapshot ready for :func:`threetears.knowledge.merge_entry_views`
    :rtype: EntrySnapshot
    """
    raw_tags = row.get("tags")
    tags: tuple[str, ...] = tuple(raw_tags) if raw_tags else ()
    scope = derive_scope(
        customer_id=row.get("customer_id"),
        user_id=row.get("user_id"),
    )
    return EntrySnapshot(
        id=_as_uuid(row["id"]),  # type: ignore[arg-type]
        scope=scope,
        origin_entry_id=_as_uuid(row.get("origin_entry_id")),
        title=row.get("title") or "",
        body=row.get("body") or "",
        tags=tags,
        always_inject=bool(row.get("always_inject")),
        datasource_id=_as_uuid(row.get("datasource_id")),
    )


def _row_to_concept_snapshot(row: dict[str, Any]) -> ConceptSnapshot:
    """Build a :class:`ConceptSnapshot` from a proxied ``concepts`` row.

    The scope is DERIVED from the row's nullability via
    :func:`threetears.knowledge.derive_scope` (D1) -- never read from a stored
    column. ``aliases`` and ``tags`` arrive from JSONB columns as lists (or
    ``None``); they are normalized to tuples for the frozen snapshot.
    ``sql_fragment`` / ``caveats`` ride through verbatim (curated context for the
    model, never executed server-side). The UUID columns (``id`` /
    ``origin_concept_id`` / ``datasource_table_id``) arrive as STRINGS over the
    NATS proxy boundary and are coerced back to :class:`UUID` via :func:`_as_uuid`
    so the shared merge's ``id.bytes`` sort + id-identity chain linking work.

    :param row: dict view of one ``concepts`` row from the proxy
    :ptype row: dict[str, Any]
    :return: immutable snapshot ready for :func:`threetears.knowledge.merge_concept_views`
    :rtype: ConceptSnapshot
    """
    raw_aliases = row.get("aliases")
    aliases: tuple[str, ...] = tuple(raw_aliases) if raw_aliases else ()
    raw_tags = row.get("tags")
    tags: tuple[str, ...] = tuple(raw_tags) if raw_tags else ()
    scope = derive_scope(
        customer_id=row.get("customer_id"),
        user_id=row.get("user_id"),
    )
    return ConceptSnapshot(
        id=_as_uuid(row["id"]),  # type: ignore[arg-type]
        scope=scope,
        origin_concept_id=_as_uuid(row.get("origin_concept_id")),
        name=row.get("name") or "",
        aliases=aliases,
        definition=row.get("definition") or "",
        datasource_table_id=_as_uuid(row.get("datasource_table_id")),
        datasource_table_ref=build_table_ref(
            row.get("bound_schema_name"),
            row.get("bound_table_name"),
        ),
        sql_fragment=row.get("sql_fragment"),
        caveats=row.get("caveats"),
        tags=tags,
        always_inject=bool(row.get("always_inject")),
    )


class PlaybookEntryCollection(SchemaBackedCollection[PlaybookEntryEntity]):
    """Agent-side three-tier collection over ``platform.playbook_entries``.

    Re-implements the hub's playbook-entry collection for the agent pod over the
    ``system.platform.rbac`` proxy pool. The schema OMITS the ``embedding`` column
    from the projection so the proxy pool never decodes a pgvector value; the
    similarity ranker fetches embeddings separately via :meth:`fetch_embeddings`.

    The only bespoke queries are :meth:`list_visible_to_user` (a caller-visibility
    filtered list whose WHERE clause is evaluated SQL-side -- the trust boundary is
    the SQL itself, never a python post-filter) and :meth:`list_own_drafts` (the
    author-private draft read).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="playbook_entries",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("playbook_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True),
            Column("user_id", UUID_TYPE, nullable=True),
            Column("visibility", STRING_TYPE),
            Column("origin_entry_id", UUID_TYPE, nullable=True),
            Column("title", STRING_TYPE),
            Column("body", STRING_TYPE),
            Column("tags", JSONB_TYPE),
            Column("datasource_id", UUID_TYPE),
            Column("always_inject", BOOL_TYPE),
            # draft lifecycle + turn provenance (knowledge-task-06 / v019).
            # ``status`` carries ``server_default='active'`` so the agent never
            # has to name it (the agent never WRITES this table -- the hub
            # emitter does); declaring the columns keeps the agent-side schema in
            # agreement with the hub table so the own-drafts read projects them.
            Column("status", STRING_TYPE, server_default="'active'"),
            Column("conversation_id", UUID_TYPE, nullable=True),
            Column("message_id_source", UUID_TYPE, nullable=True),
            Column("turn_count", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """Return database table name.

        :return: table name string
        :rtype: str
        """
        return "playbook_entries"

    @property
    def entity_class(self) -> type[PlaybookEntryEntity]:
        """Return entity class for this collection.

        :return: PlaybookEntryEntity class
        :rtype: type[PlaybookEntryEntity]
        """
        return PlaybookEntryEntity

    async def list_visible_to_user(
        self,
        user_id: UUID,
        *,
        datasource_id: UUID | None = None,
        customer_scope: UUID,
    ) -> list[EntrySnapshot]:
        """List caller-visible entries as merge snapshots, domain-filtered.

        The trust boundary is the SQL itself: the WHERE clause admits a row iff
        (a) it is platform-scope (``customer_id IS NULL``) OR its customer is one
        the caller's RBAC grants admit, AND (b) it is NOT a user-scope row owned by
        a DIFFERENT user (``user_id IS NULL OR user_id = <caller>``) -- composed as
        ONE fragment from the shared
        :func:`threetears.agent.acl.three_scope_visibility_clause` (the SAME copy
        the hub runs, D10), evaluated over the ``system.platform.rbac`` proxy pool.
        Optional ``datasource_id`` narrows the result to the anchoring datasource;
        ``None`` returns every visible entry. When ``datasource_id`` is a CUSTOMER
        datasource D linked to a canonical platform-shared datasource P
        (``D.origin_datasource_id = P``), the gather widens to
        ``datasource_id IN (D, P)`` (KNW-77) so platform-scope knowledge anchored
        on P composes into D's effective view; D's own customer/user entries stay
        private to D. ``status = 'active'`` excludes correction-harvest DRAFTS from
        retrieval (KNW-06 / D5).

        Cache-bypass: rbac-visibility filtered list -- inherently a cross-table
        JOIN against ``role_assignments`` / ``group_members`` the by-pk Collection
        abstraction cannot express, so this reads through the proxy pool directly.

        :param user_id: caller (conversation end-user) UUID
        :ptype user_id: UUID
        :param datasource_id: optional datasource-anchor restriction
        :ptype datasource_id: UUID | None
        :param customer_scope: the conversation's hub-authenticated customer the
            broker clamps this Class-B read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: caller-visible entry snapshots for the requested domain ordered by
            ``date_created`` ascending
        :rtype: list[EntrySnapshot]
        """
        result: list[EntrySnapshot] = []
        if self.l3_pool is not None:
            visibility_clause, params = three_scope_visibility_clause(
                user_id=user_id,
                customer_id_column="pe.customer_id",
                user_id_column="pe.user_id",
                param_offset=1,
            )
            sql = (
                "SELECT pe.id, pe.customer_id, pe.user_id, "
                "pe.origin_entry_id, pe.title, pe.body, pe.tags, "
                "pe.datasource_id, pe.always_inject "
                "FROM playbook_entries pe "
                f"WHERE ({visibility_clause}) "
                "AND pe.status = 'active' "
            )
            if datasource_id is not None:
                # KNW-77: gather across the origin link. the IN-set is {D, P}
                # where P = D.origin_datasource_id (resolved by the correlated
                # subquery over platform.datasources on the same rbac-read proxy
                # pool -- a fresh per-call read, no module state). a NULL origin
                # yields {D}, so an unlinked datasource keeps the single-anchor
                # behaviour.
                params.append(datasource_id)
                idx = len(params)
                sql += (
                    f" AND pe.datasource_id IN (${idx}, "
                    f"(SELECT origin_datasource_id FROM datasources "
                    f"WHERE id = ${idx}))"
                )
            sql += " ORDER BY pe.date_created ASC"

            rows = await self.l3_pool.fetch(sql, *params, customer_scope=customer_scope)
            for row in rows:
                result.append(_row_to_snapshot(dict(row)))
        return result

    async def list_own_drafts(
        self,
        user_id: UUID,
        *,
        customer_scope: UUID,
    ) -> list[DraftView]:
        """List the caller's OWN correction-harvest entry drafts (KNW-54).

        Returns ONLY ``status='draft'`` entries owned by ``user_id``, read over the
        ``system.platform.rbac`` proxy pool (a SELECT the broker carve-out admits).
        Drafts are private to their author, so this is keyed strictly on
        ``user_id`` ownership -- no RBAC-customer widening, no peer visibility. This
        is the ONLY agent read path that surfaces drafts;
        :meth:`list_visible_to_user` filters them out of retrieval.

        Cache-bypass: owner+status predicate scan over the partial draft-owner
        index; not a by-pk read.

        :param user_id: authoring user UUID; only this user's drafts are returned
        :ptype user_id: UUID
        :param customer_scope: conversation customer the broker clamps this Class-B
            read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: the caller's own entry drafts as :class:`DraftView`
        :rtype: list[DraftView]
        """
        result: list[DraftView] = []
        if self.l3_pool is not None:
            sql = (
                "SELECT pe.id, pe.title, pe.body, pe.datasource_id, "
                "pe.conversation_id, pe.turn_count "
                "FROM playbook_entries pe "
                "WHERE pe.user_id = $1 AND pe.status = 'draft' "
                "ORDER BY pe.date_created ASC"
            )
            rows = await self.l3_pool.fetch(sql, user_id, customer_scope=customer_scope)
            for row in rows:
                data = dict(row)
                result.append(
                    DraftView(
                        draft_id=_as_uuid(data["id"]),  # type: ignore[arg-type]
                        target="entry",
                        title=data.get("title") or "",
                        body=data.get("body") or "",
                        related_domain=str(data.get("datasource_id") or ""),
                        conversation_id=_as_uuid(data.get("conversation_id")),
                        turn_count=data.get("turn_count"),
                    ),
                )
        return result

    async def fetch_embeddings(
        self,
        ids: list[UUID],
        *,
        customer_scope: UUID,
    ) -> dict[UUID, list[float]]:
        """Fetch stored embeddings for the given entry ids (KNW-95).

        The SEPARATE, bounded vector read the situational ranker uses: the hot
        :meth:`list_visible_to_user` projection OMITS ``embedding``; this reads
        vectors ONLY for the small set of situational candidates about to be
        ranked, via the ``embedding::text`` cast. Rows whose ``embedding`` is NULL
        are OMITTED from the result so the caller falls back to stable-order for
        them. Returns an empty map when ``ids`` is empty.

        :param ids: situational candidate entry ids to fetch vectors for
        :ptype ids: list[UUID]
        :param customer_scope: conversation customer the broker clamps this Class-B
            read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: map of entry id to its embedding vector (NULL-embedding ids
            omitted)
        :rtype: dict[UUID, list[float]]
        """
        return await _fetch_embeddings(self.l3_pool, "playbook_entries", ids, customer_scope=customer_scope)


class ConceptCollection(SchemaBackedCollection[ConceptEntity]):
    """Agent-side three-tier collection over ``platform.concepts``.

    Re-implements the hub's concept collection for the agent pod over the
    ``system.platform.rbac`` proxy pool. The schema OMITS the ``embedding`` column
    from the projection so the proxy pool never decodes a pgvector value; the
    similarity ranker fetches vectors separately via :meth:`fetch_embeddings`.

    The only bespoke queries are :meth:`list_visible_to_user` (a caller-visibility
    filtered list evaluated SQL-side, exactly mirroring the entry collection) and
    :meth:`list_own_drafts` (the author-private draft read).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="concepts",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True),
            Column("user_id", UUID_TYPE, nullable=True),
            Column("visibility", STRING_TYPE),
            Column("origin_concept_id", UUID_TYPE, nullable=True),
            Column("name", STRING_TYPE),
            Column("aliases", JSONB_TYPE),
            Column("definition", STRING_TYPE),
            Column("datasource_id", UUID_TYPE),
            Column("datasource_table_id", UUID_TYPE, nullable=True),
            Column("sql_fragment", STRING_TYPE, nullable=True),
            Column("caveats", STRING_TYPE, nullable=True),
            Column("tags", JSONB_TYPE),
            Column("always_inject", BOOL_TYPE),
            # draft lifecycle + turn provenance (knowledge-task-06 / v019).
            # ``status`` carries ``server_default='active'`` (the agent never
            # writes this table; the hub emitter does). declaring the columns
            # keeps the agent-side schema in agreement with the hub table so the
            # own-drafts read projects them.
            Column("status", STRING_TYPE, server_default="'active'"),
            Column("conversation_id", UUID_TYPE, nullable=True),
            Column("message_id_source", UUID_TYPE, nullable=True),
            Column("turn_count", INT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """Return database table name.

        :return: table name string
        :rtype: str
        """
        return "concepts"

    @property
    def entity_class(self) -> type[ConceptEntity]:
        """Return entity class for this collection.

        :return: ConceptEntity class
        :rtype: type[ConceptEntity]
        """
        return ConceptEntity

    async def list_visible_to_user(
        self,
        user_id: UUID,
        *,
        datasource_id: UUID | None = None,
        datasource_table_id: UUID | None = None,
        customer_scope: UUID,
    ) -> list[ConceptSnapshot]:
        """List caller-visible concepts as merge snapshots, domain-filtered.

        The trust boundary is the SQL itself: the WHERE clause admits a row iff
        (a) it is platform-scope OR its customer is one the caller's RBAC grants
        admit, AND (b) it is NOT a user-scope row owned by a DIFFERENT user --
        composed as ONE fragment from the shared
        :func:`threetears.agent.acl.three_scope_visibility_clause` (D10), evaluated
        over the ``system.platform.rbac`` proxy pool. Optional ``datasource_id``
        narrows to concepts anchored on a datasource; the finer
        ``datasource_table_id`` narrows to a single bound table. When
        ``datasource_id`` is a CUSTOMER datasource D linked to a platform-shared
        datasource P, the gather widens to ``datasource_id IN (D, P)`` (KNW-77). A
        LEFT JOIN carries the bound table's ``schema.table`` NAME through with the
        concept (the renderer emits the name, never the raw ``datasource_table_id``
        UUID the agent cannot resolve). ``status = 'active'`` excludes DRAFTS from
        retrieval (KNW-06 / D5).

        Cache-bypass: rbac-visibility filtered list -- a cross-table JOIN the by-pk
        Collection abstraction cannot express, so this reads through the proxy pool
        directly.

        :param user_id: caller (conversation end-user) UUID
        :ptype user_id: UUID
        :param datasource_id: optional datasource-anchor restriction
        :ptype datasource_id: UUID | None
        :param datasource_table_id: optional binding-domain restriction
        :ptype datasource_table_id: UUID | None
        :param customer_scope: the conversation's hub-authenticated customer the
            broker clamps this Class-B read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: caller-visible concept snapshots for the requested binding domain
            ordered by ``date_created`` ascending
        :rtype: list[ConceptSnapshot]
        """
        result: list[ConceptSnapshot] = []
        if self.l3_pool is not None:
            visibility_clause, params = three_scope_visibility_clause(
                user_id=user_id,
                customer_id_column="co.customer_id",
                user_id_column="co.user_id",
                param_offset=1,
            )
            sql = (
                "SELECT co.id, co.customer_id, co.user_id, "
                "co.origin_concept_id, co.name, co.aliases, co.definition, "
                "co.datasource_id, co.datasource_table_id, co.sql_fragment, "
                "co.caveats, co.tags, co.always_inject, "
                "dt.schema_name AS bound_schema_name, "
                "dt.table_name AS bound_table_name "
                "FROM concepts co "
                "LEFT JOIN datasource_tables dt "
                "ON dt.id = co.datasource_table_id "
                f"WHERE ({visibility_clause}) "
                "AND co.status = 'active' "
            )
            if datasource_id is not None:
                params.append(datasource_id)
                idx = len(params)
                sql += (
                    f" AND co.datasource_id IN (${idx}, "
                    f"(SELECT origin_datasource_id FROM datasources "
                    f"WHERE id = ${idx}))"
                )
            if datasource_table_id is not None:
                params.append(datasource_table_id)
                sql += f" AND co.datasource_table_id = ${len(params)}"
            sql += " ORDER BY co.date_created ASC"

            rows = await self.l3_pool.fetch(sql, *params, customer_scope=customer_scope)
            for row in rows:
                result.append(_row_to_concept_snapshot(dict(row)))
        return result

    async def list_own_drafts(
        self,
        user_id: UUID,
        *,
        customer_scope: UUID,
    ) -> list[DraftView]:
        """List the caller's OWN correction-harvest concept drafts (KNW-54).

        Returns ONLY ``status='draft'`` concepts owned by ``user_id`` over the
        rbac-read proxy pool. Private to the author; keyed strictly on ``user_id``
        ownership. The SOLE agent read path that surfaces draft concepts.

        Cache-bypass: owner+status predicate scan over the partial draft-owner
        index; not a by-pk read.

        :param user_id: authoring user UUID; only this user's drafts are returned
        :ptype user_id: UUID
        :param customer_scope: conversation customer the broker clamps this Class-B
            read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: the caller's own concept drafts as :class:`DraftView`
        :rtype: list[DraftView]
        """
        result: list[DraftView] = []
        if self.l3_pool is not None:
            sql = (
                "SELECT co.id, co.name, co.definition, co.datasource_id, "
                "co.conversation_id, co.turn_count "
                "FROM concepts co "
                "WHERE co.user_id = $1 AND co.status = 'draft' "
                "ORDER BY co.date_created ASC"
            )
            rows = await self.l3_pool.fetch(sql, user_id, customer_scope=customer_scope)
            for row in rows:
                data = dict(row)
                result.append(
                    DraftView(
                        draft_id=_as_uuid(data["id"]),  # type: ignore[arg-type]
                        target="concept",
                        title=data.get("name") or "",
                        body=data.get("definition") or "",
                        related_domain=str(data.get("datasource_id") or ""),
                        conversation_id=_as_uuid(data.get("conversation_id")),
                        turn_count=data.get("turn_count"),
                    ),
                )
        return result

    async def fetch_embeddings(
        self,
        ids: list[UUID],
        *,
        customer_scope: UUID,
    ) -> dict[UUID, list[float]]:
        """Fetch stored embeddings for the given concept ids (KNW-95).

        The concept mirror of :meth:`PlaybookEntryCollection.fetch_embeddings`: the
        SEPARATE, bounded vector read the situational ranker uses for the concept
        candidates about to be ranked, via the ``embedding::text`` cast.
        NULL-embedding ids are OMITTED so the ranker falls back to stable-order for
        them.

        :param ids: situational candidate concept ids to fetch vectors for
        :ptype ids: list[UUID]
        :param customer_scope: conversation customer the broker clamps this Class-B
            read to (broker-isolation-task-01)
        :ptype customer_scope: UUID
        :return: map of concept id to its embedding vector (NULL-embedding ids
            omitted)
        :rtype: dict[UUID, list[float]]
        """
        return await _fetch_embeddings(self.l3_pool, "concepts", ids, customer_scope=customer_scope)
