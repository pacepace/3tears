"""Agent-side three-tier knowledge collections -- playbook entries + concepts.

Re-implements the hub's playbook-entry / concept collections for the agent pod:
the agent owns its OWN collections over the SAME ``playbook_entries`` /
``concepts`` tables (reverse-importing the hub class is banned),
constructed over the ``system.platform.rbac`` NATS proxy pool. Each schema OMITS
the ``embedding`` column from the projection so the proxy pool never decodes a
pgvector value on the hot per-turn list path; the similarity ranker fetches
vectors separately via the bounded :meth:`fetch_embeddings` read (the
``embedding::text`` cast, the proxy read-path fix).

Both collections are built entirely from 3tears primitives:
:class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`,
:func:`threetears.agent.acl.visible_rows` (the SAME three-scope read rule
the hub's own listings read by), and the ``threetears.knowledge`` snapshot / scope
types. The only bespoke reads are :meth:`list_visible_to_user` (an
rbac-visibility filtered list, read by key) and :meth:`list_own_drafts`
(the author-private draft read); both bypass the by-pk L1/L2 caches: neither is a
read of one row by its primary key.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar
from uuid import UUID

from threetears.agent.acl import ACCESS_TABLES, visible_rows
from threetears.core.collections.delete_cascade import DeleteCascade
from threetears.core.collections.generation import WRITE_GENERATION
from threetears.core.collections.scan_cache import ScanCacheKey
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.keyed import Fragment, KeyBatch, Selected, Sort, read_by, read_joined, read_keyed, with_options
from threetears.knowledge import (
    ConceptSnapshot,
    EntrySnapshot,
    build_table_ref,
    derive_scope,
)
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.eviction import Declared, Policy

#: Every table the concept visibility scan reads, and every table the access decision that narrows
#: it reads. The access tables are not optional: the visibility clause JOINs role_assignments,
#: group_members and namespaces, and the broker narrows an agent's read through the roles
#: its grants hold (roles, reached through groups), so a revoked grant or a role that lost a
#: permission must evict the cached result. Declaring only the data table would turn a staleness
#: window into an authorization one. datasources is read by the origin-link subquery of a
#: datasource-scoped scan, so linking or unlinking a datasource evicts the widened (or narrowed)
#: knowledge set. Nothing here ages: a scan is cached only while every one of these tables is
#: followed (ScanCache).
_CONCEPT_SCAN_DEPENDS_ON: tuple[str, ...] = ("concepts", "datasource_tables", "datasources", *ACCESS_TABLES)

#: Same, for the entry scan.
_ENTRY_SCAN_DEPENDS_ON: tuple[str, ...] = ("playbook_entries", "datasources", *ACCESS_TABLES)

#: the tables beyond the access tables that a knowledge scan depends on: a process that caches
#: these scans follows them (``threetears.epoch.follow_tables``, from ``3tears-epoch``) beside the
#: access tables its ACL cache follows
KNOWLEDGE_SCAN_TABLES: tuple[str, ...] = ("playbook_entries", "concepts", "datasources", "datasource_tables")


def _scan_cache_for(collection: Any) -> Any:
    """return the pod's scan cache, or ``None`` when there is nowhere to cache.

    A collection can be constructed without a registry (bare, in tests, or in a
    minimal embedding). It still has to serve reads -- just uncached -- so this
    degrades to ``None`` rather than raising. Caching is an optimisation; the
    query is the contract.

    :param collection: the collection asking for its cache
    :ptype collection: Any
    :return: the registry's scan cache, or ``None``
    :rtype: Any
    """
    registry = collection.registry
    return None if registry is None else registry.scan_cache


from threetears.agent.knowledge.entities import ConceptEntity, PlaybookEntryEntity
from threetears.agent.knowledge.integration import DraftView
from threetears.core.tenancy import tenant_column

__all__ = ["KNOWLEDGE_SCAN_TABLES", "ConceptCollection", "PlaybookEntryCollection"]


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


# WHY the visible listings read through visible_rows: the listing admitted each row through a correlated ``EXISTS`` over
# the caller's grants (three_scope_visibility_clause), run for every row of the table it scanned. The caller's knowledge
# customers are now read once, by the caller's memberships through the grants' own indexes, and the rows a page at a
# time through the hub's ``<table>_visible_by_customer (customer_id, date_created, id)`` (hub v177): a platform row, or
# one of those customers, and never another user's own -- the three-scope read rule (D10), as the hub's own listings
# read it. Over the ``system.platform.rbac`` proxy pool, each statement clamped to the conversation's customer
# (broker-isolation-task-01). Equivalence: tests/integration/test_same_results_visible.py; plans: test_index_plans.py.

#: the order a visible listing reads in: oldest first, the ``id`` key breaking ties (the hub's ``OLDEST_FIRST``)
_OLDEST_FIRST: tuple[Sort, ...] = (Sort("date_created", cast="timestamptz"), Sort("id", cast="uuid"))


#: the columns an entry snapshot is built from, and the listing's order
_ENTRY_VISIBLE_COLUMNS: tuple[str, ...] = (
    "id",
    "customer_id",
    "user_id",
    "origin_entry_id",
    "title",
    "body",
    "tags",
    "datasource_id",
    "always_inject",
    "date_created",
)

#: the columns a concept snapshot is built from, and the listing's order; its bound table's name is joined after
_CONCEPT_VISIBLE_COLUMNS: tuple[str, ...] = (
    "id",
    "customer_id",
    "user_id",
    "origin_concept_id",
    "name",
    "aliases",
    "definition",
    "datasource_id",
    "datasource_table_id",
    "sql_fragment",
    "caveats",
    "tags",
    "always_inject",
    "date_created",
)


async def _with_bound_tables(pool: Any, rows: list[dict[str, Any]], *, customer_scope: UUID) -> None:
    """carry each concept's bound table's ``schema.table`` NAME (``bound_schema_name`` / ``bound_table_name``), None
    where it binds none or the table is not read: the renderer emits the name, never the raw ``datasource_table_id``
    the agent cannot resolve. One keyed read of ``datasource_tables`` by the ids the concepts hold, what the
    hand-written ``LEFT JOIN datasource_tables dt ON dt.id = co.datasource_table_id`` read beside them.

    :param pool: the rbac-read proxy pool
    :ptype pool: Any
    :param rows: the visible concepts, completed in place
    :ptype rows: list[dict[str, Any]]
    :param customer_scope: the conversation's customer the broker clamps the read to
    :ptype customer_scope: UUID
    """
    bound = [row for row in rows if row["datasource_table_id"] is not None]
    tables = (
        await read_joined(
            with_options(pool, customer_scope=customer_scope),
            bound,
            "datasource_tables",
            on={"id": "datasource_table_id"},
            columns=["id", "schema_name", "table_name"],
            casts={"id": "uuid"},
        )
        if bound
        else {}
    )
    for row in rows:
        (table, *_) = tables.get((row["datasource_table_id"],), [None])
        row["bound_schema_name"] = None if table is None else table["schema_name"]
        row["bound_table_name"] = None if table is None else table["table_name"]


def _active_anchored(datasource_id: UUID | None) -> list[Fragment]:
    """a visible knowledge listing's own filters: ``status = 'active'`` (correction-harvest drafts stay out of
    retrieval, KNW-06 / D5) and, given a datasource D, the rows anchored on D or on the platform datasource D's
    ``origin_datasource_id`` links to (KNW-77: ``IN (D, P)``; a NULL origin leaves ``{D}``, an unlinked datasource
    keeps the single-anchor behaviour).

    :param datasource_id: the anchoring datasource, or None for every anchor
    :ptype datasource_id: UUID | None
    :return: the ``where=`` filters, each from ``$1``
    :rtype: list[Fragment]
    """
    where = [Fragment("t.\"status\" = 'active'")]
    if datasource_id is not None:
        # the origin resolved in the same statement, a point read of the hub's ``datasources`` by its key on the
        # same rbac-read proxy pool: a fresh per-call read, no module state
        where.append(
            Fragment(
                # a datasource marked gone (ROW_STAMPED) names no origin
                't."datasource_id" IN ($1, (SELECT origin_datasource_id FROM datasources WHERE id = $1 AND NOT gone))',
                (datasource_id,),
            )
        )
    return where


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


#: what the vector read selects: the id, and the vector as its bracketed text (``embedding::text``)
_EMBEDDING_COLUMNS: tuple[str | Selected, ...] = ("id", Selected("t.embedding::text", "embedding"))


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
    :meth:`ConceptCollection.fetch_embeddings`: ``id, embedding::text`` of
    the ids, a keyed read by primary key (:func:`threetears.core.keyed.read_keyed`,
    the customer scope on each statement) over the rbac-read proxy pool. The ``::text`` cast returns the bracketed textual vector
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
        rows = await read_keyed(
            with_options(l3_pool, customer_scope=customer_scope),
            table_name,
            keys=KeyBatch(("id",), [list(ids)], ("uuid",)),
            columns=_EMBEDDING_COLUMNS,
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
    """Agent-side three-tier collection over the hub's ``playbook_entries``.

    Re-implements the hub's playbook-entry collection for the agent pod over the
    ``system.platform.rbac`` proxy pool. The schema OMITS the ``embedding`` column
    from the projection so the proxy pool never decodes a pgvector value; the
    similarity ranker fetches embeddings separately via :meth:`fetch_embeddings`.

    The only bespoke queries are :meth:`list_visible_to_user` (a caller-visibility
    filtered list whose WHERE clause is evaluated SQL-side -- the trust boundary is
    the SQL itself, never a python post-filter) and :meth:`list_own_drafts` (the
    author-private draft read).
    """

    tenancy = tenant_column("customer_id")

    #: switched on (epoch-task-06): its writes advance the table's write generation, so a cache
    #: derived from it (a visibility scan) is evicted when a broadcast is missed, not timed out
    # so its schema declares ``write_generation=WRITE_GENERATION``, which this class reads (one declaration)

    primary_key_column: str = "id"
    #: playbook_entries_origin_entry_id_fkey (hub v012, ``ON DELETE SET NULL``): a shadow loses its link to a deleted
    #: original. The row is marked gone in place (``ROW_STAMPED``), so the database's action never runs: 3tears sets
    #: each shadow's ``origin_entry_id`` NULL by key, its epoch risen, in the delete's transaction
    #: (``apply_delete_cascade``)
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (
        DeleteCascade("playbook_entries", "origin_entry_id", "SET NULL"),
    )
    schema = TableSchema(
        name="playbook_entries",
        write_generation=WRITE_GENERATION,
        # one row a key, carrying its last write's epoch (append-only L3, decision 3): a restore of a lost change log
        # reads its rows stamped since X a bucket at a time, and a delete marks a row gone in place, so the restore
        # hears it. Stamped, not versioned: the hub edits a row in place, a key at a time (its scope immutable through
        # an update, a promotion a new row), never a page of rows at once. Its DDL is the hub's, so the hub's migration
        # stamps it (threetears.agent.knowledge.stamp_tables.STAMPED_TABLES)
        row_epoch=ROW_STAMPED,
        # the tenant on the schema too, so the restore declaration names it
        tenancy=tenant_column("customer_id"),
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("playbook_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True),
            Column("user_id", UUID_TYPE, nullable=True),
            Column("visibility", STRING_TYPE, server_default="'private'"),
            Column("origin_entry_id", UUID_TYPE, nullable=True),
            Column("title", STRING_TYPE),
            Column("body", STRING_TYPE),
            Column("tags", JSONB_TYPE, server_default="'[]'::jsonb"),
            Column("datasource_id", UUID_TYPE),
            Column("always_inject", BOOL_TYPE, server_default="false"),
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
        # the hub's own-drafts index (hub v019): a user's drafts, the only rows it holds
        indexes=(
            Index(
                "playbook_entries_draft_owner_idx",
                "user_id",
                "status",
                where="status = 'draft'",
                purpose="packages/agent/knowledge/src/threetears/agent/knowledge/collections.py::PlaybookEntryCollection.list_own_drafts",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_draft_owner_indexes",
            ),
            # the hub's customer-led listing index (hub v177): a caller's visible entries, their customers' rows in
            # (date_created, id) order, a page at a time
            Index(
                "playbook_entries_visible_by_customer",
                "customer_id",
                "date_created",
                "id",
                asc=("date_created", "id"),
                purpose="packages/agent/knowledge/src/threetears/agent/knowledge/collections.py::PlaybookEntryCollection.list_visible_to_user",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_visible_by_customer_indexes",
            ),
            # the shadows a delete of an original reaches (its ``ON DELETE SET NULL`` on origin_entry_id, taken
            # by 3tears for a gone mark: delete_cascades above), read by read_delete_cascade's keyset read by
            # origin_entry_id: the rows naming the original alone. The hub's migration builds it (the table's DDL
            # is the hub's); declared here because 3tears reads by it
            Index(
                "playbook_entries_by_origin_entry",
                "origin_entry_id",
                where="origin_entry_id IS NOT NULL",
                purpose="packages/core/src/threetears/core/collections/delete_cascade.py::_children",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_origin_indexes",
            ),
        ),
        cas_column="date_updated",
        l1_eviction=Declared(Policy.EVICTABLE, "rows cached on a read by key; a miss reads L2/L3 and refills"),
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

        The trust boundary is the SQL itself: each page admits a row iff
        (a) it is platform-scope (``customer_id IS NULL``) OR its customer is one
        the caller's RBAC grants admit, AND (b) it is NOT a user-scope row owned by
        a DIFFERENT user (``user_id IS NULL OR user_id = <caller>``) -- the
        three-scope read rule (D10) as :func:`threetears.agent.acl.visible_rows`
        reads it by key, over the ``system.platform.rbac``
        proxy pool.
        Optional ``datasource_id`` narrows the result to the anchoring datasource;
        ``None`` returns every visible entry. When ``datasource_id`` is a CUSTOMER
        datasource D linked to a canonical platform-shared datasource P
        (``D.origin_datasource_id = P``), the gather widens to
        ``datasource_id IN (D, P)`` (KNW-77) so platform-scope knowledge anchored
        on P composes into D's effective view; D's own customer/user entries stay
        private to D. ``status = 'active'`` excludes correction-harvest DRAFTS from
        retrieval (KNW-06 / D5).

        Cache-bypass: rbac-visibility filtered list -- the caller's grants, then the
        rows they admit, which no by-pk Collection read expresses, so this reads
        through the proxy pool directly.

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
            cache = _scan_cache_for(self)
            cache_key = ScanCacheKey("playbook_entries", user_id, datasource_id, customer_scope)
            cached = None if cache is None else cache.get(cache_key)
            if cached is None:
                # the token is taken BEFORE the read: a write evicted while the read is in
                # flight makes put() refuse, instead of caching the pre-write rows where
                # that eviction can no longer reach them.
                token = None if cache is None else cache.begin_read(_ENTRY_SCAN_DEPENDS_ON)
                cached = await visible_rows(
                    with_options(self.l3_pool, customer_scope=customer_scope),
                    "playbook_entries",
                    columns=_ENTRY_VISIBLE_COLUMNS,
                    user_id=user_id,
                    namespace_type="knowledge",
                    customer_column="customer_id",
                    user_column="user_id",
                    order=_OLDEST_FIRST,
                    where=_active_anchored(datasource_id),
                )
                if cache is not None and token is not None:
                    cache.put(cache_key, cached, token=token)
            for row in cached:
                result.append(_row_to_snapshot(row))
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
            rows = await read_by(
                with_options(self.l3_pool, customer_scope=customer_scope),
                "playbook_entries",
                by={"user_id": user_id, "status": "draft"},
                columns=("id", "title", "body", "datasource_id", "conversation_id", "turn_count"),
                records=True,
                order=[Sort("date_created")],
            )
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
    """Agent-side three-tier collection over the hub's ``concepts``.

    Re-implements the hub's concept collection for the agent pod over the
    ``system.platform.rbac`` proxy pool. The schema OMITS the ``embedding`` column
    from the projection so the proxy pool never decodes a pgvector value; the
    similarity ranker fetches vectors separately via :meth:`fetch_embeddings`.

    The only bespoke queries are :meth:`list_visible_to_user` (a caller-visibility
    filtered list evaluated SQL-side, exactly mirroring the entry collection) and
    :meth:`list_own_drafts` (the author-private draft read).
    """

    tenancy = tenant_column("customer_id")

    #: switched on (epoch-task-06): its writes advance the table's write generation, so a cache
    #: derived from it (a visibility scan) is evicted when a broadcast is missed, not timed out
    # so its schema declares ``write_generation=WRITE_GENERATION``, which this class reads (one declaration)

    primary_key_column: str = "id"
    #: concepts_origin_concept_id_fkey (hub v013, ``ON DELETE SET NULL``): a shadow loses its link to a deleted
    #: original. The row is marked gone in place (``ROW_STAMPED``), so the database's action never runs: 3tears sets
    #: each shadow's ``origin_concept_id`` NULL by key, its epoch risen, in the delete's transaction
    #: (``apply_delete_cascade``)
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (DeleteCascade("concepts", "origin_concept_id", "SET NULL"),)
    schema = TableSchema(
        name="concepts",
        write_generation=WRITE_GENERATION,
        # one row a key, carrying its last write's epoch (append-only L3, decision 3): a restore of a lost change log
        # reads its rows stamped since X a bucket at a time, and a delete marks a row gone in place, so the restore
        # hears it. Stamped, not versioned: the hub edits a row in place, a key at a time (its scope immutable through
        # an update, a promotion a new row), never a page of rows at once. Its DDL is the hub's, so the hub's migration
        # stamps it (threetears.agent.knowledge.stamp_tables.STAMPED_TABLES)
        row_epoch=ROW_STAMPED,
        # the tenant on the schema too, so the restore declaration names it
        tenancy=tenant_column("customer_id"),
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, nullable=True),
            Column("user_id", UUID_TYPE, nullable=True),
            Column("visibility", STRING_TYPE, server_default="'private'"),
            Column("origin_concept_id", UUID_TYPE, nullable=True),
            Column("name", STRING_TYPE),
            Column("aliases", JSONB_TYPE, server_default="'[]'::jsonb"),
            Column("definition", STRING_TYPE),
            Column("datasource_id", UUID_TYPE),
            Column("datasource_table_id", UUID_TYPE, nullable=True),
            Column("sql_fragment", STRING_TYPE, nullable=True),
            Column("caveats", STRING_TYPE, nullable=True),
            Column("tags", JSONB_TYPE, server_default="'[]'::jsonb"),
            Column("always_inject", BOOL_TYPE, server_default="false"),
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
        # the hub's own-drafts index (hub v019): a user's drafts, the only rows it holds
        indexes=(
            Index(
                "concepts_draft_owner_idx",
                "user_id",
                "status",
                where="status = 'draft'",
                purpose="packages/agent/knowledge/src/threetears/agent/knowledge/collections.py::ConceptCollection.list_own_drafts",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_draft_owner_indexes",
            ),
            # the hub's customer-led listing index (hub v177): a caller's visible concepts, their customers' rows in
            # (date_created, id) order, a page at a time
            Index(
                "concepts_visible_by_customer",
                "customer_id",
                "date_created",
                "id",
                asc=("date_created", "id"),
                purpose="packages/agent/knowledge/src/threetears/agent/knowledge/collections.py::ConceptCollection.list_visible_to_user",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_visible_by_customer_indexes",
            ),
            # the shadows a delete of an original reaches (its ``ON DELETE SET NULL`` on origin_concept_id, taken
            # by 3tears for a gone mark: delete_cascades above), read by read_delete_cascade's keyset read by
            # origin_concept_id: the rows naming the original alone. The hub's migration builds it (the table's DDL
            # is the hub's); declared here because 3tears reads by it
            Index(
                "concepts_by_origin_concept",
                "origin_concept_id",
                where="origin_concept_id IS NOT NULL",
                purpose="packages/core/src/threetears/core/collections/delete_cascade.py::_children",
                plan_test="packages/agent/knowledge/tests/integration/test_index_plans.py::test_origin_indexes",
            ),
        ),
        cas_column="date_updated",
        l1_eviction=Declared(Policy.EVICTABLE, "rows cached on a read by key; a miss reads L2/L3 and refills"),
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

        The trust boundary is the SQL itself: each page admits a row iff
        (a) it is platform-scope OR its customer is one the caller's RBAC grants
        admit, AND (b) it is NOT a user-scope row owned by a DIFFERENT user -- the
        three-scope read rule (D10) as :func:`threetears.agent.acl.visible_rows`
        reads it by key, over the ``system.platform.rbac``
        proxy pool. Optional ``datasource_id``
        narrows to concepts anchored on a datasource; the finer
        ``datasource_table_id`` narrows to a single bound table. When
        ``datasource_id`` is a CUSTOMER datasource D linked to a platform-shared
        datasource P, the gather widens to ``datasource_id IN (D, P)`` (KNW-77). A
        keyed read of the bound tables (:func:`_with_bound_tables`) carries the bound
        table's ``schema.table`` NAME through with the concept (the renderer emits the name, never the raw ``datasource_table_id``
        UUID the agent cannot resolve). ``status = 'active'`` excludes DRAFTS from
        retrieval (KNW-06 / D5).

        Cache-bypass: rbac-visibility filtered list -- the caller's grants, then the
        rows they admit, which no by-pk Collection read expresses, so this reads
        through the proxy pool directly.

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
            cache = _scan_cache_for(self)
            cache_key = ScanCacheKey(
                "concepts",
                user_id,
                datasource_id,
                datasource_table_id,
                customer_scope,
            )
            cached = None if cache is None else cache.get(cache_key)
            if cached is None:
                # the token is taken BEFORE the read: a write evicted while the read is in
                # flight makes put() refuse, instead of caching the pre-write rows where
                # that eviction can no longer reach them.
                token = None if cache is None else cache.begin_read(_CONCEPT_SCAN_DEPENDS_ON)
                where = _active_anchored(datasource_id)
                if datasource_table_id is not None:
                    where.append(Fragment('t."datasource_table_id" = $1', (datasource_table_id,)))
                cached = await visible_rows(
                    with_options(self.l3_pool, customer_scope=customer_scope),
                    "concepts",
                    columns=_CONCEPT_VISIBLE_COLUMNS,
                    user_id=user_id,
                    namespace_type="knowledge",
                    customer_column="customer_id",
                    user_column="user_id",
                    order=_OLDEST_FIRST,
                    where=where,
                )
                await _with_bound_tables(self.l3_pool, cached, customer_scope=customer_scope)
                if cache is not None and token is not None:
                    cache.put(cache_key, cached, token=token)
            for row in cached:
                result.append(_row_to_concept_snapshot(row))
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
            rows = await read_by(
                with_options(self.l3_pool, customer_scope=customer_scope),
                "concepts",
                by={"user_id": user_id, "status": "draft"},
                columns=("id", "name", "definition", "datasource_id", "conversation_id", "turn_count"),
                records=True,
                order=[Sort("date_created")],
            )
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
