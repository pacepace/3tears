"""read-time merge across the three scopes of playbook entries (D4).

the effective view of a domain's playbook knowledge is the ADDITIVE
UNION of the caller-visible platform + customer + user entries
attached to that domain (KNW-11). an entry that carries an
``origin_entry_id`` SHADOWS the ancestor it points at: whole-body
replace at read time, nearest scope wins (user > customer > platform,
D4). entries WITHOUT an origin link pass through untouched — two
contradictory unlinked entries are both emitted (the author chose not
to declare a shadow, so neither is suppressed).

shadow resolution is WHOLE-ENTRY: the nearest-scope member of a chain
is emitted verbatim and the rest are dropped from the effective view.
entry bodies are NEVER concatenated or spliced — an entry body is one
atomic procedure and merging two procedures hands the model
contradictory instructions (D4). this module deliberately reuses the
effective/layered SPLIT pattern (``XxxEffective`` / ``XxxLayered``)
but NOT any per-field merge rules.

the ``always_inject`` invariant flag (KNW-17 / D9) RIDES THROUGH shadow
resolution like any other field: the nearest-scope winner's flag
governs. a user may shadow a platform invariant with a non-invariant
override (or the reverse) — the merge does not special-case the flag.
the flag is surfaced on :class:`EntryEffective` so the SDK retrieval
node can honour the trim-exemption tier; the budget logic itself lives
in the SDK node, not here.

cycle safety is a WRITE-time concern: :func:`assert_no_origin_cycle`
walks the prospective ``origin_entry_id`` chain with a visited set and
bounded depth, rejecting a cycle before it can be persisted. the
read-time merge therefore assumes acyclic chains; it still guards its
own walk with a visited set so a corrupted row never spins the merge.

this module is the SINGLE authority on the entry merge shared by the
hub (eval fingerprint builder) and the SDK (turn-time retrieval node):
both import :func:`merge_entry_views` from here so a fingerprint and a
live turn agree byte-for-byte on the effective view (knowledge-task-04).
the group-by-root + nearest-scope-wins WALK itself is shared one level
down in :mod:`threetears.knowledge.chains`
(:func:`resolve_shadow_chains`) so the concept merge (knowledge-task-03)
reuses the identical resolution; this module only adapts entries onto
that generic walk and surfaces the entry-shaped effective / layered
views.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from threetears.observe import get_logger

from threetears.knowledge.chains import resolve_shadow_chains
from threetears.knowledge.scope import Scope

__all__ = [
    "MAX_SHADOW_CHAIN_DEPTH",
    "EntryEffective",
    "EntryEnforcement",
    "EntryLayered",
    "EntrySnapshot",
    "OriginCycleError",
    "assert_no_origin_cycle",
    "merge_entry_views",
]

log = get_logger(__name__)


#: bound on shadow-chain length. a chain longer than this is treated as
#: pathological (likely a cycle or an abuse) and rejected at write time.
#: three real scopes (user -> customer -> platform) need a depth of at
#: most three links; the generous bound leaves room without admitting a
#: runaway walk.
MAX_SHADOW_CHAIN_DEPTH = 32


class OriginCycleError(ValueError):
    """raised when an ``origin_entry_id`` chain forms a cycle.

    rejected at WRITE time: a cyclic shadow chain has no nearest-scope
    member and would spin the read-time walk. the workflow surfaces
    this as a 4xx at the create / update endpoint rather than
    persisting an un-resolvable row.
    """


class EntryEnforcement(BaseModel):
    """machine-checkable query constraint carried by a governing entry.

    the deterministic query guard (query-enforcement-task-03) reads this
    structured shape to reject an aggregate over the governed ``table``
    that omits the ``required_predicates``; the human-readable entry
    ``body`` is untouched (the verify node still reads it). this is the
    MINIMAL vocabulary the real traps need — it is deliberately NOT a
    general SQL policy language; grow it only when a concrete trap needs
    a new field.

    a Pydantic ``BaseModel`` (not a stdlib dataclass) so it round-trips
    through FastAPI request validation, the OpenAPI schema, AND the SDK
    codegen — the path proven for ``WorkspaceConfig`` (codegen follows
    ``BaseModel`` / ``Enum`` refs, not dataclasses). serialized at the
    persistence + HTTP borders via :meth:`pydantic.BaseModel.model_dump`
    and reconstructed via :meth:`pydantic.BaseModel.model_validate`.

    ``frozen=True`` makes the constraint object immutable (matching
    :class:`EntrySnapshot`'s frozen-dataclass immutability); nothing
    hashes the snapshot (``required_predicates`` is a list, so a snapshot
    carrying enforcement is unhashable regardless).

    :ivar table: table within the entry's datasource this rule governs
    :ivar required_predicates: column names that MUST appear in a filter
        on any aggregate over ``table`` (matched case-insensitively by
        the analyzer)
    :ivar forbidden_bare_aggregate: reject ``SUM`` / ``COUNT`` / ``AVG``
        / ``MIN`` / ``MAX`` over ``table`` lacking the required predicates
    :ivar canonical_sql: the single correct query for the governed total,
        returned verbatim as the suggested fix (``None`` when there is no
        canonical form)
    :ivar note: short human reason, surfaced in the rejection message
    """

    model_config = ConfigDict(frozen=True)

    table: str
    required_predicates: list[str]
    forbidden_bare_aggregate: bool
    # optional: a table with no single canonical total (e.g. a tall
    # metric table queried per-metric) carries none. without the default
    # pydantic treats ``str | None`` as required-but-nullable, which 422s a
    # constraint that legitimately omits it.
    canonical_sql: str | None = None
    note: str


@dataclass(frozen=True)
class EntrySnapshot:
    """immutable snapshot of one playbook-entry row for the merge.

    the merge consumes snapshots, not live entities, so the algorithm
    is pure and unit-testable without a Collection. every field is the
    value at that entry's own scope; ``origin_entry_id`` is the shadow
    link (``None`` for an unlinked entry).

    :ivar id: entry id
    :ivar scope: derived scope of this entry (user / customer / platform)
    :ivar origin_entry_id: ancestor this entry shadows, or ``None``
    :ivar title: short procedure title
    :ivar body: atomic procedure body (consumed verbatim; never spliced)
    :ivar tags: routing / search tags
    :ivar always_inject: KNW-17 invariant flag (rides through shadow
        resolution like any field)
    :ivar datasource_id: attached datasource domain, or ``None``
    :ivar namespace_id: attached namespace domain, or ``None``
    :ivar enforcement: optional machine-checkable query constraint the
        deterministic guard reads (query-enforcement-task-01); ``None``
        (the default for every existing entry) means no deterministic
        guard — the verify node still reads the entry ``body``. a
        snapshot-level passthrough: the merge applies NO per-field
        semantics to it (the winner's value governs, like
        ``always_inject``)
    """

    id: UUID
    scope: Scope
    origin_entry_id: UUID | None = None
    title: str = ""
    body: str = ""
    tags: tuple[str, ...] = ()
    always_inject: bool = False
    datasource_id: UUID | None = None
    namespace_id: UUID | None = None
    enforcement: EntryEnforcement | None = None


@dataclass(frozen=True)
class EntryEffective:
    """effective (winner) view of one shadow chain returned to callers.

    the winner is the nearest-scope member of the chain (D4). when the
    winner shadows an ancestor, ``shadows_scope`` names the scope of
    the NEAREST shadowed ancestor so the task-07 provenance footer can
    disclose the override (D5). ``always_inject`` is the winner's own
    flag (KNW-17) — the SDK retrieval node reads it to exempt the entry
    from similarity-trim.

    :ivar entry: the winning entry snapshot (emitted verbatim)
    :ivar shadows_scope: scope of the nearest shadowed ancestor, or
        ``None`` when the winner shadows nothing
    """

    entry: EntrySnapshot
    shadows_scope: Scope | None = None

    @property
    def always_inject(self) -> bool:
        """expose the winner's invariant flag at the effective view.

        the SDK retrieval node reads this to decide whether the entry
        is exempt from similarity-trim / token-budget eviction (KNW-17).

        :return: the winning entry's ``always_inject`` flag
        :rtype: bool
        """
        return self.entry.always_inject


@dataclass(frozen=True)
class EntryLayered:
    """layered provenance view of one shadow chain (KNW-12 / D5).

    retains the winner plus the shadowed ancestors (nearest-first) with
    their scope labels so a UI / review surface can show "this user
    entry shadows the platform entry: ...". the read-time effective
    view drops the ancestors; the layered view keeps them for display.

    :ivar effective: the winner (same object the effective view emits)
    :ivar shadowed: shadowed ancestors ordered nearest-scope-first
        (the member directly under the winner first, the chain root
        last); empty for an unlinked entry
    """

    effective: EntryEffective
    shadowed: tuple[EntrySnapshot, ...] = field(default_factory=tuple)


def assert_no_origin_cycle(
    *,
    entry_id: UUID,
    origin_entry_id: UUID | None,
    origin_of: Mapping[UUID, UUID | None],
) -> None:
    """reject an ``origin_entry_id`` assignment that would form a cycle.

    walks the prospective chain from the proposed ``origin_entry_id``
    upward through ``origin_of`` (the ``id -> origin_entry_id`` map of
    the already-persisted entries), with a visited set seeded by
    ``entry_id`` so a self-link or any back-link onto the entry being
    written is caught. also rejects a chain that exceeds
    :data:`MAX_SHADOW_CHAIN_DEPTH` (a runaway / pathological link).

    called at WRITE time by the entry create / update path; the source
    entry is never mutated (the check is read-only over the map).

    :param entry_id: id of the entry being written (its own link is
        ``origin_entry_id``)
    :ptype entry_id: UUID
    :param origin_entry_id: the prospective shadow link, or ``None``
        (``None`` is trivially acyclic)
    :ptype origin_entry_id: UUID | None
    :param origin_of: map of every persisted entry id to its
        ``origin_entry_id`` (used to walk ancestors)
    :ptype origin_of: Mapping[UUID, UUID | None]
    :return: nothing on success
    :rtype: None
    :raises OriginCycleError: when the chain revisits ``entry_id`` or
        any earlier node, or exceeds the depth bound
    """
    if origin_entry_id is None:
        return
    visited: set[UUID] = {entry_id}
    cursor: UUID | None = origin_entry_id
    depth = 0
    while cursor is not None:
        depth += 1
        if depth > MAX_SHADOW_CHAIN_DEPTH:
            raise OriginCycleError(
                f"origin_entry_id chain from {entry_id} exceeds max depth {MAX_SHADOW_CHAIN_DEPTH} (likely a cycle)",
            )
        if cursor in visited:
            raise OriginCycleError(
                f"origin_entry_id cycle detected writing {entry_id}: chain revisits {cursor}",
            )
        visited.add(cursor)
        cursor = origin_of.get(cursor)


def merge_entry_views(
    entries: list[EntrySnapshot],
) -> tuple[list[EntryEffective], list[EntryLayered]]:
    """merge a domain's caller-visible entries into effective + layered.

    additive union of the supplied entries (KNW-11), then whole-entry
    shadow resolution per chain: entries linked by ``origin_entry_id``
    are grouped into chains, and each chain emits exactly its
    nearest-scope member (D4). unlinked entries each form a
    single-member chain and pass through untouched.

    the group-by-root + nearest-scope-wins walk is delegated to the
    shared :func:`threetears.knowledge.chains.resolve_shadow_chains` (the
    same walk the concept merge uses); this function only adapts entry
    snapshots onto that walk and wraps the resolved chains into the
    entry-shaped :class:`EntryEffective` / :class:`EntryLayered` views.

    the supplied list is the ALREADY-FILTERED caller-visible set for one
    domain — visibility + domain routing happen in the serving SQL, not
    here. the merge is a pure function of the snapshots.

    returns both views together so callers that need both (effective
    for injection, layered for the provenance footer) do not double-walk
    the data. ordering of the returned lists is stable: chains are
    emitted in ascending winner scope rank (platform first, then
    customer, then user) and ties broken by winner id, so two merges of
    the same input produce the same order.

    :param entries: caller-visible entry snapshots for one domain
    :ptype entries: list[EntrySnapshot]
    :return: tuple ``(effective_views, layered_views)`` aligned by index
    :rtype: tuple[list[EntryEffective], list[EntryLayered]]
    """
    resolved = resolve_shadow_chains(
        entries,
        id_of=lambda e: e.id,
        scope_of=lambda e: e.scope,
        origin_id_of=lambda e: e.origin_entry_id,
    )

    effective_views: list[EntryEffective] = []
    layered_views: list[EntryLayered] = []
    for chain in resolved:
        effective = EntryEffective(
            entry=chain.winner,
            shadows_scope=chain.shadows_scope,
        )
        effective_views.append(effective)
        layered_views.append(
            EntryLayered(effective=effective, shadowed=chain.shadowed),
        )
    return effective_views, layered_views
