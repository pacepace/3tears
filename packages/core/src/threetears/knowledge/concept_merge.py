"""read-time merge across the three scopes of concepts (D4 / D5).

a concept is a governed mapping from a business term ("active users",
"early vote") to its canonical data binding: a definition prose, an
optional datasource-table reference + SQL-fragment binding, caveats,
and a name + aliases the retrieval path matches on. concepts ride the
same three-scope ladder and whole-entry shadow mechanics as playbook
entries (knowledge-task-02): a concept that carries an
``origin_concept_id`` SHADOWS the ancestor it points at — whole-entry
replace at read time, nearest scope wins (user > customer > platform,
D4). the group-by-root + nearest-scope-wins WALK is the shared
:func:`threetears.knowledge.chains.resolve_shadow_chains`; this module
only adapts concept snapshots onto it and surfaces the concept-shaped
effective / layered views.

AMBIGUITY (knowledge-task-03 Implementation-Note 2): the SAME concept
name at DIFFERENT scopes WITHOUT an origin link is allowed but is a
silent-divergence hazard (D5) — two governed definitions of "active
users" competing with no declared shadow. the effective view flags such
a winner ``ambiguous: True`` so the agent and the provenance footer can
disclose the competing definitions rather than silently picking one
(the anti-pattern the shard explicitly forbids). a name collision is
detected after shadow resolution: if two SURVIVING winners share a
case-insensitive name, every winner with that name is flagged. a name
collision that resolved via a declared shadow link collapses to one
winner; that winner is still flagged ambiguous if it competes by name
with a SEPARATE unlinked winner (name_counts counts EVERY surviving
winner, so a shadow winner that shares a name with an independent
surviving winner is correctly flagged). a collision WITHIN one scope is
a WRITE-time 409 (the hub enforces it), so it never reaches this merge.

the ``always_inject`` invariant flag (KNW-25 / D9) rides through shadow
resolution like any field: the nearest-scope winner's flag governs. it
is surfaced on :class:`ConceptEffective` so the SDK retrieval node can
honour the trim-exemption tier; the budget logic lives in the SDK node,
not here.

this module is the SINGLE authority on the concept merge shared by the
hub (eval fingerprint builder — task-04) and the SDK (turn-time
retrieval node): both import :func:`merge_concept_views` from here so a
fingerprint and a live turn agree byte-for-byte on the effective view.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from threetears.observe import get_logger

from threetears.knowledge.chains import resolve_shadow_chains
from threetears.knowledge.scope import Scope

__all__ = [
    "ConceptEffective",
    "ConceptLayered",
    "ConceptSnapshot",
    "merge_concept_views",
]

log = get_logger(__name__)


@dataclass(frozen=True)
class ConceptSnapshot:
    """immutable snapshot of one concept row for the merge.

    the merge consumes snapshots, not live entities, so the algorithm
    is pure and unit-testable without a Collection. every field is the
    value at that concept's own scope; ``origin_concept_id`` is the
    shadow link (``None`` for an unlinked concept).

    the binding (``datasource_table_id`` + ``sql_fragment``) is curated
    raw material for the model — ``sql_fragment`` is a filter / expression
    fragment, NOT a runnable query, and is never executed or interpolated
    by the platform (knowledge-task-03 Note 3); SQL safety stays at the
    datasource tool boundary.

    :ivar id: concept id
    :ivar scope: derived scope of this concept (user / customer / platform)
    :ivar origin_concept_id: ancestor this concept shadows, or ``None``
    :ivar name: canonical business term (e.g. "active users")
    :ivar aliases: alternate names the retrieval path also matches on
    :ivar definition: definition prose (consumed verbatim; never spliced)
    :ivar datasource_table_id: bound datasource table, or ``None`` (an
        unbound concept carries definition + caveats only)
    :ivar sql_fragment: curated filter / expression fragment, or ``None``
        (NOT a runnable query; never executed server-side)
    :ivar caveats: caveats prose, or ``None``
    :ivar tags: routing / search tags
    :ivar always_inject: KNW-25 invariant flag (rides through shadow
        resolution like any field)
    """

    id: UUID
    scope: Scope
    origin_concept_id: UUID | None = None
    name: str = ""
    aliases: tuple[str, ...] = ()
    definition: str = ""
    datasource_table_id: UUID | None = None
    sql_fragment: str | None = None
    caveats: str | None = None
    tags: tuple[str, ...] = ()
    always_inject: bool = False


@dataclass(frozen=True)
class ConceptEffective:
    """effective (winner) view of one shadow chain returned to callers.

    the winner is the nearest-scope member of the chain (D4). when the
    winner shadows an ancestor, ``shadows_scope`` names the scope of the
    NEAREST shadowed ancestor so the task-07 provenance footer can
    disclose the override (D5). ``always_inject`` is the winner's own
    flag (KNW-25).

    ``ambiguous`` is the concept-specific divergence flag (Note 2): it is
    ``True`` when another SURVIVING winner shares this winner's
    case-insensitive name WITHOUT a declared shadow link between them —
    two competing governed definitions the agent must disclose rather
    than silently pick. a shadowed collision (one winner, the rest
    folded under it) is never ambiguous.

    :ivar concept: the winning concept snapshot (emitted verbatim)
    :ivar shadows_scope: scope of the nearest shadowed ancestor, or
        ``None`` when the winner shadows nothing
    :ivar ambiguous: ``True`` when a competing same-name unlinked winner
        survives the merge (D5 silent-divergence disclosure)
    """

    concept: ConceptSnapshot
    shadows_scope: Scope | None = None
    ambiguous: bool = False

    @property
    def always_inject(self) -> bool:
        """expose the winner's invariant flag at the effective view.

        the SDK retrieval node reads this to decide whether the concept
        is exempt from similarity-trim / token-budget eviction (KNW-25).

        :return: the winning concept's ``always_inject`` flag
        :rtype: bool
        """
        return self.concept.always_inject


@dataclass(frozen=True)
class ConceptLayered:
    """layered provenance view of one shadow chain (D5).

    retains the winner plus the shadowed ancestors (nearest-first) with
    their scope labels so a UI / review surface can show "this user
    concept shadows the customer concept: ...". the read-time effective
    view drops the ancestors; the layered view keeps them for display.

    :ivar effective: the winner (same object the effective view emits)
    :ivar shadowed: shadowed ancestors ordered nearest-scope-first
        (the member directly under the winner first, the chain root
        last); empty for an unlinked concept
    """

    effective: ConceptEffective
    shadowed: tuple[ConceptSnapshot, ...] = field(default_factory=tuple)


def merge_concept_views(
    concepts: list[ConceptSnapshot],
) -> tuple[list[ConceptEffective], list[ConceptLayered]]:
    """merge a domain's caller-visible concepts into effective + layered.

    additive union of the supplied concepts, then whole-entry shadow
    resolution per chain via the shared
    :func:`threetears.knowledge.chains.resolve_shadow_chains` (the same
    walk the entry merge uses): concepts linked by ``origin_concept_id``
    are grouped into chains, and each chain emits exactly its
    nearest-scope member (D4). unlinked concepts each form a
    single-member chain and pass through untouched.

    after resolution, AMBIGUITY is flagged (Note 2): any surviving
    winner whose case-insensitive name is shared by another surviving
    winner is marked ``ambiguous`` — two competing governed definitions
    with no declared shadow, which the agent and footer must disclose
    (D5). a name collision that resolved via a declared shadow link
    collapses to one winner; that winner is still flagged ambiguous if
    it competes by name with a SEPARATE unlinked winner.

    the supplied list is the ALREADY-FILTERED caller-visible set for one
    domain — visibility + domain routing happen in the serving SQL, not
    here. the merge is a pure function of the snapshots.

    returns both views together (effective for injection, layered for
    the provenance footer) so callers do not double-walk the data.
    ordering is stable: chains are emitted in ascending winner scope
    rank (platform first, then customer, then user), ties broken by
    winner id.

    :param concepts: caller-visible concept snapshots for one domain
    :ptype concepts: list[ConceptSnapshot]
    :return: tuple ``(effective_views, layered_views)`` aligned by index
    :rtype: tuple[list[ConceptEffective], list[ConceptLayered]]
    """
    resolved = resolve_shadow_chains(
        concepts,
        id_of=lambda c: c.id,
        scope_of=lambda c: c.scope,
        origin_id_of=lambda c: c.origin_concept_id,
    )

    # ambiguity: count case-insensitive winner names AFTER shadow
    # resolution. a name held by >1 surviving winner means competing
    # unlinked definitions (a shadowed collision already folded to one
    # winner, so it never doubles up here). blank names are not counted
    # as collisions.
    name_counts: Counter[str] = Counter(
        chain.winner.name.strip().lower()
        for chain in resolved
        if chain.winner.name.strip()
    )

    effective_views: list[ConceptEffective] = []
    layered_views: list[ConceptLayered] = []
    for chain in resolved:
        normalized = chain.winner.name.strip().lower()
        ambiguous = bool(normalized) and name_counts[normalized] > 1
        effective = ConceptEffective(
            concept=chain.winner,
            shadows_scope=chain.shadows_scope,
            ambiguous=ambiguous,
        )
        effective_views.append(effective)
        layered_views.append(
            ConceptLayered(effective=effective, shadowed=chain.shadowed),
        )
    return effective_views, layered_views
