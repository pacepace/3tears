"""generic whole-entry shadow-chain resolution shared across knowledge types.

the knowledge layer has two read-time mergeable entity types so far —
playbook entries (knowledge-task-02) and concepts (knowledge-task-03) —
and both resolve shadow chains identically: group the caller-visible
snapshots into chains by ``origin_*_id`` lineage, emit exactly the
nearest-scope member of each chain (user > customer > platform, D4),
and retain the shadowed ancestors nearest-first for the layered
provenance view (D5).

the resolution is WHOLE-ENTRY: the winner is emitted verbatim and the
rest are dropped from the effective view — bodies / definitions are
never concatenated or spliced (D4). an invariant flag (``always_inject``,
KNW-17 / D9) rides through on the winner with no special-casing; this
module does not read it, the type-specific wrapper carries it on its
own snapshot.

this module is the SINGLE authority on the chain-resolution walk. the
entry merge (:func:`threetears.knowledge.merge.merge_entry_views`) and
the concept merge
(:func:`threetears.knowledge.concept_merge.merge_concept_views`) both
consume :func:`resolve_shadow_chains` rather than re-implementing the
group-by-root + nearest-scope-wins walk — a single walk so a hub
fingerprint and a live turn agree byte-for-byte on the effective view
(knowledge-task-04).

the walk is parameterized by three pure getters over the opaque
snapshot type ``T`` (``id``, ``scope``, ``origin id``) so a caller of
any snapshot shape reuses it without this module importing the concrete
types. cycle safety is a WRITE-time concern (:func:`assert_no_origin_cycle`
in :mod:`threetears.knowledge.merge`); the read-time walk still guards
itself with a visited set so a corrupted row never spins the merge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar
from uuid import UUID

from threetears.observe import get_logger

from threetears.knowledge.scope import Scope

__all__ = [
    "ResolvedChain",
    "resolve_shadow_chains",
]

log = get_logger(__name__)


#: scope precedence for "nearest scope wins": higher integer = nearer
#: the caller. user shadows customer shadows platform (D4).
_SCOPE_RANK: Mapping[Scope, int] = {
    Scope.PLATFORM: 0,
    Scope.CUSTOMER: 1,
    Scope.USER: 2,
}


#: opaque snapshot type the resolver walks; the caller supplies pure
#: getters for the three fields the walk needs (id / scope / origin id).
T = TypeVar("T")


@dataclass(frozen=True)
class ResolvedChain[T]:
    """one resolved shadow chain: the winner plus its shadowed ancestors.

    the winner is the nearest-scope member of the chain (D4). the
    shadowed ancestors are ordered nearest-scope-first (the member
    directly under the winner first, the chain root last); empty for an
    unlinked snapshot. ``shadows_scope`` is the scope of the nearest
    shadowed ancestor, or ``None`` when the winner shadows nothing — the
    type-specific effective view surfaces it for the task-07 provenance
    footer (D5).

    the type-specific merge functions wrap this into their own
    ``XxxEffective`` / ``XxxLayered`` shapes; this dataclass is the
    shared intermediate, never returned to API callers directly.

    :ivar winner: nearest-scope member of the chain (emitted verbatim)
    :ivar shadowed: shadowed ancestors ordered nearest-scope-first
    :ivar shadows_scope: scope of the nearest shadowed ancestor, or
        ``None``
    """

    winner: T
    shadowed: tuple[T, ...]
    shadows_scope: Scope | None


def _chain_root(
    snapshot_id: UUID,
    origin_of: Mapping[UUID, UUID | None],
) -> UUID:
    """walk ``origin_*_id`` links to the root of the shadow chain.

    the root is the deepest ancestor reachable from ``snapshot_id`` that
    is still present in ``origin_of`` (a snapshot whose origin id is
    ``None`` OR points outside the visible set). the walk is guarded by
    a visited set so a corrupted (cyclic) row terminates at the snapshot
    where the cycle closes rather than spinning.

    :param snapshot_id: id of the snapshot to walk up from
    :ptype snapshot_id: UUID
    :param origin_of: map of visible snapshot id to its origin id
    :ptype origin_of: Mapping[UUID, UUID | None]
    :return: the chain-root snapshot id
    :rtype: UUID
    """
    visited: set[UUID] = set()
    current = snapshot_id
    while True:
        visited.add(current)
        parent = origin_of.get(current)
        # parent anchors the chain root when it is None or points at a
        # snapshot outside the visible set (a shadowed platform snapshot
        # the caller cannot see still anchors the chain by its id, but
        # we cannot walk into it, so we stop here).
        if parent is None or parent not in origin_of:
            result = current
            break
        if parent in visited:
            # corrupted cyclic row: stop at the current node so the
            # merge still terminates. write-time validation prevents
            # this; the guard is defence in depth.
            log.warning(
                "origin id cycle encountered during merge at %s; anchoring chain here",
                current,
            )
            result = current
            break
        current = parent
    return result


def resolve_shadow_chains(
    snapshots: list[T],
    *,
    id_of: Callable[[T], UUID],
    scope_of: Callable[[T], Scope],
    origin_id_of: Callable[[T], UUID | None],
) -> list[ResolvedChain[T]]:
    """resolve a caller-visible snapshot set into nearest-scope winners.

    additive union of the supplied snapshots, then whole-entry shadow
    resolution per chain: snapshots linked by their origin id are grouped
    into chains, and each chain emits exactly its nearest-scope member
    (D4). unlinked snapshots each form a single-member chain and pass
    through untouched.

    the supplied list is the ALREADY-FILTERED caller-visible set for one
    domain — visibility + domain routing happen in the serving SQL, not
    here. the resolution is a pure function of the snapshots and the
    three getters.

    output ordering is stable: chains are emitted in ascending winner
    scope rank (platform first, then customer, then user) and ties
    broken by winner id, so two resolutions of the same input produce
    the same order.

    :param snapshots: caller-visible snapshots for one domain
    :ptype snapshots: list[T]
    :param id_of: getter returning a snapshot's id
    :ptype id_of: Callable[[T], UUID]
    :param scope_of: getter returning a snapshot's derived scope
    :ptype scope_of: Callable[[T], Scope]
    :param origin_id_of: getter returning a snapshot's origin id, or
        ``None`` for an unlinked snapshot
    :ptype origin_id_of: Callable[[T], UUID | None]
    :return: resolved chains ordered by ascending winner scope rank then
        winner id
    :rtype: list[ResolvedChain[T]]
    """
    origin_of: dict[UUID, UUID | None] = {id_of(s): origin_id_of(s) for s in snapshots}

    # group visible snapshots by chain root; each group is one chain.
    chains: dict[UUID, list[T]] = {}
    for snapshot in snapshots:
        root = _chain_root(id_of(snapshot), origin_of)
        chains.setdefault(root, []).append(snapshot)

    resolved: list[ResolvedChain[T]] = []
    for members in chains.values():
        # winner = nearest scope; tie-break on id for determinism (two
        # snapshots at the SAME scope in one chain is itself a malformed
        # shadow, but we still resolve deterministically).
        ordered = sorted(
            members,
            key=lambda s: (_SCOPE_RANK[scope_of(s)], id_of(s).bytes),
            reverse=True,
        )
        winner = ordered[0]
        shadowed = tuple(ordered[1:])
        shadows_scope = scope_of(shadowed[0]) if shadowed else None
        resolved.append(
            ResolvedChain(
                winner=winner,
                shadowed=shadowed,
                shadows_scope=shadows_scope,
            ),
        )

    # stable output ordering: platform winners first, then customer,
    # then user, ties broken by winner id.
    resolved.sort(
        key=lambda chain: (
            _SCOPE_RANK[scope_of(chain.winner)],
            id_of(chain.winner).bytes,
        ),
    )
    return resolved
