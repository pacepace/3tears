"""eligibility scope for memory retrieval, applied inside the candidate scan.

:class:`RetrievalScope` lets a caller restrict retrieval to a defined subset
of the corpus -- *"rank the top-k WITHIN this subset"* -- which none of the
fixed identity columns can express. two rules make it correct, and both are
structural here rather than left to call sites:

1. **the predicate goes in the ``WHERE`` of the candidate scan, before
   ``ORDER BY ... LIMIT``, in EVERY arm.** applied after top-k the result
   silently under-fills (a caller asking for 10 gets 3 with no signal even
   when 10 eligible rows sat just below the unfiltered cut), and
   ``hybrid_search`` runs vector and FTS arms in parallel before merging, so
   a predicate on one arm leaks ineligible rows in through the other. this
   is the entire reason scoping cannot be left to callers as a post-filter.

2. **empty is not absent.** ``restrict_to_ids=frozenset()`` means *nothing
   is eligible* -- the answer is ``[]`` -- never "no constraint". on a
   provenance-constrained (security) caller the permissive reading turns an
   empty allow-list into a full-corpus read. the same holds for empty tag
   tuples. :func:`scope_matches_nothing` is that decision, named.

predicates ride columns and an index that already shipped: ``tags`` (v025,
``idx_memories_tags`` GIN) and the primary key. ``metadata`` is deliberately
not scoped in v1 -- it has no GIN index, so a containment predicate would
seq-scan, and adding the index is a migration this change does not need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

__all__ = ["RetrievalScope", "build_scope_conditions", "scope_matches_nothing"]


@dataclass(frozen=True)
class RetrievalScope:
    """an eligibility predicate over the corpus, applied inside the candidate scan.

    every field is optional; ``None`` means "no constraint on this axis".
    an EMPTY collection is a constraint that nothing satisfies -- see
    :func:`scope_matches_nothing`.

    :ivar tags_any: row carries at least one of these tags
    :ivar tags_all: row carries every one of these tags
    :ivar restrict_to_ids: only these rows are eligible
    :ivar exclude_ids: these rows are never eligible
    """

    tags_any: tuple[str, ...] | None = None
    tags_all: tuple[str, ...] | None = None
    restrict_to_ids: frozenset[UUID] | None = None
    exclude_ids: frozenset[UUID] | None = None


def scope_matches_nothing(scope: RetrievalScope | None) -> bool:
    """whether ``scope`` was given a constraint nothing can satisfy.

    an empty ``restrict_to_ids`` allow-list, or an empty tag tuple, is a
    stated constraint with no satisfying row -- the caller said "only
    these" and named none. retrieval answers ``[]`` without touching the
    database, never treats it as absent. (``exclude_ids=frozenset()``
    excludes nothing and is NOT empty-eligibility.)

    :param scope: the scope under inspection, possibly ``None``
    :ptype scope: RetrievalScope | None
    :return: whether retrieval must return no rows
    :rtype: bool
    """
    nothing = False
    if scope is not None:
        nothing = (
            (scope.restrict_to_ids is not None and len(scope.restrict_to_ids) == 0)
            or (scope.tags_any is not None and len(scope.tags_any) == 0)
            or (scope.tags_all is not None and len(scope.tags_all) == 0)
        )
    return nothing


def build_scope_conditions(
    scope: RetrievalScope | None,
    *,
    next_param: int,
) -> tuple[list[str], list[Any], int]:
    """render ``scope`` as SQL conditions with positional parameters.

    conditions are returned in a fixed field order so parameter numbering
    is deterministic; the caller splices them into the candidate scan's
    ``WHERE`` (every arm) before ``ORDER BY ... LIMIT``. callers must
    check :func:`scope_matches_nothing` FIRST -- this builder assumes a
    satisfiable scope.

    :param scope: the scope to render, possibly ``None``
    :ptype scope: RetrievalScope | None
    :param next_param: first free positional parameter number
    :ptype next_param: int
    :return: (condition fragments, bound params, next free param number)
    :rtype: tuple[list[str], list[Any], int]
    """
    conditions: list[str] = []
    params: list[Any] = []
    if scope is not None:
        if scope.tags_any:
            conditions.append(f"tags ?| ${next_param}::text[]")
            params.append(list(scope.tags_any))
            next_param += 1
        if scope.tags_all:
            # RAW list, never pre-dumped: the family's canonical jsonb codec
            # (core/collections/asyncpg_init.py) json.dumps every jsonb
            # param itself, so a pre-encoded string double-encodes into a
            # JSON *string* the containment operator can never match.
            conditions.append(f"tags @> ${next_param}::jsonb")
            params.append(list(scope.tags_all))
            next_param += 1
        if scope.restrict_to_ids:
            conditions.append(f"memory_id = ANY(${next_param}::uuid[])")
            params.append(list(scope.restrict_to_ids))
            next_param += 1
        if scope.exclude_ids:
            conditions.append(f"NOT (memory_id = ANY(${next_param}::uuid[]))")
            params.append(list(scope.exclude_ids))
            next_param += 1
    return conditions, params, next_param
