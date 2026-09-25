"""GIN predicates that YugabyteDB's index cannot serve, rendered so they filter rows instead.

YugabyteDB implements ``USING gin`` as ``ybgin``, which can serve a scan with exactly one
required entry. A predicate that needs several -- a full-text query with an OR or a NOT
(``websearch_to_tsquery`` produces both from ordinary text: "build or publish", "-draft"),
a jsonb any-key test (``tags ?| $n``), an array overlap (``labels && $n``) -- is refused
outright when the planner picks the index::

    FeatureNotSupportedError: unsupported ybgin index scan
    DETAIL: ybgin index method cannot use more than one required scan entry: got 3.

The query fails; it does not degrade. Plain AND full-text (``plainto_tsquery``) and jsonb
containment (``@>``) have one required entry and are served by the index as usual.

:func:`gin_filter` wraps such a predicate in a boolean test. The planner cannot match
``(expr) IS TRUE`` to an index, so the predicate is evaluated against the rows the query's
other conditions (its scope columns, which carry btree indexes) have already narrowed to,
with the same result it would have had on PostgreSQL.
"""

from __future__ import annotations

from threetears.observe import get_logger

__all__ = ["gin_filter"]

log = get_logger(__name__)


def gin_filter(predicate: str) -> str:
    """render a multi-entry GIN predicate so it filters rows instead of scanning the GIN index.

    use it for any predicate on a GIN-indexed column that can need more than one required
    scan entry: ``websearch_to_tsquery`` or ``to_tsquery`` over text a person or an agent
    typed, ``?|``, ``?&`` and ``&&``. the result is the same on PostgreSQL and YugabyteDB;
    only the access path changes.

    :param predicate: SQL boolean expression on a GIN-indexed column, with its placeholders
    :ptype predicate: str
    :return: the predicate as ``(<predicate>) IS TRUE``
    :rtype: str
    :raises ValueError: when ``predicate`` is empty or only whitespace
    """
    body = predicate.strip()
    if not body:
        msg = "gin_filter needs a predicate; an empty one would render '() IS TRUE'"
        raise ValueError(msg)
    return f"({body}) IS TRUE"
