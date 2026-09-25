"""GIN predicates that YugabyteDB's index cannot serve, rendered so they filter rows instead.

YugabyteDB implements ``USING gin`` as ``ybgin``, which can serve a scan with exactly one
required entry. When the planner picks the index for a predicate that needs several, the
query is refused outright::

    FeatureNotSupportedError: unsupported ybgin index scan
    DETAIL: ybgin index method cannot use more than one required scan entry: got 3.

The query fails; it does not degrade. Measured on YugabyteDB with the index forced by plan
hint, these shapes are refused:

- full-text ``@@`` against a query that can carry OR or NOT: ``websearch_to_tsquery`` (which
  produces both from ordinary text: "build or publish", "-draft"), ``to_tsquery``, and a
  pre-built ``$n::tsquery``, in either operand order;
- jsonb any-key, ``tags ?| $n``;
- array overlap, ``labels && $n``, once it holds more than one element;
- ``pg_trgm`` similarity, ``name % $n``, over a ``gin_trgm_ops`` index.

These are served by the index as usual and need nothing: ``@@ plainto_tsquery(...)`` and
``@@ phraseto_tsquery(...)``, jsonb all-keys ``?&``, ``?``, containment ``@>`` / ``<@``, and
``ILIKE`` over a ``gin_trgm_ops`` index.
``scripts/probe-ybgin-shapes.py`` re-measures both lists against a live YugabyteDB.

:func:`gin_filter` wraps a refused shape in a boolean test. The planner cannot match
``(expr) IS TRUE`` to an index, so the predicate is evaluated against the rows the query's
other conditions have already narrowed to, with the same result it would have had on
PostgreSQL.
"""

from __future__ import annotations

from threetears.observe import get_logger

__all__ = ["gin_filter"]

log = get_logger(__name__)


def gin_filter(predicate: str) -> str:
    """render multi-entry GIN predicate so it filters rows instead of scanning GIN index.

    use it for a predicate on a GIN-indexed column that YugabyteDB's index refuses: an
    ``@@`` against ``websearch_to_tsquery``, ``to_tsquery`` or a pre-built tsquery, ``?|``,
    and ``&&``. result is same on PostgreSQL and YugabyteDB; only access path changes.

    precondition: the query's other conditions must narrow the rows on their own, through
    indexed scope columns (``agent_id``, ``user_id`` and similar). the wrapped predicate is
    then a filter over that slice. with no such condition it becomes a filter over a full
    table scan, which is correct and slow.

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
