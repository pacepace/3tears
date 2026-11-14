"""a chain of links within one table, walked by key in one statement: a row, the row its link names, and on.

The one spelling of what a module used to write as a recursive statement of its own::

    WITH RECURSIVE chain(id, origin_id, depth) AS (
        SELECT id, origin_id, 1 FROM concepts WHERE id = $1
      UNION ALL
        SELECT c.id, c.origin_id, chain.depth + 1 FROM concepts c JOIN chain ON c.id = chain.origin_id
         WHERE chain.depth < $2
    )
    SELECT id, origin_id FROM chain

is :func:`read_link_chain`: the row whose ``key`` is ``start``, then the row whose ``key`` is its ``link``, and on, at
most ``most`` rows (a stored cycle is followed round until then, as the statement follows it). Each step reads one row
by the table's key, so the walk costs its links, whatever the table holds; the no-whole-reads rule judges the call by
the statement it sends.

IO only: one statement, awaited on the caller's connection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.core.keyed._filters import sent_by
from threetears.interfaces.keyed.filters import quote_identifier, quoted_table

__all__ = [
    "link_chain_statement",
    "read_link_chain",
]

#: the walk's own column of how many links in a row is, named so no table's column is
_DEPTH = "link_chain_depth"


def link_chain_statement(table: str, *, key: str, link: str, columns: Sequence[str]) -> str:
    """the recursive statement :func:`read_link_chain` sends: ``$1`` the first row's key, ``$2`` the most rows.

    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: the column a row is named by (the table's key), a TRUSTED identifier
    :ptype key: str
    :param link: the column naming the next row's key, a TRUSTED identifier
    :ptype link: str
    :param columns: the columns answered, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :return: the statement
    :rtype: str
    :raises ValueError: when no column is answered
    """
    if not columns:
        raise ValueError(f"a link chain of {table} answers at least one column")
    carried = list(dict.fromkeys([key, link, *columns]))
    names = ", ".join(quote_identifier(column) for column in carried)
    first = ", ".join(f"t.{quote_identifier(column)}" for column in carried)
    depth = quote_identifier(_DEPTH)
    answered = ", ".join(quote_identifier(column) for column in columns)
    return (
        f"WITH RECURSIVE chain({names}, {depth}) AS ("
        f" SELECT {first}, 1 FROM {quoted_table(table)} AS t WHERE t.{quote_identifier(key)} = $1"
        f" UNION ALL SELECT {first}, chain.{depth} + 1 FROM {quoted_table(table)} AS t"
        f" JOIN chain ON t.{quote_identifier(key)} = chain.{quote_identifier(link)} WHERE chain.{depth} < $2"
        f") SELECT {answered} FROM chain"
    )


async def read_link_chain(
    conn: Any, table: str, *, key: str, link: str, start: Any, most: int, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """the chain of rows from the one ``key`` names ``start``, each the row the one before links to: what the
    hand-written recursive statement answered.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: the column a row is named by (the table's key), a TRUSTED identifier
    :ptype key: str
    :param link: the column naming the next row's key, a TRUSTED identifier
    :ptype link: str
    :param start: the first row's key
    :ptype start: Any
    :param most: the most rows walked (a stored cycle is followed round until then)
    :ptype most: int
    :param columns: the columns answered, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :return: the rows, as dicts
    :rtype: list[dict[str, Any]]
    :raises ValueError: when no column is answered, or ``most`` is under one
    """
    if isinstance(most, bool) or most < 1:
        raise ValueError(f"a link chain walks at least one row, not {most!r}")
    statement = link_chain_statement(table, key=key, link=link, columns=columns)
    return [dict(row) for row in await sent_by(conn).fetch(statement, start, most)]
