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

A stamped L3 table (``ROW_STAMPED``) is walked by the same one statement, each step its live rows alone
(``NOT t."gone"`` on the first row and on every link's), carrying the keys walked so far (``path``): the walk stops at
a gone row (the row is not answered, nor any past it), a NULL link, a row already walked (a cycle ends where it comes
back, each row answered once), or ``most`` rows. WHY the cycle stops there and not on a plain table: a plain table's
walk is the hand-written statement's twin (the rewrite proof holds it to the statement it replaced, which walked a
cycle round) and runs on a local copy too (DuckDB, SQLite: no array path); a stamped walk only ever goes to L3
(PostgreSQL, YugabyteDB: a local copy holds live rows alone and is walked as a plain one), and its callers (the hub's
origin chains) want each row once.

A versioned table (``ROW_VERSIONED``) stays refused: its row for a key is the key's newest version unless that is a
gone marker or a newer run marker covers it (``newest_relation``), a choice among the key's versions that no step of a
recursive term can make (PostgreSQL refuses an aggregate, a window or ``DISTINCT ON`` over the recursive reference,
and a guard on one version alone would walk an older version, or past a key whose newest version is gone). Its chain
is read a link a keyed read (``read_row``), each through its newest relation.

IO only: one statement, awaited on the caller's connection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.core.keyed._filters import reads_l3, refuse_versioned, sent_by
from threetears.interfaces.keyed.filters import quote_identifier, quoted_table
from threetears.interfaces.keyed.keys import StampedTable, live_guard, stamped_table

__all__ = [
    "read_link_chain",
]

#: the walk's own column of how many links in a row is, named so no table's column is
_DEPTH = "link_chain_depth"

#: a stamped walk's own column of the keys walked to a row, named so no table's column is
_PATH = "link_chain_path"


def _link_chain_statement(
    table: str, *, key: str, link: str, columns: Sequence[str], stamped: StampedTable | None = None
) -> str:
    """the recursive statement :func:`read_link_chain` sends: ``$1`` the first row's key, ``$2`` the most rows.

    For a stamped table (``stamped``), each step reads its live rows alone and no row already walked::

        WITH RECURSIVE chain("id", "origin_id", "link_chain_depth", "link_chain_path") AS (
            SELECT t."id", t."origin_id", 1, ARRAY[t."id"] FROM "concepts" AS t
             WHERE t."id" = $1 AND NOT t."gone"
          UNION ALL
            SELECT t."id", t."origin_id", chain."link_chain_depth" + 1, chain."link_chain_path" || ARRAY[t."id"]
              FROM "concepts" AS t JOIN chain ON t."id" = chain."origin_id"
             WHERE chain."link_chain_depth" < $2 AND NOT t."gone" AND NOT (t."id" = ANY(chain."link_chain_path"))
        ) SELECT "id", "origin_id" FROM chain

    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: the column a row is named by (the table's key), a TRUSTED identifier
    :ptype key: str
    :param link: the column naming the next row's key, a TRUSTED identifier
    :ptype link: str
    :param columns: the columns answered, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param stamped: the table's stamped declaration, when it is walked on L3: its live rows alone, a cycle once
    :ptype stamped: StampedTable | None
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
    keyed = f"t.{quote_identifier(key)}"
    answered = ", ".join(quote_identifier(column) for column in columns)
    if stamped is None:
        return (
            f"WITH RECURSIVE chain({names}, {depth}) AS ("
            f" SELECT {first}, 1 FROM {quoted_table(table)} AS t WHERE {keyed} = $1"
            f" UNION ALL SELECT {first}, chain.{depth} + 1 FROM {quoted_table(table)} AS t"
            f" JOIN chain ON {keyed} = chain.{quote_identifier(link)} WHERE chain.{depth} < $2"
            f") SELECT {answered} FROM chain"
        )
    path = quote_identifier(_PATH)
    live = live_guard(stamped, "t")
    return (
        f"WITH RECURSIVE chain({names}, {depth}, {path}) AS ("
        f" SELECT {first}, 1, ARRAY[{keyed}] FROM {quoted_table(table)} AS t WHERE {keyed} = $1 AND {live}"
        f" UNION ALL SELECT {first}, chain.{depth} + 1, chain.{path} || ARRAY[{keyed}] FROM {quoted_table(table)} AS t"
        f" JOIN chain ON {keyed} = chain.{quote_identifier(link)}"
        f" WHERE chain.{depth} < $2 AND {live} AND NOT ({keyed} = ANY(chain.{path}))"
        f") SELECT {answered} FROM chain"
    )


async def read_link_chain(
    conn: Any, table: str, *, key: str, link: str, start: Any, most: int, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """the chain of rows from the one ``key`` names ``start``, each the row the one before links to: what the
    hand-written recursive statement answered. On a stamped L3 table, its live rows alone, each once: the walk stops at
    a gone row, a NULL link, a row already walked, or ``most`` rows (the module's docstring says why).

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
    :param most: the most rows walked (a stored cycle is followed round until then, on a plain table)
    :ptype most: int
    :param columns: the columns answered, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :return: the rows, as dicts
    :rtype: list[dict[str, Any]]
    :raises ValueError: when no column is answered, ``most`` is under one, or the table is a versioned L3 table
    """
    if isinstance(most, bool) or most < 1:
        raise ValueError(f"a link chain walks at least one row, not {most!r}")
    # one recursive statement: no newest relation reaches its steps, so a versioned table is refused (its every
    # version); a stamped one is guarded in place, each step its live rows alone, so the walk reads no gone row
    refuse_versioned(conn, [table], f"a link chain of {table}")
    stamped = stamped_table(table, l3=reads_l3(conn))
    statement = _link_chain_statement(table, key=key, link=link, columns=columns, stamped=stamped)
    return [dict(row) for row in await sent_by(conn).fetch(statement, start, most)]
