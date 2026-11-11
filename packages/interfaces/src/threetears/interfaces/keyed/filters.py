"""the one spelling of a keyed statement's filters and names, shared by every keyed builder.

A keyed builder is handed the rows it names as ``by``: column -> a value (``"col" = $n``) or an :class:`AnyOf`
(``"col" = ANY($n::<type>[])``, every value bound at once as one array). The builder renders them here, numbering its
placeholders from where the statement's own leave off, so every builder spells a filter alike and PostgreSQL,
YugabyteDB, DuckDB and SQLite each read it the same way. Pure: text and binds, no connection.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "IN_SUBQUERY",
    "PLACEHOLDER",
    "SQL_TYPE",
    "AnyOf",
    "Derived",
    "Fragment",
    "Lock",
    "Pin",
    "Selected",
    "batched_join_hint",
    "by_conditions",
    "narrowing",
    "prefix_conditions",
    "quote_identifier",
    "quoted_table",
    "relation",
    "renumbered",
    "selected_column",
    "sqlite_numbered",
]


def quote_identifier(identifier: str) -> str:
    """an identifier quoted for SQL, so a name with spaces, capitals or a keyword survives.

    Every L1 backend quotes every table and column name it interpolates through this one
    function: SQLite and DuckDB quote alike, and a backend that quoted some statements and not
    others would create a table it could then not read. A converted extract's column names
    (``% of Exp. In``) are the case that needs it. Postgres, Yugabyte and Redshift accept the same
    spelling.

    :param identifier: a table or column name
    :ptype identifier: str
    :return: the quoted identifier, any double quote in it doubled
    :rtype: str
    """
    return '"' + identifier.replace('"', '""') + '"'


#: a positional placeholder in a fragment the caller wrote (``$1``), renumbered when the fragment joins a statement
PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\$(\d+)")

#: a SQL type a set filter's array is cast to: a name, optionally with a precision, never anything else
SQL_TYPE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*(\(\d+(,\s*\d+)?\))?$")


@dataclass(frozen=True)
class AnyOf:
    """a set filter: the column is one of ``values``.

    Bound as one array (``"col" = ANY($n::<cast>[])``), a statement's text the same however many values; or, for a
    database with no arrays (Redshift, SQLite), as a list of placeholders (``"col" IN ($n, $n+1, ...)``), each value
    bound on its own. With no cast (``cast=None``) the array is bound untyped (``"col" = ANY($n)``): the database types
    it by the column, for a caller that holds the values but not the column's type (a table it knows only by its key
    columns' names).

    :param values: the values the column may hold; none keeps no row
    :ptype values: Sequence[Any]
    :param cast: the column's SQL type, which the array is cast to (``text``, ``uuid``, ``bigint``); None binds it
        untyped, typed by the column; unused in a list
    :ptype cast: str | None
    :param as_list: bind the values as a list of placeholders, for a database with no arrays
    :ptype as_list: bool
    """

    values: Sequence[Any]
    cast: str | None = "text"
    as_list: bool = False

    def __post_init__(self) -> None:
        if self.cast is not None and not SQL_TYPE.match(self.cast):
            raise ValueError(f"AnyOf: {self.cast!r} is not a SQL type name")


@dataclass(frozen=True)
class Derived:
    """a derived relation a keyed helper reads from: a TRUSTED ``SELECT`` written by its caller, read as
    ``FROM (<sql>) AS <alias>`` (a warehouse's per-family long relation). The no-whole-reads rule judges its text on
    its own, where it is written.

    :param sql: the ``SELECT``, with no parameters of its own
    :ptype sql: str
    """

    sql: str


def relation(table: str | Derived, alias: str) -> str:
    """how a helper names the relation it reads: a table quoted segment by segment (``schema.table`` as
    ``"schema"."table"``), or a derived relation in parentheses, either under ``alias``.

    :param table: a TRUSTED table name, or a :class:`Derived` relation
    :ptype table: str | Derived
    :param alias: the alias the statement reads it by
    :ptype alias: str
    :return: ``<relation> AS <alias>``
    :rtype: str
    """
    if isinstance(table, Derived):
        return f"({table.sql}) AS {alias}"
    return f"{quoted_table(table)} AS {alias}"


#: the name PostgreSQL (and so pg_hint_plan) gives the relation of an ``IN (SELECT ...)`` it joins as a semi-join
IN_SUBQUERY: Final = "ANY_subquery"


def batched_join_hint(outer: str, inner: str) -> str:
    """the hint every keyed join carries, as the statement's first words: YugabyteDB joins ``outer`` (the keys) to
    ``inner`` (the table) by a batched nested loop, one storage request for up to ``yb_bnl_batch_size`` keys, where
    its planner would otherwise pick a plain nested loop (one request a key: 600 keys, 600 requests). PostgreSQL reads
    it as a comment; its own planner already nests a loop over the key's index.

    :param outer: the alias of the keys' relation (:data:`IN_SUBQUERY` for an ``IN (SELECT ...)``)
    :ptype outer: str
    :param inner: the alias of the table read by the keys
    :ptype inner: str
    :return: the hint and a space
    :rtype: str
    """
    return f"/*+ YbBatchedNL({outer} {inner}) */ "


def quoted_table(table: str) -> str:
    """a TRUSTED table name quoted a segment at a time: ``schema.table`` as ``"schema"."table"``.

    :param table: the name
    :ptype table: str
    :return: the quoted name
    :rtype: str
    """
    return ".".join(quote_identifier(part) for part in table.split("."))


class Lock(StrEnum):
    """the row lock a keyed read takes on the rows it answers, in the caller's transaction: a closed set, so no
    caller text reaches the statement."""

    UPDATE = "FOR UPDATE"
    UPDATE_SKIP_LOCKED = "FOR UPDATE SKIP LOCKED"
    #: refused at once (``LockNotAvailableError``) when another transaction holds a row, never waited on
    UPDATE_NOWAIT = "FOR UPDATE NOWAIT"
    NO_KEY_UPDATE = "FOR NO KEY UPDATE"
    SHARE = "FOR SHARE"

    def of(self, alias: str) -> str:
        """the clause locking only the rows of ``alias``.

        :param alias: the read table's alias
        :ptype alias: str
        :return: the clause, with its leading space
        :rtype: str
        """
        for wait in (" SKIP LOCKED", " NOWAIT"):
            if self.value.endswith(wait):
                return f" {self.value.removesuffix(wait)} OF {alias}{wait}"
        return f" {self.value} OF {alias}"


@dataclass(frozen=True)
class Fragment:
    """a boolean SQL fragment and its parameters, its placeholders numbered from ``$1`` as written.

    The shape a visibility clause takes (:func:`threetears.core.keyed.exists_keyed`, an ACL's caller clause): a
    helper renumbers its placeholders into the statement it joins (:func:`renumbered`), so the fragment is written
    once, on its own.

    :param sql: the fragment, one boolean expression
    :ptype sql: str
    :param params: its parameters, ``$1`` first
    :ptype params: tuple[Any, ...]
    """

    sql: str
    params: tuple[Any, ...] = ()

    @classmethod
    def of(cls, clause: tuple[str, Sequence[Any]]) -> Fragment:
        """the fragment a ``(sql, params)`` clause builder answered, built with its first placeholder ``$1``
        (``threetears.agent.acl.caller_visible_customer_clause(..., param_offset=1)`` and its siblings).

        :param clause: the builder's answer
        :ptype clause: tuple[str, Sequence[Any]]
        :return: the fragment
        :rtype: Fragment
        """
        sql, params = clause
        return cls(sql=sql, params=tuple(params))


def renumbered(fragment: Fragment, before: int) -> str:
    """``fragment``'s text with each ``$n`` moved past the ``before`` parameters a statement already binds.

    :param fragment: the fragment
    :ptype fragment: Fragment
    :param before: the parameters bound ahead of it
    :ptype before: int
    :return: the text, renumbered
    :rtype: str
    :raises ValueError: when the fragment names a placeholder it binds no parameter for
    """
    numbers = [int(found) for found in PLACEHOLDER.findall(fragment.sql)]
    if numbers and max(numbers) > len(fragment.params):
        raise ValueError(f"a fragment names ${max(numbers)} but binds {len(fragment.params)} parameters")
    return PLACEHOLDER.sub(lambda match: f"${int(match.group(1)) + before}", fragment.sql)


def narrowing(fragment: Fragment, before: int) -> str:
    """``fragment`` as a condition a helper ANDs after its pins: renumbered (:func:`renumbered`) and wrapped in its own
    parentheses, so whatever it holds (a top-level ``OR`` among them) only narrows the rows the pins name, never widens
    past them. Every helper writes a ``where`` (and a ``visible``) fragment through this one function: a run-time
    fragment is judged by the pins alone (the no-whole-reads rule), which holds only because of it.

    :param fragment: the fragment
    :ptype fragment: Fragment
    :param before: the parameters bound ahead of it
    :ptype before: int
    :return: ``(<the text, renumbered>)``
    :rtype: str
    :raises ValueError: when the fragment names a placeholder it binds no parameter for
    """
    return f"({renumbered(fragment, before)})"


class Pin(str):
    """a ``by`` key pinning a TRUSTED expression of the row rather than a column: ``by={Pin.expr("metadata->>'pod'"):
    pod}`` binds ``metadata->>'pod' = $n``, the expression rendered once, as written, with no alias or quoting added.

    A ``str`` so every ``by: Mapping[str, Any]`` takes it as it takes a column; an index on the same expression keys
    the read (``CREATE INDEX ... (namespace_type, (metadata->>'pod'))``), as a :class:`Sort` expression orders one.
    The expression names the table's columns as written (``t.`` when it must), with no parameters of its own.
    """

    __slots__ = ()

    @classmethod
    def expr(cls, expression: str) -> Pin:
        """the pin of ``expression``.

        :param expression: the expression, TRUSTED SQL of the row's columns
        :ptype expression: str
        :return: the pin
        :rtype: Pin
        :raises ValueError: when the expression is empty or names a placeholder
        """
        if not expression.strip():
            raise ValueError("a pinned expression names its expression")
        if PLACEHOLDER.search(expression):
            raise ValueError(f"a pinned expression takes no parameters, got {expression!r}")
        return cls(expression)


def by_conditions(
    by: Mapping[str, Any],
    *,
    first: int = 1,
    alias: str | None = None,
    quote: Callable[[str], str] = quote_identifier,
) -> tuple[list[str], list[Any]]:
    """the conditions keeping only ``by``'s rows, placeholders numbered from ``first``, and their values.

    :param by: column -> its value, or an :class:`AnyOf` of its values; columns are TRUSTED identifiers, or a
        :class:`Pin` of a TRUSTED expression
    :ptype by: Mapping[str, Any]
    :param first: the first placeholder's number
    :ptype first: int
    :param alias: the table's alias, prefixed to each column; none when None
    :ptype alias: str | None
    :param quote: how each column is spelled
    :ptype quote: Callable[[str], str]
    :return: one condition a column, and the values in placeholder order
    :rtype: tuple[list[str], list[Any]]
    """
    conditions: list[str] = []
    values: list[Any] = []
    prefix = f"{alias}." if alias else ""
    for column, value in by.items():
        place = first + len(values)
        # a pinned expression is rendered as written; a column is the alias's, quoted
        spelled = f"{column}" if isinstance(column, Pin) else f"{prefix}{quote(column)}"
        if isinstance(value, AnyOf) and value.as_list:
            listed = list(value.values)
            if not listed:
                conditions.append("1 = 0")
                continue
            conditions.append(f"{spelled} IN ({', '.join(f'${place + index}' for index in range(len(listed)))})")
            values.extend(listed)
        elif isinstance(value, AnyOf):
            typed = "" if value.cast is None else f"::{value.cast}[]"
            conditions.append(f"{spelled} = ANY(${place}{typed})")
            values.append(list(value.values))
        elif value is None:
            conditions.append(f"{spelled} IS NULL")
        else:
            conditions.append(f"{spelled} = ${place}")
            values.append(value)
    return conditions, values


#: the highest code point; a prefix's successor steps past any trailing run of it
_MAX_CHAR: Final = 0x10FFFF
#: UTF-16 surrogates are no characters of a text value: a successor stepping into them steps past them
_SURROGATES: Final = range(0xD800, 0xE000)


def _successor(prefix: str) -> str | None:
    """the least text above every text starting with ``prefix``, in code-point (``COLLATE "C"``) order; None when
    there is none (``prefix`` is all U+10FFFF).

    :param prefix: the prefix
    :ptype prefix: str
    :return: the successor, or None
    :rtype: str | None
    """
    stem = prefix.rstrip(chr(_MAX_CHAR))
    if not stem:
        return None
    following = ord(stem[-1]) + 1
    if following in _SURROGATES:
        following = _SURROGATES.stop
    return stem[:-1] + chr(following)


def prefix_conditions(
    prefix: Mapping[str, str],
    *,
    first: int = 1,
    alias: str | None = None,
) -> tuple[list[str], list[Any]]:
    """the conditions keeping only the rows whose text column starts with its prefix, as a range an index reads.

    ``{"signal": "name/"}`` is ``"signal" COLLATE "C" >= $n AND "signal" COLLATE "C" < $n+1`` with ``"name/"`` and
    ``"name0"`` bound: code-point order, where every text starting with the prefix sits in that range and no other
    does. A linguistic collation does not keep that promise (``en_US`` reads ``name/zzz`` past ``name0``), so the
    range is always the C collation's: it reads by an index on the column in C order (a ``COLLATE "C"`` column, or a
    ``(col COLLATE "C")`` index), a range (ASC) one on YugabyteDB.

    :param prefix: column -> the prefix its rows start with; columns are TRUSTED identifiers, prefixes text
    :ptype prefix: Mapping[str, str]
    :param first: the first placeholder's number
    :ptype first: int
    :param alias: the table's alias, prefixed to each column; none when None
    :ptype alias: str | None
    :return: the conditions, and the values in placeholder order
    :rtype: tuple[list[str], list[Any]]
    :raises ValueError: when a prefix is not text, or is empty (it would name every row)
    """
    conditions: list[str] = []
    values: list[Any] = []
    named = f"{alias}." if alias else ""
    for column, start in prefix.items():
        if not isinstance(start, str):
            raise ValueError(f"a prefix of {column} is text, got {type(start).__name__}")
        if not start:
            raise ValueError(f"an empty prefix of {column} names every row; name the rows")
        spelled = f'{named}{quote_identifier(column)} COLLATE "C"'
        place = first + len(values)
        conditions.append(f"{spelled} >= ${place}")
        values.append(start)
        above = _successor(start)
        if above is not None:
            conditions.append(f"{spelled} < ${place + 1}")
            values.append(above)
    return conditions, values


@dataclass(frozen=True)
class Selected:
    """a column a read selects as a TRUSTED expression under its own name, as :class:`Fragment` is a trusted SET value.

    ``Selected("CAST(copied_at AS VARCHAR)", "copied_at")`` reads the warehouse's own text of a timestamp where the
    column would arrive as a typed value: ``CAST(copied_at AS VARCHAR) AS "copied_at"``. The expression names the
    table's columns as written (``t.`` when it must), with no parameters of its own.

    :param expression: the expression, TRUSTED SQL
    :ptype expression: str
    :param alias: the name the row carries it under, a TRUSTED identifier
    :ptype alias: str
    """

    expression: str
    alias: str

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise ValueError(f"a selected column {self.alias!r} names its expression")
        if PLACEHOLDER.search(self.expression):
            raise ValueError(f"a selected column takes no parameters, got {self.expression!r}")


def selected_column(column: str | Selected, alias: str = "t") -> str:
    """one select-list item: a column of ``alias`` (``t."col"``), or a :class:`Selected` expression under its name.

    :param column: the column, or the expression
    :ptype column: str | Selected
    :param alias: the table's alias
    :ptype alias: str
    :return: the item
    :rtype: str
    """
    if isinstance(column, Selected):
        return f"{column.expression} AS {quote_identifier(column.alias)}"
    return f"{alias}.{quote_identifier(column)}"


def sqlite_numbered(query: str) -> str:
    """``query`` with each ``$n`` written as SQLite's ``?n``: a keyed statement, every placeholder numbered as every
    builder numbers it, run by SQLite's own driver (``sqlite3``), which binds ``?n`` from the same parameters.

    :param query: the statement, ``$n`` placeholders
    :ptype query: str
    :return: the statement, ``?n`` placeholders
    :rtype: str
    """
    return PLACEHOLDER.sub(r"?\1", query)
