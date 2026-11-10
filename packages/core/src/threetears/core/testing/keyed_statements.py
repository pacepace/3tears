"""read back the statement a keyed helper sent, so an in-memory store can answer it as the database would.

A unit test that stands an in-memory store in for a pool (``fetch`` / ``execute``) used to match the hand-written
SQL of the code under test. Once that code calls the :mod:`threetears.core.keyed` helpers, the store answers the
statements the helpers render instead: :func:`decode` reads one back into its table, the columns it pins, the values
it sets and its order, and the :class:`KeyedStatement` it returns answers over plain row dicts the way PostgreSQL
would (:meth:`KeyedStatement.select`, :meth:`KeyedStatement.count`, :meth:`KeyedStatement.update`,
:meth:`KeyedStatement.delete`).

It reads only what the helpers write. A statement or condition it cannot read raises :class:`ValueError`, so a store
built on it fails loudly rather than matching rows it should not.

.. code-block:: python

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        statement = decode(sql, args)
        if statement.kind == "update":
            return statement.update(self.rows)
        return statement.select(self.rows)
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "KeyedStatement",
    "decode",
]

_IDENT: Final = r'(?:\w+\.)?"?(\w+)"?'
_PLACE: Final = r"\$(\d+)(?:::[\w\[\]]+)*"
_LITERAL: Final = r"'((?:[^']|'')*)'"
_STATEMENT: Final = re.compile(
    r"^\s*(?:"
    r"(?P<select>SELECT\s+(?P<columns>.+?)\s+FROM\s+"
    + _IDENT.replace("(\\w+)", "(?P<stable>\\w+)")
    + r"(?:\s+AS\s+\w+)?)"
    r"|(?P<update>UPDATE\s+" + _IDENT.replace("(\\w+)", "(?P<utable>\\w+)") + r"(?:\s+AS\s+\w+)?\s+SET\s+(?P<sets>.+?))"
    r"|(?P<delete>DELETE\s+FROM\s+" + _IDENT.replace("(\\w+)", "(?P<dtable>\\w+)") + r"(?:\s+AS\s+\w+)?)"
    r")"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+GROUP\s+BY\s+(?P<group>.+?))?"
    r"(?:\s+ORDER\s+BY\s+(?P<order>.+?))?"
    r"(?:\s+LIMIT\s+(?P<limit>\$?\d+))?"
    r"(?:\s+RETURNING\s+(?P<returning>.+?))?"
    r"\s*$",
    re.IGNORECASE | re.DOTALL,
)
_COUNTED: Final = re.compile(
    r'^\s*\d+\s+AS\s+relation\s*,\s*"?(\w+)"?\s+AS\s+counted\s*,\s*count\(\*\)\s+AS\s+held\s*$', re.I
)
_ORDER_ITEM: Final = re.compile(_IDENT + r"(?:\s+(ASC|DESC))?(?:\s+NULLS\s+(FIRST|LAST))?$", re.IGNORECASE)
_TUPLE_IN: Final = re.compile(r"^\(([^)]*)\)\s+IN\s+\(SELECT\s+(.+)\)$", re.IGNORECASE)
_UNNEST: Final = re.compile(r"unnest\(" + _PLACE + r"\)", re.IGNORECASE)
_OPERATORS: Final = ("<=", ">=", "!=", "<>", "=", "<", ">")


@dataclass(frozen=True)
class _Condition:
    """one condition of a ``WHERE``: ``test(row)`` is whether the row passes it."""

    text: str
    test: Callable[[dict[str, Any]], bool]
    #: the column pinned to one value by ``=``, and that value; None for any other condition
    pinned: tuple[str, Any] | None = None


@dataclass(frozen=True)
class KeyedStatement:
    """one statement a keyed helper rendered, read back.

    :ivar kind: ``select``, ``count`` (a :func:`~threetears.core.keyed.counts_by` group count), ``update`` or
        ``delete``
    :ivar table: the table it names
    :ivar columns: the columns a select reads
    :ivar values: column -> the value an update sets (the bound value, or the SQL text of a fragment like ``now()``)
    :ivar pinned: column -> the value an ``=`` condition binds it to
    :ivar order: (column, descending, nulls first) for each ORDER BY item
    :ivar limit: the LIMIT, or None
    :ivar returning: the columns an update or delete returns
    :ivar group: the column a count groups by
    """

    kind: Literal["select", "count", "update", "delete"]
    table: str
    columns: tuple[str, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)
    pinned: dict[str, Any] = field(default_factory=dict)
    order: tuple[tuple[str, bool, bool], ...] = ()
    limit: int | None = None
    returning: tuple[str, ...] = ()
    group: str | None = None
    conditions: tuple[_Condition, ...] = ()

    def matches(self, row: dict[str, Any]) -> bool:
        """whether ``row`` passes every condition of the ``WHERE``.

        :param row: one row
        :ptype row: dict[str, Any]
        :return: True when it does
        :rtype: bool
        """
        return all(condition.test(row) for condition in self.conditions)

    def select(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """the rows a select answers over ``rows``: matched, ordered, limited, cut to its columns (copies).

        :param rows: the table's rows
        :ptype rows: Iterable[dict[str, Any]]
        :return: what the database would answer
        :rtype: list[dict[str, Any]]
        """
        found = [row for row in rows if self.matches(row)]
        for column, descending, nulls_first in reversed(self.order):
            present = sorted((row for row in found if row.get(column) is not None), key=lambda r: r[column])
            if descending:
                present.reverse()
            absent = [row for row in found if row.get(column) is None]
            found = absent + present if nulls_first else present + absent
        if self.limit is not None:
            found = found[: self.limit]
        return [{column: row.get(column) for column in self.columns} if self.columns else dict(row) for row in found]

    def count(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """the rows a :func:`~threetears.core.keyed.counts_by` group count answers over ``rows``.

        :param rows: the table's rows
        :ptype rows: Iterable[dict[str, Any]]
        :return: one ``{"relation", "counted", "held"}`` row per value held
        :rtype: list[dict[str, Any]]
        """
        assert self.group is not None, "not a group count"
        held: dict[Any, int] = {}
        for row in rows:
            if self.matches(row):
                held[row.get(self.group)] = held.get(row.get(self.group), 0) + 1
        return [{"relation": 0, "counted": value, "held": n} for value, n in held.items()]

    def update(self, rows: Iterable[dict[str, Any]], *, now: Any = None) -> list[dict[str, Any]]:
        """apply an update to the matching rows in place; the rows it changed (copies, cut to ``returning``).

        :param rows: the table's rows (changed in place)
        :ptype rows: Iterable[dict[str, Any]]
        :param now: what a ``now()`` SET value writes; the text ``now()`` is refused when None
        :ptype now: Any
        :return: the updated rows; its length is the statement's row count
        :rtype: list[dict[str, Any]]
        """
        changed: list[dict[str, Any]] = []
        for row in rows:
            if not self.matches(row):
                continue
            for column, value in self.values.items():
                if isinstance(value, _Sql):
                    if value.text.lower() != "now()" or now is None:
                        raise ValueError(f"a SET value the decoder cannot write: {column} = {value.text}")
                    value = now
                row[column] = value
            changed.append({column: row.get(column) for column in self.returning} if self.returning else dict(row))
        return changed

    def delete(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """remove the matching rows from ``rows`` in place; the rows removed.

        :param rows: the table's rows (changed in place)
        :ptype rows: list[dict[str, Any]]
        :return: the removed rows; its length is the statement's row count
        :rtype: list[dict[str, Any]]
        """
        gone = [row for row in rows if self.matches(row)]
        rows[:] = [row for row in rows if not self.matches(row)]
        return gone


@dataclass(frozen=True)
class _Sql:
    """a SET value written as SQL (a fragment), not bound."""

    text: str


def decode(sql: str, args: Sequence[Any]) -> KeyedStatement:
    """read back one statement a keyed helper rendered, with the values it was bound to.

    :param sql: the statement
    :ptype sql: str
    :param args: its bound values, in placeholder order
    :ptype args: Sequence[Any]
    :return: the statement, read back
    :rtype: KeyedStatement
    :raises ValueError: when the statement, or one of its conditions, is not one the helpers write
    """
    matched = _STATEMENT.match(sql)
    if matched is None:
        raise ValueError(f"not a statement the keyed helpers write: {sql}")
    parts = matched.groupdict()
    conditions = tuple(_condition(text, args) for text in _split(parts["where"] or "", " AND "))
    pinned = dict(condition.pinned for condition in conditions if condition.pinned is not None)
    returning = tuple(_column(item) for item in _split(parts["returning"] or "", ","))
    common: dict[str, Any] = {"conditions": conditions, "pinned": pinned, "returning": returning}
    if parts["update"]:
        values: dict[str, Any] = {}
        for item in _split(parts["sets"], ","):
            column, _, value = item.partition("=")
            place = re.fullmatch(r"\s*" + _PLACE + r"\s*", value)
            values[_column(column)] = args[int(place.group(1)) - 1] if place else _Sql(value.strip())
        return KeyedStatement(kind="update", table=parts["utable"], values=values, **common)
    if parts["delete"]:
        return KeyedStatement(kind="delete", table=parts["dtable"], **common)
    table = parts["stable"]
    counted = _COUNTED.match(parts["columns"])
    if counted is not None:
        return KeyedStatement(kind="count", table=table, group=counted.group(1), **common)
    if parts["group"]:
        raise ValueError(f"a grouped select the decoder cannot read: {sql}")
    order = tuple(_order(item) for item in _split(parts["order"] or "", ","))
    limit = parts["limit"]
    if limit is not None:
        limit = int(args[int(limit[1:]) - 1]) if limit.startswith("$") else int(limit)
    columns = tuple(_column(item) for item in _split(parts["columns"], ","))
    return KeyedStatement(kind="select", table=table, columns=columns, order=order, limit=limit, **common)


def _split(text: str, separator: str) -> list[str]:
    """``text`` split on ``separator`` outside parentheses and quotes."""
    pieces: list[str] = []
    depth, quoted, start, index = 0, False, 0, 0
    upper = text.upper()
    while index < len(text):
        char = text[index]
        if char == "'":
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and depth == 0 and upper.startswith(separator, index):
            pieces.append(text[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    pieces.append(text[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


def _column(text: str) -> str:
    matched = re.fullmatch(r"\s*" + _IDENT + r"\s*", text)
    if matched is None:
        raise ValueError(f"not a column the decoder can read: {text}")
    return matched.group(1)


def _order(text: str) -> tuple[str, bool, bool]:
    matched = _ORDER_ITEM.match(text.strip())
    if matched is None:
        raise ValueError(f"an ORDER BY item the decoder cannot read: {text}")
    descending = (matched.group(2) or "ASC").upper() == "DESC"
    # PostgreSQL's default: NULLS LAST ascending, NULLS FIRST descending
    nulls_first = (matched.group(3) or ("FIRST" if descending else "LAST")).upper() == "FIRST"
    return matched.group(1), descending, nulls_first


def _same(held: Any, bound: Any) -> bool:
    """a row's value against a bound one; a key bound as text (``$1::text[]::uuid[]``) compares as its text."""
    return bool(held == bound) or (isinstance(bound, str) and not isinstance(held, str) and str(held) == bound)


def _value(text: str, args: Sequence[Any]) -> Any:
    place = re.fullmatch(_PLACE, text.strip())
    if place is not None:
        return args[int(place.group(1)) - 1]
    literal = re.fullmatch(_LITERAL, text.strip())
    if literal is not None:
        return literal.group(1).replace("''", "'")
    raise ValueError(f"a value the decoder cannot read: {text}")


def _compare(operator: str, held: Any, bound: Any) -> bool:
    if operator == "=":
        return _same(held, bound)
    if operator in ("!=", "<>"):
        return held is not None and bound is not None and not _same(held, bound)
    if held is None or bound is None:
        return False
    return bool({"<": held < bound, "<=": held <= bound, ">": held > bound, ">=": held >= bound}[operator])


def _compared(operator: str, column: str, bound: Any, row: dict[str, Any]) -> bool:
    return _compare(operator, row.get(column), bound)


def _condition(text: str, args: Sequence[Any]) -> _Condition:
    """one condition the helpers write, as a test over a row."""
    if text == "1 = 0":
        return _Condition(text, lambda row: False)
    tupled = _TUPLE_IN.match(text)
    if tupled is not None:
        columns = [_column(item) for item in tupled.group(1).split(",")]
        arrays = [args[int(place) - 1] for place in _UNNEST.findall(tupled.group(2))]
        if len(arrays) != len(columns):
            raise ValueError(f"a key batch the decoder cannot read: {text}")
        keys = list(zip(*arrays, strict=True))
        return _Condition(
            text,
            lambda row: any(all(_same(row.get(c), v) for c, v in zip(columns, key, strict=True)) for key in keys),
        )
    null = re.fullmatch(_IDENT + r"\s+IS\s+(NOT\s+)?NULL", text, re.IGNORECASE)
    if null is not None:
        column, negated = null.group(1), bool(null.group(2))
        return _Condition(text, lambda row: (row.get(column) is not None) == negated)
    any_of = re.fullmatch(_IDENT + r"\s*=\s*ANY\(" + _PLACE + r"\)", text, re.IGNORECASE)
    if any_of is not None:
        column, listed = any_of.group(1), list(args[int(any_of.group(2)) - 1])
        return _Condition(text, lambda row: any(_same(row.get(column), value) for value in listed))
    listed_in = re.fullmatch(_IDENT + r"\s+IN\s+\((.+)\)", text, re.IGNORECASE)
    if listed_in is not None:
        column = listed_in.group(1)
        values = [_value(item, args) for item in _split(listed_in.group(2), ",")]
        return _Condition(text, lambda row: any(_same(row.get(column), value) for value in values))
    for operator in _OPERATORS:
        compared = re.fullmatch(_IDENT + r"\s*" + re.escape(operator) + r"\s*(.+)", text)
        if compared is not None:
            column, bound = compared.group(1), _value(compared.group(2), args)
            pinned = (column, bound) if operator == "=" else None
            return _Condition(text, functools.partial(_compared, operator, column, bound), pinned)
    raise ValueError(f"a condition the decoder cannot read: {text}")
