"""the one spelling of a keyed read's filters, shared by every helper in :mod:`threetears.core.keyed`.

A keyed helper is handed the rows it reads as ``by``: column -> a value (``"col" = $n``) or an :class:`AnyOf`
(``"col" = ANY($n::<type>[])``, every value bound at once as one array). The helper renders them here, numbering its
placeholders from where the statement's own leave off, so every helper spells a filter alike and PostgreSQL,
YugabyteDB and DuckDB each read it the same way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.core.sql_fragments import quote_identifier

__all__ = ["AnyOf", "Fragment", "by_conditions", "renumbered"]

#: a positional placeholder in a fragment the caller wrote (``$1``), renumbered when the fragment joins a statement
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\$(\d+)")

#: a SQL type a set filter's array is cast to: a name, optionally with a precision, never anything else
_SQL_TYPE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*(\(\d+(,\s*\d+)?\))?$")


@dataclass(frozen=True)
class AnyOf:
    """a set filter: the column is one of ``values``.

    Bound as one array (``"col" = ANY($n::<cast>[])``), a statement's text the same however many values; or, for a
    database with no arrays (Redshift, SQLite), as a list of placeholders (``"col" IN ($n, $n+1, ...)``), each value
    bound on its own.

    :param values: the values the column may hold; none keeps no row
    :ptype values: Sequence[Any]
    :param cast: the column's SQL type, which the array is cast to (``text``, ``uuid``, ``bigint``); unused in a list
    :ptype cast: str
    :param as_list: bind the values as a list of placeholders, for a database with no arrays
    :ptype as_list: bool
    """

    values: Sequence[Any]
    cast: str = "text"
    as_list: bool = False

    def __post_init__(self) -> None:
        if not _SQL_TYPE.match(self.cast):
            raise ValueError(f"AnyOf: {self.cast!r} is not a SQL type name")


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
    numbers = [int(found) for found in _PLACEHOLDER.findall(fragment.sql)]
    if numbers and max(numbers) > len(fragment.params):
        raise ValueError(f"a fragment names ${max(numbers)} but binds {len(fragment.params)} parameters")
    return _PLACEHOLDER.sub(lambda match: f"${int(match.group(1)) + before}", fragment.sql)


def by_conditions(
    by: Mapping[str, Any],
    *,
    first: int = 1,
    alias: str | None = None,
    quote: Callable[[str], str] = quote_identifier,
) -> tuple[list[str], list[Any]]:
    """the conditions keeping only ``by``'s rows, placeholders numbered from ``first``, and their values.

    :param by: column -> its value, or an :class:`AnyOf` of its values; columns are TRUSTED identifiers
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
        if isinstance(value, AnyOf) and value.as_list:
            listed = list(value.values)
            if not listed:
                conditions.append("1 = 0")
                continue
            conditions.append(
                f"{prefix}{quote(column)} IN ({', '.join(f'${place + index}' for index in range(len(listed)))})"
            )
            values.extend(listed)
        elif isinstance(value, AnyOf):
            conditions.append(f"{prefix}{quote(column)} = ANY(${place}::{value.cast}[])")
            values.append(list(value.values))
        elif value is None:
            conditions.append(f"{prefix}{quote(column)} IS NULL")
        else:
            conditions.append(f"{prefix}{quote(column)} = ${place}")
            values.append(value)
    return conditions, values
