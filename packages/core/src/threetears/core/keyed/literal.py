"""a keyed read written out with its values as literals, for a statement that binds no parameters.

A warehouse's ``UNLOAD`` takes its ``SELECT`` as text and binds nothing, so the keyed read an export runs cannot pass
its key values as parameters. :func:`literal_select` writes the same statement a keyed read sends -- ``by`` pinning
the key (``=`` a value, ``IN`` a set), ``where`` narrowing it -- with each value rendered as a literal of its type, so
the read is keyed by construction and the no-whole-reads rule judges the call as it judges :func:`read_by`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from threetears.core.keyed._filters import (
    _PLACEHOLDER,
    AnyOf,
    Derived,
    Fragment,
    Selected,
    by_conditions,
    narrowing,
    selected_column,
)
from threetears.core.keyed._filters import relation as _relation_sql

__all__ = ["literal_select"]


def _sql_literal(value: Any) -> str:
    """a value as a SQL literal of its type: NULL, TRUE / FALSE, an integer's digits, a finite number as its decimal
    text, a date or a time as quoted ISO text, text quoted with each quote doubled.

    :param value: the value
    :ptype value: Any
    :return: the literal
    :rtype: str
    :raises ValueError: for a number that is not finite, text holding a NUL, or a value of no type above
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal | float):
        if not math.isfinite(value):
            raise ValueError(f"a literal number is finite, got {value!r}")
        return format(Decimal(str(value)), "f")
    if isinstance(value, datetime | date):
        return "'" + (value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()) + "'"
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("a literal text holds no NUL")
        return "'" + value.replace("'", "''") + "'"
    raise ValueError(f"no SQL literal for a {type(value).__name__}")


def literal_select(
    relation: str | Derived,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    where: Sequence[Fragment] = (),
    alias: str = "t",
) -> str:
    """the ``SELECT`` a keyed read of ``relation`` sends, every value written in as a literal: a set (:class:`AnyOf`)
    as ``IN (...)`` (an empty one keeps no row), a value as ``=``, None as ``IS NULL``; each ``where`` fragment's
    parameters rendered the same way.

    :param relation: the relation, a TRUSTED name, or a derived relation
    :ptype relation: str | Derived
    :param columns: the columns, as :func:`~threetears.core.keyed.read_keyed` takes them
    :ptype columns: Sequence[str | Selected]
    :param by: the pinned key columns: column -> its value or an :class:`AnyOf` of its values; columns TRUSTED
    :ptype by: Mapping[str, Any]
    :param where: further filters, each written as ``t."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param alias: the relation's alias
    :ptype alias: str
    :return: the statement, binding no parameters
    :rtype: str
    :raises ValueError: when ``by`` pins nothing (a whole read), names no columns, or a value has no literal
    """
    if not by:
        raise ValueError("a literal keyed select pins its key (by): with none it is a whole read")
    if not columns:
        raise ValueError("a literal keyed select names its columns")
    listed = {
        column: AnyOf(value.values, value.cast, as_list=True) if isinstance(value, AnyOf) else value
        for column, value in by.items()
    }
    conditions, values = by_conditions(listed, alias=alias)
    for fragment in where:
        conditions.append(narrowing(fragment, len(values)))
        values.extend(fragment.params)
    text = " AND ".join(conditions)
    written = _PLACEHOLDER.sub(lambda match: _sql_literal(values[int(match.group(1)) - 1]), text)
    selected = ", ".join(selected_column(column, alias) for column in columns)
    return f"SELECT {selected} FROM {_relation_sql(relation, alias)} WHERE {written}"  # noqa: S608 - TRUSTED names; literals
