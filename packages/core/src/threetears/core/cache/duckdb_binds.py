"""a DuckDB statement run with its list parameters bound through Arrow, never value by value.

DuckDB's Python layer converts each bound value on its own, and for every one it tries ``import pandas`` (twice) to
recognise its type. Where pandas is not installed a failed import is never remembered, so each try is a search of
the whole import path: 100 keys bound as one list cost ~40 ms on a laptop against ~1 ms through Arrow, 10,000 keys
~2.7 s against ~5 ms, and far more on a starved host. :func:`run_bound` makes every list parameter of
:data:`_ARROW_LIST_FROM` values or more one Arrow column of one row, sets it as a DuckDB variable in one statement,
and runs the statement with that parameter read from the variable (``getvariable``, folded to a constant when the
statement is bound: the plan is the one the bound list gets). Short lists and scalars are bound as before: two
probes cost less than the Arrow round. A list Arrow cannot type as DuckDB would (nested values, UUIDs, decimals,
mixed kinds, integers past 64 bits, only None) is bound as before, slow but exact; so is every parameter of a statement whose
placeholders cannot be matched to its parameters.

Integer lists are typed as DuckDB types a bound Python list: ``INTEGER[]`` when every value fits 32 bits, else
``BIGINT[]``.

The variable and its one-row relation are the connection's (a cursor is a connection of its own): a statement on
another cursor never sees them. The variable is reset once the statement's answer is read, so a page of keys is not
held between statements.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeVar

try:
    import duckdb

    _DUCKDB_ERROR: tuple[type[BaseException], ...] = (duckdb.Error,)
except ImportError:  # NOSILENT: DuckDB is optional; without it no connection reaches run_bound
    _DUCKDB_ERROR = ()

try:
    import pyarrow as pa
    import pyarrow.compute as pc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; without it every parameter is bound as before
    _PYARROW = False

__all__ = ["run_bound"]

_T = TypeVar("_T")

#: the fewest values a list parameter must hold to be bound through Arrow; below it, binding value by value is as fast
_ARROW_LIST_FROM: Final = 4

#: the DuckDB variable a statement's Arrow-bound lists are read from (a struct, a field ``p<n>`` a parameter)
_VARIABLE: Final = "_threetears_bound"

#: the one-row relation the variable is set from, registered only while it is
_RELATION: Final = "_threetears_bound_row"

#: a placeholder (``$n`` or ``?``), or a string, quoted name or comment, which holds none
_TOKENS: Final = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/|\$(\d+)|(\?)", re.DOTALL)

_INT32: Final = (-(2**31), 2**31 - 1)


def _column(values: Sequence[Any]) -> Any:
    """a list parameter's values as one Arrow list (of one row) typed as DuckDB types the bound list, or None when
    Arrow cannot type it so.

    :param values: the list
    :ptype values: Sequence[Any]
    :return: the list array, or None
    :rtype: pyarrow.ListArray | None
    """
    try:
        array = pa.array(list(values))
    except pa.ArrowException, OverflowError, TypeError:
        return None  # NOSILENT: a list Arrow cannot type is bound value by value, as before (slow but exact)
    kind = array.type
    if pa.types.is_integer(kind):
        bounds = pc.min_max(array)
        low, high = bounds["min"].as_py(), bounds["max"].as_py()
        if low is not None and _INT32[0] <= low and high <= _INT32[1]:
            array = array.cast(pa.int32())
    elif not (
        pa.types.is_floating(kind)
        or pa.types.is_string(kind)
        or pa.types.is_boolean(kind)
        or pa.types.is_binary(kind)
        or pa.types.is_timestamp(kind)
        or pa.types.is_date32(kind)
    ):
        return None
    return pa.ListArray.from_arrays(pa.array([0, len(array)], type=pa.int32()), array)


def _placeholders(query: str) -> list[re.Match[str]] | None:
    """the statement's placeholders in order, all ``$n`` or all ``?``; None for a mix.

    :param query: the statement
    :ptype query: str
    :return: each placeholder's match
    :rtype: list[re.Match[str]] | None
    """
    found = [m for m in _TOKENS.finditer(query) if m.group(1) or m.group(2)]
    numbered = {bool(m.group(1)) for m in found}
    return found if len(numbered) <= 1 else None


def _through_arrow(query: str, params: Sequence[Any]) -> tuple[str, list[Any], Any] | None:
    """the statement rewritten to read its long lists from the variable, the parameters still bound, and the one-row
    table to set the variable from; None when it is to be bound as it is.

    :param query: the statement, ``$n`` or ``?`` placeholders
    :ptype query: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :return: the statement, its remaining parameters and the table; or None
    :rtype: tuple[str, list[Any], pyarrow.Table] | None
    """
    long = [i for i, p in enumerate(params) if isinstance(p, list | tuple) and len(p) >= _ARROW_LIST_FROM]
    if not _PYARROW or not long:
        return None
    found = _placeholders(query)
    if not found:
        return None
    numbered = bool(found[0].group(1))
    asked = [int(m.group(1)) - 1 for m in found] if numbered else list(range(len(found)))
    if numbered and (max(asked) >= len(params) or set(asked) != set(range(len(params)))):
        return None
    if not numbered and len(found) != len(params):
        return None
    columns = {i: _column(params[i]) for i in long}
    if any(column is None for column in columns.values()):
        return None
    kept = [i for i in range(len(params)) if i not in columns]
    renumbered = {i: n + 1 for n, i in enumerate(kept)}
    pieces: list[str] = []
    at = 0
    for m, i in zip(found, asked, strict=True):
        pieces.append(query[at : m.start()])
        if i in columns:
            pieces.append(f"(getvariable('{_VARIABLE}').p{i + 1})")
        else:
            pieces.append(f"${renumbered[i]}" if numbered else "?")
        at = m.end()
    pieces.append(query[at:])
    row = pa.table({"b": pa.StructArray.from_arrays(list(columns.values()), names=[f"p{i + 1}" for i in columns])})
    return "".join(pieces), [params[i] for i in kept], row


def run_bound(conn: Any, query: str, params: Sequence[Any], answer: Callable[[Any], _T]) -> _T:
    """``answer(conn.execute(query, params))``, each list parameter of :data:`_ARROW_LIST_FROM` values or more bound
    through Arrow (module docstring): the same rows, the same plan, none of DuckDB's per-value pandas probes.

    ``answer`` reads the result before anything else runs on the connection (a DuckDB connection holds one result).

    :param conn: the DuckDB connection or cursor, held by the caller alone
    :ptype conn: duckdb.DuckDBPyConnection
    :param query: the statement, ``$n`` or ``?`` placeholders
    :ptype query: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :param answer: what to make of the result: its rows, its status, its Arrow
    :ptype answer: Callable[[duckdb.DuckDBPyConnection], _T]
    :return: the answer
    :rtype: _T
    """
    plan = _through_arrow(query, params)
    if plan is None:
        return answer(conn.execute(query, list(params)))
    rewritten, bound, row = plan
    conn.register(_RELATION, row)
    try:
        conn.execute(f"SET VARIABLE {_VARIABLE} = (SELECT b FROM {_RELATION})")  # noqa: S608
    finally:
        conn.unregister(_RELATION)
    try:
        answered = answer(conn.execute(rewritten, bound))
    except BaseException:
        try:
            conn.execute(f"RESET VARIABLE {_VARIABLE}")
        except _DUCKDB_ERROR:
            pass  # NOSILENT: the statement's failure aborted the transaction; it is raised, and the next run resets
        raise
    conn.execute(f"RESET VARIABLE {_VARIABLE}")
    return answered
