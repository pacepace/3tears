"""a DuckDB statement run with its parameters bound through Arrow, never value by value.

DuckDB's Python layer converts each bound value on its own, and for every one it tries ``import pandas`` (twice) to
recognise its type, twice more for a list. Where pandas is not installed a failed import is never remembered, so each
try is a search of the whole import path (~0.4 ms on a laptop's workspace venv): 100 keys bound as one list cost
~40 ms against ~1 ms through Arrow, 10,000 keys ~2.7 s against ~5 ms, and far more on a starved host.
:func:`run_bound` makes each parameter one Arrow field of one row, sets that row as a DuckDB variable in one
statement, and runs the statement with each such parameter read from the variable (``getvariable``, folded to a
constant when the statement is bound: the plan is the one the bound value gets). No Python value is bound then.

The Arrow round (register, set, reset) costs about what six probes do (measured: ~1.6 ms of the calling thread's CPU
against ~0.27 ms a probe), so it is taken when the statement holds an Arrow array, a list of :data:`_ARROW_LIST_FROM`
values or more, or values whose probes come to :data:`_ARROW_FROM_PROBES` or more (a long list's: five scalars, a
margin for a pod, whose shorter import path makes a probe cheaper); once taken, every parameter Arrow can type rides
in it, scalars and short lists too, at no further cost. A statement of a few scalars binds them as before: their
probes cost less than the round. ``None`` is bound as it is (DuckDB probes nothing for
it), and so is a value Arrow cannot type as DuckDB would (nested values, decimals, mixed kinds, integers past 64 bits,
a list of only None): slow but exact. Every parameter of a statement whose placeholders cannot be matched to its
parameters is bound as before.

Integers are typed as DuckDB types a bound Python integer or list: ``INTEGER`` when every value fits 32 bits, else
``BIGINT``.

The variable and its one-row relation are the connection's (a cursor is a connection of its own): a statement on
another cursor never sees them. The variable is reset once the statement's answer is read, so a page of keys is not
held between statements.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeVar

from threetears.interfaces.keyed.filters import SQL_TOKEN

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

__all__ = ["KEYS", "run_bound", "run_joined"]

_T = TypeVar("_T")

#: the fewest values a list parameter must hold to take the statement through Arrow; below it, binding value by value
#: is as fast
_ARROW_LIST_FROM: Final = 4

#: the fewest pandas probes (two a value bound, two more a list) a statement's parameters must cost to take it through
#: Arrow without a long list: those of a list of :data:`_ARROW_LIST_FROM` values, so five scalars take it, four do not
_ARROW_FROM_PROBES: Final = 2 + 2 * _ARROW_LIST_FROM

#: the DuckDB variable a statement's Arrow-bound parameters are read from (a struct, a field ``p<n>`` a parameter)
_VARIABLE: Final = "_threetears_bound"

#: the one-row relation the variable is set from, registered only while it is
_RELATION: Final = "_threetears_bound_row"

#: a placeholder (``$n``, its number the ``n`` group, or ``?``, the ``q`` group), or a string, quoted name,
#: dollar-quoted string or comment, which holds none: the keyed builders' one reading (``SQL_TOKEN``) and DuckDB's ``?``
_TOKENS: Final = re.compile(SQL_TOKEN.pattern + r"|(?P<q>\?)", re.DOTALL)

_INT32: Final = (-(2**31), 2**31 - 1)

#: where a :func:`run_joined` statement names its keys' relation (``... IN ({keys})``)
KEYS: Final = "{keys}"

#: the relation :func:`run_joined` registers a statement's keys as, only while the statement runs
_KEYS_RELATION: Final = "_threetears_keys"


def _typed(values: Sequence[Any]) -> Any:
    """``values`` as one Arrow array typed as DuckDB types the same values bound as a list, or None when Arrow cannot
    type them so.

    :param values: the values
    :ptype values: Sequence[Any]
    :return: the array, or None
    :rtype: pyarrow.Array | None
    """
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array):
        # already Arrow (a copy's keys held as text): bound as they are, never a Python object a value
        array = values
    else:
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
        or pa.types.is_time64(kind)
        or pa.types.is_duration(kind)
        # DuckDB reads Arrow's UUID extension as UUID, the type a bound uuid.UUID gets
        or getattr(kind, "extension_name", None) == "arrow.uuid"
    ):
        return None
    return array


def _column(values: Sequence[Any]) -> Any:
    """a list parameter's values as one Arrow list (of one row) typed as DuckDB types the bound list, or None when
    Arrow cannot type it so.

    :param values: the list
    :ptype values: Sequence[Any]
    :return: the list array, or None
    :rtype: pyarrow.ListArray | None
    """
    array = _typed(values)
    if array is None:
        return None
    return pa.ListArray.from_arrays(pa.array([0, len(array)], type=pa.int32()), array)


def _placeholders(query: str) -> list[re.Match[str]] | None:
    """the statement's placeholders in order, all ``$n`` or all ``?``; None for a mix.

    :param query: the statement
    :ptype query: str
    :return: each placeholder's match
    :rtype: list[re.Match[str]] | None
    """
    found = [m for m in _TOKENS.finditer(query) if m.group("n") or m.group("q")]
    numbered = {bool(m.group("n")) for m in found}
    return found if len(numbered) <= 1 else None


def _probes(param: Any) -> int:
    """the pandas probes DuckDB makes binding ``param`` value by value: none for None, two a value, two more a list.

    :param param: the parameter
    :ptype param: Any
    :return: the probes
    :rtype: int
    """
    if param is None:
        return 0
    if isinstance(param, list | tuple):
        return 2 + 2 * len(param)
    return 2


def _field(param: Any) -> Any:
    """``param`` as one Arrow field of the variable's one row, typed as DuckDB types it bound; None when it stays bound
    (None itself, or a value Arrow cannot type so).

    :param param: the parameter: an Arrow array or a list (the row's one list), or a scalar
    :ptype param: Any
    :return: the one-row array, or None
    :rtype: pyarrow.Array | None
    """
    if param is None:
        return None
    if isinstance(param, pa.Array | pa.ChunkedArray | list | tuple):
        return _column(param)
    return _typed((param,))


def _through_arrow(query: str, params: Sequence[Any]) -> tuple[str, list[Any], Any] | None:
    """the statement rewritten to read its parameters from the variable, the parameters still bound (None, and values
    Arrow cannot type), and the one-row table to set the variable from; None when it is to be bound as it is.

    :param query: the statement, ``$n`` or ``?`` placeholders
    :ptype query: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :return: the statement, its remaining parameters and the table; or None
    :rtype: tuple[str, list[Any], pyarrow.Table] | None
    """
    if not _PYARROW or not params:
        return None
    # an Arrow array (a key batch's column, bound in Arrow by threetears.core.keyed.binding) at any length: DuckDB
    # binds no Arrow array as a parameter, and it is never made Python values here
    arrays = [isinstance(p, pa.Array | pa.ChunkedArray) for p in params]
    long = any(isinstance(p, list | tuple) and len(p) >= _ARROW_LIST_FROM for p in params)
    if not (any(arrays) or long or sum(_probes(p) for p in params) >= _ARROW_FROM_PROBES):
        return None
    found = _placeholders(query)
    if not found:
        return None
    numbered = bool(found[0].group("n"))
    asked = [int(m.group("n")) - 1 for m in found] if numbered else list(range(len(found)))
    if numbered and (max(asked) >= len(params) or set(asked) != set(range(len(params)))):
        return None
    if not numbered and len(found) != len(params):
        return None
    fields = {i: _field(p) for i, p in enumerate(params)}
    if any(fields[i] is None for i, array in enumerate(arrays) if array):
        return None  # an Arrow array Arrow cannot type as DuckDB would: every parameter bound as before (_plain)
    columns = {i: field for i, field in fields.items() if field is not None}
    if not any(arrays) and sum(_probes(params[i]) for i in columns) < _ARROW_FROM_PROBES:
        return None  # what Arrow can carry saves fewer probes than the round costs
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


def _plain(params: Sequence[Any]) -> list[Any]:
    """the parameters as DuckDB binds them value by value: an Arrow array a statement could not read from the
    variable (its placeholders not matched to its parameters) made its Python values, slow but exact."""
    if not _PYARROW:
        return list(params)
    return [p.to_pylist() if isinstance(p, pa.Array | pa.ChunkedArray) else p for p in params]


def run_bound(conn: Any, query: str, params: Sequence[Any], answer: Callable[[Any], _T]) -> _T:
    """``answer(conn.execute(query, params))``, its parameters bound through Arrow when a long list or enough values
    make the round pay (module docstring): the same rows, the same plan, none of DuckDB's per-value pandas probes.

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
        return answer(conn.execute(query, _plain(params)))
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


def run_joined(conn: Any, query: str, columns: Sequence[Sequence[Any]], answer: Callable[[Any], _T]) -> _T:
    """``answer`` of ``query`` with :data:`KEYS` read as a relation of ``columns``: a row a key, its nth column the
    nth of ``columns`` (a composite key's parts, value for value), registered as one Arrow table while the statement
    runs. The statement joins it as a table DuckDB holds (``(a, b) IN ({keys})`` is a semi-join on it), with no list
    bound, no variable set and none reset. Columns Arrow cannot type as DuckDB would are bound as lists through
    :func:`run_bound` and unnested instead: the same rows, slower.

    ``answer`` reads the result before the relation goes (a DuckDB connection holds one result).

    :param conn: the DuckDB connection or cursor, held by the caller alone
    :ptype conn: duckdb.DuckDBPyConnection
    :param query: the statement, naming the keys' relation as :data:`KEYS` and taking no parameters
    :ptype query: str
    :param columns: each key column's values, all as long
    :ptype columns: Sequence[Sequence[Any]]
    :param answer: what to make of the result: its rows, its status, its Arrow
    :ptype answer: Callable[[duckdb.DuckDBPyConnection], _T]
    :return: the answer
    :rtype: _T
    :raises ValueError: when the columns differ in length, or the statement names no :data:`KEYS`
    """
    if KEYS not in query:
        raise ValueError(f"the statement names no {KEYS}: {query}")
    if len({len(column) for column in columns}) > 1:
        raise ValueError(f"the key columns differ in length: {[len(column) for column in columns]}")
    arrays = [_typed(column) for column in columns] if _PYARROW else [None]
    if all(array is not None for array in arrays):
        names = [f"k{n + 1}" for n in range(len(columns))]
        conn.register(_KEYS_RELATION, pa.table(dict(zip(names, arrays, strict=True))))
        try:
            return answer(conn.execute(query.replace(KEYS, f"SELECT {', '.join(names)} FROM {_KEYS_RELATION}")))
        finally:
            conn.unregister(_KEYS_RELATION)
    unnested = ", ".join("unnest(?)" for _ in columns)
    return run_bound(conn, query.replace(KEYS, f"SELECT {unnested}"), [list(column) for column in columns], answer)
