"""the use-the-helper rewrite's proof: each recorded rewrite gives what the statement it replaced gave.

The rewrite tool (``python -m threetears.enforcement.no_whole_reads.rewrite``) records each statement it replaced in the
repo's manifest: the old statement, the helper and its table, columns and pins, each parameter's argument as written.
ONE generated test module per repo walks the manifest and, for each entry:

- :func:`assert_rewrite_same` seeds the entry's table, on a real migrated database (PostgreSQL and YugabyteDB, the same
  calls on each), for every case of :func:`~threetears.core.testing.same_results.assert_same_results` (empty, one, many,
  NULLs, ties, a tenant boundary, a page boundary and one past it), and compares the old statement with the helper call
  as that harness compares them: every row, every column, as multisets. A ``fetchrow`` or ``fetchval`` rewrite is held
  twice: every row its key names (the old statement whole against :func:`~threetears.core.keyed.read_by`), and the one
  row or value (the old statement under ``LIMIT 1`` against :func:`~threetears.core.keyed.read_row` /
  :func:`~threetears.core.keyed.read_value`). A listing (``"shape": "list"``, its ``order`` the statement's
  ``ORDER BY`` terms) is read through ``read_by(order=...)`` and compared in order too: the same rows, in the order's
  sequence of sort values (rows tied on the sort may come in either order, as the statement left them).
- :func:`assert_source_says` checks the module still holds the helper call the manifest records, in the function it
  names, its table, columns and each pin bound to the same argument: the proof is of the code that runs.

**Counts** (``"shape": "count"``): a ``count(*)`` replaced by :func:`~threetears.core.keyed.aggregate_keyed`'s one
row's count, or an ``EXISTS (SELECT 1 ...)`` by :func:`~threetears.core.keyed.holds_any` walking the entry's ``walk``.
Over the same seed cases, the old statement's one value (``SELECT (<old>) AS answer``) is compared with the helper's:
the same count, the same yes or no, an empty table included.

**Sweeps** (``"shape": "sweep"``): an expiry ``DELETE`` replaced by :func:`~threetears.core.keyed.delete_before`. Its
table is seeded with rows expired before the cutoff, rows after it, rows on it and rows with no expiry; each statement
runs in a transaction rolled back after it, with a batch holding every expired row and one holding fewer. With room
for all, both delete exactly the same rows; with fewer, each deletes the batch's count of expired rows, the helper the
oldest (its ``ORDER BY``; the old statement named no order). The source check finds the ``delete_before`` call in the
function with the recorded key and column, its collection ``self``.

**Writes** (``"shape": "write"``): an ``UPDATE`` / ``DELETE`` by equality replaced by
:func:`~threetears.core.keyed.update_keyed` / :func:`~threetears.core.keyed.delete_keyed`. Over the same seed cases,
each side runs on its own seeded table in a transaction rolled back after it, and the table as each leaves it (every
column of every row) is compared: the same rows written, to the same values, and no other. A ``SET`` parameter binds
a value no seeded row holds (the write shows), ``now()`` the transaction's own time on both sides (it is the same
within a transaction only per side: compared by being set, not by its instant); an increment (``col = col + 1``) and an
expression (``GREATEST(col, $1)``: its parameters bound as the set column's type) are sent as ``Increment`` and
``Fragment``. A ``RETURNING`` write's returned rows
are compared too (the entry's ``returning`` columns: the old statement's every returned row against the helper's
``rows``). The source check finds the call's table, ``by`` pins, ``values`` and ``returning`` as recorded.

**Seeding.** Each row is built from the table as migrated (its columns, types, NOT NULLs, defaults, enums and unique
keys, read from the catalog), so a table needs no hand-written row factory: every
column the read pins as a parameter is shared by a tenant's rows (the key names many rows), or unique a row where a
unique key would otherwise collide; a column pinned to a literal holds it on two rows in three and another value on the
third; every other column is unique a row, NULL on the NULLs case's odd rows where nullable. The parameters are the
values of the first row (tenant 0); an ``= ANY($n)`` pin's array holds the first row's value and a second tenant's
(``AnyOf`` on the helper's side, with the statement's cast). Foreign keys and triggers are held off while seeding
(``session_replication_role = replica``: the connection must be a superuser's, as a test database's is), and the
table's CHECK constraints are dropped in the test database (:func:`_drop_checks`: the statements are then compared over
more rows than production can hold, never fewer). A pgvector ``vector(n)`` column is seeded as the text of ``n``
dimensions. A column no generic value fits takes ``overrides``: column -> value of a :class:`Seeded`, for the seeded
rows and for a write's SET parameters (:func:`set_value`).
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from uuid import NAMESPACE_OID, uuid5

from threetears.core.collections.asyncpg_init import register_vector_text_codec
from threetears.core.keyed import (
    AnyOf,
    Count,
    Fragment,
    Increment,
    keyset_page,
    Selected,
    Sort,
    aggregate_keyed,
    delete_before,
    delete_keyed,
    holds_any,
    index_hint,
    read_by,
    read_row,
    read_value,
    serving_index,
    update_keyed,
)
from threetears.core.testing.plans import StatementPlan, explain_statement, on_yugabyte
from threetears.core.testing.same_results import Seeded, assert_same_results

__all__ = [
    "assert_rewrite_same",
    "assert_source_says",
    "entry_id",
    "explain_index_read",
    "insert_duplicate",
    "load_manifest",
    "row_factory",
]

#: per table, column -> its value for a seed, where the catalog's row does not fit
_Overrides = Mapping[str, Mapping[str, Callable[[Seeded], Any]]]

_BASE_TIME: Final = datetime(2026, 1, 1, tzinfo=UTC)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """the manifest's entries.

    :param path: the manifest
    :ptype path: Path
    :return: the entries; none when the file is missing
    :rtype: list[dict[str, Any]]
    """
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return entries


def entry_id(entry: Mapping[str, Any]) -> str:
    """a test id for an entry: ``<path>::<function>:<helper>(<table>)``.

    :param entry: the entry
    :ptype entry: Mapping[str, Any]
    :return: the id
    :rtype: str
    """
    pinned = ",".join(pin["column"] for pin in entry.get("pins", ()))
    return f"{entry['path']}::{entry['function']}:{entry['helper']}({entry['table']}:{pinned})"


@dataclass
class CatalogColumn:
    """one column of a table as its catalog describes it: what a seeded row's value is built from."""

    name: str
    type: str
    category: str
    element_category: str
    nullable: bool
    has_default: bool
    generated: bool
    labels: tuple[str, ...] = ()


@dataclass
class CatalogTable:
    """a table's columns and unique sets as its catalog describes them: the shape a seeded row fills."""

    columns: dict[str, CatalogColumn] = field(default_factory=dict)
    unique: list[frozenset[str]] = field(default_factory=list)


async def _drop_checks(conn: Any, table: str) -> None:
    """drop the table's CHECK constraints, in the test database the proof runs on.

    A CHECK narrows which rows the table may hold; the two statements are compared over the rows seeded, so without
    it they are compared over more rows than production can hold, never fewer: a difference on any row fails, and no
    generic row has to satisfy a multi-column rule. Indexes stay, so each statement is planned as it is in production.
    """
    found = await conn.fetchval("SELECT to_regclass($1)::oid", table)
    for check in await conn.fetch("SELECT conname FROM pg_constraint WHERE conrelid = $1 AND contype = 'c'", found):
        name = check["conname"].replace('"', '""')
        await conn.execute(f'ALTER TABLE {table} DROP CONSTRAINT "{name}"')


async def _vectors_as_text(conn: Any) -> None:
    """bind and read pgvector's ``vector`` as its text on ``conn``, as the code under test sends it (``$n::text``
    cast to the vector, ``col::text`` read back).

    asyncpg has no vector codec. On PostgreSQL the extension's type has an ordinary OID and asyncpg falls back to its
    text form; on YugabyteDB ``vector`` is a built-in type (OID 8078), which asyncpg refuses outright ("unhandled
    standard data type") when a seeded row binds one or the table's state is read. A text codec covers both.
    """
    await register_vector_text_codec(conn)


async def _catalog(conn: Any, table: str) -> CatalogTable:
    """the table as migrated: its columns, their types and constraints, its unique keys (``conn`` then binds and
    reads a vector as its text: :func:`_vectors_as_text`)."""
    await _vectors_as_text(conn)
    found = await conn.fetchval("SELECT to_regclass($1)::oid", table)
    if found is None:
        raise AssertionError(f"{table}: no such table on the test database's search path")
    attributes = await conn.fetch(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS type, t.typcategory::text AS typcategory,"
        " t.typtype::text AS typtype, a.atttypid, COALESCE(e.typcategory::text, '') AS element_category,"
        " a.attnotnull, a.atthasdef, a.attidentity::text AS attidentity, a.attgenerated::text AS attgenerated"
        " FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid"
        " LEFT JOIN pg_type e ON e.oid = t.typelem AND t.typcategory = 'A'"
        " WHERE a.attrelid = $1 AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
        found,
    )
    shape = CatalogTable()
    for attribute in attributes:
        labels: tuple[str, ...] = ()
        if attribute["typtype"] == "e":
            labels = tuple(
                row["enumlabel"]
                for row in await conn.fetch(
                    "SELECT enumlabel FROM pg_enum WHERE enumtypid = $1 ORDER BY enumsortorder", attribute["atttypid"]
                )
            )
        shape.columns[attribute["attname"]] = CatalogColumn(
            name=attribute["attname"],
            type=attribute["type"],
            category=attribute["typcategory"],
            element_category=attribute["element_category"],
            nullable=not attribute["attnotnull"],
            has_default=attribute["atthasdef"],
            generated=attribute["attidentity"] == "a" or attribute["attgenerated"].strip("\x00") != "",
            labels=labels,
        )
    for index in await conn.fetch(
        "SELECT pg_get_indexdef(indexrelid) AS definition FROM pg_index WHERE indrelid = $1 AND indisunique", found
    ):
        # the key's columns only: a partial index's predicate (`` WHERE namespace_type <> 'workspace'``) keeps rows
        # out of the key, so a row differing there from another still collides on the key while both are in it
        body = index["definition"].split(" USING ", 1)[-1].split(" WHERE ", 1)[0]
        named = {name for name in re.findall(r"\b(\w+)\b", body) if name in shape.columns}
        if named:
            shape.unique.append(frozenset(named))
    return shape


def _text_of(column: CatalogColumn, n: int) -> str:
    width = re.search(r"\((\d+)\)", column.type)
    text = f"{column.name}-{n}"
    if width is not None:
        limit = int(width.group(1))
        text = text[-limit:] if len(text) > limit else text
    return text


def _numeric_of(column: CatalogColumn, n: int) -> Decimal:
    """``n`` as a value a ``numeric(precision, scale)`` column holds: its last ``precision`` digits, ``scale`` of them
    after the point, so each ``n`` below ``10 ** precision`` gives its own value and none overflows the column."""
    typed = re.search(r"\((\d+)\s*,\s*(\d+)\)", column.type)
    if typed is None:
        return Decimal(n)
    precision, scale = int(typed.group(1)), int(typed.group(2))
    return Decimal(n % 10**precision).scaleb(-scale)


def seed_scalar(column: CatalogColumn, category: str, n: int) -> Any:
    """a value of a column's type, distinct for each ``n``."""
    kind = column.type.split("(")[0].removesuffix("[]").strip()
    if column.labels:
        return column.labels[n % len(column.labels)]
    if kind == "uuid":
        return uuid5(NAMESPACE_OID, f"{column.name}:{n}")
    if category == "S":
        return _text_of(column, n)
    if category == "B":
        return n % 2 == 0
    if category == "N":
        if kind in {"smallint", "integer", "bigint"}:
            return n
        if kind in {"real", "double precision"}:
            return float(n)
        return _numeric_of(column, n)
    if kind == "timestamp with time zone":
        return _BASE_TIME + timedelta(seconds=n)
    if kind == "timestamp without time zone":
        return (_BASE_TIME + timedelta(seconds=n)).replace(tzinfo=None)
    if kind == "date":
        return date(2026, 1, 1) + timedelta(days=n)
    if kind == "time without time zone":
        return time(n // 3600 % 24, n // 60 % 60, n % 60)
    if kind == "interval":
        return timedelta(seconds=n)
    if kind in {"jsonb", "json"}:
        # as its text: a connection with no json codec (asyncpg's default) takes and gives json as text
        return json.dumps({"n": n})
    if kind == "bytea":
        return f"{n}".encode()
    if kind == "inet":
        return f"10.0.{n // 256 % 256}.{n % 256}"
    if kind.rsplit(".", 1)[-1] == "vector":
        # pgvector's text form, as many dimensions as the column declares (format_type: [public.]vector(n))
        dimensions = re.search(r"\((\d+)\)", column.type)
        if dimensions is None:
            raise AssertionError(
                f"column {column.name} ({column.type}): a vector of no declared width; give it an override"
            )
        return "[" + ",".join([str(n), *["0"] * (int(dimensions.group(1)) - 1)]) + "]"
    raise AssertionError(f"column {column.name} ({column.type}): no generic value; give it an override")


def _value(column: CatalogColumn, n: int) -> Any:
    if column.category == "A":
        return [seed_scalar(column, column.element_category, n)]
    return seed_scalar(column, column.category, n)


def set_value(column: CatalogColumn, n: int, overrides: Mapping[str, Callable[[Seeded], Any]]) -> Any:
    """the value a write's SET parameter ``n`` binds for ``column``: the table's override for the column when given
    (handed a seed past every seeded row, so no row holds it before the write), else the column's generic value.

    :param column: the column the parameter is set into
    :ptype column: CatalogColumn
    :param n: the value's number, past the seeded rows' (900_000 up)
    :ptype n: int
    :param overrides: the table's column -> value of a seed
    :ptype overrides: Mapping[str, Callable[[Seeded], Any]]
    :return: the parameter's value
    :rtype: Any
    """
    if column.name in overrides:
        return overrides[column.name](Seeded(index=n, case="sets", tenant=n, null=False, tie=n))
    return _value(column, n)


def _other(column: CatalogColumn, literal: Any) -> Any:
    """a value of the column other than ``literal``."""
    choices = column.labels
    if choices:
        others = [choice for choice in choices if choice != literal]
        if not others:
            raise AssertionError(f"column {column.name}: no value other than {literal!r} to seed")
        return others[0]
    if isinstance(literal, bool):
        return not literal
    if isinstance(literal, int):
        return literal + 1
    other = _value(column, 999_999)
    return other if other != literal else _value(column, 999_998)


def row_factory(
    shape: CatalogTable,
    entry: Mapping[str, Any],
    overrides: Mapping[str, Callable[[Seeded], Any]],
    *,
    nulled: bool = False,
) -> tuple[Callable[[Seeded], dict[str, Any]], list[Any]]:
    """the row a seed puts in the entry's table, and the old statement's parameters (the first row's).

    ``nulled``: each null-safe pin (``IS NOT DISTINCT FROM $n``) is bound None, and two rows of three hold NULL in
    its column, the third a value -- the case a plain ``= $n`` would read as no row."""
    pins = {pin["column"]: pin for pin in entry["pins"]}
    for column in pins:
        if column not in shape.columns:
            raise AssertionError(f"{entry['table']}: the read pins {column}, which the table has no column of")
    bound = {column for column, pin in pins.items() if pin["kind"] in _BOUND}
    per_row = {column for column in shape.columns if column not in pins}
    for unique in shape.unique:
        # a unique key (a partial one's by its key columns alone) holding a bound column: the bound column is
        # unique a row unless another of its columns already is (a bool, an enum or a pinned column never is)
        apart = [
            column
            for column in unique - bound
            if column not in pins and shape.columns[column].category != "B" and not shape.columns[column].labels
        ]
        if not apart:
            per_row |= unique & bound

    alone = {next(iter(unique)) for unique in shape.unique if len(unique) == 1}

    def value(column: CatalogColumn, seed: Seeded) -> Any:
        pin = pins.get(column.name)
        if column.name in overrides:
            return overrides[column.name](seed)
        if pin is not None and pin["kind"] == "literal":
            return pin["literal"] if seed.index % 3 != 2 else _other(column, pin["literal"])
        if pin is not None and (pin["kind"] == "null" or (nulled and pin["kind"] == "nullsafe")):
            return None if seed.index % 3 != 2 else _value(column, seed.index)
        if pin is not None and pin["kind"] == "anylit":
            # two rows of three hold one of the listed values (each listed value in turn), the third none of them
            listed = list(pin["literal"])
            if seed.index % 3 != 2:
                return listed[seed.index % len(listed)]
            other = _other(column, listed[0])
            while other in listed:
                other = _other(column, other)
            return other
        if pin is None and column.nullable and seed.null:
            return None
        n = seed.index if column.name in per_row else seed.tenant
        if column.name in alone and column.category == "B":
            # a bool unique on its own (a partial "one utility model" index) holds True on one row only
            return n == 0
        return _value(column, n)

    seeded = [
        column
        for column in shape.columns.values()
        if not column.generated and (column.name in overrides or column.name in pins or _seedable(column))
    ]

    def row(seed: Seeded) -> dict[str, Any]:
        return {column.name: value(column, seed) for column in seeded}

    first = Seeded(index=0, case="params", tenant=0, null=False, tie=0)
    # an ANY pin's array: the first row's value and a second tenant's (or row's), so the set is read as a set
    second = Seeded(index=1, case="params", tenant=1, null=False, tie=0)
    params: list[Any] = [None] * len({pin["param"] for pin in pins.values() if pin["kind"] in _BOUND})
    for column, pin in pins.items():
        if pin["kind"] == "param" or (pin["kind"] == "nullsafe" and not nulled):
            params[pin["param"] - 1] = value(shape.columns[column], first)
        elif pin["kind"] == "nullsafe":
            params[pin["param"] - 1] = None
        elif pin["kind"] == "any":
            params[pin["param"] - 1] = [value(shape.columns[column], first), value(shape.columns[column], second)]
    return row, params


#: the pin kinds bound as a parameter: ``= $n``, ``= ANY($n)`` and ``IS NOT DISTINCT FROM $n``
_BOUND: Final[frozenset[str]] = frozenset({"param", "any", "nullsafe"})


def _numbers(pin: Mapping[str, Any]) -> tuple[int, ...]:
    """the statement's parameters a manifest pin or SET binds (a fragment's, in order; none for a literal)."""
    if pin["kind"] in {"fragment", "guard"}:
        return tuple(pin["params"])
    return (pin["param"],) if pin.get("param") else ()


def _seedable(column: CatalogColumn) -> bool:
    """whether a generic value is written to an unpinned column (else it takes its default or NULL)."""
    try:
        _value(column, 0)
    except AssertionError:
        if column.nullable or column.has_default:
            return False
        raise
    return True


def _by(entry: Mapping[str, Any], params: Sequence[Any]) -> dict[str, Any]:
    by: dict[str, Any] = {}
    for pin in entry["pins"]:
        if pin["kind"] in {"param", "nullsafe"}:
            # a null-safe pin bound None is sent IS NULL, as IS NOT DISTINCT FROM NULL reads
            by[pin["column"]] = params[pin["param"] - 1]
        elif pin["kind"] == "anylit":
            by[pin["column"]] = AnyOf(tuple(pin["literal"]), cast=None)
        elif pin["kind"] == "any":
            by[pin["column"]] = AnyOf(params[pin["param"] - 1], cast=pin.get("cast"))
        elif pin["kind"] == "literal":
            by[pin["column"]] = pin["literal"]
        else:
            by[pin["column"]] = None
    return by


def _where(entry: Mapping[str, Any], params: Sequence[Any]) -> list[Fragment]:
    """the entry's guards (each ``WHERE`` conjunct other than an equality) as the ``where=`` fragments its helper call
    takes, bound to the statement's parameters."""
    return [
        Fragment(guard["literal"], tuple(params[number - 1] for number in guard["params"]))
        for guard in entry.get("where", ())
    ]


def _guard_params(
    shape: CatalogTable,
    entry: Mapping[str, Any],
    params: list[Any],
    bound: set[int],
    overrides: Mapping[str, Callable[[Seeded], Any]],
) -> list[Any]:
    """``params`` grown to every parameter the entry's guards name, each one no pin binds (not in ``bound``) given the
    first seeded row's value of the column its guard compares it with: a range from that row keeps some rows and
    leaves others."""
    first = Seeded(index=0, case="params", tenant=0, null=False, tie=0)
    total = max([len(params), *(number for guard in entry.get("where", ()) for number in guard["params"])])
    grown = [*params, *([None] * (total - len(params)))]
    for guard in entry.get("where", ()):
        for number, compared in zip(guard["params"], guard["compared"], strict=True):
            if number in bound:
                continue
            if compared == ":bool":
                # a truth value standing on its own (``($1 OR col <> 'x')``): False, so the predicate beside it decides
                grown[number - 1] = False
                bound.add(number)
                continue
            column = shape.columns.get(compared)
            if column is None:
                raise AssertionError(f"{entry['table']}: a guard compares ${number} with {compared}, no column of it")
            grown[number - 1] = overrides[compared](first) if compared in overrides else _value(column, 0)
            bound.add(number)
    return grown


def _guarded_params(
    shape: CatalogTable,
    entry: Mapping[str, Any],
    overrides: Mapping[str, Callable[[Seeded], Any]],
    *,
    nulled: bool,
) -> tuple[Callable[[Seeded], dict[str, Any]], list[Any]]:
    """:func:`row_factory`'s row and parameters for a read whose guards name parameters too: the pins numbered
    ``$1..`` among themselves for the row factory, each then placed at its own number, the guards' own after."""
    if not entry.get("where"):
        return row_factory(shape, entry, overrides, nulled=nulled)
    numbered = sorted({pin["param"] for pin in entry["pins"] if pin["kind"] in _BOUND})
    compact = {number: place + 1 for place, number in enumerate(numbered)}
    pins = [{**pin, "param": compact[pin["param"]]} if pin["kind"] in _BOUND else pin for pin in entry["pins"]]
    row, pinned = row_factory(shape, {**entry, "pins": pins}, overrides, nulled=nulled)
    params: list[Any] = [None] * max(numbered or [0])
    for number, place in compact.items():
        params[number - 1] = pinned[place - 1]
    return row, _guard_params(shape, entry, params, set(compact), overrides)


async def assert_rewrite_same(conn: Any, entry: Mapping[str, Any], *, overrides: _Overrides | None = None) -> None:
    """the entry's helper call gives what its old statement gave, over every seed case, on ``conn``'s database.

    :param conn: an asyncpg connection to a migrated test database, as a superuser
    :ptype conn: Any
    :param entry: the manifest entry
    :ptype entry: Mapping[str, Any]
    :param overrides: per table, column -> value of a seed, where the generic row does not fit
    :ptype overrides: _Overrides | None
    :return: nothing
    :rtype: None
    :raises AssertionError: naming each case whose rows differ
    """
    if entry.get("shape") == "sweep":
        await _assert_sweep_same(conn, entry)
        return
    if entry.get("shape") == "write":
        await _assert_write_same(conn, entry, overrides=overrides)
        return
    if entry.get("shape") == "count":
        await _assert_count_same(conn, entry, overrides=overrides, nulled=False)
        if any(pin["kind"] == "nullsafe" for pin in entry["pins"]):
            await _assert_count_same(conn, entry, overrides=overrides, nulled=True)
        return
    await _assert_point_same(conn, entry, overrides=overrides, nulled=False)
    if any(pin["kind"] == "nullsafe" for pin in entry["pins"]):
        await _assert_point_same(conn, entry, overrides=overrides, nulled=True)


def _sort(term: Mapping[str, Any]) -> Sort:
    """a listing entry's ``ORDER BY`` term as the ``Sort`` its ``read_by`` call takes."""
    return Sort(term["column"], descending=term["descending"], nulls=term["nulls"])


async def _assert_point_same(
    conn: Any, entry: Mapping[str, Any], *, overrides: _Overrides | None, nulled: bool
) -> None:
    """the read entry's helper call gives what its old statement gave, over every seed case; ``nulled``: each
    null-safe pin bound None over rows holding NULL there (:func:`row_factory`)."""
    table = entry["table"]
    await _drop_checks(conn, table)
    shape = await _catalog(conn, table)
    row, params = _guarded_params(shape, entry, (overrides or {}).get(table, {}), nulled=nulled)
    by = _by(entry, params)
    where = _where(entry, params)
    columns = _selected(entry["columns"])
    old_sql = entry["old_sql"]
    order = [_sort(term) for term in entry.get("order", ())]
    await conn.execute("SET session_replication_role = replica")
    try:
        if entry.get("shape") == "first":
            await _assert_first_same(conn, entry, row, params, by, order, where)
            return
        await assert_same_results(
            conn,
            table=table,
            row=row,
            old_sql=old_sql,
            params=params,
            helper=lambda c: read_by(c, table, by=by, columns=columns, records=True, order=order, where=where),
            order_by=[term["column"] for term in entry.get("order", ())] or None,
            ordered=bool(order),
        )
        if entry["helper"] == "read_row":

            async def one_row(c: Any) -> list[Any]:
                found = await read_row(c, table, by=by, columns=columns, records=True, where=where)
                return [] if found is None else [found]

            await assert_same_results(
                conn,
                table=table,
                row=row,
                old_sql=f"SELECT * FROM ({old_sql}) AS o LIMIT 1",  # noqa: S608 - the manifest's own statement
                params=params,
                helper=one_row,
            )
        elif entry["helper"] == "read_value":
            column = columns[0]

            async def one_value(c: Any) -> list[dict[str, Any]]:
                return [{column: await read_value(c, table, by=by, column=column, where=where)}]

            await assert_same_results(
                conn,
                table=table,
                row=row,
                old_sql=f'SELECT (SELECT o."{column}" FROM ({old_sql}) AS o LIMIT 1) AS "{column}"',  # noqa: S608
                params=params,
                helper=one_value,
            )
    finally:
        await conn.execute("SET session_replication_role = DEFAULT")


async def _assert_count_same(
    conn: Any, entry: Mapping[str, Any], *, overrides: _Overrides | None, nulled: bool
) -> None:
    """a count entry's helper answers what its old statement answered, over every seed case: the count of the rows
    its pins name (``aggregate_keyed``), or whether any is held (``holds_any``)."""
    table = entry["table"]
    await _drop_checks(conn, table)
    shape = await _catalog(conn, table)
    row, params = _guarded_params(shape, entry, (overrides or {}).get(table, {}), nulled=nulled)
    by = _by(entry, params)
    where = _where(entry, params)

    async def answered(c: Any) -> list[dict[str, Any]]:
        if entry["helper"] == "holds_any":
            return [{"answer": await holds_any(c, table, order=entry["walk"], by=by)}]
        counted = await aggregate_keyed(c, table, aggregates={"n": Count()}, by=by, where=where)
        return [{"answer": counted[0]["n"]}]

    await conn.execute("SET session_replication_role = replica")
    try:
        await assert_same_results(
            conn,
            table=table,
            row=row,
            old_sql=f'SELECT ({entry["old_sql"]}) AS "answer"',  # noqa: S608 - the manifest's own statement
            params=params,
            helper=answered,
        )
    finally:
        await conn.execute("SET session_replication_role = DEFAULT")


#: the rows a top-n read's ``LIMIT $n`` is bound to in the comparison: inside the seed cases' 12- and 25-row tables,
#: so the boundary falls among the rows (and among the tied ones)
_TOP_N: Final[int] = 3


async def _assert_first_same(
    conn: Any,
    entry: Mapping[str, Any],
    row: Callable[[Seeded], dict[str, Any]],
    params: list[Any],
    by: dict[str, Any],
    order: list[Sort],
    where: list[Fragment],
) -> None:
    """a first-of or top-n entry (shape ``first``): ``keyset_page``'s first page, its order completed by the key, gives
    the old ``LIMIT`` statement's rows (:func:`~threetears.core.testing.same_results.assert_same_results`,
    ``limited``: exact before the boundary, sort values at it)."""
    table = entry["table"]
    limit = entry["limit"]
    if "param" in limit:
        params = [*params, _TOP_N]
        rows = _TOP_N
    else:
        rows = int(limit["literal"])
    own = entry["order"][: entry["key_from"]]

    async def first_page(c: Any) -> list[Any]:
        return (
            await keyset_page(
                c, table, columns=_selected(entry["columns"]), by=by, where=where, order=order, limit=rows, records=True
            )
        ).rows

    await assert_same_results(
        conn,
        table=table,
        row=row,
        old_sql=entry["old_sql"],
        params=params,
        helper=first_page,
        order_by=[term["column"] for term in own],
        ordered=True,
        limited=True,
    )


async def _assert_write_same(conn: Any, entry: Mapping[str, Any], *, overrides: _Overrides | None) -> None:
    """the write entry's helper call leaves the table as its old statement left it, over every seed case.

    :param conn: an asyncpg connection to a migrated test database, as a superuser
    :ptype conn: Any
    :param entry: the manifest entry (``shape`` ``write``)
    :ptype entry: Mapping[str, Any]
    :param overrides: per table, column -> value of a seed, where the generic row does not fit
    :ptype overrides: _Overrides | None
    :return: nothing
    :rtype: None
    :raises AssertionError: naming each case whose table after the write differs
    """
    await _assert_write_cases(conn, entry, overrides=overrides, nulled=False)
    if any(pin["kind"] == "nullsafe" for pin in entry["pins"]):
        await _assert_write_cases(conn, entry, overrides=overrides, nulled=True)


async def _assert_write_cases(
    conn: Any, entry: Mapping[str, Any], *, overrides: _Overrides | None, nulled: bool
) -> None:
    """the write entry over every seed case; ``nulled``: each null-safe pin bound None (:func:`row_factory`)."""
    table = entry["table"]
    await _drop_checks(conn, table)
    shape = await _catalog(conn, table)
    # the pins numbered $1.. among themselves for the row factory, then each placed at its own number
    numbered = sorted({pin["param"] for pin in entry["pins"] if pin["kind"] in _BOUND})
    compact = {number: place + 1 for place, number in enumerate(numbered)}
    pins = [{**pin, "param": compact[pin["param"]]} if pin["kind"] in _BOUND else pin for pin in entry["pins"]]
    row, pinned = row_factory(shape, {**entry, "pins": pins}, (overrides or {}).get(table, {}), nulled=nulled)
    total = max(
        [number for pin in (*entry["pins"], *entry["sets"], *entry.get("where", ())) for number in _numbers(pin)] or [0]
    )
    params: list[Any] = [None] * total
    for number, place in compact.items():
        params[number - 1] = pinned[place - 1]
    for offset, pin in enumerate(entry["sets"]):
        for number in _numbers(pin):
            if params[number - 1] is not None:
                continue
            column = shape.columns.get(pin["column"])
            if column is None:
                raise AssertionError(f"{table}: the write sets {pin['column']}, which the table has no column of")
            # a value no seeded row holds, so the write shows in the table after it (a fragment's parameters are
            # taken as the set column's type: GREATEST(col, $1), COALESCE($1::jsonb, col))
            params[number - 1] = set_value(column, 900_000 + offset, (overrides or {}).get(table, {}))
    set_numbers = {number for pin in entry["sets"] for number in _numbers(pin)}
    params = _guard_params(shape, entry, params, {*compact, *set_numbers}, (overrides or {}).get(table, {}))
    by = _by(entry, params)
    where = _where(entry, params)
    values: dict[str, Any] = {}
    for pin in entry["sets"]:
        if pin["kind"] == "param":
            values[pin["column"]] = params[pin["param"] - 1]
        elif pin["kind"] == "literal":
            values[pin["column"]] = pin["literal"]
        elif pin["kind"] == "now":
            values[pin["column"]] = Fragment("now()")
        elif pin["kind"] == "increment":
            values[pin["column"]] = Increment(pin["literal"])
        elif pin["kind"] == "fragment":
            values[pin["column"]] = Fragment(pin["literal"], tuple(params[number - 1] for number in pin["params"]))
        else:
            values[pin["column"]] = None

    returning = list(entry.get("returning") or ())

    async def helper(c: Any) -> list[Any]:
        if entry["helper"] == "update_keyed":
            written = await update_keyed(c, table, values=values, by=by, returning=returning, where=where)
        else:
            written = await delete_keyed(c, table, by=by, returning=returning, where=where)
        return written.rows

    # now() differs between the two sides' transactions: compared by whether it was set, not by its instant
    now_columns = {pin["column"] for pin in entry["sets"] if pin["kind"] == "now"}
    kept = [name for name in shape.columns if name not in now_columns]
    listed = ", ".join(f'"{name}"' for name in kept)
    stamped = "".join(f', "{name}" IS NOT NULL AS "{name}__set"' for name in sorted(now_columns))
    await conn.execute("SET session_replication_role = replica")
    try:
        await assert_same_results(
            conn,
            table=table,
            row=row,
            old_sql=entry["old_sql"],
            params=params,
            helper=helper,
            columns=returning or None,
            state_sql=f"SELECT {listed}{stamped} FROM {table}",  # noqa: S608 - the test's own table
        )
    finally:
        await conn.execute("SET session_replication_role = DEFAULT")


def _write_says(call: ast.Call, entry: Mapping[str, Any]) -> bool:
    """whether a write helper call names the entry's table, pins and values as recorded."""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if len(call.args) != 2 or not (isinstance(call.args[1], ast.Constant) and call.args[1].value == entry["table"]):
        return False

    def matches(written: ast.expr | None, pins: Sequence[Mapping[str, Any]]) -> bool:
        if not isinstance(written, ast.Dict) or len(written.keys) != len(pins):
            return False
        for key, value, pin in zip(written.keys, written.values, pins, strict=True):
            if not (isinstance(key, ast.Constant) and key.value == pin["column"]):
                return False
            if not _pin_says(value, pin, entry["args"]):
                return False
        return True

    if not matches(keywords.get("by"), entry["pins"]) or not _where_says(keywords, entry):
        return False
    returning = keywords.get("returning")
    if entry.get("returning"):
        if not isinstance(returning, ast.Tuple | ast.List):
            return False
        if [element.value for element in returning.elts if isinstance(element, ast.Constant)] != entry["returning"]:
            return False
    elif returning is not None:
        return False
    if entry["helper"] == "update_keyed":
        return matches(keywords.get("values"), entry["sets"])
    return "values" not in keywords


def _where_says(keywords: Mapping[str | None, ast.expr], entry: Mapping[str, Any]) -> bool:
    """whether a helper call's ``where=`` holds the entry's guards, each ``Fragment(<text>, (<arguments>))`` as recorded
    and in order; none written for none recorded."""
    guards = entry.get("where") or ()
    written = keywords.get("where")
    if not guards:
        return written is None
    if not isinstance(written, ast.List | ast.Tuple) or len(written.elts) != len(guards):
        return False
    return all(_pin_says(value, guard, entry["args"]) for value, guard in zip(written.elts, guards, strict=True))


def _pin_says(value: ast.expr, pin: Mapping[str, Any], args: Sequence[str]) -> bool:
    """whether a helper call's pinned or set value is the entry's: its argument, ``AnyOf(<argument>, cast=...)``,
    ``Fragment("now()")``, the literal, or None."""
    if pin["kind"] in {"param", "nullsafe"}:
        return _same_expression(value, args[pin["param"] - 1])
    if pin["kind"] == "anylit":
        listed = ", ".join(repr(literal) for literal in pin["literal"])
        return _same_expression(value, f"AnyOf(({listed}{',' if len(pin['literal']) == 1 else ''}), cast=None)")
    if pin["kind"] == "any":
        return _same_expression(value, f"AnyOf({args[pin['param'] - 1]}, cast={pin.get('cast')!r})")
    if pin["kind"] == "now":
        return _same_expression(value, 'Fragment("now()")')
    if pin["kind"] == "increment":
        return _same_expression(value, f"Increment({pin['literal']!r})")
    if pin["kind"] in {"fragment", "guard"}:
        bound = ", ".join(args[number - 1] for number in pin["params"])
        tail = "," if len(pin["params"]) == 1 else ""
        return _same_expression(value, f"Fragment({pin['literal']!r}, ({bound}{tail}))")
    return isinstance(value, ast.Constant) and value.value == (pin["literal"] if pin["kind"] == "literal" else None)


def _function(tree: ast.Module, qualified: str) -> ast.AST | None:
    if qualified == "<module>":
        return tree
    node: ast.AST = tree
    for part in qualified.split("."):
        children = [
            child
            for child in ast.iter_child_nodes(node)
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and child.name == part
        ]
        if not children:
            return None
        node = children[0]
    return node


def _same_expression(written: ast.expr, recorded: str) -> bool:
    return ast.dump(written) == ast.dump(ast.parse(recorded.strip(), mode="eval").body)


def _module_columns(tree: ast.Module, name: str) -> list[Any] | None:
    """the elements of the module-level tuple or list of literals ``name`` is assigned, or None when it is not one."""
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return _column_elements(value)
    return None


def _column_element(element: ast.expr) -> Any:
    """one written column: a literal's value, or a ``Selected("<expression>", "<alias>")`` as the manifest records
    it (``{"expr", "alias"}``); None for anything else."""
    if isinstance(element, ast.Constant):
        return element.value
    if (
        isinstance(element, ast.Call)
        and isinstance(element.func, ast.Name)
        and element.func.id == "Selected"
        and len(element.args) == 2
        and not element.keywords
        and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in element.args)
    ):
        expression, alias = (arg.value for arg in element.args if isinstance(arg, ast.Constant))
        return {"expr": expression, "alias": alias}
    return None


def _column_elements(value: ast.expr) -> list[Any] | None:
    """a tuple or list of written columns (:func:`_column_element`), or None when any is not one."""
    if not isinstance(value, ast.Tuple | ast.List):
        return None
    elements = [_column_element(element) for element in value.elts]
    return None if any(element is None for element in elements) else elements


def _selected(columns: Sequence[Any] | None) -> tuple[Any, ...] | None:
    """a manifest entry's columns as a helper takes them: each ``{"expr", "alias"}`` a :class:`Selected`."""
    if columns is None:
        return None
    return tuple(column if isinstance(column, str) else Selected(column["expr"], column["alias"]) for column in columns)


def _written_columns(columns: ast.expr | None, tree: ast.Module) -> list[Any] | None:
    """the columns a call names: a tuple or list of literals written in the call, or a module-level name assigned one."""
    if isinstance(columns, ast.Tuple | ast.List):
        return _column_elements(columns)
    if isinstance(columns, ast.Name):
        return _module_columns(tree, columns.id)
    return None


#: the rows a sweep's table is seeded with: this many expired before the cutoff, as many after it, plus one on it and
#: one with no expiry, each under one of two purposes
_SWEEP_EXPIRED: Final = 12


async def _sweep_rows(conn: Any, entry: Mapping[str, Any]) -> tuple[list[str], datetime]:
    """seed the sweep's table, inside the caller's transaction: its rows' keys, and the cutoff."""
    table, key, column = entry["table"], entry["key"], entry["column"]
    shape = await _catalog(conn, table)
    if column not in shape.columns or any(part not in shape.columns for part in key):
        raise AssertionError(f"{table}: the sweep names {key} and {column}, which the table does not hold")
    cutoff = _BASE_TIME
    expiries: list[datetime | None] = [cutoff - timedelta(minutes=n + 1) for n in range(_SWEEP_EXPIRED)]
    expiries += [cutoff + timedelta(minutes=n + 1) for n in range(_SWEEP_EXPIRED)]
    expiries += [cutoff, None]
    seeded = [c for c in shape.columns.values() if not c.generated and c.name not in {*key, column} and _seedable(c)]
    names = [*key, column, *(c.name for c in seeded)]
    for n, expiry in enumerate(expiries):
        values: list[Any] = [f"{part}-{n % 2}" if part == key[0] else f"{part}-{n}" for part in key]
        values += [expiry, *(_value(c, n) for c in seeded)]
        places = ", ".join(f"${i}" for i in range(1, len(values) + 1))
        quoted = ", ".join(f'"{name}"' for name in names)
        await conn.execute(f"INSERT INTO {table} ({quoted}) VALUES ({places})", *values)  # noqa: S608 - test table
    return names, cutoff


async def _left(conn: Any, entry: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    key = ", ".join(f'"{part}"' for part in entry["key"])
    return {tuple(row) for row in await conn.fetch(f"SELECT {key} FROM {entry['table']}")}  # noqa: S608 - test table


async def _assert_sweep_same(conn: Any, entry: Mapping[str, Any]) -> None:
    """the sweep entry's ``delete_before`` deletes what its old statement deleted, on ``conn``'s database.

    :param conn: an asyncpg connection to a migrated test database
    :ptype conn: Any
    :param entry: the manifest entry (``shape`` ``sweep``)
    :ptype entry: Mapping[str, Any]
    :return: nothing
    :rtype: None
    :raises AssertionError: naming the batch whose deleted rows differ
    """
    table, key, column = entry["table"], entry["key"], entry["column"]
    await _drop_checks(conn, table)
    for batch in (_SWEEP_EXPIRED + 5, _SWEEP_EXPIRED // 3):
        deleted: dict[str, set[tuple[Any, ...]]] = {}
        expired: set[tuple[Any, ...]] = set()
        oldest: set[tuple[Any, ...]] = set()
        for side in ("old", "helper"):
            transaction = conn.transaction()
            await transaction.start()
            try:
                await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - the test's own table, in its transaction
                _, cutoff = await _sweep_rows(conn, entry)
                before = await _left(conn, entry)
                named = ", ".join(f'"{part}"' for part in key)
                expired = {
                    tuple(row)
                    for row in await conn.fetch(
                        f'SELECT {named} FROM {table} WHERE "{column}" < $1 ORDER BY "{column}"',  # noqa: S608
                        cutoff,
                    )
                }
                ordered = await conn.fetch(
                    f'SELECT {named} FROM {table} WHERE "{column}" < $1 ORDER BY "{column}" LIMIT {batch}',  # noqa: S608
                    cutoff,
                )
                oldest = {tuple(row) for row in ordered}
                if side == "old":
                    await conn.execute(entry["old_sql"], cutoff, batch)
                else:
                    await delete_before(
                        table,
                        collection=None,
                        conn=conn,
                        key=key,
                        column=column,
                        before=cutoff,
                        page=batch,
                        most_pages=1,
                    )
                deleted[side] = before - await _left(conn, entry)
            finally:
                await transaction.rollback()
        if batch >= len(expired):
            assert deleted["old"] == deleted["helper"] == expired, (
                f"{table} batch {batch}: the old statement deleted {sorted(deleted['old'])}, delete_before "
                f"{sorted(deleted['helper'])}, the expired rows are {sorted(expired)}"
            )
        else:
            assert len(deleted["old"]) == len(deleted["helper"]) == batch, (table, batch, deleted)
            assert deleted["old"] <= expired and deleted["helper"] == oldest, (table, batch, deleted, oldest)


def _sweep_says(call: ast.Call, entry: Mapping[str, Any]) -> bool:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    collection = keywords.get("collection")
    if not (isinstance(collection, ast.Name) and collection.id == "self"):
        return False
    column = keywords.get("column")
    if not (isinstance(column, ast.Constant) and column.value == entry["column"]):
        return False
    key = keywords.get("key")
    if not isinstance(key, ast.Tuple | ast.List):
        return False
    written = [element.value for element in key.elts if isinstance(element, ast.Constant)]
    return written == list(entry["key"]) and bool(call.args) and ast.unparse(call.args[0]) == entry["table_argument"]


def _says(call: ast.Call, entry: Mapping[str, Any], tree: ast.Module) -> bool:
    if entry.get("shape") == "sweep":
        return _sweep_says(call, entry)
    if entry.get("shape") == "write":
        return _write_says(call, entry)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if len(call.args) != 2 or not (isinstance(call.args[1], ast.Constant) and call.args[1].value == entry["table"]):
        return False
    by = keywords.get("by")
    if not isinstance(by, ast.Dict) or len(by.keys) != len(entry["pins"]):
        return False
    for key, value, pin in zip(by.keys, by.values, entry["pins"], strict=True):
        if not (isinstance(key, ast.Constant) and key.value == pin["column"]):
            return False
        if not _pin_says(value, pin, entry["args"]):
            return False
    if not _where_says(keywords, entry):
        return False
    if entry.get("shape") == "count":
        return _count_says(keywords, entry)
    if entry["helper"] == "read_value":
        column = keywords.get("column")
        return isinstance(column, ast.Constant) and column.value == entry["columns"][0]
    if entry.get("shape") == "first" and not _first_says(keywords, entry):
        return False
    records = keywords.get("records")
    if not (isinstance(records, ast.Constant) and records.value is True):
        return False
    if not _order_says(keywords.get("order"), entry.get("order", ())):
        return False
    columns = keywords.get("columns")
    if entry["columns"] is None:
        return columns is None or (isinstance(columns, ast.Constant) and columns.value is None)
    written: list[Any] | None = _written_columns(columns, tree)
    return written is not None and written == list(entry["columns"])


def _count_says(keywords: Mapping[str | None, ast.expr], entry: Mapping[str, Any]) -> bool:
    """whether a count's helper call walks the entry's ``walk`` (``holds_any``) or counts every row
    (``aggregate_keyed(aggregates={"n": Count()})``), and asks nothing else."""
    if entry["helper"] == "holds_any":
        order = keywords.get("order")
        if not isinstance(order, ast.List | ast.Tuple):
            return False
        written = [element.value for element in order.elts if isinstance(element, ast.Constant)]
        return written == list(entry["walk"]) and set(keywords) == {"order", "by"}
    aggregates = keywords.get("aggregates")
    return (
        aggregates is not None
        and _same_expression(aggregates, '{"n": Count()}')
        and set(keywords) == {"aggregates", "by"} | ({"where"} if entry.get("where") else set())
    )


def _first_says(keywords: Mapping[str | None, ast.expr], entry: Mapping[str, Any]) -> bool:
    """whether a ``keyset_page`` call reads the first-of entry's ``LIMIT`` (its argument, or its literal) and, for a
    ``SELECT *``, every column (``columns=None``)."""
    limit = entry["limit"]
    recorded = entry["args"][limit["param"] - 1] if "param" in limit else str(limit["literal"])
    limited = keywords.get("limit")
    if limited is None or not _same_expression(limited, recorded):
        return False
    every = keywords.get("columns")
    return entry["columns"] is not None or (isinstance(every, ast.Constant) and every.value is None)


def _order_says(order: ast.expr | None, terms: Sequence[Mapping[str, Any]]) -> bool:
    """whether a read's ``order=`` is the entry's sort, term for term (none written for none recorded)."""
    if not terms:
        return order is None
    if not isinstance(order, ast.List | ast.Tuple) or len(order.elts) != len(terms):
        return False
    for written, term in zip(order.elts, terms, strict=True):
        parts = [repr(term["column"])]
        if term["descending"]:
            parts.append("descending=True")
        if term["nulls"] is not None:
            parts.append(f"nulls={term['nulls']!r}")
        if not _same_expression(written, f"Sort({', '.join(parts)})"):
            return False
    return True


def assert_source_says(repo_root: Path, entry: Mapping[str, Any]) -> None:
    """the entry's module holds, in the function it names, the helper call the entry records.

    :param repo_root: the repo
    :ptype repo_root: Path
    :param entry: the manifest entry
    :ptype entry: Mapping[str, Any]
    :return: nothing
    :rtype: None
    :raises AssertionError: when the function or the call is not there as recorded
    """
    tree = ast.parse((repo_root / entry["path"]).read_text(encoding="utf-8"))
    function = _function(tree, entry["function"])
    assert function is not None, f"{entry['path']}: no function {entry['function']}"
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry["helper"]
    ]
    assert any(_says(call, entry, tree) for call in calls), (
        f"{entry['path']}::{entry['function']}: no {entry['helper']} call of {entry['table']} as the manifest records it"
        f" (by {[pin['column'] for pin in entry.get('pins', ())]}, args {entry.get('args')})"
    )


#: the rows an index plan's table is seeded with, and the tenants they are split between: a bound value names one
#: tenant's rows (a twentieth of the table), so reading through the index is the plan's only sensible choice
_PLAN_ROWS: Final[int] = 400
_PLAN_TENANTS: Final[int] = 20


async def explain_index_read(conn: Any, entry: Mapping[str, Any]) -> StatementPlan:
    """the plan of an index's read (an index-purpose manifest entry: its ``table``, ``sql`` and ``pins``) over its
    table seeded on ``conn``'s migrated test database, everything rolled back after.

    The table is seeded as :func:`assert_rewrite_same` seeds it (:func:`row_factory`), :data:`_PLAN_ROWS` rows over
    :data:`_PLAN_TENANTS` tenants, each pinned column bound to the first tenant's value; PostgreSQL then gathers the
    table's statistics. A sequential scan is priced out (``enable_seqscan``), so the plan shows which index the read's
    shape goes through rather than whether a seed this small is worth one: the test asks that it is the entry's.

    Each bound column but the last names :data:`_PLAN_SPREAD` times more rows than the one after it (the last names
    one tenant's), and a read that orders names :data:`_PLAN_SPREAD` tenants: an index shorter than the read's bound
    columns (the primary key's lead) finds more rows than the index under test, and a sort a page needs costs more
    than walking the index in order.

    :param conn: an asyncpg connection to a migrated test database, as a superuser
    :ptype conn: Any
    :param entry: the manifest entry
    :ptype entry: Mapping[str, Any]
    :return: the read's analysed plan
    :rtype: StatementPlan
    """
    table = entry["table"]
    yugabyte = await on_yugabyte(conn)
    transaction = conn.transaction()
    await transaction.start()
    try:
        await _drop_checks(conn, table)
        shape = await _catalog(conn, table)
        row, params = row_factory(shape, entry, _coarser_leads(shape, entry))
        await conn.execute("SET LOCAL session_replication_role = replica")
        rows = [
            row(Seeded(index=index, case="plan", tenant=index % _PLAN_TENANTS, null=False, tie=index))
            for index in range(_PLAN_ROWS)
        ]
        await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - a test's own table, in a transaction rolled back
        columns = list(rows[0])
        marks = ", ".join(f"${number}" for number in range(1, len(columns) + 1))
        await conn.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})",  # noqa: S608 - trusted test identifiers
            [tuple(seeded[column] for column in columns) for seeded in rows],
        )
        if not yugabyte:
            await conn.execute(f"ANALYZE {table}")
        await conn.execute("SET LOCAL enable_seqscan = off")
        hint = _declared_hint(entry, params) if yugabyte else ""
        return await explain_statement(conn, hint + entry["sql"], *params, yugabyte=yugabyte)
    finally:
        await transaction.rollback()


#: an index-purpose entry's read's sort, as the keyed helpers write it (the table read as ``t``)
_ENTRY_ORDER: Final = re.compile(r'\bORDER BY t\."(\w+)"( DESC)?', re.IGNORECASE)


def _declared_hint(entry: Mapping[str, Any], params: Sequence[Any]) -> str:
    """the hint a keyed helper starts the entry's read with on YugabyteDB, as production sends it: the declared index
    serving its pins and order (:func:`~threetears.core.keyed.serving_index`, over the tables its purposes' modules
    declare), named to pg_hint_plan; nothing for a read not written over ``t`` or one only the primary key serves.

    :param entry: the manifest entry (its ``table``, ``sql``, ``pins`` and ``purpose``)
    :ptype entry: Mapping[str, Any]
    :param params: the values its read is bound to
    :ptype params: Sequence[Any]
    :return: the hint and a space, or the empty string
    :rtype: str
    """
    if f'FROM "{entry["table"]}" AS t' not in entry["sql"]:
        return ""
    for purpose in str(entry.get("purpose", "")).split(";"):
        path = purpose.strip().split("::", 1)[0]
        if "/src/" in path and path.endswith(".py"):
            # its module declares the table (a TableSchema registers its indexes when its class is built)
            importlib.import_module(path.split("/src/", 1)[1].removesuffix(".py").replace("/", "."))
    by = {pin["column"]: params[pin["param"] - 1] for pin in entry["pins"] if pin["kind"] == "param"}
    order = [Sort(column, descending=bool(down)) for column, down in _ENTRY_ORDER.findall(entry["sql"])]
    index = serving_index(entry["table"], by, order)
    return "" if index is None or index.primary else index_hint(index)


async def insert_duplicate(conn: Any, entry: Mapping[str, Any]) -> None:
    """insert into the entry's table (an index-purpose manifest entry declaring a unique index's ``constraint``) a
    row, then a second one equal to it on the index's columns (its ``param`` pins) and on its partial ``WHERE``'s
    literals (its ``literal`` pins), differing on every other column; everything rolled back after.

    The second insert is the one the unique index refuses: the caller asserts the database raises, naming the index.
    The table's CHECK constraints are dropped and its triggers off as :func:`explain_index_read`'s seed has them, so
    only the index stands between the two rows.

    :param conn: an asyncpg connection to a migrated test database, as a superuser
    :ptype conn: Any
    :param entry: the manifest entry (its ``table`` and ``pins``)
    :ptype entry: Mapping[str, Any]
    :return: nothing; the second insert raises when the index holds
    :rtype: None
    """
    table = entry["table"]
    transaction = conn.transaction()
    await transaction.start()
    try:
        await _drop_checks(conn, table)
        shape = await _catalog(conn, table)
        # each index column held to one value (a literal pin both rows hold), every other column its row's own
        held = {
            **entry,
            "pins": [
                {**pin, "kind": "literal", "literal": _value(shape.columns[pin["column"]], 0)}
                if pin["kind"] == "param"
                else pin
                for pin in entry["pins"]
            ],
        }
        row, _ = row_factory(shape, held, {})
        await conn.execute("SET LOCAL session_replication_role = replica")
        first, second = (
            row(Seeded(index=index, case="refuses", tenant=index, null=False, tie=index)) for index in (0, 1)
        )
        columns = list(first)
        marks = ", ".join(f"${number}" for number in range(1, len(columns) + 1))
        insert = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})"  # noqa: S608 - trusted test identifiers
        await conn.execute(insert, *(first[column] for column in columns))
        await conn.execute(insert, *(second[column] for column in columns))
    finally:
        await transaction.rollback()


#: how many more rows each leading bound column of an index plan's read names than the column after it
_PLAN_SPREAD: Final[int] = 5


def _coarser_leads(shape: CatalogTable, entry: Mapping[str, Any]) -> dict[str, Callable[[Seeded], Any]]:
    """the seed values of an index plan's bound columns: each one :data:`_PLAN_SPREAD` times coarser than the next
    (:func:`explain_index_read`), the last one tenant's, or :data:`_PLAN_SPREAD` tenants' when the read orders.

    :param shape: the table as migrated
    :ptype shape: CatalogTable
    :param entry: the manifest entry (its ``pins`` in the index's order, its ``sql``)
    :ptype entry: Mapping[str, Any]
    :return: column -> its value of a seed
    :rtype: dict[str, Callable[[Seeded], Any]]
    """
    bound = [pin["column"] for pin in entry["pins"]]
    last = _PLAN_SPREAD if " ORDER BY " in entry["sql"] else 1
    overrides: dict[str, Callable[[Seeded], Any]] = {}
    for place, name in enumerate(bound):
        coarse = last * _PLAN_SPREAD ** (len(bound) - 1 - place)
        if coarse > 1:
            overrides[name] = _coarse_value(shape.columns[name], coarse)
    return overrides


def _coarse_value(column: CatalogColumn, coarse: int) -> Callable[[Seeded], Any]:
    """a column's seed value shared by ``coarse`` tenants in a row."""

    def value(seed: Seeded) -> Any:
        return _value(column, seed.tenant // coarse)

    return value
