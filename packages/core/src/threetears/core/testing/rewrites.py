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
  :func:`~threetears.core.keyed.read_value`).
- :func:`assert_source_says` checks the module still holds the helper call the manifest records, in the function it
  names, its table, columns and each pin bound to the same argument: the proof is of the code that runs.

**Seeding.** Each row is built from the table as migrated (its columns, types, NOT NULLs, defaults, enums and unique
keys, read from the catalog), so a table needs no hand-written row factory: every
column the read pins as a parameter is shared by a tenant's rows (the key names many rows), or unique a row where a
unique key would otherwise collide; a column pinned to a literal holds it on two rows in three and another value on the
third; every other column is unique a row, NULL on the NULLs case's odd rows where nullable. The parameters are the
values of the first row (tenant 0). Foreign keys and triggers are held off while seeding
(``session_replication_role = replica``: the connection must be a superuser's, as a test database's is), and the
table's CHECK constraints are dropped in the test database (:func:`_drop_checks`: the statements are then compared over
more rows than production can hold, never fewer). A column no generic value fits takes ``overrides``: column -> value
of a :class:`Seeded`.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from uuid import NAMESPACE_OID, uuid5

from threetears.core.keyed import read_by, read_row, read_value
from threetears.core.testing.same_results import Seeded, assert_same_results

__all__ = [
    "assert_rewrite_same",
    "assert_source_says",
    "entry_id",
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
    pinned = ",".join(pin["column"] for pin in entry["pins"])
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


async def _catalog(conn: Any, table: str) -> CatalogTable:
    """the table as migrated: its columns, their types and constraints, its unique keys."""
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
        body = index["definition"].split(" USING ", 1)[-1]
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
        return Decimal(n)
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
    raise AssertionError(f"column {column.name} ({column.type}): no generic value; give it an override")


def _value(column: CatalogColumn, n: int) -> Any:
    if column.category == "A":
        return [seed_scalar(column, column.element_category, n)]
    return seed_scalar(column, column.category, n)


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
    shape: CatalogTable, entry: Mapping[str, Any], overrides: Mapping[str, Callable[[Seeded], Any]]
) -> tuple[Callable[[Seeded], dict[str, Any]], list[Any]]:
    """the row a seed puts in the entry's table, and the old statement's parameters (the first row's)."""
    pins = {pin["column"]: pin for pin in entry["pins"]}
    for column in pins:
        if column not in shape.columns:
            raise AssertionError(f"{entry['table']}: the read pins {column}, which the table has no column of")
    bound = {column for column, pin in pins.items() if pin["kind"] == "param"}
    per_row = {column for column in shape.columns if column not in pins}
    for unique in shape.unique:
        # a unique key (a partial one's predicate columns counted in) holding a bound column: the bound column is
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
        if pin is not None and pin["kind"] == "null":
            return None if seed.index % 3 != 2 else _value(column, seed.index)
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
    params: list[Any] = [None] * len({pin["param"] for pin in pins.values() if pin["kind"] == "param"})
    for column, pin in pins.items():
        if pin["kind"] == "param":
            params[pin["param"] - 1] = value(shape.columns[column], first)
    return row, params


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
        if pin["kind"] == "param":
            by[pin["column"]] = params[pin["param"] - 1]
        elif pin["kind"] == "literal":
            by[pin["column"]] = pin["literal"]
        else:
            by[pin["column"]] = None
    return by


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
    table = entry["table"]
    await _drop_checks(conn, table)
    shape = await _catalog(conn, table)
    row, params = row_factory(shape, entry, (overrides or {}).get(table, {}))
    by = _by(entry, params)
    columns = entry["columns"]
    old_sql = entry["old_sql"]
    await conn.execute("SET session_replication_role = replica")
    try:
        await assert_same_results(
            conn,
            table=table,
            row=row,
            old_sql=old_sql,
            params=params,
            helper=lambda c: read_by(c, table, by=by, columns=columns, records=True),
        )
        if entry["helper"] == "read_row":

            async def one_row(c: Any) -> list[Any]:
                found = await read_row(c, table, by=by, columns=columns, records=True)
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
                return [{column: await read_value(c, table, by=by, column=column)}]

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
            if isinstance(value, ast.Tuple | ast.List) and all(isinstance(e, ast.Constant) for e in value.elts):
                return [element.value for element in value.elts if isinstance(element, ast.Constant)]
            return None
    return None


def _written_columns(columns: ast.expr | None, tree: ast.Module) -> list[Any] | None:
    """the columns a call names: a tuple or list of literals written in the call, or a module-level name assigned one."""
    if isinstance(columns, ast.Tuple | ast.List):
        if not all(isinstance(element, ast.Constant) for element in columns.elts):
            return None
        return [element.value for element in columns.elts if isinstance(element, ast.Constant)]
    if isinstance(columns, ast.Name):
        return _module_columns(tree, columns.id)
    return None


def _says(call: ast.Call, entry: Mapping[str, Any], tree: ast.Module) -> bool:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if len(call.args) != 2 or not (isinstance(call.args[1], ast.Constant) and call.args[1].value == entry["table"]):
        return False
    by = keywords.get("by")
    if not isinstance(by, ast.Dict) or len(by.keys) != len(entry["pins"]):
        return False
    for key, value, pin in zip(by.keys, by.values, entry["pins"], strict=True):
        if not (isinstance(key, ast.Constant) and key.value == pin["column"]):
            return False
        if pin["kind"] == "param":
            if not _same_expression(value, entry["args"][pin["param"] - 1]):
                return False
        elif not (
            isinstance(value, ast.Constant) and value.value == (pin["literal"] if pin["kind"] == "literal" else None)
        ):
            return False
    if entry["helper"] == "read_value":
        column = keywords.get("column")
        return isinstance(column, ast.Constant) and column.value == entry["columns"][0]
    records = keywords.get("records")
    if not (isinstance(records, ast.Constant) and records.value is True):
        return False
    columns = keywords.get("columns")
    if entry["columns"] is None:
        return columns is None
    written: list[Any] | None = _written_columns(columns, tree)
    return written is not None and written == list(entry["columns"])


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
        f" (by {[pin['column'] for pin in entry['pins']]}, args {entry['args']})"
    )
