"""the use-the-helper rewrite: each hand-written statement a helper can send, rewritten onto that helper, with a record.

The use-the-helper rule (:mod:`~threetears.enforcement.no_whole_reads.use_the_helper`) finds every statement a module
writes as text against a data table. This tool takes those findings and rewrites, in place, every one it can rewrite
with certainty; it leaves every other untouched and lists it with the reason (a refusal). Each rewrite is recorded in
the repo's manifest (the old statement, the helper call that replaced it, its table and how each parameter is bound),
and ONE generated test module per repo holds each recorded rewrite to the statement it replaced
(:mod:`threetears.core.testing.rewrites`: seeded from the migrated table, ``assert_same_results`` old vs new, on
PostgreSQL and YugabyteDB), and checks the source still says what the manifest says.

**Point reads** (shape ``point``), the first batch: ``await <conn>.fetch|fetchrow|fetchval(SQL, *args)`` where SQL is
a plain literal (written at the call, or a constant used only there) of the form::

    SELECT <columns> | * FROM <table> WHERE <col> = $n [AND <col> = $m | <col> = <literal> | <col> IS NULL
        | <col> = ANY($k::<type>[]) | <bool col> | NOT <bool col> ...]

becomes :func:`~threetears.core.keyed.read_by` (``fetch``), :func:`~threetears.core.keyed.read_row` (``fetchrow``) or
:func:`~threetears.core.keyed.read_value` (``fetchval``), each column pinned in ``by=`` exactly as the statement bound it
(``= $n`` typed by the column; ``= ANY($k::uuid[])`` as ``AnyOf(arg, cast="uuid")``, ``= ANY($k)`` with ``cast=None``;
a bare boolean column as ``True``, ``NOT`` one as ``False``), ``records=True`` so the rows are what the connection
answered before. Refused, and listed: an interpolated statement, an ``IN`` / ``OR`` / range / expression predicate, an
``ORDER BY`` or
``LIMIT``, an alias or expression in the select list, a cast on a parameter, a ``fetchval`` of ``*``, a call with
starred or keyword arguments, a parameter used twice whose argument is not a plain name, a constant used elsewhere, and
a statement the no-whole-reads rule already finds unkeyed (that one needs a real fix first). After rewriting, the walk is
run again: a rewritten function must hold no whole-read finding it did not hold before.

**Keyed writes** (shape ``write``): ``UPDATE <table> SET <col> = $n | <literal> | NULL | now(), ... WHERE <pins>`` and
``DELETE FROM <table> WHERE <pins>`` become :func:`~threetears.core.keyed.update_keyed` /
:func:`~threetears.core.keyed.delete_keyed` with the same ``values`` and ``by``. Sent by an awaited ``execute`` whose
status is discarded; or, with a ``RETURNING`` of plain columns, by an awaited ``fetch`` / ``fetchrow`` / ``fetchval``,
the helper then called with ``returning=`` and its answer read as that method answered (``.rows``, ``.row``, ``.value``:
a list of rows, the first row or None, the first row's first column or None).

Splicing is by the call's ``ast`` span; the touched files are then put through ``ruff check --select I --fix`` (the
new import sorted in) and ``ruff format``, which every repo runs.

Usage::

    python -m threetears.enforcement.no_whole_reads.rewrite point --repo <root> \\
        --config tests.enforcement.test_no_whole_reads:_CONFIG --manifest tests/helper_rewrites.json [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, _hits, _producers
from threetears.enforcement.no_whole_reads.use_the_helper import _helper_hits

__all__ = ["main"]

#: the connection methods a point read is sent through, onto the helper that replaces each
_HELPER_OF: Final[dict[str, str]] = {"fetch": "read_by", "fetchrow": "read_row", "fetchval": "read_value"}

#: the connection method a write is sent through (its status discarded)
_WRITE_METHODS: Final[frozenset[str]] = frozenset({"execute"})

#: a write's helper, by its verb
_WRITE_HELPER: Final[dict[str, str]] = {"update": "update_keyed", "delete": "delete_keyed"}

#: where the helpers are imported from
_HELPER_MODULE: Final[str] = "threetears.core.keyed"

_IDENT: Final[str] = r"[a-z_][a-z0-9_]*"
_KEYWORDS: Final[frozenset[str]] = frozenset({"where", "order", "limit", "group", "join", "on", "left", "inner"})
_SELECT: Final[re.Pattern[str]] = re.compile(
    rf"^select\s+(?P<columns>.+?)\s+from\s+(?P<table>{_IDENT}(?:\.{_IDENT})?)"
    rf"(?:\s+(?:as\s+)?(?P<alias>{_IDENT}))?\s+where\s+(?P<where>.+?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_COLUMN: Final[re.Pattern[str]] = re.compile(rf"^(?:(?P<qualifier>{_IDENT})\.)?(?P<column>{_IDENT})$")
_PARAM_PIN: Final[re.Pattern[str]] = re.compile(rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s*=\s*\$(?P<n>\d+)$")
_LITERAL_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s*=\s*(?P<literal>'(?:[^']|'')*'|true|false|-?\d+)$", re.IGNORECASE
)
_NULL_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s+is\s+null$", re.IGNORECASE
)
_ANY_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s*=\s*any\s*\(\s*\$(?P<n>\d+)"
    rf"(?:\s*::\s*(?P<cast>{_IDENT})\s*\[\s*\])?\s*\)$",
    re.IGNORECASE,
)
_BOOL_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<not>not\s+)?(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})$", re.IGNORECASE
)
#: words a bare-column conjunct never is (a bare boolean column pins it TRUE)
_NOT_COLUMNS: Final[frozenset[str]] = frozenset({"true", "false", "null", "not", "and", "or"})
_AND: Final[re.Pattern[str]] = re.compile(r"\s+and\s+", re.IGNORECASE)
_QUOTED: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")


@dataclass(frozen=True)
class Pin:
    """one column a point read pins.

    :ivar column: the column
    :ivar kind: ``param`` (``= $n``), ``any`` (``= ANY($n)``, ``= ANY($n::<type>[])``), ``literal`` (``= 'x'``,
        ``= TRUE``, ``= 3``, a bare boolean column for TRUE and ``NOT`` one for FALSE) or ``null`` (``IS NULL``)
    :ivar param: the parameter's number, for ``param`` and ``any``
    :ivar literal: the literal's Python value, for ``literal``
    :ivar cast: the array's element type, for ``any`` (None: the array was bound untyped)
    :ivar params: a ``fragment`` SET's parameters, in the order its expression first names them
    """

    column: str
    kind: str
    param: int | None = None
    literal: str | bool | int | None = None
    cast: str | None = None
    #: a ``fragment`` SET (``col = GREATEST(col, $n)``): its expression is ``literal``, renumbered from ``$1`` in the
    #: order of these, the statement's own parameter numbers; an ``increment`` (``col = col + 1``) holds its amount in
    #: ``literal``
    params: tuple[int, ...] = ()

    def numbers(self) -> tuple[int, ...]:
        """the statement's parameters this pin binds.

        :return: the parameter numbers
        :rtype: tuple[int, ...]
        """
        return self.params if self.kind == "fragment" else (() if self.param is None else (self.param,))


@dataclass(frozen=True)
class PointSql:
    """a plain point read, parsed.

    :ivar table: the table as written (``schema.table`` kept)
    :ivar columns: the selected columns, None for ``*``
    :ivar pins: each pinned column, in the statement's order
    """

    table: str
    columns: tuple[str, ...] | None
    pins: tuple[Pin, ...]


@dataclass(frozen=True)
class Refusal:
    """a statement the tool leaves as it is.

    :ivar path: repo-relative module
    :ivar line: the statement's line
    :ivar function: its function
    :ivar shape: its use-the-helper shape
    :ivar table: its table
    :ivar reason: why it was not rewritten
    :ivar sql: the statement, one line
    """

    path: str
    line: int
    function: str
    shape: str
    table: str
    reason: str
    sql: str


@dataclass
class Rewrite:
    """one rewrite, as the manifest records it.

    :ivar path: repo-relative module
    :ivar function: the function holding it
    :ivar line: the statement's line before the rewrite
    :ivar table: the table
    :ivar old_sql: the statement replaced, as written
    :ivar method: ``fetch``, ``fetchrow`` or ``fetchval``
    :ivar helper: ``read_by``, ``read_row`` or ``read_value``
    :ivar columns: the columns read, None for every column
    :ivar pins: each pinned column (:class:`Pin` as a dict)
    :ivar args: each parameter's argument as written, ``$1`` first
    :ivar new_call: the call written in the statement's place (before formatting)
    :ivar shape: ``point`` or ``write``
    :ivar sets: a write's ``SET`` columns
    """

    path: str
    function: str
    line: int
    table: str
    old_sql: str
    method: str
    helper: str
    columns: list[str] | None
    pins: list[dict[str, Any]]
    args: list[str]
    new_call: str = ""
    #: ``point`` (a read), or ``write`` (an ``UPDATE`` / ``DELETE`` by equality, :func:`plan_write_rewrites`)
    shape: str = "point"
    #: a write's ``SET`` columns (:class:`Pin` as a dict; ``now`` for ``now()``), none for a ``DELETE`` or a read
    sets: list[dict[str, Any]] = field(default_factory=list)
    #: the columns a write's helper call returns (its ``RETURNING``; a ``fetchval``'s first column only), none when
    #: the statement returned nothing
    returning: list[str] = field(default_factory=list)
    # the edits that make it, not recorded: (start line, start col, end line, end col, text) in the module's bytes
    edits: list[tuple[int, int, int, int, str]] = field(default_factory=list, repr=False)

    def record(self) -> dict[str, Any]:
        """the manifest's entry.

        :return: the entry
        :rtype: dict[str, Any]
        """
        entry = asdict(self)
        entry.pop("edits")
        return entry


def _literal(text: str) -> str | bool | int:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text.startswith("'"):
        return text[1:-1].replace("''", "'")
    return int(text)


def parse_point(sql: str) -> PointSql | str:
    """a plain point read, or why it is not one.

    :param sql: the statement
    :ptype sql: str
    :return: the parsed read, or the refusal's reason
    :rtype: PointSql | str
    """
    text = " ".join(sql.split())
    if "{" in text or "}" in text:
        return "interpolated"
    if text.startswith("/*") or "--" in text:
        return "a comment or hint in the statement"
    match = _SELECT.match(text)
    if match is None:
        return "not SELECT <columns> FROM <table> WHERE ..."
    alias = match.group("alias")
    if alias is not None and alias.lower() in _KEYWORDS:
        return "not SELECT <columns> FROM <table> WHERE ..."
    where = match.group("where")
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", where)
    if re.search(r"\b(order\s+by|limit|offset|group\s+by|for\s+update|for\s+share)\b", masked, re.IGNORECASE):
        return "ORDER BY / LIMIT / GROUP BY / a row lock"
    qualifiers = {None, alias, match.group("table").split(".")[-1]}
    columns: tuple[str, ...] | None
    if match.group("columns").strip() == "*":
        columns = None
    else:
        picked = []
        for item in match.group("columns").split(","):
            column = _COLUMN.match(item.strip())
            if column is None or column.group("qualifier") not in qualifiers:
                return "an expression, alias or DISTINCT in the select list"
            picked.append(column.group("column"))
        columns = tuple(picked)
    pins = _pins(where, masked, qualifiers)
    if isinstance(pins, str):
        return pins
    numbers = sorted({pin.param for pin in pins if pin.param is not None})
    if numbers != list(range(1, len(numbers) + 1)):
        return "parameters not numbered $1..$n"
    if not any(pin.kind in {"param", "any"} for pin in pins):
        return "no bound parameter (literals only): a constant read, judged by hand"
    return PointSql(table=match.group("table").lower(), columns=columns, pins=tuple(pins))


def _pins(where: str, masked: str, qualifiers: set[str | None]) -> list[Pin] | str:
    """each column a ``WHERE`` of equalities pins (``= $n``, ``= <literal>``, ``IS NULL``, joined by ``AND``), or why
    it is not one.

    :param where: the ``WHERE`` clause's text
    :ptype where: str
    :param masked: the same text, its quoted literals masked
    :ptype masked: str
    :param qualifiers: the names a column may be qualified by (None: unqualified)
    :ptype qualifiers: set[str | None]
    :return: the pins, or the refusal's reason
    :rtype: list[Pin] | str
    """
    pins: list[Pin] = []
    starts = [0, *(found.end() for found in _AND.finditer(masked))]
    ends = [found.start() for found in _AND.finditer(masked)] + [len(where)]
    for start, end in zip(starts, ends, strict=True):
        conjunct = where[start:end].strip()
        for pattern, kind in (
            (_PARAM_PIN, "param"),
            (_LITERAL_PIN, "literal"),
            (_NULL_PIN, "null"),
            (_ANY_PIN, "any"),
            (_BOOL_PIN, "bool"),
        ):
            pinned = pattern.match(conjunct)
            if pinned is not None and not (kind == "bool" and pinned.group("column").lower() in _NOT_COLUMNS):
                break
        else:
            return f"a predicate other than equality: {conjunct!r}"
        if pinned.group("q") not in qualifiers:
            return f"a column of another relation: {conjunct!r}"
        name = pinned.group("column").lower()
        if kind == "param":
            pins.append(Pin(name, "param", param=int(pinned.group("n"))))
        elif kind == "any":
            cast = pinned.group("cast")
            pins.append(Pin(name, "any", param=int(pinned.group("n")), cast=cast.lower() if cast else None))
        elif kind == "literal":
            pins.append(Pin(name, "literal", literal=_literal(pinned.group("literal"))))
        elif kind == "bool":
            pins.append(Pin(name, "literal", literal=pinned.group("not") is None))
        else:
            pins.append(Pin(name, "null"))
    if len({pin.column for pin in pins}) != len(pins):
        return "a column pinned twice"
    return pins


@dataclass(frozen=True)
class WriteSql:
    """a keyed write by equality, parsed: ``UPDATE <table> SET <col> = <value>, ... WHERE <pins>`` or ``DELETE FROM
    <table> WHERE <pins>``.

    :ivar kind: ``update`` or ``delete``
    :ivar table: the table as written (``schema.table`` kept)
    :ivar sets: each column an update sets (:class:`Pin`: ``param``, ``literal``, ``null``, or ``now`` for ``now()``)
    :ivar pins: each pinned column, in the statement's order
    :ivar returning: the columns its ``RETURNING`` names, in order; none when it returns nothing
    """

    kind: str
    table: str
    sets: tuple[Pin, ...]
    pins: tuple[Pin, ...]
    returning: tuple[str, ...] = ()


_UPDATE: Final[re.Pattern[str]] = re.compile(
    rf"^update\s+(?P<table>{_IDENT}(?:\.{_IDENT})?)\s+set\s+(?P<sets>.+?)\s+where\s+(?P<where>.+?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_DELETE: Final[re.Pattern[str]] = re.compile(
    rf"^delete\s+from\s+(?P<table>{_IDENT}(?:\.{_IDENT})?)\s+where\s+(?P<where>.+?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SET_PARAM: Final[re.Pattern[str]] = re.compile(rf"^(?P<column>{_IDENT})\s*=\s*\$(?P<n>\d+)$")
_SET_LITERAL: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<column>{_IDENT})\s*=\s*(?P<literal>'(?:[^']|'')*'|true|false|-?\d+)$", re.IGNORECASE
)
_SET_NULL: Final[re.Pattern[str]] = re.compile(rf"^(?P<column>{_IDENT})\s*=\s*null$", re.IGNORECASE)
_SET_NOW: Final[re.Pattern[str]] = re.compile(rf"^(?P<column>{_IDENT})\s*=\s*now\(\)$", re.IGNORECASE)
_SET_INCREMENT: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<column>{_IDENT})\s*=\s*(?P<same>{_IDENT})\s*(?P<sign>[-+])\s*(?P<amount>\d+)$", re.IGNORECASE
)
_SET_EXPRESSION: Final[re.Pattern[str]] = re.compile(rf"^(?P<column>{_IDENT})\s*=\s*(?P<expression>.+)$", re.DOTALL)
_PARAMETER: Final[re.Pattern[str]] = re.compile(r"\$(\d+)")


def _top_level_commas(masked: str) -> list[int]:
    """where ``masked`` (its quoted literals masked) has a comma outside every parenthesis."""
    depth = 0
    found = []
    for at, char in enumerate(masked):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            found.append(at)
    return found


def _fragment(column: str, expression: str) -> Pin | str:
    """a SET expression (``GREATEST(col, $1)``, ``COALESCE($5::jsonb, col)``) as a ``fragment`` pin, its parameters
    renumbered from ``$1``; or why it is not one (a subquery, or no parameter and no column of its own: judged by
    hand)."""
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", expression)
    if re.search(r"\bselect\b", masked, re.IGNORECASE):
        return f"a SET subquery: {column} = {expression!r}"
    order: list[int] = []
    for number in _PARAMETER.findall(masked):
        if int(number) not in order:
            order.append(int(number))
    renumbered = _PARAMETER.sub(lambda found: f"${order.index(int(found.group(1))) + 1}", expression)
    return Pin(column, "fragment", literal=" ".join(renumbered.split()), params=tuple(order))


_RETURNING: Final[re.Pattern[str]] = re.compile(r"\s+returning\s+(?P<columns>[^;]+?)\s*;?\s*$", re.IGNORECASE)


def _returning(text: str, masked: str) -> tuple[str, tuple[str, ...]] | str:
    """a write's text without its trailing ``RETURNING`` and the plain columns it names, or why they are not plain.

    :param text: the statement, one line
    :ptype text: str
    :param masked: the same text, its quoted literals masked
    :ptype masked: str
    :return: the text before ``RETURNING`` and the columns (none when it has no ``RETURNING``), or the refusal
    :rtype: tuple[str, tuple[str, ...]] | str
    """
    found = _RETURNING.search(masked)
    if found is None:
        return text, ()
    columns = []
    for item in found.group("columns").split(","):
        column = re.fullmatch(_IDENT, item.strip(), re.IGNORECASE)
        if column is None:
            return f"a RETURNING other than plain columns: {found.group('columns').strip()!r}"
        columns.append(column.group().lower())
    if len(set(columns)) != len(columns):
        return "a column returned twice"
    return text[: found.start()], tuple(columns)


def parse_write(sql: str) -> WriteSql | str:
    """a keyed write by equality, or why it is not one: every ``SET`` a column to a parameter, a literal, ``NULL`` or
    ``now()``; every ``WHERE`` conjunct an equality :func:`parse_point` takes; a ``RETURNING`` of plain columns at
    most; no alias, join, subquery or expression.

    :param sql: the statement
    :ptype sql: str
    :return: the parsed write, or the refusal's reason
    :rtype: WriteSql | str
    """
    text = " ".join(sql.split())
    if "{" in text or "}" in text:
        return "interpolated"
    if text.startswith("/*") or "--" in text:
        return "a comment or hint in the statement"
    masked_text = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", text)
    returned = _returning(text, masked_text)
    if isinstance(returned, str):
        return returned
    text, returning = returned
    masked_text = masked_text[: len(text)]
    if re.search(r"\breturning\b", masked_text, re.IGNORECASE):
        return "a RETURNING not at the statement's end"
    if re.search(
        r"\b(from|using)\b",
        masked_text[len("delete from ") :] if masked_text.lower().startswith("delete") else masked_text,
        re.IGNORECASE,
    ):
        return "a join (FROM / USING)"
    update = _UPDATE.match(text)
    delete = _DELETE.match(text) if update is None else None
    if update is None and delete is None:
        return "not UPDATE <table> SET ... WHERE ... nor DELETE FROM <table> WHERE ..."
    match = update if update is not None else delete
    assert match is not None
    table = match.group("table")
    where = match.group("where")
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", where)
    pins = _pins(where, masked, {None, table.split(".")[-1]})
    if isinstance(pins, str):
        return pins
    sets: list[Pin] = []
    if update is not None:
        assignments = update.group("sets")
        masked_sets = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", assignments)
        commas = _top_level_commas(masked_sets)
        starts = [0, *(at + 1 for at in commas)]
        ends = [*commas, len(assignments)]
        for start, end in zip(starts, ends, strict=True):
            assignment = assignments[start:end].strip()
            for pattern, kind in (
                (_SET_PARAM, "param"),
                (_SET_LITERAL, "literal"),
                (_SET_NULL, "null"),
                (_SET_NOW, "now"),
                (_SET_INCREMENT, "increment"),
                (_SET_EXPRESSION, "fragment"),
            ):
                found = pattern.match(assignment)
                if found is not None and not (
                    kind == "increment" and found.group("same").lower() != found.group("column").lower()
                ):
                    break
            else:
                return f"a SET other than a column = expression: {assignment!r}"
            name = found.group("column").lower()
            if kind == "param":
                sets.append(Pin(name, "param", param=int(found.group("n"))))
            elif kind == "literal":
                sets.append(Pin(name, "literal", literal=_literal(found.group("literal"))))
            elif kind == "increment":
                amount = int(found.group("amount"))
                sets.append(Pin(name, "increment", literal=-amount if found.group("sign") == "-" else amount))
            elif kind == "fragment":
                fragment = _fragment(name, found.group("expression").strip())
                if isinstance(fragment, str):
                    return fragment
                sets.append(fragment)
            else:
                sets.append(Pin(name, kind))
        if len({pin.column for pin in sets}) != len(sets):
            return "a column set twice"
        if any(pin.kind == "now" and pin.column in returning for pin in sets):
            return "a now() column returned (each side's transaction stamps its own instant)"
    numbers = sorted({number for pin in (*sets, *pins) for number in pin.numbers()})
    if numbers != list(range(1, len(numbers) + 1)):
        return "parameters not numbered $1..$n"
    if not any(pin.kind in {"param", "any"} for pin in pins):
        return "no bound parameter in the WHERE (literals only): a constant write, judged by hand"
    return WriteSql(
        kind="update" if update is not None else "delete",
        table=table.lower(),
        sets=tuple(sets),
        pins=tuple(pins),
        returning=returning,
    )


@dataclass
class _Module:
    """one parsed module, its parents and its source."""

    path: Path
    source: str
    tree: ast.Module
    parents: dict[int, ast.AST]

    @classmethod
    def read(cls, path: Path) -> _Module:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        return cls(path, source, tree, parents)

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def segment(self, node: ast.AST) -> str:
        text = ast.get_source_segment(self.source, node)
        if text is None:
            raise ValueError(f"no source for {ast.dump(node)[:80]}")
        return text

    def scope(self, node: ast.AST) -> ast.AST:
        """the function or module a node is in."""
        current = self.parent(node)
        while current is not None and not isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.Module):
            current = self.parent(current)
        return current if current is not None else self.tree


def _plain_simple(node: ast.expr) -> bool:
    """a name, an attribute chain of names or a constant: safe to write twice."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name | ast.Constant)


def _call_of(module: _Module, use: ast.expr, methods: frozenset[str] = frozenset(_HELPER_OF)) -> ast.Call | str:
    """the awaited call of one of ``methods`` (``fetch``/``fetchrow``/``fetchval`` for a read; ``execute``, its status
    discarded, for a write) whose first argument is ``use``, or why it is not one."""
    call = module.parent(use)
    if not (isinstance(call, ast.Call) and call.args and call.args[0] is use):
        return "not a call's first argument nor a constant"
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in methods):
        return f"sent by {ast.unparse(call.func)[:60]}, not {'/'.join(sorted(methods))}"
    awaited = module.parent(call)
    if not isinstance(awaited, ast.Await):
        return "the call is not awaited where it is made"
    if methods == _WRITE_METHODS and not isinstance(module.parent(awaited), ast.Expr):
        return "the statement's status is read (the helper answers a count, not a status)"
    if call.keywords:
        return "keyword arguments to the call (timeout=, ...)"
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return "starred arguments"
    return call


def _sent_by(
    module: _Module, literal: ast.Constant, methods: frozenset[str] = frozenset(_HELPER_OF)
) -> tuple[list[ast.Call], ast.Assign | ast.AnnAssign | None] | str:
    """every awaited ``fetch``/``fetchrow``/``fetchval`` call sending a statement literal (written at the call, or a
    constant -- ``.strip()`` of one too -- every one of whose uses is such a call), and the constant; or why not."""
    holder: ast.AST = literal
    parent = module.parent(literal)
    if (
        isinstance(parent, ast.Attribute)
        and parent.attr == "strip"
        and isinstance(module.parent(parent), ast.Call)
        and not module.parent(parent).args  # type: ignore[union-attr]
    ):
        holder = module.parent(parent)  # type: ignore[assignment]
        parent = module.parent(holder)
    if isinstance(parent, ast.Assign | ast.AnnAssign) and parent.value is holder:
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return "bound to something other than one name"
        name = targets[0].id
        scope = module.scope(parent)
        uses = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        ]
        if not uses:
            return f"its constant {name} is never used"
        calls = []
        for use in uses:
            call = _call_of(module, use, methods)
            if isinstance(call, str):
                return f"its constant {name}: {call}"
            calls.append(call)
        return calls, parent
    if holder is not literal:
        return "not a call's first argument nor a constant"
    call = _call_of(module, literal, methods)
    return call if isinstance(call, str) else ([call], None)


def _function_at(module: _Module, node: ast.AST) -> str:
    """the qualified name of the function (or class) holding ``node``, ``<module>`` at the top, as the walk names it."""
    names: list[str] = []
    current = module.parent(node)
    while current is not None:
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(current.name)
        current = module.parent(current)
    return ".".join(reversed(names)) or "<module>"


def _python_literal(value: str | bool | int | None) -> str:
    return repr(value)


def _render(rewrite: Rewrite, receiver: str) -> str:
    """the helper call, written in the statement's call's place."""
    by = "{" + ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.pins) + "}"
    parts = [receiver, repr(rewrite.table), f"by={by}"]
    if rewrite.helper == "read_value":
        assert rewrite.columns is not None
        parts.append(f"column={rewrite.columns[0]!r}")
    else:
        if rewrite.columns is not None:
            listed = ", ".join(repr(column) for column in rewrite.columns)
            parts.append(f"columns=({listed}{',' if len(rewrite.columns) == 1 else ''})")
        parts.append("records=True")
    return f"{rewrite.helper}({', '.join(parts)})"


def _value_text(pin: Mapping[str, Any], args: Sequence[str]) -> str:
    """a pinned or set column's value as the helper call writes it: the argument, an ``AnyOf`` of it, a literal,
    None, or ``now()``."""
    if pin["kind"] == "param":
        return str(args[pin["param"] - 1])
    if pin["kind"] == "any":
        return f"AnyOf({args[pin['param'] - 1]}, cast={pin.get('cast')!r})"
    if pin["kind"] == "increment":
        return f"Increment({pin['literal']!r})"
    if pin["kind"] == "fragment":
        bound = "".join(f"{args[number - 1]}, " for number in pin["params"])
        return f"Fragment({pin['literal']!r}, ({bound.rstrip(' ') if len(pin['params']) == 1 else bound[:-2]}))"
    if pin["kind"] == "literal":
        return _python_literal(pin["literal"])
    if pin["kind"] == "now":
        return 'Fragment("now()")'
    return "None"


#: what a ``RETURNING`` write's helper answer is read through, by the connection method it replaces
_RETURNED: Final[dict[str, str]] = {"fetch": "rows", "fetchrow": "row", "fetchval": "value"}


def _render_write(rewrite: Rewrite, receiver: str) -> str:
    """the write helper call, written in the ``execute`` call's place; for a ``RETURNING`` write, the awaited call
    read as the ``fetch`` (``.rows``), ``fetchrow`` (``.row``) or ``fetchval`` (``.value``) it replaces, written in
    the awaited call's place."""
    by = "{" + ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.pins) + "}"
    parts = [receiver, repr(rewrite.table)]
    if rewrite.helper == "update_keyed":
        values = ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.sets)
        parts.append(f"values={{{values}}}")
    parts.append(f"by={by}")
    if not rewrite.returning:
        return f"{rewrite.helper}({', '.join(parts)})"
    listed = ", ".join(repr(column) for column in rewrite.returning)
    parts.append(f"returning=({listed}{',' if len(rewrite.returning) == 1 else ''})")
    return f"(await {rewrite.helper}({', '.join(parts)})).{_RETURNED[rewrite.method]}"


def _statement_literal(module: _Module, line: int, verb: str = "select") -> ast.Constant | str:
    """the string literal holding the statement at ``line``, one starting with ``verb``."""
    found = [
        node
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Constant | ast.JoinedStr) and getattr(node, "lineno", None) == line
    ]
    if any(isinstance(node, ast.JoinedStr) for node in found):
        return "interpolated (an f-string)"
    strings = [
        node
        for node in found
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.match(rf"\s*{verb}\b", node.value, re.I)
    ]
    if len(strings) != 1:
        return f"{len(strings)} {verb.upper()} literals on its line"
    return strings[0]


def _referenced_elsewhere(repo_root: Path, path: Path, name: str) -> bool:
    """whether a constant's name is named in any other module or migration of the repo (an index purpose, a test)."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for other in repo_root.rglob("*"):
        if other.suffix not in {".py", ".sql", ".yaml"} or other == path or ".venv" in other.parts:
            continue
        if "node_modules" in other.parts or ".git" in other.parts:
            continue
        try:
            if pattern.search(other.read_text(encoding="utf-8")):
                return True
        except UnicodeDecodeError, OSError:  # NOSILENT: a file no text can be read from names no constant
            continue
    return False


def _constant_edit(module: _Module, constant: ast.Assign | ast.AnnAssign) -> tuple[int, int, int, int, str]:
    """the edit removing a constant's assignment, its ``#:`` comment lines above it with it."""
    lines = module.source.splitlines()
    first = constant.lineno
    while first > 1 and lines[first - 2].lstrip().startswith("#:"):
        first -= 1
    last = constant.end_lineno or constant.lineno
    return (first, 0, last + 1, 0, "")


def plan_point_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every point finding of the repo, as the rewrite it takes or the refusal it gets.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    whole_hits, scan = _hits(config)
    unkeyed: dict[tuple[str, int], str] = {(hit.path, hit.line): hit.symbol for hit in whole_hits}
    hits = [
        hit for hit in _helper_hits(scan.statements, config.helper_modules, _producers(config)) if hit.shape == "point"
    ]
    modules: dict[str, _Module] = {}
    rewrites: list[Rewrite] = []
    refusals: list[Refusal] = []
    for hit in hits:
        statement = hit.statement
        one_line = " ".join(statement.text.split())

        def refuse(reason: str) -> None:
            refusals.append(
                Refusal(statement.path, statement.line, statement.function, hit.shape, hit.table, reason, one_line)
            )

        if (statement.path, statement.line) in unkeyed:
            refuse(
                f"the no-whole-reads rule finds it unkeyed ({unkeyed[statement.path, statement.line]}): a real fix first"
            )
            continue
        parsed = parse_point(statement.text)
        if isinstance(parsed, str):
            refuse(parsed)
            continue
        module = modules.get(statement.path)
        if module is None:
            module = modules[statement.path] = _Module.read(config.repo_root / statement.path)
        literal = _statement_literal(module, statement.line)
        if isinstance(literal, str):
            refuse(literal)
            continue
        old_sql = literal.value
        assert isinstance(old_sql, str)
        if " ".join(old_sql.split()) != " ".join(statement.text.split()):
            refuse("the literal at its line is not the statement the walk found")
            continue
        sent = _sent_by(module, literal)
        if isinstance(sent, str):
            refuse(sent)
            continue
        calls, constant = sent
        if constant is not None:
            target = constant.targets[0] if isinstance(constant, ast.Assign) else constant.target
            assert isinstance(target, ast.Name)
            if _referenced_elsewhere(config.repo_root, module.path, target.id):
                refuse(f"its constant {target.id} is named elsewhere in the repo")
                continue
        numbers = sorted({pin.param for pin in parsed.pins if pin.param is not None})
        uses = Counter(pin.param for pin in parsed.pins if pin.param is not None)
        planned: list[Rewrite] = []
        for call in calls:
            arguments = call.args[1:]
            assert isinstance(call.func, ast.Attribute)
            method = call.func.attr
            problem = ""
            if len(arguments) != len(numbers):
                problem = f"{len(arguments)} arguments for {len(numbers)} parameters"
            elif any(count > 1 and not _plain_simple(arguments[number - 1]) for number, count in uses.items()):
                problem = "a parameter used twice whose argument is not a plain name"
            elif method == "fetchval" and parsed.columns is None:
                problem = "fetchval of SELECT *"
            if problem:
                refuse(problem)
                planned = []
                break
            rewrite = Rewrite(
                path=statement.path,
                function=_function_at(module, call) if constant is not None else statement.function,
                line=statement.line,
                table=parsed.table,
                old_sql=old_sql,
                method=method,
                helper=_HELPER_OF[method],
                columns=None if parsed.columns is None else list(parsed.columns),
                pins=[asdict(pin) for pin in parsed.pins],
                args=[module.segment(argument) for argument in arguments],
            )
            rewrite.new_call = _render(rewrite, module.segment(call.func.value))
            assert call.end_lineno is not None and call.end_col_offset is not None
            rewrite.edits.append((call.lineno, call.col_offset, call.end_lineno, call.end_col_offset, rewrite.new_call))
            planned.append(rewrite)
        if planned and constant is not None:
            planned[0].edits.append(_constant_edit(module, constant))
        rewrites.extend(planned)
    return rewrites, refusals


def plan_write_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every write finding of the repo that is an ``UPDATE`` / ``DELETE`` by equality (:func:`parse_write`), sent by
    an awaited ``execute`` whose status is discarded, as the :func:`~threetears.core.keyed.update_keyed` /
    :func:`~threetears.core.keyed.delete_keyed` call it takes, or the refusal it gets.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    whole_hits, scan = _hits(config)
    unkeyed: dict[tuple[str, int], str] = {(hit.path, hit.line): hit.symbol for hit in whole_hits}
    hits = [
        hit for hit in _helper_hits(scan.statements, config.helper_modules, _producers(config)) if hit.shape == "write"
    ]
    modules: dict[str, _Module] = {}
    rewrites: list[Rewrite] = []
    refusals: list[Refusal] = []
    for hit in hits:
        statement = hit.statement
        one_line = " ".join(statement.text.split())

        def refuse(reason: str) -> None:
            refusals.append(
                Refusal(statement.path, statement.line, statement.function, hit.shape, hit.table, reason, one_line)
            )

        if (statement.path, statement.line) in unkeyed:
            refuse(
                f"the no-whole-reads rule finds it unkeyed ({unkeyed[statement.path, statement.line]}): a real fix first"
            )
            continue
        parsed = parse_write(statement.text)
        if isinstance(parsed, str):
            refuse(parsed)
            continue
        module = modules.get(statement.path)
        if module is None:
            module = modules[statement.path] = _Module.read(config.repo_root / statement.path)
        literal = _statement_literal(module, statement.line, parsed.kind)
        if isinstance(literal, str):
            refuse(literal)
            continue
        old_sql = literal.value
        assert isinstance(old_sql, str)
        if " ".join(old_sql.split()) != " ".join(statement.text.split()):
            refuse("the literal at its line is not the statement the walk found")
            continue
        sent = _sent_by(module, literal, frozenset(_HELPER_OF) if parsed.returning else _WRITE_METHODS)
        if isinstance(sent, str):
            refuse(sent)
            continue
        calls, constant = sent
        if constant is not None:
            target = constant.targets[0] if isinstance(constant, ast.Assign) else constant.target
            assert isinstance(target, ast.Name)
            if _referenced_elsewhere(config.repo_root, module.path, target.id):
                refuse(f"its constant {target.id} is named elsewhere in the repo")
                continue
        bound = [number for pin in (*parsed.sets, *parsed.pins) for number in pin.numbers()]
        numbers = sorted(set(bound))
        uses = Counter(bound)
        planned: list[Rewrite] = []
        for call in calls:
            arguments = call.args[1:]
            problem = ""
            if len(arguments) != len(numbers):
                problem = f"{len(arguments)} arguments for {len(numbers)} parameters"
            elif any(count > 1 and not _plain_simple(arguments[number - 1]) for number, count in uses.items()):
                problem = "a parameter used twice whose argument is not a plain name"
            if problem:
                refuse(problem)
                planned = []
                break
            assert isinstance(call.func, ast.Attribute)
            method = call.func.attr
            returning = list(parsed.returning[:1] if method == "fetchval" else parsed.returning)
            rewrite = Rewrite(
                path=statement.path,
                function=_function_at(module, call) if constant is not None else statement.function,
                line=statement.line,
                table=parsed.table,
                old_sql=old_sql,
                method=method,
                helper=_WRITE_HELPER[parsed.kind],
                columns=None,
                pins=[asdict(pin) for pin in parsed.pins],
                args=[module.segment(argument) for argument in arguments],
                shape="write",
                sets=[asdict(pin) for pin in parsed.sets],
                returning=returning,
            )
            rewrite.new_call = _render_write(rewrite, module.segment(call.func.value))
            # a RETURNING write replaces the awaited call (the helper's answer is read through .rows/.row/.value)
            awaited = module.parent(call)
            assert isinstance(awaited, ast.Await)
            replaced: ast.expr = awaited if returning else call
            assert replaced.end_lineno is not None and replaced.end_col_offset is not None
            rewrite.edits.append(
                (replaced.lineno, replaced.col_offset, replaced.end_lineno, replaced.end_col_offset, rewrite.new_call)
            )
            planned.append(rewrite)
        if planned and constant is not None:
            planned[0].edits.append(_constant_edit(module, constant))
        rewrites.extend(planned)
    return rewrites, refusals


def _splice(source: str, edits: list[tuple[int, int, int, int, str]]) -> str:
    """the source with each edit made; positions are 1-based lines and UTF-8 byte columns, as ``ast`` gives them."""
    lines = source.encode("utf-8").splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    data = source.encode("utf-8")

    def offset(line: int, column: int) -> int:
        return starts[line - 1] + column if line - 1 < len(lines) else len(data)

    placed = sorted(((offset(a, b), offset(c, d), text) for a, b, c, d, text in edits), reverse=True)
    for (start, end, _), (next_start, _, _) in zip(placed[1:], placed, strict=False):
        if end > next_start:
            raise ValueError("overlapping edits")
    for start, end, text in placed:
        data = data[:start] + text.encode("utf-8") + data[end:]
    return data.decode("utf-8")


def _with_import(source: str, names: set[str]) -> str:
    """the source importing ``names`` from the helpers' module: added to an existing top-level ``from`` import of it,
    or as a new line after the module's last top-level import."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == _HELPER_MODULE and node.level == 0:
            have = {alias.name for alias in node.names if alias.asname is None}
            wanted = sorted(have | names | {f"{a.name} as {a.asname}" for a in node.names if a.asname})
            text = f"from {_HELPER_MODULE} import {', '.join(wanted)}\n"
            assert node.end_lineno is not None
            return "".join(lines[: node.lineno - 1]) + text + "".join(lines[node.end_lineno :])
    last = 0
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            last = node.end_lineno or node.lineno
        elif last:
            break
    text = f"from {_HELPER_MODULE} import {', '.join(sorted(names))}\n"
    return "".join(lines[:last]) + text + "".join(lines[last:])


def apply_rewrites(repo_root: Path, rewrites: list[Rewrite], *, ruff: str | None = "ruff") -> list[str]:
    """make every rewrite, import the helpers, and format the touched modules.

    :param repo_root: the repo
    :ptype repo_root: Path
    :param rewrites: the rewrites
    :ptype rewrites: list[Rewrite]
    :param ruff: the ruff to run (the repo's own); none when None (the source is left as spliced)
    :ptype ruff: str | None
    :return: the touched modules, repo-relative
    :rtype: list[str]
    """
    by_path: dict[str, list[Rewrite]] = defaultdict(list)
    for rewrite in rewrites:
        by_path[rewrite.path].append(rewrite)
    for relative, mine in by_path.items():
        path = repo_root / relative
        source = _splice(path.read_text(encoding="utf-8"), [edit for rewrite in mine for edit in rewrite.edits])
        names = {rewrite.helper for rewrite in mine}
        if any(pin["kind"] in {"now", "fragment"} for rewrite in mine for pin in rewrite.sets):
            names.add("Fragment")
        if any(pin["kind"] == "increment" for rewrite in mine for pin in rewrite.sets):
            names.add("Increment")
        if any(pin["kind"] == "any" for rewrite in mine for pin in rewrite.pins):
            names.add("AnyOf")
        source = _with_import(source, names)
        ast.parse(source)
        path.write_text(source, encoding="utf-8")
    touched = sorted(by_path)
    if touched and ruff is not None:
        subprocess.run([ruff, "check", "--select", "I", "--fix", "--quiet", *touched], cwd=repo_root, check=False)
        subprocess.run([ruff, "format", "--quiet", *touched], cwd=repo_root, check=True)
    return touched


def write_manifest(path: Path, rewrites: list[Rewrite]) -> None:
    """add the rewrites to the repo's manifest (one entry a rewrite, kept across batches).

    :param path: the manifest
    :ptype path: Path
    :param rewrites: the rewrites
    :ptype rewrites: list[Rewrite]
    :return: nothing
    :rtype: None
    """
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    seen = {(entry["path"], entry["function"], entry["old_sql"], tuple(entry["args"])) for entry in entries}
    for rewrite in rewrites:
        if (rewrite.path, rewrite.function, rewrite.old_sql, tuple(rewrite.args)) not in seen:
            entries.append(rewrite.record())
    path.write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n", encoding="utf-8")


_TEST_MODULE: Final[
    str
] = '''"""GENERATED by ``python -m threetears.enforcement.no_whole_reads.rewrite test-module``: do not edit by hand.

Every statement the use-the-helper rewrite replaced (``{manifest}``) is held to the statement it replaced:
:func:`threetears.core.testing.rewrites.assert_rewrite_same` seeds its table on a migrated database for every case of
``assert_same_results`` and compares the old statement with the helper call, row for row, on PostgreSQL and on
YugabyteDB; :func:`~threetears.core.testing.rewrites.assert_source_says` checks the module still makes the recorded call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from threetears.core.testing.rewrites import assert_rewrite_same, assert_source_says, entry_id, load_manifest
{overrides_import}
__all__: list[str] = []

_REPO = Path(__file__).resolve().parents[{depth}]
_ENTRIES = load_manifest(_REPO / "{manifest}")


@pytest.mark.parametrize("entry", _ENTRIES, ids=entry_id)
def test_the_module_makes_the_recorded_call(entry: dict[str, Any]) -> None:
    """Flips if a rewritten module no longer makes the helper call its manifest entry records."""
    assert_source_says(_REPO, entry)


@pytest.mark.parametrize("entry", _ENTRIES, ids=entry_id)
async def test_on_postgresql_the_helper_gives_what_the_statement_gave(entry: dict[str, Any], {pg}: Any) -> None:
    """Flips if a helper call reads other rows than the statement it replaced, on PostgreSQL."""
    await assert_rewrite_same({pg}, entry, overrides={overrides})


@pytest.mark.parametrize("entry", _ENTRIES, ids=entry_id)
async def test_on_yugabytedb_the_helper_gives_what_the_statement_gave(entry: dict[str, Any], {yb}: Any) -> None:
    """Flips if a helper call reads other rows than the statement it replaced, on YugabyteDB."""
    await assert_rewrite_same({yb}, entry, overrides={overrides})
'''


def render_test_module(
    module: Path, manifest: str, *, pg_fixture: str, yb_fixture: str, overrides: str | None = None
) -> str:
    """the repo's one generated equivalence test module.

    :param module: where it is written, repo-relative (its depth finds the repo root)
    :ptype module: Path
    :param manifest: the manifest, repo-relative
    :ptype manifest: str
    :param pg_fixture: the repo's fixture of a superuser connection to a migrated PostgreSQL database
    :ptype pg_fixture: str
    :param yb_fixture: the repo's fixture of a superuser connection to a migrated YugabyteDB database
    :ptype yb_fixture: str
    :param overrides: ``module:attribute`` of the repo's seed overrides, none when None
    :ptype overrides: str | None
    :return: the module's text
    :rtype: str
    """
    imported = ""
    name = "None"
    if overrides is not None:
        source, _, name = overrides.partition(":")
        imported = f"\nfrom {source} import {name}\n"
    return _TEST_MODULE.format(
        manifest=manifest,
        depth=len(module.parts) - 1,
        pg=pg_fixture,
        yb=yb_fixture,
        overrides_import=imported,
        overrides=name,
    )


def _after_check(config: NoWholeReadsConfig, before: list[Any], rewrites: list[Rewrite]) -> list[str]:
    """each rewritten function's whole-read findings the rewrite added (none, when it is sound)."""
    after, _ = _hits(config)
    functions = {(rewrite.path, rewrite.function) for rewrite in rewrites}
    old = Counter((hit.path, hit.function, hit.symbol) for hit in before if (hit.path, hit.function) in functions)
    new = Counter((hit.path, hit.function, hit.symbol) for hit in after if (hit.path, hit.function) in functions)
    return [f"{path} {function}: {symbol}" for path, function, symbol in (new - old).elements()]


def _config(repo: Path, spec: str) -> NoWholeReadsConfig:
    module_name, _, attribute = spec.partition(":")
    sys.path.insert(0, str(repo))
    config = getattr(importlib.import_module(module_name), attribute)
    assert isinstance(config, NoWholeReadsConfig)
    return config


def _say(line: str) -> None:
    """write one line of the command's output to stdout: the command's product, so a stream write, not a log record.

    :param line: the line
    :ptype line: str
    """
    sys.stdout.write(f"{line}\n")


def main(argv: list[str] | None = None) -> int:
    """the command line: ``point`` (reads), ``write`` (keyed ``UPDATE`` / ``DELETE`` by equality), ``test-module``.

    :param argv: the arguments
    :ptype argv: list[str] | None
    :return: the exit status: 0, or 1 when the after-check found a rewrite that added a whole-read finding
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="threetears.enforcement.no_whole_reads.rewrite")
    parser.add_argument("shape", choices=["point", "write", "test-module"])
    parser.add_argument("--module", type=Path, help="test-module: where the generated test goes, repo-relative")
    parser.add_argument("--pg-fixture", default="conn")
    parser.add_argument("--yb-fixture", default="yugabyte_conn")
    parser.add_argument("--overrides", help="test-module: module:attribute of the repo's seed overrides")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", default="", help="module:attribute of the repo's NoWholeReadsConfig")
    parser.add_argument("--manifest", type=Path, required=True, help="repo-relative manifest path")
    parser.add_argument("--refusals", type=Path, help="where to write the refusal list (JSON)")
    parser.add_argument("--ruff", default="ruff")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    repo = options.repo.resolve()
    if options.shape == "test-module":
        text = render_test_module(
            options.module,
            str(options.manifest),
            pg_fixture=options.pg_fixture,
            yb_fixture=options.yb_fixture,
            overrides=options.overrides,
        )
        (repo / options.module).write_text(text, encoding="utf-8")
        subprocess.run([options.ruff, "format", "--quiet", str(options.module)], cwd=repo, check=True)
        _say(f"wrote {options.module}")
        return 0
    config = _config(repo, options.config)
    before, _ = _hits(config)
    rewrites, refusals = (plan_write_rewrites if options.shape == "write" else plan_point_rewrites)(config)
    reasons = Counter(refusal.reason.split(":")[0].split(" (")[0] for refusal in refusals)
    _say(f"{options.shape}: {len(rewrites)} rewrites, {len(refusals)} refusals")
    for reason, count in reasons.most_common():
        _say(f"  refused {count:4d}  {reason}")
    if options.refusals is not None:
        options.refusals.write_text(
            json.dumps([asdict(refusal) for refusal in refusals], indent=1) + "\n", encoding="utf-8"
        )
    if options.dry_run:
        for rewrite in rewrites:
            _say(f"  {rewrite.path}:{rewrite.line} {rewrite.function}: {rewrite.new_call}")
        return 0
    touched = apply_rewrites(repo, rewrites, ruff=options.ruff)
    write_manifest(repo / options.manifest, rewrites)
    added = _after_check(config, before, rewrites)
    _say(f"rewrote {len(rewrites)} statements in {len(touched)} modules; manifest {options.manifest}")
    for line in added:
        _say(f"  AFTER-CHECK: a rewrite added a whole-read finding: {line}")
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
