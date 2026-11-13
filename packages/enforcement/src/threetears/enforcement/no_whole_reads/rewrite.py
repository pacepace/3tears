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

**Listings** (shape ``list``): the same statement as a point read, sent by ``fetch``, with a trailing ``ORDER BY`` of
plain columns (``col [ASC|DESC] [NULLS FIRST|LAST]``, or none), becomes :func:`~threetears.core.keyed.read_by` with
``order=[Sort(...)]``, term for term: the same rows, in the same order, ties as the engine leaves them. Refused: a
``LIMIT`` / ``OFFSET`` (a page, not a listing), an expression or alias in the order, a listing sent by ``fetchrow`` /
``fetchval`` (its first row is a first-of read), and every point-read refusal.

**Keyed writes** (shape ``write``): ``UPDATE <table> SET <col> = $n | <literal> | NULL | now(), ... WHERE <pins>`` and
``DELETE FROM <table> WHERE <pins>`` become :func:`~threetears.core.keyed.update_keyed` /
:func:`~threetears.core.keyed.delete_keyed` with the same ``values`` and ``by``. Sent by an awaited ``execute`` whose
status is discarded; or, with a ``RETURNING`` of plain columns, by an awaited ``fetch`` / ``fetchrow`` / ``fetchval``,
the helper then called with ``returning=`` and its answer read as that method answered (``.rows``, ``.row``, ``.value``:
a list of rows, the first row or None, the first row's first column or None). A write whose status is read is
rewritten too when every read is one a count gives exactly (ruling 2026-11-12; there is no ``KeyedWrite.status``): the
awaited call becomes ``(await helper(...)).count``, read where it was awaited or bound once to a local, and each read
becomes the count's: ``status == "UPDATE n"`` / ``!=`` (the statement's own verb) as ``count == n``,
``status.endswith(" n")`` as ``count == n``, ``int(status.split()[-1])`` / ``parse_rowcount(status)`` /
``rows_affected(status)`` as the count, ``status.split()[-1] == "n"`` as ``count == n``. Any other read (returned,
logged, another verb's tag) is refused, and a status parser the rewrite leaves unused is no longer imported.

Splicing is by the call's ``ast`` span; the touched files are then put through ``ruff check --select I --fix`` (the
new import sorted in) and ``ruff format``, which every repo runs.

Usage::

    python -m threetears.enforcement.no_whole_reads.rewrite point --repo <root> \\
        --config tests.enforcement.test_no_whole_reads:_CONFIG --manifest tests/helper_rewrites.json [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import json
import keyword
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
#: ``col IS NOT DISTINCT FROM $n``: the column equals the argument, or holds NULL where the argument is None -- what
#: ``by={col: arg}`` sends (``= $n``, or ``IS NULL`` for None)
_NULLSAFE_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s+is\s+not\s+distinct\s+from\s+\$(?P<n>\d+)$", re.IGNORECASE
)
#: one literal of an ``IN`` list: a quoted text or an integer
_IN_LITERAL: Final[str] = r"'(?:[^']|'')*'|-?\d+"
#: ``col IN ('a', 'b')``: the column is one of a fixed set of literals -- ``by={col: AnyOf(('a', 'b'), cast=None)}``
_IN_LITERALS_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})\s+in\s*\(\s*(?P<literals>(?:{_IN_LITERAL})"
    rf"(?:\s*,\s*(?:{_IN_LITERAL}))*)\s*\)$",
    re.IGNORECASE,
)
_BOOL_PIN: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<not>not\s+)?(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})$", re.IGNORECASE
)
#: words a bare-column conjunct never is (a bare boolean column pins it TRUE)
_NOT_COLUMNS: Final[frozenset[str]] = frozenset({"true", "false", "null", "not", "and", "or"})
#: the pin kinds bound as a parameter: ``= $n``, ``= ANY($n)`` and ``IS NOT DISTINCT FROM $n``
_BOUND_KINDS: Final[frozenset[str]] = frozenset({"param", "any", "nullsafe"})
_AND: Final[re.Pattern[str]] = re.compile(r"\s+and\s+", re.IGNORECASE)
#: a listing's trailing ``ORDER BY`` (shape ``list``): split off the ``WHERE`` before its pins are read
_ORDER_BY: Final[re.Pattern[str]] = re.compile(r"\s+order\s+by\s+(?P<order>.+)$", re.IGNORECASE | re.DOTALL)
#: one ``ORDER BY`` term: a plain column, its direction and where its NULLs fall as written
_ORDER_TERM: Final[re.Pattern[str]] = re.compile(
    rf"^(?:(?P<q>{_IDENT})\.)?(?P<column>{_IDENT})(?:\s+(?P<direction>asc|desc))?(?:\s+nulls\s+(?P<nulls>first|last))?$",
    re.IGNORECASE,
)
_QUOTED: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")


@dataclass(frozen=True)
class Pin:
    """one column a point read pins.

    :ivar column: the column
    :ivar kind: ``param`` (``= $n``), ``any`` (``= ANY($n)``, ``= ANY($n::<type>[])``), ``literal`` (``= 'x'``,
        ``= TRUE``, ``= 3``, a bare boolean column for TRUE and ``NOT`` one for FALSE), ``null`` (``IS NULL``),
        ``nullsafe`` (``IS NOT DISTINCT FROM $n``: sent as ``= $n``, or ``IS NULL`` for a None argument) or ``anylit``
        (``IN ('a', 'b')``: an ``AnyOf`` of the literals, typed by the column)
    :ivar param: the parameter's number, for ``param``, ``any`` and ``nullsafe``
    :ivar literal: the literal's Python value, for ``literal``; the literals, for ``anylit``
    :ivar cast: the array's element type, for ``any`` (None: the array was bound untyped)
    :ivar params: a ``fragment`` SET's parameters, in the order its expression first names them
    """

    column: str
    kind: str
    param: int | None = None
    literal: str | bool | int | tuple[str | bool | int, ...] | None = None
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
    :ivar order: a listing's sort (shape ``list``): each ``ORDER BY`` term as ``{"column", "descending", "nulls"}``;
        empty for a point read
    """

    table: str
    columns: tuple[str, ...] | None
    pins: tuple[Pin, ...]
    order: tuple[dict[str, Any], ...] = ()


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
    #: a listing's sort (shape ``list``), each ``ORDER BY`` term as ``{"column", "descending", "nulls"}``: the
    #: ``read_by`` call's ``order``; none for a point read or a write
    order: list[dict[str, Any]] = field(default_factory=list)
    # the edits that make it, not recorded: (start line, start col, end line, end col, text) in the module's bytes
    edits: list[tuple[int, int, int, int, str]] = field(default_factory=list, repr=False)
    # not recorded: index purpose entries naming the replaced constant, each (repo-relative file, old text, new text)
    repoints: list[tuple[str, str, str]] = field(default_factory=list, repr=False)
    # not recorded: what is left for a hand (a comment or docstring still naming the constant)
    notes: list[str] = field(default_factory=list, repr=False)
    # not recorded: names to import beside the helpers, module -> names (a dict of helper calls' annotation)
    imports: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def record(self) -> dict[str, Any]:
        """the manifest's entry.

        :return: the entry
        :rtype: dict[str, Any]
        """
        entry = asdict(self)
        for kept_out in ("edits", "repoints", "notes", "imports"):
            entry.pop(kept_out)
        return entry


def _literal(text: str) -> str | bool | int:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text.startswith("'"):
        return text[1:-1].replace("''", "'")
    return int(text)


def parse_point(sql: str, *, ordered: bool = False) -> PointSql | str:
    """a plain point read, or why it is not one.

    :param sql: the statement
    :ptype sql: str
    :param ordered: a listing (shape ``list``): a trailing ``ORDER BY`` of plain columns is read as its sort
    :ptype ordered: bool
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
    qualifiers = {None, alias, match.group("table").split(".")[-1]}
    order: list[dict[str, Any]] = []
    sorted_by = _ORDER_BY.search(masked) if ordered else None
    if sorted_by is not None:
        terms = where[sorted_by.start("order") :]
        where, masked = where[: sorted_by.start()], masked[: sorted_by.start()]
        for term in terms.split(","):
            found = _ORDER_TERM.match(term.strip())
            if found is None or found.group("q") not in qualifiers:
                return "an ORDER BY term other than a plain column [ASC|DESC] [NULLS FIRST|LAST] (or a LIMIT)"
            nulls = found.group("nulls")
            order.append(
                {
                    "column": found.group("column"),
                    "descending": (found.group("direction") or "asc").lower() == "desc",
                    "nulls": None if nulls is None else nulls.lower(),
                }
            )
    if re.search(r"\b(order\s+by|limit|offset|group\s+by|for\s+update|for\s+share)\b", masked, re.IGNORECASE):
        return "ORDER BY / LIMIT / GROUP BY / a row lock"
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
    if columns is not None and any(term["column"] not in columns for term in order):
        return "an ORDER BY column it does not select (the equivalence proof compares the order by its values)"
    numbers = sorted({pin.param for pin in pins if pin.param is not None})
    if numbers != list(range(1, len(numbers) + 1)):
        return "parameters not numbered $1..$n"
    if not any(pin.kind in _BOUND_KINDS for pin in pins):
        return "no bound parameter (literals only): a constant read, judged by hand"
    return PointSql(table=match.group("table").lower(), columns=columns, pins=tuple(pins), order=tuple(order))


def parse_list(sql: str) -> PointSql | str:
    """a plain listing: a point read's pins, its rows in a trailing ``ORDER BY`` of plain columns (or in none), or why
    it is not one.

    :param sql: the statement
    :ptype sql: str
    :return: the parsed read, or the refusal's reason
    :rtype: PointSql | str
    """
    return parse_point(sql, ordered=True)


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
            (_NULLSAFE_PIN, "nullsafe"),
            (_IN_LITERALS_PIN, "anylit"),
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
        elif kind == "nullsafe":
            pins.append(Pin(name, "nullsafe", param=int(pinned.group("n"))))
        elif kind == "anylit":
            listed = tuple(_literal(found.group()) for found in re.finditer(_IN_LITERAL, pinned.group("literals")))
            pins.append(Pin(name, "anylit", literal=listed))
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
        r"(?<!distinct )\b(from|using)\b",
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
    if not any(pin.kind in _BOUND_KINDS for pin in pins):
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
        read = _status_reading(module, awaited)
        if isinstance(read, str):
            return read
    if call.keywords:
        return "keyword arguments to the call (timeout=, ...)"
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return "starred arguments"
    return call


#: a status compared with a literal: ``status == "UPDATE 0"``, the verb and the count
_STATUS_TAG: Final[re.Pattern[str]] = re.compile(r"^(?P<verb>UPDATE|DELETE) (?P<count>\d+)$")
#: a status's ending: ``status.endswith(" 0")``, the count after its space
_STATUS_END: Final[re.Pattern[str]] = re.compile(r"^ (?P<count>\d+)$")
#: the functions that read a status as its count (``parse_rowcount(status)``, ``rows_affected(status)``)
_STATUS_PARSERS: Final[frozenset[str]] = frozenset({"parse_rowcount", "rows_affected"})
#: where a status read rewritten into a comparison is written bare; anywhere else it is parenthesised
_BARE_CONTEXTS: Final[tuple[type[ast.AST], ...]] = (
    ast.If,
    ast.While,
    ast.Assert,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.BoolOp,
    ast.IfExp,
    ast.Expr,
    ast.keyword,
    ast.comprehension,
)


@dataclass(frozen=True)
class _StatusRead:
    """how a write's status is read, as the helper's count reads it (ruling 2026-11-12: the caller reads
    ``(await helper(...)).count``; there is no ``KeyedWrite.status``).

    :ivar local: the name the status is bound to (``status = await conn.execute(...)``), None when it is read where
        it is awaited
    :ivar binding: that binding (its annotation, if any, becomes ``int``)
    :ivar reads: each place the status is read: the node replaced, and its text with ``{}`` for the count
    """

    local: str | None
    binding: ast.Assign | ast.AnnAssign | None
    reads: tuple[tuple[ast.expr, str], ...]


def _minus_one(node: ast.expr) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _int_constant(node.operand) == 1


def _int_constant(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _last_word(module: _Module, status: ast.expr) -> ast.Subscript | None:
    """``status.split()[-1]`` (``split(" ")``, ``rsplit()``, ``rsplit(" ", 1)``; ``[1]`` too: an UPDATE or DELETE
    status is two words), the subscript; None when ``status`` is not read so."""
    method = module.parent(status)
    if not (isinstance(method, ast.Attribute) and method.value is status and method.attr in ("split", "rsplit")):
        return None
    call = module.parent(method)
    if not (isinstance(call, ast.Call) and call.func is method and not call.keywords):
        return None
    separator = call.args[0] if call.args else None
    if separator is not None and not (isinstance(separator, ast.Constant) and separator.value in (" ", None)):
        return None
    if len(call.args) > 2 or (len(call.args) == 2 and (method.attr != "rsplit" or _int_constant(call.args[1]) != 1)):
        return None
    subscript = module.parent(call)
    if not (isinstance(subscript, ast.Subscript) and subscript.value is call):
        return None
    return subscript if _minus_one(subscript.slice) or _int_constant(subscript.slice) == 1 else None


def _compared(
    module: _Module, node: ast.expr, other: ast.expr, verbs: frozenset[str] | None, *, word: bool
) -> tuple[ast.expr, str] | str | None:
    """``node == <literal>`` / ``!=`` (either side) as the count compared; None when ``node`` is in no comparison."""
    compare = module.parent(node)
    if not isinstance(compare, ast.Compare) or len(compare.ops) != 1:
        return None
    if not isinstance(compare.ops[0], ast.Eq | ast.NotEq):
        return None
    if not (isinstance(other, ast.Constant) and isinstance(other.value, str)):
        return None
    if word:
        if not other.value.isascii() or not other.value.isdigit():
            return None
        count = int(other.value)
    else:
        tag = _STATUS_TAG.match(other.value)
        if tag is None:
            return None
        if verbs is not None and verbs != {tag["verb"].lower()}:
            return f"the statement's status is compared with {other.value!r}, not its own verb's tag"
        count = int(tag["count"])
    op = "==" if isinstance(compare.ops[0], ast.Eq) else "!="
    return compare, ("{} " + op + f" {count}" if compare.left is node else f"{count} {op} " + "{}")


def _one_read(module: _Module, status: ast.expr, verbs: frozenset[str] | None) -> tuple[ast.expr, str] | str:
    """one place a write's status is read, as the node it replaces and its text with ``{}`` for the count; or why it
    cannot be read as a count."""
    parent = module.parent(status)
    # status == "UPDATE 0" / "DELETE 1" != status
    if isinstance(parent, ast.Compare):
        other = parent.comparators[0] if parent.left is status else parent.left
        compared = _compared(module, status, other, verbs, word=False)
        if compared is not None:
            return compared
    # parse_rowcount(status) / rows_affected(status)
    if (
        isinstance(parent, ast.Call)
        and parent.args == [status]
        and not parent.keywords
        and (parent.func.id if isinstance(parent.func, ast.Name) else getattr(parent.func, "attr", None))
        in _STATUS_PARSERS
    ):
        return parent, "{}"
    # status.endswith(" 0")
    if isinstance(parent, ast.Attribute) and parent.attr == "endswith":
        call = module.parent(parent)
        if (
            isinstance(call, ast.Call)
            and call.func is parent
            and len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            end = _STATUS_END.match(call.args[0].value)
            if end is not None:
                text = "{} == " + end["count"]
                return call, f"({text})" if not isinstance(module.parent(call), _BARE_CONTEXTS) else text
    # int(status.split()[-1]); status.split()[-1] == "0"
    word = _last_word(module, status)
    if word is not None:
        holder = module.parent(word)
        if (
            isinstance(holder, ast.Call)
            and isinstance(holder.func, ast.Name)
            and holder.func.id == "int"
            and holder.args == [word]
            and not holder.keywords
        ):
            return holder, "{}"
        if isinstance(holder, ast.Compare):
            other = holder.comparators[0] if holder.left is word else holder.left
            compared = _compared(module, word, other, verbs, word=True)
            if compared is not None:
                return compared
    shown: ast.AST = parent if parent is not None else status
    if isinstance(shown, ast.Attribute) and isinstance(module.parent(shown), ast.Call):
        shown = module.parent(shown)  # type: ignore[assignment]
    spelled = module.segment(shown)
    return (
        f"the statement's status is read as {' '.join(spelled.split())[:60]!r}, which no count reads "
        "(the helper answers a count, not a status)"
    )


def _status_reading(module: _Module, awaited: ast.Await, verbs: frozenset[str] | None = None) -> _StatusRead | str:
    """every read of the status of the awaited write ``awaited`` (read where it is awaited, or bound once to a local
    and read there), each as the helper's count reads it; or why one cannot be (refused, and listed).

    :param module: the module
    :ptype module: _Module
    :param awaited: the awaited ``execute``
    :ptype awaited: ast.Await
    :param verbs: the write's verbs (``update``/``delete``), so a compared tag is the statement's own; None to take
        either
    :ptype verbs: frozenset[str] | None
    :return: the reads, or the refusal
    :rtype: _StatusRead | str
    """
    binding = module.parent(awaited)
    if not isinstance(binding, ast.Assign | ast.AnnAssign) or binding.value is not awaited:
        read = _one_read(module, awaited, verbs)
        return read if isinstance(read, str) else _StatusRead(None, None, (read,))
    targets = binding.targets if isinstance(binding, ast.Assign) else [binding.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return "the statement's status is bound to something other than one name"
    local = targets[0].id
    scope = module.scope(binding)
    if any(isinstance(node, ast.Global | ast.Nonlocal) and local in node.names for node in ast.walk(scope)):
        return f"the statement's status {local} is global or nonlocal"
    named = _loads(scope, local)
    if [node for node in named if not isinstance(node.ctx, ast.Load)] != [targets[0]]:
        return f"the statement's status {local} is bound more than once"
    reads = []
    for node in named:
        if node is targets[0]:
            continue
        read = _one_read(module, node, verbs)
        if isinstance(read, str):
            return read
        reads.append(read)
    return _StatusRead(local, binding, tuple(reads))


def _status_edits(
    module: _Module, call: ast.Call, text: str, verbs: frozenset[str]
) -> list[tuple[int, int, int, int, str]] | str:
    """the edits that put ``text`` (the helper's call, awaited where ``call`` was) in ``call``'s place: the call
    alone when its status is discarded; when it is read, the awaited call read as its ``.count`` and each read of the
    status as that count reads it (:func:`_status_reading`); or why the status cannot be read so.

    :return: the edits, or the refusal
    :rtype: list[tuple[int, int, int, int, str]] | str
    """
    awaited = module.parent(call)
    assert isinstance(awaited, ast.Await)
    if isinstance(module.parent(awaited), ast.Expr):
        assert call.end_lineno is not None and call.end_col_offset is not None
        return [(call.lineno, call.col_offset, call.end_lineno, call.end_col_offset, text)]
    read = _status_reading(module, awaited, verbs)
    if isinstance(read, str):
        return read
    count = f"(await {text}).count"
    edits: list[tuple[int, int, int, int, str]] = []
    if read.local is None:
        ((node, template),) = read.reads
        assert node.end_lineno is not None and node.end_col_offset is not None
        return [(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, template.format(count))]
    assert awaited.end_lineno is not None and awaited.end_col_offset is not None
    edits.append((awaited.lineno, awaited.col_offset, awaited.end_lineno, awaited.end_col_offset, count))
    if isinstance(read.binding, ast.AnnAssign):
        annotation = read.binding.annotation
        assert annotation.end_lineno is not None and annotation.end_col_offset is not None
        edits.append(
            (annotation.lineno, annotation.col_offset, annotation.end_lineno, annotation.end_col_offset, "int")
        )
    for node, template in read.reads:
        assert node.end_lineno is not None and node.end_col_offset is not None
        edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, template.format(read.local)))
    return edits


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
        if rewrite.order:
            parts.append(f"order=[{', '.join(_sort_text(term) for term in rewrite.order)}]")
    return f"{rewrite.helper}({', '.join(parts)})"


def _sort_text(term: Mapping[str, Any]) -> str:
    """one ``ORDER BY`` term as the helper call writes it: ``Sort("col")``, with ``descending=True`` and ``nulls=`` as
    the statement wrote them (the engine's own NULL order otherwise, as before).

    :param term: the term, ``{"column", "descending", "nulls"}``
    :ptype term: Mapping[str, Any]
    :return: the ``Sort`` expression
    :rtype: str
    """
    parts = [repr(term["column"])]
    if term["descending"]:
        parts.append("descending=True")
    if term["nulls"] is not None:
        parts.append(f"nulls={term['nulls']!r}")
    return f"Sort({', '.join(parts)})"


def _value_text(pin: Mapping[str, Any], args: Sequence[str]) -> str:
    """a pinned or set column's value as the helper call writes it: the argument, an ``AnyOf`` of it, a literal,
    None, or ``now()``."""
    if pin["kind"] in {"param", "nullsafe"}:
        return str(args[pin["param"] - 1])
    if pin["kind"] == "anylit":
        listed = ", ".join(_python_literal(value) for value in pin["literal"])
        return f"AnyOf(({listed}{',' if len(pin['literal']) == 1 else ''}), cast=None)"
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
    """the string literal holding the statement at ``line``, one starting with ``verb`` (a status tag the statement's
    status is compared with, ``"DELETE 0"``, is not one)."""
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
        and not _STATUS_TAG.match(node.value)
    ]
    if len(strings) != 1:
        return f"{len(strings)} {verb.upper()} literals on its line"
    return strings[0]


@dataclass
class _Mentions:
    """where else in the repo a statement constant's name is written.

    :ivar used: a mention that is a use (code naming it in another module, a statement or config naming it): the
        constant stays
    :ivar purposes: the files holding an index purpose entry ``<its module>::<name>``, which the rewrite repoints to
        the function now making the call
    :ivar prose: ``path:line`` of each comment or docstring naming it, which the rewrite leaves (listed, to reword
        by hand)
    """

    used: list[str] = field(default_factory=list)
    purposes: list[Path] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)


#: the files whose mentions of a constant are judged: code, statements, config, and the index declaration ledgers
_MENTION_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".sql", ".yaml", ".txt"})

#: calls of a mapping that read only its keys: a test holding a dict's keys to an enum names it, and stays true
_KEY_READS: Final[frozenset[str]] = frozenset({"set", "len", "sorted", "list", "frozenset", "tuple"})


def _key_only(parents: Mapping[int, ast.AST], node: ast.expr) -> bool:
    """whether a mention of a mapping reads only its keys: ``set(D)``, ``len(D)``, ``k in D``, ``D.keys()``."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Call) and parent.args == [node] and not parent.keywords:
        return isinstance(parent.func, ast.Name) and parent.func.id in _KEY_READS
    if isinstance(parent, ast.Compare) and node in parent.comparators:
        return all(isinstance(op, ast.In | ast.NotIn) for op in parent.ops)
    if isinstance(parent, ast.Attribute) and parent.attr == "keys":
        call = parents.get(id(parent))
        return isinstance(call, ast.Call) and call.func is parent and not call.args
    return False


def _python_mentions(text: str, name: str, *, mapping: bool) -> tuple[list[int], list[int]]:
    """the lines of a Python module where ``name`` is a use, and where it is prose (a comment or a docstring).

    :param text: the module's source
    :ptype text: str
    :param name: the constant's name
    :ptype name: str
    :param mapping: whether the constant is a mapping, so a mention reading only its keys is no use of its values
    :ptype mapping: bool
    :return: the use lines, and the prose lines
    :rtype: tuple[list[int], list[int]]
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    try:
        tree = ast.parse(text)
    except SyntaxError:  # NOSILENT: a module that does not parse is judged by its text: every mention a use
        return [number for number, line in enumerate(text.splitlines(), 1) if pattern.search(line)], []
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    used: list[int] = []
    code_lines: set[int] = set()
    for node in ast.walk(tree):
        named = (isinstance(node, ast.Name) and node.id == name) or (
            isinstance(node, ast.Attribute) and node.attr == name
        )
        if isinstance(node, ast.alias) and name in {node.name, node.asname}:
            used.append(node.lineno)
            code_lines.add(node.lineno)
        elif named:
            assert isinstance(node, ast.Name | ast.Attribute)
            code_lines.add(node.lineno)
            if not (mapping and _key_only(parents, node)):
                used.append(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and pattern.search(node.value)
            and node.lineno not in docstrings
        ):
            used.append(node.lineno)
            code_lines.add(node.lineno)
    prose = [
        number
        for number, line in enumerate(text.splitlines(), 1)
        if pattern.search(line) and number not in code_lines and number not in used
    ]
    return sorted(set(used)), prose


def _mentions(repo_root: Path, path: Path, name: str, *, mapping: bool = False) -> _Mentions:
    """every mention of a constant's name in the repo's other modules, migrations and declaration ledgers, sorted
    into uses, index purpose entries naming it, and prose.

    :param repo_root: the repo
    :ptype repo_root: Path
    :param path: the constant's module
    :ptype path: Path
    :param name: the constant's name
    :ptype name: str
    :param mapping: whether the constant is a mapping (a mention reading only its keys is no use of its values)
    :ptype mapping: bool
    :return: the mentions
    :rtype: _Mentions
    """
    relative = path.relative_to(repo_root).as_posix()
    purpose = re.compile(rf"{re.escape(relative)}::{re.escape(name)}\b")
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    found = _Mentions()
    for other in sorted(repo_root.rglob("*")):
        if other.suffix not in _MENTION_SUFFIXES or other == path:
            continue
        if {".venv", "node_modules", ".git"} & set(other.parts):
            continue
        try:
            text = other.read_text(encoding="utf-8")
        except UnicodeDecodeError, OSError:  # NOSILENT: a file no text can be read from names no constant
            continue
        if not pattern.search(text):
            continue
        where = other.relative_to(repo_root).as_posix()
        if purpose.search(text):
            found.purposes.append(other)
            # each purpose entry is the purpose's text, not code: judged with the entries taken out
            text = purpose.sub(relative + "::", text)
            if not pattern.search(text):
                continue
        if other.suffix == ".py":
            used, prose = _python_mentions(text, name, mapping=mapping)
            found.used += [f"{where}:{number}" for number in used]
            found.prose += [f"{where}:{number}" for number in prose]
        elif other.suffix == ".txt":
            found.prose += [f"{where}:{n}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
        else:
            found.used += [f"{where}:{n}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
    return found


def _module_mentions_left(module: _Module, name: str, constant: ast.Assign | ast.AnnAssign) -> list[int]:
    """the lines of the constant's own module, outside its assignment, that still name it once its uses are gone:
    a comment or docstring naming it is left as it is (listed)."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    first, last = constant.lineno, constant.end_lineno or constant.lineno
    code = {
        node.lineno
        for node in ast.walk(module.tree)
        if (isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name)
    }
    return [
        number
        for number, line in enumerate(module.source.splitlines(), 1)
        if pattern.search(line) and not first <= number <= last and number not in code
    ]


def _elsewhere(
    repo_root: Path, module: _Module, constant: ast.Assign | ast.AnnAssign | None, calls: Sequence[ast.Call]
) -> tuple[list[tuple[str, str, str]], list[str]] | str:
    """a statement constant's mentions outside the calls that send it: the index purpose entries to repoint (to the
    functions now making the calls) and the prose left to reword; or why the constant stays (another module uses it).

    :param repo_root: the repo
    :ptype repo_root: Path
    :param module: the constant's module
    :ptype module: _Module
    :param constant: its assignment, None for a statement written at its call
    :ptype constant: ast.Assign | ast.AnnAssign | None
    :param calls: the calls sending it
    :ptype calls: Sequence[ast.Call]
    :return: the repoints (file, old entry, new entry) and the notes, or the refusal
    :rtype: tuple[list[tuple[str, str, str]], list[str]] | str
    """
    if constant is None:
        return [], []
    target = constant.targets[0] if isinstance(constant, ast.Assign) else constant.target
    assert isinstance(target, ast.Name)
    name = target.id
    mapping = isinstance(constant.value, ast.Dict)
    found = _mentions(repo_root, module.path, name, mapping=mapping)
    if found.used:
        return f"its constant {name} is named elsewhere in the repo ({', '.join(found.used[:3])})"
    relative = module.path.relative_to(repo_root).as_posix()
    repoints: list[tuple[str, str, str]] = []
    if found.purposes:
        readers = sorted({_function_at(module, call) for call in calls})
        if mapping or "<module>" in readers:
            return f"its constant {name} is named in an index purpose, and no one function makes its calls"
        old = f"{relative}::{name}"
        new = "; ".join(f"{relative}::{reader}" for reader in readers)
        repoints = [(other.relative_to(repo_root).as_posix(), old, new) for other in found.purposes]
    if mapping:
        # the dict stays (its values become helper calls), so every mention of it stays true
        return repoints, []
    prose = found.prose + [f"{relative}:{number}" for number in _module_mentions_left(module, name, constant)]
    return repoints, [f"{where}: still names {name} (prose; reword by hand)" for where in prose]


def _constant_edit(module: _Module, constant: ast.Assign | ast.AnnAssign) -> tuple[int, int, int, int, str]:
    """the edit removing a constant's assignment, its ``#:`` comment lines above it with it."""
    lines = module.source.splitlines()
    first = constant.lineno
    while first > 1 and lines[first - 2].lstrip().startswith("#:"):
        first -= 1
    last = constant.end_lineno or constant.lineno
    return (first, 0, last + 1, 0, "")


#: the type a dict of statements' values take once each is the helper call it maps to
_CALLABLE_VALUE: Final[str] = "Callable[..., Awaitable[Any]]"

#: what a dict of helper calls needs imported for its annotation: module -> names
_CALLABLE_IMPORTS: Final[dict[str, frozenset[str]]] = {
    "collections.abc": frozenset({"Awaitable", "Callable"}),
    "typing": frozenset({"Any"}),
}


def _parameter_names(pins: Sequence[Pin], count: int) -> list[str]:
    """a name for each of a statement's parameters, ``$1`` first: the column it pins or sets, ``p<n>`` where that
    is not a plain new name (a keyword, a builtin, ``conn``, a name taken twice, a fragment's parameter)."""
    names: dict[int, str] = {}
    for pin in pins:
        if pin.kind in _BOUND_KINDS and pin.param is not None and pin.param not in names:
            names[pin.param] = pin.column
    taken: Counter[str] = Counter(names.values())
    plain = {
        number: name
        for number, name in names.items()
        if taken[name] == 1
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and name not in dir(builtins)
        and name != "conn"
    }
    return [plain.get(number, f"p{number}") for number in range(1, count + 1)]


@dataclass(frozen=True)
class _Shape:
    """how one shape's statements are parsed and sent."""

    shape: str
    parse: Any
    methods: frozenset[str]


def _value_sql(value: ast.expr) -> str | None:
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


def _annotation_edit(module: _Module, constant: ast.Assign | ast.AnnAssign) -> tuple[int, int, int, int, str] | str:
    """the edit retyping a dict of statements' values (``dict[K, str]``, ``Final[...]`` of one) as the helper calls
    they become; none needed for an unannotated dict."""
    if isinstance(constant, ast.Assign):
        return (0, 0, 0, 0, "")
    node = constant.annotation
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Final":
        node = node.slice
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"dict", "Mapping"}
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[1], ast.Name)
        and node.slice.elts[1].id == "str"
    ):
        return f"annotated {ast.unparse(constant.annotation)}, not dict[K, str]"
    value = node.slice.elts[1]
    assert value.end_lineno is not None and value.end_col_offset is not None
    return (value.lineno, value.col_offset, value.end_lineno, value.end_col_offset, _CALLABLE_VALUE)


def _plan_dict(
    config: NoWholeReadsConfig,
    module: _Module,
    literal: ast.Constant,
    shape: _Shape,
    pending: Mapping[tuple[str, int], Any],
    unkeyed: Mapping[tuple[str, int], str],
) -> list[Rewrite] | str:
    """a dict of statements (``D = {key: "SELECT ...", ...}`` at the module's top) every use of which sends one value
    (``await conn.fetch(D[key], *args)``): each value rewritten into a ``lambda conn, <params>:`` of the helper call
    it maps to, each call into ``D[key](conn, *args)``; or why not. Every value must be a statement of the shape the
    rule finds (and not unkeyed), all of one parameter count, every call of one method and argument count.

    :return: one rewrite per value (the first holds the call and annotation edits), or the refusal
    :rtype: list[Rewrite] | str
    """
    mapping = module.parent(literal)
    assert isinstance(mapping, ast.Dict)
    constant = module.parent(mapping)
    if not (
        isinstance(constant, ast.Assign | ast.AnnAssign)
        and constant.value is mapping
        and module.parent(constant) is module.tree
    ):
        return "a dict of statements not bound at the module's top"
    targets = constant.targets if isinstance(constant, ast.Assign) else [constant.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return "a dict of statements bound to something other than one name"
    name = targets[0].id
    if any(key is None for key in mapping.keys):
        return f"its dict {name} spreads another mapping"
    relative = module.path.relative_to(config.repo_root).as_posix()
    parsed: list[Any] = []
    for value in mapping.values:
        sql = _value_sql(value)
        if sql is None:
            return f"its dict {name} holds a value other than a statement literal"
        if (relative, value.lineno) in unkeyed:
            return f"its dict {name}: the no-whole-reads rule finds a value unkeyed: a real fix first"
        if (relative, value.lineno) not in pending:
            return f"its dict {name} holds a value the rule finds no {shape.shape} statement in"
        one = shape.parse(sql)
        if isinstance(one, str):
            return f"its dict {name}: {one}"
        if shape.shape == "write" and one.returning:
            return f"its dict {name}: a RETURNING write (its answer is read through the awaited call)"
        parsed.append(one)
    counts = {len({n for pin in (*getattr(one, "sets", ()), *one.pins) for n in pin.numbers()}) for one in parsed}
    if len(counts) != 1:
        return f"its dict {name}: values of different parameter counts"
    count = counts.pop()
    calls: list[ast.Call] = []
    for node in ast.walk(module.tree):
        if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
            continue
        if _key_only(module.parents, node):
            continue
        subscript = module.parent(node)
        if not (isinstance(subscript, ast.Subscript) and subscript.value is node):
            return f"its dict {name} is used other than indexed at a call"
        call = _call_of(module, subscript, shape.methods)
        if isinstance(call, str):
            return f"its dict {name}: {call}"
        calls.append(call)
    if not calls:
        return f"its dict {name} is never sent"
    methods = {call.func.attr for call in calls if isinstance(call.func, ast.Attribute)}
    if len(methods) != 1:
        return f"its dict {name} is sent by {'/'.join(sorted(methods))}: one helper per value"
    method = methods.pop()
    if any(len(call.args) - 1 != count for call in calls):
        return f"its dict {name}: a call's arguments are not its values' {count} parameters"
    if shape.shape != "write" and method == "fetchval" and any(one.columns is None for one in parsed):
        return f"its dict {name}: fetchval of SELECT *"
    elsewhere = _elsewhere(config.repo_root, module, constant, calls)
    if isinstance(elsewhere, str):
        return elsewhere
    retyped = _annotation_edit(module, constant)
    if isinstance(retyped, str):
        return f"its dict {name} is {retyped}"
    planned: list[Rewrite] = []
    for value, one in zip(mapping.values, parsed, strict=True):
        args = _parameter_names([*getattr(one, "sets", ()), *one.pins], count)
        rewrite = Rewrite(
            path=relative,
            function="<module>",
            line=value.lineno,
            table=one.table,
            old_sql=_value_sql(value) or "",
            method=method,
            helper=_HELPER_OF[method] if shape.shape != "write" else _WRITE_HELPER[one.kind],
            columns=None if shape.shape == "write" or one.columns is None else list(one.columns),
            pins=[asdict(pin) for pin in one.pins],
            args=args,
            shape=shape.shape,
            sets=[asdict(pin) for pin in getattr(one, "sets", ())],
            order=list(getattr(one, "order", ())),
        )
        call_text = _render(rewrite, "conn") if shape.shape != "write" else _render_write(rewrite, "conn")
        rewrite.new_call = f"lambda conn, {', '.join(args)}: {call_text}" if args else f"lambda conn: {call_text}"
        assert value.end_lineno is not None and value.end_col_offset is not None
        rewrite.edits.append((value.lineno, value.col_offset, value.end_lineno, value.end_col_offset, rewrite.new_call))
        planned.append(rewrite)
    first = planned[0]
    verbs = frozenset(one.kind for one in parsed) if shape.shape == "write" else frozenset()
    for call in calls:
        assert isinstance(call.func, ast.Attribute) and call.end_lineno is not None and call.end_col_offset is not None
        rest = "".join(f", {module.segment(argument)}" for argument in call.args[1:])
        text = f"{module.segment(call.args[0])}({module.segment(call.func.value)}{rest})"
        if shape.shape != "write":
            first.edits.append((call.lineno, call.col_offset, call.end_lineno, call.end_col_offset, text))
            continue
        edits = _status_edits(module, call, text, verbs)
        if isinstance(edits, str):
            return f"its dict {name}: {edits}"
        first.edits.extend(edits)
    if retyped[0]:
        first.edits.append(retyped)
        first.imports = {module_name: set(names) for module_name, names in _CALLABLE_IMPORTS.items()}
    first.repoints, first.notes = elsewhere
    return planned


def _module_constant(module: _Module, literal: ast.expr) -> tuple[ast.Assign | ast.AnnAssign, str] | None:
    """the module-top assignment of one name a statement literal is, and the name; None when it is not one."""
    constant = module.parent(literal)
    if not (
        isinstance(constant, ast.Assign | ast.AnnAssign)
        and constant.value is literal
        and module.parent(constant) is module.tree
    ):
        return None
    targets = constant.targets if isinstance(constant, ast.Assign) else [constant.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    return constant, targets[0].id


def _loads(scope: ast.AST, name: str) -> list[ast.Name]:
    return [node for node in ast.walk(scope) if isinstance(node, ast.Name) and node.id == name]


def _choice_of(module: _Module, literal: ast.Constant) -> ast.IfExp | None:
    """the conditional choosing between two statement constants (``x = A if c else B``) a constant's one use is in,
    or None when its use is not one."""
    found = _module_constant(module, literal)
    if found is None:
        return None
    uses = [node for node in _loads(module.tree, found[1]) if isinstance(node.ctx, ast.Load)]
    if len(uses) != 1:
        return None
    choice = module.parent(uses[0])
    return choice if isinstance(choice, ast.IfExp) and uses[0] in (choice.body, choice.orelse) else None


def _plan_choice(
    config: NoWholeReadsConfig,
    module: _Module,
    choice: ast.IfExp,
    shape: _Shape,
    pending: Mapping[tuple[str, int], Any],
    unkeyed: Mapping[tuple[str, int], str],
) -> list[Rewrite] | str:
    """two statement constants chosen by a conditional bound to a local (``x = A if c else B``) that the very next
    statement sends (``await conn.execute(x, *args)``): the call rewritten into ``(<A's helper call> if c else <B's>)``,
    the local's binding and both constants removed; or why not. Both must be statements of the shape the rule finds
    (and not unkeyed), of the call's argument count; the condition is evaluated where it was, before the arguments.

    :return: the two rewrites (the first holds the edits), or the refusal
    :rtype: list[Rewrite] | str
    """
    relative = module.path.relative_to(config.repo_root).as_posix()
    chosen: list[tuple[ast.Assign | ast.AnnAssign, str, ast.Constant, Any]] = []
    for side in (choice.body, choice.orelse):
        if not isinstance(side, ast.Name):
            return "a conditional choosing something other than two constants"
        bound = [
            statement
            for statement in module.tree.body
            if isinstance(statement, ast.Assign | ast.AnnAssign)
            and statement.value is not None
            and _module_constant(module, statement.value) == (statement, side.id)
        ]
        if len(bound) != 1 or bound[0].value is None or _value_sql(bound[0].value) is None:
            return f"its choice's {side.id} is not one module statement constant"
        literal = bound[0].value
        assert isinstance(literal, ast.Constant)
        if len([node for node in _loads(module.tree, side.id) if isinstance(node.ctx, ast.Load)]) != 1:
            return f"its choice's {side.id} is used other than in the choice"
        if (relative, literal.lineno) in unkeyed:
            return f"its choice's {side.id}: the no-whole-reads rule finds it unkeyed: a real fix first"
        if (relative, literal.lineno) not in pending:
            return f"its choice's {side.id} is no {shape.shape} statement the rule finds"
        one = shape.parse(literal.value)
        if isinstance(one, str):
            return f"its choice's {side.id}: {one}"
        if shape.shape == "write" and one.returning:
            return f"its choice's {side.id}: a RETURNING write"
        chosen.append((bound[0], side.id, literal, one))
    binding = module.parent(choice)
    if not (
        isinstance(binding, ast.Assign)
        and binding.value is choice
        and len(binding.targets) == 1
        and isinstance(binding.targets[0], ast.Name)
    ):
        return "a choice of statements not bound to one local"
    local = binding.targets[0].id
    scope = module.scope(binding)
    if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        return "a choice of statements outside a function"
    named = _loads(scope, local)
    loads = [node for node in named if isinstance(node.ctx, ast.Load)]
    if len(loads) != 1 or len(named) != 2:
        return f"its local {local} is bound or read more than once"
    block = next(
        (
            body
            for node in ast.walk(scope)
            for body in (getattr(node, "body", None), getattr(node, "orelse", None))
            if isinstance(body, list) and binding in body
        ),
        None,
    )
    assert block is not None
    after = block.index(binding) + 1
    if after >= len(block) or not any(node is loads[0] for node in ast.walk(block[after])):
        return f"its local {local} is not sent by the statement right after its binding"
    call = _call_of(module, loads[0], shape.methods)
    if isinstance(call, str):
        return call
    assert isinstance(call.func, ast.Attribute)
    method = call.func.attr
    arguments = call.args[1:]
    planned: list[Rewrite] = []
    edits: list[tuple[int, int, int, int, str]] = []
    repoints: list[tuple[str, str, str]] = []
    notes: list[str] = []
    for constant, name, literal, one in chosen:
        bound_numbers = [n for pin in (*getattr(one, "sets", ()), *one.pins) for n in pin.numbers()]
        if len(arguments) != len(set(bound_numbers)):
            return f"its choice's {name}: {len(arguments)} arguments for {len(set(bound_numbers))} parameters"
        if any(
            count > 1 and not _plain_simple(arguments[number - 1]) for number, count in Counter(bound_numbers).items()
        ):
            return "a parameter used twice whose argument is not a plain name"
        if shape.shape != "write" and method == "fetchval" and one.columns is None:
            return "fetchval of SELECT *"
        elsewhere = _elsewhere(config.repo_root, module, constant, [call])
        if isinstance(elsewhere, str):
            return elsewhere
        repoints += elsewhere[0]
        notes += elsewhere[1]
        rewrite = Rewrite(
            path=relative,
            function=_function_at(module, call),
            line=literal.lineno,
            table=one.table,
            old_sql=str(literal.value),
            method=method,
            helper=_HELPER_OF[method] if shape.shape != "write" else _WRITE_HELPER[one.kind],
            columns=None if shape.shape == "write" or one.columns is None else list(one.columns),
            pins=[asdict(pin) for pin in one.pins],
            args=[module.segment(argument) for argument in arguments],
            shape=shape.shape,
            sets=[asdict(pin) for pin in getattr(one, "sets", ())],
            order=list(getattr(one, "order", ())),
        )
        receiver = module.segment(call.func.value)
        rewrite.new_call = _render(rewrite, receiver) if shape.shape != "write" else _render_write(rewrite, receiver)
        edits.append(_constant_edit(module, constant))
        planned.append(rewrite)
    assert call.end_lineno is not None and call.end_col_offset is not None
    text = f"({planned[0].new_call} if {module.segment(choice.test)} else {planned[1].new_call})"
    if shape.shape != "write":
        edits.append((call.lineno, call.col_offset, call.end_lineno, call.end_col_offset, text))
    else:
        sent = _status_edits(module, call, text, frozenset(one.kind for *_, one in chosen))
        if isinstance(sent, str):
            return sent
        edits.extend(sent)
    edits.append((binding.lineno, 0, (binding.end_lineno or binding.lineno) + 1, 0, ""))
    planned[0].edits, planned[0].repoints, planned[0].notes = edits, repoints, notes
    return planned


_POINT: Final[_Shape] = _Shape("point", parse_point, frozenset(_HELPER_OF))
#: a listing is sent by ``fetch`` only: its rows, every one, in its order (``read_by(order=)``)
_LIST: Final[_Shape] = _Shape("list", parse_list, frozenset({"fetch"}))
_WRITE: Final[_Shape] = _Shape("write", parse_write, _WRITE_METHODS)


def plan_point_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every point finding of the repo, as the rewrite it takes or the refusal it gets.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    return _plan_reads(config, _POINT)


def plan_list_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every listing finding of the repo (``SELECT ... WHERE <pins> [ORDER BY <columns>]`` sent by ``fetch``), as the
    :func:`~threetears.core.keyed.read_by` call with ``order=`` it takes, or the refusal it gets. The sort is the
    statement's own, term for term: the same rows in the same order, ties as the engine leaves them, as before.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    return _plan_reads(config, _LIST)


def _plan_reads(config: NoWholeReadsConfig, shape: _Shape) -> tuple[list[Rewrite], list[Refusal]]:
    """every read finding of ``shape`` (``point`` or ``list``) in the repo, as the rewrite it takes or the refusal it
    gets."""
    whole_hits, scan = _hits(config)
    unkeyed: dict[tuple[str, int], str] = {(hit.path, hit.line): hit.symbol for hit in whole_hits}
    hits = [
        hit
        for hit in _helper_hits(scan.statements, config.helper_modules, _producers(config))
        if hit.shape == shape.shape
    ]
    modules: dict[str, _Module] = {}
    rewrites: list[Rewrite] = []
    refusals: list[Refusal] = []
    pending = {(hit.statement.path, hit.statement.line): hit for hit in hits}
    groups: dict[int, list[Rewrite] | str] = {}
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
        parsed = shape.parse(statement.text)
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
        choice = _choice_of(module, literal)
        if isinstance(module.parent(literal), ast.Dict) or choice is not None:
            held = module.parent(literal) if choice is None else choice
            group = groups.get(id(held))
            if group is None:
                group = groups[id(held)] = (
                    _plan_dict(config, module, literal, shape, pending, unkeyed)
                    if choice is None
                    else _plan_choice(config, module, choice, shape, pending, unkeyed)
                )
                if not isinstance(group, str):
                    rewrites.extend(group)
            if isinstance(group, str):
                refuse(group)
            continue
        sent = _sent_by(module, literal)
        if isinstance(sent, str):
            refuse(sent)
            continue
        calls, constant = sent
        elsewhere = _elsewhere(config.repo_root, module, constant, calls)
        if isinstance(elsewhere, str):
            refuse(elsewhere)
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
            elif method not in shape.methods:
                problem = f"a listing sent by {method}: its first row is a first-of read, not a listing"
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
                shape=shape.shape,
                order=list(parsed.order),
            )
            rewrite.new_call = _render(rewrite, module.segment(call.func.value))
            assert call.end_lineno is not None and call.end_col_offset is not None
            rewrite.edits.append((call.lineno, call.col_offset, call.end_lineno, call.end_col_offset, rewrite.new_call))
            planned.append(rewrite)
        if planned and constant is not None:
            planned[0].edits.append(_constant_edit(module, constant))
            planned[0].repoints, planned[0].notes = elsewhere
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
    pending = {(hit.statement.path, hit.statement.line): hit for hit in hits}
    groups: dict[int, list[Rewrite] | str] = {}
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
        choice = _choice_of(module, literal)
        if isinstance(module.parent(literal), ast.Dict) or choice is not None:
            held = module.parent(literal) if choice is None else choice
            group = groups.get(id(held))
            if group is None:
                group = groups[id(held)] = (
                    _plan_dict(config, module, literal, _WRITE, pending, unkeyed)
                    if choice is None
                    else _plan_choice(config, module, choice, _WRITE, pending, unkeyed)
                )
                if not isinstance(group, str):
                    rewrites.extend(group)
            if isinstance(group, str):
                refuse(group)
            continue
        sent = _sent_by(module, literal, frozenset(_HELPER_OF) if parsed.returning else _WRITE_METHODS)
        if isinstance(sent, str):
            refuse(sent)
            continue
        calls, constant = sent
        elsewhere = _elsewhere(config.repo_root, module, constant, calls)
        if isinstance(elsewhere, str):
            refuse(elsewhere)
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
            if returning:
                assert awaited.end_lineno is not None and awaited.end_col_offset is not None
                rewrite.edits.append(
                    (awaited.lineno, awaited.col_offset, awaited.end_lineno, awaited.end_col_offset, rewrite.new_call)
                )
            else:
                # its status discarded, the call alone; read, the helper's count in its place
                edits = _status_edits(module, call, rewrite.new_call, frozenset({parsed.kind}))
                if isinstance(edits, str):
                    refuse(edits)
                    planned = []
                    break
                rewrite.edits.extend(edits)
            planned.append(rewrite)
        if planned and constant is not None:
            planned[0].edits.append(_constant_edit(module, constant))
            planned[0].repoints, planned[0].notes = elsewhere
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


def _with_import(source: str, names: set[str], module: str = _HELPER_MODULE) -> str:
    """the source importing ``names`` from ``module`` (the helpers' module by default): added to an existing
    top-level ``from`` import of it, or as a new line after the module's last top-level import."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module and node.level == 0:
            have = {alias.name for alias in node.names if alias.asname is None}
            wanted = sorted(have | names | {f"{a.name} as {a.asname}" for a in node.names if a.asname})
            text = f"from {module} import {', '.join(wanted)}\n"
            assert node.end_lineno is not None
            return "".join(lines[: node.lineno - 1]) + text + "".join(lines[node.end_lineno :])
    last = 0
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            last = node.end_lineno or node.lineno
        elif last:
            break
    text = f"from {module} import {', '.join(sorted(names))}\n"
    return "".join(lines[:last]) + text + "".join(lines[last:])


def _without_unused(source: str, names: frozenset[str]) -> str:
    """the source with each ``from ... import`` of one of ``names`` (a status parser a rewritten status read no longer
    calls) removed when nothing else in the module names it (code, or ``__all__``).

    :param source: the module, rewritten
    :ptype source: str
    :param names: the imported names to drop when unused
    :ptype names: frozenset[str]
    :return: the module
    :rtype: str
    """
    tree = ast.parse(source)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    edits: list[tuple[int, int, int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        dropped = [alias for alias in node.names if (alias.asname or alias.name) in names - used]
        if not dropped:
            continue
        kept = [alias for alias in node.names if alias not in dropped]
        assert node.end_lineno is not None and node.end_col_offset is not None
        if kept:
            text = f"from {'.' * node.level}{node.module or ''} import {', '.join(ast.unparse(a) for a in kept)}"
            edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, text))
        else:
            edits.append((node.lineno, 0, node.end_lineno + 1, 0, ""))
    return _splice(source, edits) if edits else source


def apply_rewrites(repo_root: Path, rewrites: list[Rewrite], *, ruff: str | None = "ruff") -> list[str]:
    """make every rewrite, import the helpers, and format the touched modules.

    :param repo_root: the repo
    :ptype repo_root: Path
    :param rewrites: the rewrites
    :ptype rewrites: list[Rewrite]
    :param ruff: the ruff to run (the repo's own); none when None (the source is left as spliced)
    :ptype ruff: str | None
    :return: the touched modules, repo-relative (each file whose index purpose entry is repointed is rewritten too,
        as text, and not formatted)
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
        if any(pin["kind"] in {"any", "anylit"} for rewrite in mine for pin in rewrite.pins):
            names.add("AnyOf")
        if any(rewrite.order for rewrite in mine):
            names.add("Sort")
        source = _with_import(source, names)
        for rewrite in mine:
            for imported, wanted in rewrite.imports.items():
                source = _with_import(source, wanted, imported)
        source = _without_unused(source, _STATUS_PARSERS)
        ast.parse(source)
        path.write_text(source, encoding="utf-8")
    touched = sorted(by_path)
    for rewrite in rewrites:
        for relative, old, new in rewrite.repoints:
            path = repo_root / relative
            text = path.read_text(encoding="utf-8")
            path.write_text(re.sub(rf"{re.escape(old)}\b", lambda _: new, text), encoding="utf-8")
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

#: its own mark puts it in the repo's integration suite (it needs a database) and out of the unit run
pytestmark = pytest.mark.integration

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
    """the command line: ``point`` (reads), ``list`` (listings in their ``ORDER BY``), ``write`` (keyed ``UPDATE`` /
    ``DELETE`` by equality), ``test-module``.

    :param argv: the arguments
    :ptype argv: list[str] | None
    :return: the exit status: 0, or 1 when the after-check found a rewrite that added a whole-read finding
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="threetears.enforcement.no_whole_reads.rewrite")
    parser.add_argument("shape", choices=["point", "list", "write", "test-module"])
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
    planner = {"write": plan_write_rewrites, "list": plan_list_rewrites}.get(options.shape, plan_point_rewrites)
    rewrites, refusals = planner(config)
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
            for relative, old, new in rewrite.repoints:
                _say(f"    purpose to repoint in {relative}: {old} -> {new}")
            for note in rewrite.notes:
                _say(f"    LEFT: {note}")
        return 0
    touched = apply_rewrites(repo, rewrites, ruff=options.ruff)
    write_manifest(repo / options.manifest, rewrites)
    added = _after_check(config, before, rewrites)
    _say(f"rewrote {len(rewrites)} statements in {len(touched)} modules; manifest {options.manifest}")
    for rewrite in rewrites:
        for relative, old, new in rewrite.repoints:
            _say(f"  purpose repointed in {relative}: {old} -> {new}")
        for note in rewrite.notes:
            _say(f"  LEFT: {note}")
    for line in added:
        _say(f"  AFTER-CHECK: a rewrite added a whole-read finding: {line}")
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
