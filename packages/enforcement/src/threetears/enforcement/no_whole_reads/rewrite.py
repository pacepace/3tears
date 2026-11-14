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

**First-of and top-n reads** (shape ``first``, ruling 2026-11-13): a listing's statement with a trailing ``LIMIT 1``
or ``LIMIT $n`` after its ``ORDER BY``, sent by ``fetch`` (the top n), ``fetchrow`` (the first row) or ``fetchval``
(the first row's first column), becomes the first page of :func:`~threetears.core.keyed.keyset_page` with
``records=True``: the statement's own sort, completed by the table's declared key (its columns the pins and the sort
do not already hold, ascending), so which of the rows tied at the boundary it keeps is the key's choice, not the
engine's. ``fetch`` reads the page's ``.rows``; ``fetchrow`` its first row or None; ``fetchval`` that row's column or
None. Refused: no ``ORDER BY`` (the rows kept are the engine's choice), an ``OFFSET``, a table with no declared key,
a key column the statement does not select, the ``LIMIT`` parameter not the last, and every listing refusal.

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

**Counts and existence** (shape ``count``, the use-the-helper ``count`` findings): a point read's ``FROM <table>
WHERE <pins>`` under ``SELECT count(*) [AS <name>]``, or wrapped as ``SELECT EXISTS (SELECT 1 FROM <table> WHERE
<pins>) [AS <name>]``, sent by an awaited ``fetchval``. A count becomes ``(await aggregate_keyed(conn, <table>,
aggregates={"n": Count()}, by=<pins>))[0]["n"]`` (one row however many it counts, 0 for none); an existence test
:func:`~threetears.core.keyed.holds_any` with ``order=`` the columns of the table's declared key or index its pins
lead (the primary key first, then a unique key, then an index): one row read, at most. Refused: any other select list
(a sum, a max, a column beside the count, a ``GROUP BY``), an ``EXISTS`` of anything but ``SELECT 1``, an existence
test whose pins lead no declared key or index, a count sent by ``fetch`` / ``fetchrow`` (read by a column name the
helper does not answer under), and every point-read refusal.

**Guards** (every read and write shape, ruling 2026-11-14): a ``WHERE`` conjunct other than an equality beside the
equalities (a range ``col > $n``, ``col <> $n``, an optional pin ``($n::text IS NULL OR col = $n)``, a ``NOT (...)``)
is sent as a ``where=`` :class:`~threetears.core.keyed.Fragment` beside the ``by=`` pins: its qualifiers dropped (the
helper reads one relation), its parameters renumbered from ``$1`` and bound to the same arguments. The rows read or
written are the same (the conjunction is the same). The equalities must still bind a parameter, and the no-whole-reads
rule must still find the statement keyed by them. Refused: a top-level ``OR`` (the ``AND`` beside it binds first), a
``BETWEEN``, a subquery, another relation's column, a ``?`` placeholder, an existence test (``holds_any`` takes no
guard), and a guard parameter no pin binds that its predicate compares with no column (the generated proof binds it
to that column's seeded value).

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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, _hits, _producers, _undeclared_statements
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql
from threetears.enforcement.no_whole_reads.use_the_helper import _helper_hits
from threetears.enforcement.no_whole_reads.walker import WholeReadScan, _enum_strings

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
    #: ``literal``; a ``guard``'s (a predicate other than equality, sent as a ``where=`` fragment beside the pins)
    #: likewise, its text renumbered from ``$1``
    params: tuple[int, ...] = ()
    #: a ``guard``'s: for each of ``params``, the column the predicate compares it with (``""``: none it names), the
    #: column whose seeded value the generated proof binds it to when no pin binds that parameter
    compared: tuple[str, ...] = ()

    def numbers(self) -> tuple[int, ...]:
        """the statement's parameters this pin binds.

        :return: the parameter numbers
        :rtype: tuple[int, ...]
        """
        if self.kind in {"fragment", "guard"}:
            return self.params
        return () if self.param is None else (self.param,)


@dataclass(frozen=True)
class PointSql:
    """a plain point read, parsed.

    :ivar table: the table as written (``schema.table`` kept)
    :ivar columns: the selected columns, None for ``*``
    :ivar pins: each pinned column, in the statement's order
    :ivar order: a listing's sort (shape ``list``): each ``ORDER BY`` term as ``{"column", "descending", "nulls"}``;
        empty for a point read
    :ivar limit: a first-of or top-n read's ``LIMIT`` (shape ``first``): ``{"param": n}`` or ``{"literal": n}``
    """

    table: str
    #: each selected item: a plain column, or a :data:`Selected <threetears.core.keyed.Selected>` expression as
    #: ``{"expr", "alias"}`` (the expression over the helper's row ``t``)
    columns: tuple[str | dict[str, str], ...] | None
    pins: tuple[Pin, ...]
    order: tuple[dict[str, Any], ...] = ()
    limit: dict[str, int] | None = None
    #: a count's (shape ``count``): ``count`` for ``SELECT count(*)``, ``exists`` for ``SELECT EXISTS (SELECT 1 ...)``;
    #: empty for every other read
    aggregate: str = ""
    #: each ``WHERE`` conjunct other than an equality (a range, ``<>``, an optional pin, ``NOT (...)``), as a ``guard``
    #: pin: the helper's ``where=`` fragments, ANDed beside its ``by=`` pins
    guards: tuple[Pin, ...] = ()


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
    #: a first-of or top-n read's ``LIMIT`` (shape ``first``): ``{"param": n}`` or ``{"literal": n}``; not recorded
    #: for any other shape
    limit: dict[str, int] | None = None
    #: shape ``first``: how many of ``order``'s terms are the statement's own; the rest complete it by the key
    key_from: int | None = None
    #: shape ``count``, an existence test (``holds_any``): the declared key or index its probe walks (``order=``)
    walk: list[str] = field(default_factory=list)
    #: each ``WHERE`` conjunct other than an equality (a ``guard`` :class:`Pin` as a dict): the call's ``where=``
    #: fragments; not recorded when there is none
    where: list[dict[str, Any]] = field(default_factory=list)
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
        if self.shape != "first":
            entry.pop("limit")
            entry.pop("key_from")
        if self.helper != "holds_any":
            entry.pop("walk")
        if not self.where:
            entry.pop("where")
        return entry


class _TopSelect:
    """:data:`_SELECT` matched on a statement's top level (each parenthesized part and quoted literal blanked, so a
    sub-select in the select list does not end it), its groups read back from the statement as written."""

    def __init__(self, text: str, match: re.Match[str]) -> None:
        self._text = text
        self._match = match

    def group(self, name: str) -> Any:
        start, end = self._match.span(name)
        return None if start < 0 else self._text[start:end]


def _top_select(text: str) -> _TopSelect | None:
    """:data:`_SELECT` over ``text``'s top level, or None."""
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", text)
    blanked, depth = [], 0
    for char in masked:
        if char == ")":
            depth -= 1
        blanked.append(char if depth == 0 or char in "()" else "_")
        if char == "(":
            depth += 1
    match = _SELECT.match("".join(blanked))
    return None if match is None else _TopSelect(text, match)


def _literal(text: str) -> str | bool | int:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text.startswith("'"):
        return text[1:-1].replace("''", "'")
    return int(text)


def parse_point(sql: str, *, ordered: bool = False, guards: bool = False) -> PointSql | str:
    """a plain point read, or why it is not one.

    :param sql: the statement
    :ptype sql: str
    :param ordered: a listing (shape ``list``): a trailing ``ORDER BY`` of plain columns is read as its sort
    :ptype ordered: bool
    :param guards: each ``WHERE`` conjunct other than an equality is kept as a guard (the helper's ``where=``), not
        refused; at least one equality must still bind a parameter
    :ptype guards: bool
    :return: the parsed read, or the refusal's reason
    :rtype: PointSql | str
    """
    text = " ".join(sql.split())
    if "{" in text or "}" in text:
        return "interpolated"
    if text.startswith("/*") or "--" in text:
        return "a comment or hint in the statement"
    match = _top_select(text)
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
    columns: tuple[str | dict[str, str], ...] | None
    if match.group("columns").strip() == "*":
        columns = None
    else:
        picked = _select_list(match.group("columns"), match.group("table"), alias)
        if isinstance(picked, str):
            return picked
        columns = picked
    kept: list[Pin] | None = [] if guards else None
    pins = _pins(where, masked, qualifiers, kept)
    if isinstance(pins, str):
        return pins
    if columns is not None and any(term["column"] not in column_names(columns) for term in order):
        return "an ORDER BY column it does not select (the equivalence proof compares the order by its values)"
    numbers = sorted({number for pin in (*pins, *(kept or ())) for number in pin.numbers()})
    if numbers != list(range(1, len(numbers) + 1)):
        return "parameters not numbered $1..$n"
    if not any(pin.kind in _BOUND_KINDS for pin in pins):
        return "no bound parameter (literals only): a constant read, judged by hand"
    return PointSql(
        table=match.group("table").lower(),
        columns=columns,
        pins=tuple(pins),
        order=tuple(order),
        guards=tuple(kept or ()),
    )


#: a select-list item ``<expression> AS <alias>``: the expression, the alias it is answered under
_SELECTED_ITEM: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<expr>.+?)\s+as\s+(?P<alias>{_IDENT})$", re.IGNORECASE | re.DOTALL
)


def _reads_unaliased(masked: str, table: str) -> bool:
    """whether ``masked`` reads ``table`` with no alias of its own (``FROM table WHERE ...``, ``FROM table)``)."""
    for found in re.finditer(rf'\bfrom\s+"?{re.escape(table)}"?(?![\w"])\s*(?P<next>{_IDENT})?', masked, re.IGNORECASE):
        following = (found.group("next") or "").lower()
        if not following or following in _KEYWORDS:
            return True
    return False


def column_names(columns: Sequence[str | Mapping[str, str]]) -> list[str]:
    """the names a select list's rows answer under: each plain column, each expression's alias.

    :param columns: the parsed select list
    :ptype columns: Sequence[str | Mapping[str, str]]
    :return: the names, in order
    :rtype: list[str]
    """
    return [column if isinstance(column, str) else column["alias"] for column in columns]


def _select_list(text: str, table: str, alias: str | None) -> tuple[str | dict[str, str], ...] | str:
    """a select list's items: a plain column (qualified by the table or its alias, or not), or ``<expression> AS
    <alias>`` -- the expression respelled over the helper's row ``t`` (each ``<table>.`` / ``<alias>.`` it names
    becomes ``t.``), sent as ``Selected(expression, alias)``; or why it is not one.

    Refused: ``DISTINCT``, an expression binding a parameter (the helper numbers only its pins), an unnamed expression,
    and one reading the same table again unaliased (its own ``<table>.`` would be the inner read's, not the row's)."""
    bare = table.split(".")[-1]
    qualifiers = {None, alias, bare}
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", text)
    if re.match(r"\s*distinct\b", masked, re.IGNORECASE):
        return "an expression, alias or DISTINCT in the select list"
    cuts = [-1, *_top_level_commas(masked), len(text)]
    picked: list[str | dict[str, str]] = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        item, item_masked = text[start + 1 : end].strip(), masked[start + 1 : end].strip()
        column = _COLUMN.match(item)
        if column is not None:
            if column.group("qualifier") not in qualifiers:
                return "a select-list column of another relation"
            picked.append(column.group("column"))
            continue
        named = _SELECTED_ITEM.match(item_masked)
        if named is None:
            return "an expression in the select list with no alias (AS <name>)"
        if re.search(r"\$\d", item_masked):
            return "a select-list expression binding a parameter"
        names = {name for name in (alias, bare) if name}
        if _reads_unaliased(item_masked, bare):
            return f"a select-list expression reading {bare} again unaliased"
        expression = item[: named.end("expr")]
        for name in names:
            # only outside quoted text: the masked text says where those are
            spans = [
                found.span()
                for found in re.finditer(rf'(?<![\w."]){re.escape(name)}\.', item_masked[: named.end("expr")])
            ]
            for begin, finish in reversed(spans):
                expression = expression[:begin] + "t." + expression[finish:]
        picked.append({"expr": " ".join(expression.split()), "alias": named.group("alias")})
    return tuple(picked)


def parse_list(sql: str, *, guards: bool = False) -> PointSql | str:
    """a plain listing: a point read's pins, its rows in a trailing ``ORDER BY`` of plain columns (or in none), or why
    it is not one.

    :param sql: the statement
    :ptype sql: str
    :param guards: as :func:`parse_point`'s
    :ptype guards: bool
    :return: the parsed read, or the refusal's reason
    :rtype: PointSql | str
    """
    return parse_point(sql, ordered=True, guards=guards)


#: a first-of or top-n read's trailing ``LIMIT`` (shape ``first``)
_LIMIT: Final[re.Pattern[str]] = re.compile(r"\s+limit\s+(?P<limit>\d+|\$\d+)\s*$", re.IGNORECASE)


def parse_first(sql: str, *, guards: bool = False) -> PointSql | str:
    """a first-of or top-n read: a listing's pins and ``ORDER BY`` of plain columns with a trailing ``LIMIT 1`` /
    ``LIMIT $n``, or why it is not one.

    :param sql: the statement
    :ptype sql: str
    :param guards: as :func:`parse_point`'s
    :ptype guards: bool
    :return: the parsed read, its ``limit`` set, or the refusal's reason
    :rtype: PointSql | str
    """
    text = " ".join(sql.split())
    found = _LIMIT.search(text)
    if found is None:
        return "no trailing LIMIT 1 / LIMIT $n after its ORDER BY (an OFFSET, or no LIMIT)"
    parsed = parse_point(text[: found.start()], ordered=True, guards=guards)
    if isinstance(parsed, str):
        return parsed
    if not parsed.order:
        return "a LIMIT with no ORDER BY: which rows it keeps is the engine's choice"
    written = found.group("limit")
    if not written.startswith("$"):
        return replace(parsed, limit={"literal": int(written)}) if int(written) > 0 else "LIMIT 0"
    numbers = {number for pin in (*parsed.pins, *parsed.guards) for number in pin.numbers()}
    if int(written[1:]) != len(numbers) + 1:
        return "the LIMIT's parameter is not the statement's last"
    return replace(parsed, limit={"param": int(written[1:])})


#: ``count(*)``, alone in a count's select list, under any name
_COUNT_ITEM: Final[re.Pattern[str]] = re.compile(rf"^count\s*\(\s*\*\s*\)(?:\s+(?:as\s+)?{_IDENT})?$", re.IGNORECASE)
#: ``SELECT EXISTS (SELECT 1 FROM ...) [AS <name>]``: the inner read, its select list ``1``
_EXISTS: Final[re.Pattern[str]] = re.compile(
    rf"^select\s+exists\s*\(\s*select\s+1\s+(?P<inner>from\s+.+)\)(?:\s+(?:as\s+)?{_IDENT})?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def parse_count(sql: str, *, guards: bool = False) -> PointSql | str:
    """a count (``SELECT count(*) FROM <table> WHERE <pins>``) or an existence test (``SELECT EXISTS (SELECT 1 FROM
    <table> WHERE <pins>)``) of a point read's rows, or why it is not one.

    :param sql: the statement
    :ptype sql: str
    :param guards: as :func:`parse_point`'s
    :ptype guards: bool
    :return: the parsed read, its ``aggregate`` ``count`` or ``exists`` and no columns, or the refusal's reason
    :rtype: PointSql | str
    """
    text = " ".join(sql.split())
    if "{" in text or "}" in text:
        return "interpolated"
    exists = _EXISTS.match(text)
    if exists is not None:
        parsed = parse_point(f"SELECT * {exists.group('inner')}", guards=guards)
        aggregate = "exists"
    else:
        match = _SELECT.match(text)
        if match is None:
            return "not SELECT count(*) / EXISTS (SELECT 1 ...) FROM <table> WHERE ..."
        if _COUNT_ITEM.match(match.group("columns").strip()) is None:
            return "a select list other than count(*) alone (a sum, a max, a column, a GROUP BY)"
        parsed = parse_point(f"{text[: match.start('columns')]}*{text[match.end('columns') :]}", guards=guards)
        aggregate = "count"
    if isinstance(parsed, str):
        return parsed
    if aggregate == "exists" and parsed.guards:
        return f"an existence test with a predicate other than equality: {parsed.guards[0].literal!r}"
    return replace(parsed, columns=None, aggregate=aggregate)


def _probe_walk(parsed: PointSql, declared: Any) -> list[str] | str:
    """the columns an existence test's probe walks: those of the table's declared key or index whose leading columns
    its pins hold most of (the primary key first among equals, then a unique key, then an index that covers every
    row; a pin past the walked prefix narrows the rows walked), or why none leads with a pin.

    :param parsed: the existence test
    :ptype parsed: PointSql
    :param declared: the table's declaration in the walk's schema catalogue, None when it has none
    :ptype declared: Any
    :return: the walk's columns, or the refusal's reason
    :rtype: list[str] | str
    """
    if declared is None:
        return f"no declared key or index of {parsed.table} for its probe to walk"
    pinned = {pin.column.lower() for pin in parsed.pins}
    indexes = [
        index.columns for index in declared.indexes.values() if not index.partial and not index.method and index.columns
    ]
    best: tuple[str, ...] = ()
    held = 0
    for columns in [*declared.primary_keys, *declared.unique_keys, *indexes]:
        lead = 0
        while lead < len(columns) and columns[lead].lower() in pinned:
            lead += 1
        if lead > held:
            best, held = tuple(columns), lead
    if not held:
        return f"its pins {sorted(pinned)} lead no declared key or index of {parsed.table}"
    return list(best)


def _key_completion(parsed: PointSql, declared: Any) -> list[dict[str, Any]] | str:
    """the sort terms that complete a first-of read's order by its table's key: none when the pins and the sort
    already hold a declared key; else, of the declared keys (the primary key first among equals, then each
    ``UNIQUE``), the columns of the one they hold most of, ascending, when those are selected and an index holds them.

    An index holds the completed order when, its leading columns pinned to one value set aside, it goes on with the
    order's columns and then the completion's: only then does the first page read its rows off the index in order.
    Without one the database reads every row the pins leave and sorts them (YugabyteDB has no incremental sort), so
    the read is refused until such an index is declared.

    :param parsed: the read
    :ptype parsed: PointSql
    :param declared: the table's declaration in the walk's schema catalogue, None when it has none
    :ptype declared: Any
    :return: the terms to append, or the refusal's reason
    :rtype: list[dict[str, Any]] | str
    """
    if declared is None or not declared.primary_keys:
        return f"no declared key of {parsed.table} to complete its order with"
    pinned = {pin.column.lower() for pin in parsed.pins if pin.kind in _ONE_VALUE_KINDS}
    ordered = [term["column"].lower() for term in parsed.order]
    held = pinned | set(ordered)
    keys = [*declared.primary_keys, *(key for key in declared.unique_keys if key not in declared.primary_keys)]
    if any({column.lower() for column in key} <= held for key in keys):
        return []
    selected = None if parsed.columns is None else {column.lower() for column in column_names(parsed.columns)}
    reasons: list[str] = []
    for key in sorted(keys, key=lambda key: sum(column.lower() not in held for column in key)):
        completing = [column for column in key if column.lower() not in held]
        unselected = [column for column in completing if selected is not None and column.lower() not in selected]
        if unselected:
            reasons.append(f"the key column(s) {unselected} completing the order are not selected")
        elif not _index_holds(declared, pinned, [*ordered, *(column.lower() for column in completing)]):
            reasons.append(
                f"no declared index of {parsed.table} holds its order completed by the key "
                f"({', '.join([*ordered, *completing])}, after the pins {sorted(pinned)}): declare that index first"
            )
        else:
            return [{"column": column, "descending": False, "nulls": None} for column in completing]
    return reasons[0]


def _index_holds(declared: Any, pinned: set[str], order: list[str]) -> bool:
    """whether a declared key or plain (not hnsw/gin) index of the table, its leading columns pinned to one value set
    aside, goes on with ``order``'s lower-cased columns, so a read pinned and ordered so walks it in order.

    :param declared: the table's declaration in the walk's schema catalogue
    :ptype declared: Any
    :param pinned: the read's columns pinned to one value, lower-cased
    :ptype pinned: set[str]
    :param order: the completed order's columns, lower-cased
    :ptype order: list[str]
    :return: whether one does
    :rtype: bool
    """
    indexes = [index.columns for index in declared.indexes.values() if not index.method and index.columns]
    for columns in [*declared.primary_keys, *declared.unique_keys, *indexes]:
        walked = [column.lower() for column in columns]
        while walked and walked[0] in pinned:
            walked.pop(0)
        if walked[: len(order)] == order:
            return True
    return False


#: pin kinds holding a column to one value (or NULL) on every row read: no key column they hold needs ordering
_ONE_VALUE_KINDS: Final[frozenset[str]] = frozenset({"param", "literal", "null", "nullsafe"})


def _top_level(masked: str, pattern: re.Pattern[str]) -> list[re.Match[str]] | str:
    """each match of ``pattern`` in ``masked`` (its quoted literals masked) outside every parenthesis, or why the
    parentheses do not balance."""
    depth = 0
    depths = []
    for char in masked:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return "unbalanced parentheses"
        depths.append(depth)
    if depth:
        return "unbalanced parentheses"
    return [found for found in pattern.finditer(masked) if depths[found.start()] == 0]


#: a column written with a qualifier (``t.col``), not a cast's type (``::public.vector``) nor a number (``1.5``)
_QUALIFIED: Final[re.Pattern[str]] = re.compile(rf"(?<![\w.$:])(?P<q>{_IDENT})\.(?P<column>{_IDENT})\b", re.IGNORECASE)
_TOP_OR: Final[re.Pattern[str]] = re.compile(r"\s+or\s+", re.IGNORECASE)
#: a comparison of a column with a parameter, either way round (``col >= $2``, ``$2::timestamptz < col``)
_COMPARISON: Final[str] = r"(?:=|<>|!=|<=|>=|<|>|\s+is\s+(?:not\s+)?distinct\s+from\s+)"
#: a guard's ``compared`` for a parameter standing as a truth value on its own, which the proof binds False
_BOOL_OPERAND: Final[str] = ":bool"
#: words that read as a column in a comparison and are not one
_SQL_WORDS: Final[frozenset[str]] = frozenset({"null", "true", "false", "not", "and", "or", "is", "any", "all"})


def _guard(conjunct: str, qualifiers: set[str | None]) -> Pin | str:
    """a ``WHERE`` conjunct other than an equality as a ``guard`` pin: its qualifiers dropped (the helper reads one
    relation), its parameters renumbered from ``$1``, each paired with the column it is compared with; or why it cannot
    be one (a subquery, a column of another relation, a ``?`` placeholder, an interpolation).

    :param conjunct: the conjunct, as written
    :ptype conjunct: str
    :param qualifiers: the names a column may be qualified by (None: unqualified)
    :ptype qualifiers: set[str | None]
    :return: the guard, or the refusal's reason
    :rtype: Pin | str
    """
    masked = _QUOTED.sub(lambda found: "'" + "x" * (len(found.group()) - 2) + "'", conjunct)
    if re.search(r"\bselect\b", masked, re.IGNORECASE):
        return f"a subquery in a predicate: {conjunct!r}"
    if re.search(r"\?\s*(?:$|[),])", masked):
        # a ? placeholder (SQLite's), not jsonb's ? operator (which a quoted key follows)
        return f"a ? placeholder in a predicate: {conjunct!r}"
    if re.search(r"\bbetween\b", masked, re.IGNORECASE):
        return f"a BETWEEN (its AND is not a conjunction): {conjunct!r}"
    for found in _QUALIFIED.finditer(masked):
        if found.group("q").lower() not in {qualifier.lower() for qualifier in qualifiers if qualifier}:
            return f"a column of another relation: {conjunct!r}"
    text = "".join(_unqualified(conjunct, masked))
    order: list[int] = []
    for number in _PARAMETER.findall(text):
        if int(number) not in order:
            order.append(int(number))
    renumbered = _PARAMETER.sub(lambda found: f"${order.index(int(found.group(1))) + 1}", text)
    compared = []
    for place in range(1, len(order) + 1):
        parameter = rf"\${place}(?!\d)(?:\s*::\s*{_IDENT}(?:\[\])?)*"
        paired = re.search(
            rf"(?P<column>{_IDENT})\s*{_COMPARISON}\s*{parameter}", renumbered, re.IGNORECASE
        ) or re.search(rf"{parameter}\s*{_COMPARISON}\s*(?P<column>{_IDENT})", renumbered, re.IGNORECASE)
        column = paired.group("column").lower() if paired is not None else ""
        if column in _SQL_WORDS:
            column = ""
        if not column and re.search(
            rf"(?:^|(?<!\w)\(|\bor\b|\band\b|\bnot\b)\s*\${place}(?!\d)(?:\s*::\s*bool(?:ean)?)?\s*(?:$|\)|\bor\b|\band\b)",
            renumbered,
            re.IGNORECASE,
        ):
            # a parameter standing as a truth value on its own (``($1 OR col <> 'x')``): the proof binds it False, so
            # the predicate beside it decides
            column = _BOOL_OPERAND
        compared.append(column)
    return Pin("", "guard", literal=" ".join(renumbered.split()), params=tuple(order), compared=tuple(compared))


def _unqualified(text: str, masked: str) -> list[str]:
    """``text`` cut around each qualifier :data:`_QUALIFIED` finds in ``masked``, the qualifiers left out."""
    pieces = []
    at = 0
    for found in _QUALIFIED.finditer(masked):
        pieces.append(text[at : found.start()])
        at = found.start("column")
    pieces.append(text[at:])
    return pieces


def _pins(where: str, masked: str, qualifiers: set[str | None], guards: list[Pin] | None = None) -> list[Pin] | str:
    """each column a ``WHERE`` of equalities pins (``= $n``, ``= <literal>``, ``IS NULL``, joined by ``AND``), or why
    it is not one.

    With ``guards``, every other conjunct (a range, ``<>``, an optional pin ``($5::text IS NULL OR col = $5)``, a
    ``NOT (...)``) is appended there as a ``guard`` pin instead of refused: the helper's ``where=`` fragments, ANDed
    beside the pins, so the rows read are the same. A top-level ``OR`` (which the conjunction would misread) is refused.

    :param where: the ``WHERE`` clause's text
    :ptype where: str
    :param masked: the same text, its quoted literals masked
    :ptype masked: str
    :param qualifiers: the names a column may be qualified by (None: unqualified)
    :ptype qualifiers: set[str | None]
    :param guards: where the conjuncts other than equalities go; refused when None
    :ptype guards: list[Pin] | None
    :return: the pins, or the refusal's reason
    :rtype: list[Pin] | str
    """
    pins: list[Pin] = []
    ands = _top_level(masked, _AND)
    if isinstance(ands, str):
        return ands
    ors = _top_level(masked, _TOP_OR)
    if guards is not None and ors:
        return "an OR outside every parenthesis (the AND beside it binds first)"
    starts = [0, *(found.end() for found in ands)]
    ends = [found.start() for found in ands] + [len(where)]
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
            if guards is None:
                return f"a predicate other than equality: {conjunct!r}"
            guard = _guard(conjunct, qualifiers)
            if isinstance(guard, str):
                return guard
            guards.append(guard)
            continue
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
    #: as :attr:`PointSql.guards`
    guards: tuple[Pin, ...] = ()


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


def parse_write(sql: str, *, guards: bool = False) -> WriteSql | str:
    """a keyed write by equality, or why it is not one: every ``SET`` a column to a parameter, a literal, ``NULL`` or
    ``now()``; every ``WHERE`` conjunct an equality :func:`parse_point` takes; a ``RETURNING`` of plain columns at
    most; no alias, join, subquery or expression.

    :param sql: the statement
    :ptype sql: str
    :param guards: as :func:`parse_point`'s
    :ptype guards: bool
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
    kept: list[Pin] | None = [] if guards else None
    pins = _pins(where, masked, {None, table.split(".")[-1]}, kept)
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
    numbers = sorted({number for pin in (*sets, *pins, *(kept or ())) for number in pin.numbers()})
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
        guards=tuple(kept or ()),
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


def _class_constant_uses(module: _Module, owner: ast.ClassDef, name: str) -> list[ast.expr] | str:
    """every read of the class-level constant ``name`` of ``owner``: ``self.name`` / ``cls.name`` /
    ``<Class>.name`` anywhere in the module, and the bare name in the class body itself; or why it is left (read
    through another object, which the rewrite cannot follow)."""
    uses: list[ast.expr] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
            if not (isinstance(node.value, ast.Name) and node.value.id in {"self", "cls", owner.name}):
                return f"its class constant {name} is read through {ast.unparse(node.value)}"
            uses.append(node)
    for statement in owner.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        uses += [
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        ]
    return uses


def _sent_by(
    module: _Module, literal: ast.Constant | ast.JoinedStr, methods: frozenset[str] = frozenset(_HELPER_OF)
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
        owner = module.parent(parent)
        if isinstance(owner, ast.ClassDef):
            # a class-level constant: sent as self.X / cls.X / Class.X (anywhere in the module), or by its bare name
            # in the class body; read through any other object, it is left
            uses = _class_constant_uses(module, owner, name)
            if isinstance(uses, str):
                return uses
        else:
            scope = module.scope(parent)
            named = [node for node in ast.walk(scope) if isinstance(node, ast.Name) and node.id == name]
            uses = [node for node in named if isinstance(node.ctx, ast.Load)]
            if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef) and len(named) - len(uses) > 1:
                return f"its local {name} is bound more than once (a statement chosen by branch)"
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


def _unseeded_guard(parsed: PointSql | WriteSql) -> str:
    """why the generated proof could not bind a guard's parameter (no pin binds it and the guard compares it with no
    column, so no seeded value is its), or ``""``."""
    pinned = {number for pin in (*parsed.pins, *getattr(parsed, "sets", ())) for number in pin.numbers()}
    for guard in parsed.guards:
        for number, column in zip(guard.params, guard.compared, strict=True):
            if number not in pinned and not column:
                return f"a predicate's parameter ${number} compared with no column: {guard.literal!r}"
    return ""


def _where_text(rewrite: Rewrite) -> str:
    """the helper call's ``where=`` argument (`` where=[Fragment(...), ...]``, with its leading ``, ``), or ``""``."""
    if not rewrite.where:
        return ""
    return f", where=[{', '.join(_value_text(guard, rewrite.args) for guard in rewrite.where)}]"


def _render(rewrite: Rewrite, receiver: str) -> str:
    """the helper call, written in the statement's call's place."""
    by = "{" + ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.pins) + "}"
    parts = [receiver, repr(rewrite.table), f"by={by}"]
    if rewrite.where:
        parts.append(_where_text(rewrite).removeprefix(", "))
    if rewrite.helper == "read_value":
        assert rewrite.columns is not None
        parts.append(f"column={rewrite.columns[0]!r}")
    else:
        if rewrite.columns is not None:
            parts.append(f"columns={_columns_text(rewrite.columns)}")
        parts.append("records=True")
        if rewrite.order:
            parts.append(f"order=[{', '.join(_sort_text(term) for term in rewrite.order)}]")
    return f"{rewrite.helper}({', '.join(parts)})"


def _render_first(rewrite: Rewrite, receiver: str) -> str:
    """the first page of ``keyset_page``, written in the awaited ``fetch`` / ``fetchrow`` / ``fetchval`` call's place
    and read as that method answered: the page's rows, its first row or None, that row's column or None."""
    assert rewrite.limit is not None
    by = "{" + ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.pins) + "}"
    columns = "None" if rewrite.columns is None else _columns_text(rewrite.columns)
    limit = str(rewrite.args[rewrite.limit["param"] - 1]) if "param" in rewrite.limit else str(rewrite.limit["literal"])
    order = ", ".join(_sort_text(term) for term in rewrite.order)
    page = (
        f"(await keyset_page({receiver}, {rewrite.table!r}, columns={columns}, by={by}{_where_text(rewrite)}, "
        f"order=[{order}], limit={limit}, records=True))"
    )
    if rewrite.method == "fetch":
        return f"{page}.rows"
    if rewrite.method == "fetchrow":
        return f"next(iter({page}.rows), None)"
    assert rewrite.columns is not None
    return f"next((row[{rewrite.columns[0]!r}] for row in {page}.rows), None)"


def _columns_text(columns: Sequence[str | Mapping[str, str]]) -> str:
    """a helper call's ``columns=``: a tuple of each plain column and each ``Selected(expression, alias)``."""
    listed = ", ".join(
        repr(column) if isinstance(column, str) else f"Selected({column['expr']!r}, {column['alias']!r})"
        for column in columns
    )
    return f"({listed}{',' if len(columns) == 1 else ''})"


def _render_count(rewrite: Rewrite, receiver: str) -> str:
    """a count's helper, written in the awaited ``fetchval``'s place: ``aggregate_keyed``'s one row's count, or the
    awaited ``holds_any``."""
    by = "{" + ", ".join(f"{pin['column']!r}: {_value_text(pin, rewrite.args)}" for pin in rewrite.pins) + "}"
    if rewrite.helper == "holds_any":
        walk = ", ".join(repr(column) for column in rewrite.walk)
        return f"(await holds_any({receiver}, {rewrite.table!r}, order=[{walk}], by={by}))"
    return (
        f"(await aggregate_keyed({receiver}, {rewrite.table!r}, aggregates={{'n': Count()}}, by={by}"
        f"{_where_text(rewrite)}))[0]['n']"
    )


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
    if pin["kind"] in {"fragment", "guard"}:
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
    if rewrite.where:
        parts.append(_where_text(rewrite).removeprefix(", "))
    if not rewrite.returning:
        return f"{rewrite.helper}({', '.join(parts)})"
    listed = ", ".join(repr(column) for column in rewrite.returning)
    parts.append(f"returning=({listed}{',' if len(rewrite.returning) == 1 else ''})")
    return f"(await {rewrite.helper}({', '.join(parts)})).{_RETURNED[rewrite.method]}"


def _module_text(module: _Module, value: ast.expr, seen: frozenset[str] = frozenset()) -> str | None:
    """the text a module-level expression holds, read without running it: a string literal, ``+`` of texts,
    ``"<sep>".join(<a tuple or list of string literals>)``, or a module constant bound once to one of those; None
    for anything else (a value computed at run time)."""
    if isinstance(value, ast.Constant):
        return value.value if isinstance(value.value, str) else None
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        left, right = _module_text(module, value.left, seen), _module_text(module, value.right, seen)
        return None if left is None or right is None else left + right
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "join"
        and isinstance(value.func.value, ast.Constant)
        and isinstance(value.func.value.value, str)
        and len(value.args) == 1
        and not value.keywords
    ):
        listed = value.args[0]
        if isinstance(listed, ast.Name):
            bound = _module_binding(module, listed.id)
            if bound is None or listed.id in seen:
                return None
            seen, listed = seen | {listed.id}, bound
        if not isinstance(listed, ast.Tuple | ast.List):
            return None
        parts = [_module_text(module, element, seen) for element in listed.elts]
        texts = [part for part in parts if part is not None]
        return None if len(texts) != len(parts) else value.func.value.value.join(texts)
    if isinstance(value, ast.Name) and value.id not in seen:
        bound = _module_binding(module, value.id)
        return None if bound is None else _module_text(module, bound, seen | {value.id})
    return None


def _module_binding(module: _Module, name: str) -> ast.expr | None:
    """the value ``name`` is bound to at the module's top, when it is bound exactly once in the whole module (no other
    assignment, ``global`` or loop target anywhere rebinds it); None otherwise."""
    bindings = [
        node
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store | ast.Del)
    ]
    if len(bindings) != 1 or any(isinstance(node, ast.Global) and name in node.names for node in ast.walk(module.tree)):
        return None
    for statement in module.tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            return value
    return None


def _fstring_text(module: _Module, node: ast.JoinedStr) -> tuple[str, str] | str:
    """an f-string statement whose every interpolation is a module constant of text (:func:`_module_text`), as the
    statement it sends and as the walk writes it (each interpolation ``{}``); or why it is not one."""
    sent: list[str] = []
    walked: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            sent.append(part.value)
            walked.append(part.value)
            continue
        assert isinstance(part, ast.FormattedValue)
        member = _enum_value(module, part.value) if part.conversion == -1 and part.format_spec is None else None
        if member is not None:
            # a text enum member's value (``{IntentionStatus.OPEN.value}``): the literal the walk writes in
            sent.append(member)
            walked.append(member)
            continue
        if part.conversion != -1 or part.format_spec is not None or not isinstance(part.value, ast.Name):
            return f"interpolated: {{{module.segment(part.value)}}} is not a module constant"
        text = _module_text(module, part.value)
        if text is None:
            return f"interpolated: {{{part.value.id}}} is not a module constant of text written in the module"
        sent.append(text)
        walked.append("{}")
    return "".join(sent), "".join(walked)


def _enum_value(module: _Module, value: ast.expr) -> str | None:
    """the text of a text enum member's value (``IntentionStatus.OPEN.value``), its class written in the module or
    imported by ``from <module> import <class>`` from a module beside it under one of the file's source roots; None
    for anything else."""
    if not (isinstance(value, ast.Attribute) and value.attr == "value" and isinstance(value.value, ast.Attribute)):
        return None
    owner = value.value.value
    if not isinstance(owner, ast.Name):
        return None
    key = f"{owner.id}.{value.value.attr}.value"
    for statement in module.tree.body:
        if isinstance(statement, ast.ClassDef) and statement.name == owner.id:
            return _enum_strings(statement).get(key)
    for statement in module.tree.body:
        if not (isinstance(statement, ast.ImportFrom) and statement.module and not statement.level):
            continue
        named = next((alias for alias in statement.names if (alias.asname or alias.name) == owner.id), None)
        if named is None:
            continue
        relative = Path(*statement.module.split(".")).with_suffix(".py")
        source = next((root / relative for root in module.path.parents if (root / relative).is_file()), None)
        if source is None:
            return None
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == named.name:
                return _enum_strings(node).get(f"{named.name}.{value.value.attr}.value")
        return None
    return None


def _literal_sent(module: _Module, literal: ast.Constant | ast.JoinedStr) -> str:
    """the statement a literal sends: its text, or an f-string's with each module constant written in (``""`` when an
    interpolation is not one, which no walk's text equals)."""
    if isinstance(literal, ast.Constant):
        return str(literal.value)
    resolved = _fstring_text(module, literal)
    return "" if isinstance(resolved, str) else resolved[0]


def _sent_text(config: NoWholeReadsConfig, modules: dict[str, _Module], statement: Any) -> str:
    """the statement a finding's call sends: the walk's text, or, for an f-string whose every interpolation is a
    module constant of text, that text with each constant written in (checked against the walk's ``{}`` spelling);
    ``"!<reason>"`` when it cannot be read."""
    if "{" not in statement.text:
        return str(statement.text)
    module = modules.get(statement.path)
    if module is None:
        module = modules[statement.path] = _Module.read(config.repo_root / statement.path)
    verb = statement.text.split(None, 1)[0] if statement.text.split() else "select"
    found = _statement_literal(module, statement.line, verb.lower(), fstrings=True)
    if isinstance(found, str):
        return f"!{found}"
    if not isinstance(found, ast.JoinedStr):
        return "!interpolated"
    resolved = _fstring_text(module, found)
    if isinstance(resolved, str):
        return f"!{resolved}"
    sent, walked = resolved
    if " ".join(walked.split()) != " ".join(statement.text.split()):
        return "!the f-string at its line is not the statement the walk found"
    return sent


def _statement_literal(
    module: _Module, line: int, verb: str = "select", *, fstrings: bool = False
) -> ast.Constant | ast.JoinedStr | str:
    """the string literal holding the statement at ``line``, one starting with ``verb`` (a status tag the statement's
    status is compared with, ``"DELETE 0"``, is not one)."""
    found = [
        node
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Constant | ast.JoinedStr) and getattr(node, "lineno", None) == line
    ]
    if any(isinstance(node, ast.JoinedStr) for node in found):
        if not fstrings:
            return "interpolated (an f-string)"
        joined = [
            node
            for node in found
            if isinstance(node, ast.JoinedStr)
            and node.values
            and isinstance(node.values[0], ast.Constant)
            and re.match(rf"\s*{verb}\b", str(node.values[0].value), re.I)
        ]
        if len(joined) != 1:
            return f"{len(joined)} interpolated {verb.upper()} statements on its line"
        return joined[0]
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
    if constant is None or isinstance(module.scope(constant), ast.FunctionDef | ast.AsyncFunctionDef):
        # a statement written at its call, or bound to a function's local: no other module can name it
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
    """how one shape's statements are parsed and sent.

    :ivar shape: the rewrite's shape
    :ivar parse: its statement parser
    :ivar methods: the connection methods it rewrites
    :ivar finds: the use-the-helper shape whose findings it takes; ``shape`` when empty
    """

    shape: str
    parse: Any
    methods: frozenset[str]
    finds: str = ""


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
    if method == "fetchval" and any(not isinstance((one.columns or ("",))[0], str) for one in parsed):
        return f"its dict {name}: fetchval of a select-list expression"
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
        if method == "fetchval" and not isinstance((one.columns or ("",))[0], str):
            return "fetchval of a select-list expression (read_value reads a column)"
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
#: a first-of or top-n read is a listing finding with a trailing LIMIT, sent by any of the three
_FIRST: Final[_Shape] = _Shape("first", parse_first, frozenset(_HELPER_OF), finds="list")
_WRITE: Final[_Shape] = _Shape("write", parse_write, _WRITE_METHODS)
#: a count or an existence test is sent by an awaited ``fetchval``: its one value
_COUNT: Final[_Shape] = _Shape("count", parse_count, frozenset({"fetchval"}))


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


def plan_first_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every first-of or top-n finding of the repo (a listing's statement with a trailing ``LIMIT 1`` / ``LIMIT $n``),
    as the first page of :func:`~threetears.core.keyed.keyset_page` it takes, its order completed by the table's
    key, or the refusal it gets. A listing finding with no ``LIMIT`` is refused here and planned by ``list``.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    return _plan_reads(config, _FIRST)


def plan_count_rewrites(config: NoWholeReadsConfig) -> tuple[list[Rewrite], list[Refusal]]:
    """every count finding of the repo that is a ``count(*)`` or an ``EXISTS (SELECT 1 ...)`` of a point read's rows
    sent by an awaited ``fetchval`` (:func:`parse_count`), as the :func:`~threetears.core.keyed.aggregate_keyed` or
    :func:`~threetears.core.keyed.holds_any` call it takes, or the refusal it gets.

    :param config: the repo's no-whole-reads config
    :ptype config: NoWholeReadsConfig
    :return: the rewrites, and the refusals
    :rtype: tuple[list[Rewrite], list[Refusal]]
    """
    return _plan_reads(config, _COUNT)


#: a null-safe pin's comparison, ``col IS NOT DISTINCT FROM $n``
_NULL_SAFE: Final[re.Pattern[str]] = re.compile(r"\bIS\s+NOT\s+DISTINCT\s+FROM\b", re.IGNORECASE)


def _keyed_once_sent(scan: WholeReadScan, statement: Any) -> bool:
    """whether a statement the rule finds unkeyed for its null-safe pins reads through a declared key or index once
    each is sent as the helper sends it: ``col IS NOT DISTINCT FROM $n`` is no index condition (both engines filter
    it), where the helper's ``by=`` sends ``col = $n``, or ``col IS NULL`` for a None, each one. Judged with every
    declared index counted as purposed (the index-purpose tool's walk), so that tool then names the rewritten read
    as its index's purpose; a statement unkeyed for anything else stays refused.

    :param scan: the walk the findings came from (its schema)
    :ptype scan: WholeReadScan
    :param statement: the finding's statement
    :ptype statement: Any
    :return: whether it is keyed once sent through the helper
    :rtype: bool
    """
    text = str(statement.text)
    if not _NULL_SAFE.search(text):
        return False
    schema = scan.schema
    was, schema.discovering = schema.discovering, True
    try:
        reader = Reader(schema=schema, names=frozenset({f"{statement.path}::{statement.function}"}))
        return not judge_sql(_NULL_SAFE.sub("=", text), reader)
    finally:
        schema.discovering = was


def _plan_reads(config: NoWholeReadsConfig, shape: _Shape) -> tuple[list[Rewrite], list[Refusal]]:
    """every read finding of ``shape`` (``point``, ``list``, ``first`` or ``count``) in the repo, as the rewrite it takes or the refusal it
    gets."""
    whole_hits, scan = _hits(config)
    unkeyed: dict[tuple[str, int], str] = {(hit.path, hit.line): hit.symbol for hit in whole_hits}
    hits = [
        hit
        for hit in _helper_hits(
            _undeclared_statements(config, scan, whole_hits),
            config.helper_modules,
            _producers(config),
            caller_judged=config.caller_judged_modules,
        )
        if hit.shape == (shape.finds or shape.shape)
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

        if (statement.path, statement.line) in unkeyed and not _keyed_once_sent(scan, statement):
            refuse(
                f"the no-whole-reads rule finds it unkeyed ({unkeyed[statement.path, statement.line]}): a real fix first"
            )
            continue
        text = _sent_text(config, modules, statement)
        if text.startswith("!"):
            refuse(text[1:])
            continue
        parsed = shape.parse(text, guards=True)
        if isinstance(parsed, str):
            refuse(parsed)
            continue
        unproven = _unseeded_guard(parsed)
        if unproven:
            refuse(unproven)
            continue
        completion: list[dict[str, Any]] = []
        if parsed.limit is not None:
            completed = _key_completion(parsed, scan.schema.table(parsed.table))
            if isinstance(completed, str):
                refuse(completed)
                continue
            completion = completed
        walk: list[str] = []
        if parsed.aggregate == "exists":
            walked = _probe_walk(parsed, scan.schema.table(parsed.table))
            if isinstance(walked, str):
                refuse(walked)
                continue
            walk = walked
        module = modules.get(statement.path)
        if module is None:
            module = modules[statement.path] = _Module.read(config.repo_root / statement.path)
        literal = _statement_literal(module, statement.line, fstrings=True)
        if isinstance(literal, str):
            refuse(literal)
            continue
        old_sql = _literal_sent(module, literal)
        assert isinstance(old_sql, str)
        if " ".join(old_sql.split()) != " ".join(text.split()):
            refuse("the literal at its line is not the statement the walk found")
            continue
        if isinstance(literal, ast.JoinedStr) and not isinstance(module.parent(literal), ast.Call):
            refuse("an interpolated statement not written at the call sending it")
            continue
        held_in = isinstance(literal, ast.Constant) and isinstance(module.parent(literal), ast.Dict)
        choice = _choice_of(module, literal) if isinstance(literal, ast.Constant) else None
        if (parsed.limit is not None or parsed.aggregate) and (held_in or choice is not None):
            refuse(f"a {'count' if parsed.aggregate else 'first-of read'} held in a dict or chosen by a branch")
            continue
        if (held_in or choice is not None) and isinstance(literal, ast.Constant):
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
        bound = [number for pin in (*parsed.pins, *parsed.guards) for number in pin.numbers()]
        numbers = sorted(set(bound))
        if parsed.limit is not None and "param" in parsed.limit:
            numbers.append(parsed.limit["param"])
        uses = Counter(bound)
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
            elif method == "fetchval" and parsed.columns is None and not parsed.aggregate:
                problem = "fetchval of SELECT *"
            elif method == "fetchval" and not isinstance((parsed.columns or ("",))[0], str):
                problem = "fetchval of a select-list expression (read_value reads a column)"
            elif method not in shape.methods and parsed.aggregate:
                problem = f"a count sent by {method}: read by a column name the helper does not answer under"
            elif method not in shape.methods:
                problem = f"a listing sent by {method}: its first row is a first-of read, not a listing"
            elif (parsed.limit is not None or parsed.aggregate) and not isinstance(module.parent(call), ast.Await):
                problem = "not awaited where it is sent"
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
                order=[*parsed.order, *completion],
                limit=parsed.limit,
                key_from=len(parsed.order) if parsed.limit is not None else None,
                walk=walk,
                where=[asdict(guard) for guard in parsed.guards],
            )
            spliced: ast.expr = call
            if parsed.aggregate:
                rewrite.helper = "holds_any" if parsed.aggregate == "exists" else "aggregate_keyed"
                rewrite.new_call = _render_count(rewrite, module.segment(call.func.value))
                awaited = module.parent(call)
                assert isinstance(awaited, ast.Await)
                spliced = awaited
            elif parsed.limit is None:
                rewrite.new_call = _render(rewrite, module.segment(call.func.value))
            else:
                rewrite.helper = "keyset_page"
                rewrite.new_call = _render_first(rewrite, module.segment(call.func.value))
                awaited = module.parent(call)
                assert isinstance(awaited, ast.Await)
                spliced = awaited
            assert spliced.end_lineno is not None and spliced.end_col_offset is not None
            rewrite.edits.append(
                (spliced.lineno, spliced.col_offset, spliced.end_lineno, spliced.end_col_offset, rewrite.new_call)
            )
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
        hit
        for hit in _helper_hits(
            _undeclared_statements(config, scan, whole_hits),
            config.helper_modules,
            _producers(config),
            caller_judged=config.caller_judged_modules,
        )
        if hit.shape == "write"
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
        text = _sent_text(config, modules, statement)
        if text.startswith("!"):
            refuse(text[1:])
            continue
        parsed = parse_write(text, guards=True)
        if isinstance(parsed, str):
            refuse(parsed)
            continue
        module = modules.get(statement.path)
        if module is None:
            module = modules[statement.path] = _Module.read(config.repo_root / statement.path)
        literal = _statement_literal(module, statement.line, parsed.kind, fstrings=True)
        if isinstance(literal, str):
            refuse(literal)
            continue
        old_sql = _literal_sent(module, literal)
        assert isinstance(old_sql, str)
        if " ".join(old_sql.split()) != " ".join(text.split()):
            refuse("the literal at its line is not the statement the walk found")
            continue
        if isinstance(literal, ast.JoinedStr) and not isinstance(module.parent(literal), ast.Call):
            refuse("an interpolated statement not written at the call sending it")
            continue
        held_in = isinstance(literal, ast.Constant) and isinstance(module.parent(literal), ast.Dict)
        choice = _choice_of(module, literal) if isinstance(literal, ast.Constant) else None
        if (held_in or choice is not None) and isinstance(literal, ast.Constant):
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
        unproven = _unseeded_guard(parsed)
        if unproven:
            refuse(unproven)
            continue
        bound = [number for pin in (*parsed.sets, *parsed.pins, *parsed.guards) for number in pin.numbers()]
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
                where=[asdict(guard) for guard in parsed.guards],
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
        if "aggregate_keyed" in names:
            names.add("Count")
        if any(pin["kind"] in {"now", "fragment"} for rewrite in mine for pin in rewrite.sets) or any(
            rewrite.where for rewrite in mine
        ):
            names.add("Fragment")
        if any(pin["kind"] == "increment" for rewrite in mine for pin in rewrite.sets):
            names.add("Increment")
        if any(pin["kind"] in {"any", "anylit"} for rewrite in mine for pin in rewrite.pins):
            names.add("AnyOf")
        if any(rewrite.order for rewrite in mine):
            names.add("Sort")
        if any(not isinstance(column, str) for rewrite in mine for column in rewrite.columns or ()):
            names.add("Selected")
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
    """the command line: ``point`` (reads), ``list`` (listings in their ``ORDER BY``), ``first`` (first-of and top-n
    reads), ``write`` (keyed ``UPDATE`` / ``DELETE`` by equality), ``count`` (counts and existence tests),
    ``test-module``.

    :param argv: the arguments
    :ptype argv: list[str] | None
    :return: the exit status: 0, or 1 when the after-check found a rewrite that added a whole-read finding
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="threetears.enforcement.no_whole_reads.rewrite")
    parser.add_argument("shape", choices=["point", "list", "first", "write", "count", "test-module"])
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
    planner = {
        "write": plan_write_rewrites,
        "list": plan_list_rewrites,
        "first": plan_first_rewrites,
        "count": plan_count_rewrites,
    }.get(options.shape, plan_point_rewrites)
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
