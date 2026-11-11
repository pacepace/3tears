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

    SELECT <columns> | * FROM <table> WHERE <col> = $n [AND <col> = $m | <col> = <literal> | <col> IS NULL ...]

becomes :func:`~threetears.core.keyed.read_by` (``fetch``), :func:`~threetears.core.keyed.read_row` (``fetchrow``) or
:func:`~threetears.core.keyed.read_value` (``fetchval``), each column pinned in ``by=`` exactly as the statement bound it
(``= $n`` typed by the column), ``records=True`` so the rows are what the connection answered before. Refused, and
listed: an interpolated statement, an ``IN`` / ``ANY`` / ``OR`` / range / expression predicate, an ``ORDER BY`` or
``LIMIT``, an alias or expression in the select list, a cast on a parameter, a ``fetchval`` of ``*``, a call with
starred or keyword arguments, a parameter used twice whose argument is not a plain name, a constant used elsewhere, and
a statement the no-whole-reads rule already finds unkeyed (that one needs a real fix first). After rewriting, the walk is
run again: a rewritten function must hold no whole-read finding it did not hold before.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, _hits, _producers
from threetears.enforcement.no_whole_reads.use_the_helper import _helper_hits

__all__ = ["main"]

#: the connection methods a point read is sent through, onto the helper that replaces each
_HELPER_OF: Final[dict[str, str]] = {"fetch": "read_by", "fetchrow": "read_row", "fetchval": "read_value"}

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
_AND: Final[re.Pattern[str]] = re.compile(r"\s+and\s+", re.IGNORECASE)
_QUOTED: Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")


@dataclass(frozen=True)
class Pin:
    """one column a point read pins.

    :ivar column: the column
    :ivar kind: ``param`` (``= $n``), ``literal`` (``= 'x'``, ``= TRUE``, ``= 3``) or ``null`` (``IS NULL``)
    :ivar param: the parameter's number, for ``param``
    :ivar literal: the literal's Python value, for ``literal``
    """

    column: str
    kind: str
    param: int | None = None
    literal: str | bool | int | None = None


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
    if re.search(r"[()]|\bor\b|\bin\b|\bany\b|\bnot\b|<|>|::|\blike\b|\bbetween\b|->", masked, re.IGNORECASE):
        return "a predicate other than equality (IN, ANY, OR, range, cast, expression)"
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
    pins: list[Pin] = []
    starts = [0, *(found.end() for found in _AND.finditer(masked))]
    ends = [found.start() for found in _AND.finditer(masked)] + [len(where)]
    for start, end in zip(starts, ends, strict=True):
        conjunct = where[start:end].strip()
        for pattern, kind in ((_PARAM_PIN, "param"), (_LITERAL_PIN, "literal"), (_NULL_PIN, "null")):
            pinned = pattern.match(conjunct)
            if pinned is not None:
                break
        else:
            return f"a predicate other than equality: {conjunct!r}"
        if pinned.group("q") not in qualifiers:
            return f"a column of another relation: {conjunct!r}"
        name = pinned.group("column").lower()
        if kind == "param":
            pins.append(Pin(name, "param", param=int(pinned.group("n"))))
        elif kind == "literal":
            pins.append(Pin(name, "literal", literal=_literal(pinned.group("literal"))))
        else:
            pins.append(Pin(name, "null"))
    if len({pin.column for pin in pins}) != len(pins):
        return "a column pinned twice"
    numbers = sorted({pin.param for pin in pins if pin.param is not None})
    if numbers != list(range(1, len(numbers) + 1)):
        return "parameters not numbered $1..$n"
    if not any(pin.kind == "param" for pin in pins):
        return "no bound parameter (literals only): a constant read, judged by hand"
    return PointSql(table=match.group("table").lower(), columns=columns, pins=tuple(pins))


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


def _call_of(module: _Module, use: ast.expr) -> ast.Call | str:
    """the awaited ``fetch``/``fetchrow``/``fetchval`` call whose first argument is ``use``, or why it is not one."""
    call = module.parent(use)
    if not (isinstance(call, ast.Call) and call.args and call.args[0] is use):
        return "not a call's first argument nor a constant"
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in _HELPER_OF):
        return f"sent by {ast.unparse(call.func)[:60]}, not fetch/fetchrow/fetchval"
    if not isinstance(module.parent(call), ast.Await):
        return "the call is not awaited where it is made"
    if call.keywords:
        return "keyword arguments to the call (timeout=, ...)"
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return "starred arguments"
    return call


def _sent_by(module: _Module, literal: ast.Constant) -> tuple[list[ast.Call], ast.Assign | ast.AnnAssign | None] | str:
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
            call = _call_of(module, use)
            if isinstance(call, str):
                return f"its constant {name}: {call}"
            calls.append(call)
        return calls, parent
    if holder is not literal:
        return "not a call's first argument nor a constant"
    call = _call_of(module, literal)
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
    entries = []
    for pin in rewrite.pins:
        if pin["kind"] == "param":
            value = rewrite.args[pin["param"] - 1]
        elif pin["kind"] == "literal":
            value = _python_literal(pin["literal"])
        else:
            value = "None"
        entries.append(f"{pin['column']!r}: {value}")
    by = "{" + ", ".join(entries) + "}"
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


def _statement_literal(module: _Module, line: int) -> ast.Constant | str:
    """the string literal holding the statement at ``line``."""
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
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.match(r"\s*select\b", node.value, re.I)
    ]
    if len(strings) != 1:
        return f"{len(strings)} SELECT literals on its line"
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
        except UnicodeDecodeError, OSError:
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
        source = _with_import(source, {rewrite.helper for rewrite in mine})
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


def main(argv: list[str] | None = None) -> int:
    """the command line: ``point`` (the only shape so far).

    :param argv: the arguments
    :ptype argv: list[str] | None
    :return: the exit status: 0, or 1 when the after-check found a rewrite that added a whole-read finding
    :rtype: int
    """
    parser = argparse.ArgumentParser(prog="threetears.enforcement.no_whole_reads.rewrite")
    parser.add_argument("shape", choices=["point", "test-module"])
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
        print(f"wrote {options.module}")
        return 0
    config = _config(repo, options.config)
    before, _ = _hits(config)
    rewrites, refusals = plan_point_rewrites(config)
    reasons = Counter(refusal.reason.split(":")[0].split(" (")[0] for refusal in refusals)
    print(f"point: {len(rewrites)} rewrites, {len(refusals)} refusals")
    for reason, count in reasons.most_common():
        print(f"  refused {count:4d}  {reason}")
    if options.refusals is not None:
        options.refusals.write_text(
            json.dumps([asdict(refusal) for refusal in refusals], indent=1) + "\n", encoding="utf-8"
        )
    if options.dry_run:
        for rewrite in rewrites:
            print(f"  {rewrite.path}:{rewrite.line} {rewrite.function}: {rewrite.new_call}")
        return 0
    touched = apply_rewrites(repo, rewrites, ruff=options.ruff)
    write_manifest(repo / options.manifest, rewrites)
    added = _after_check(config, before, rewrites)
    print(f"rewrote {len(rewrites)} statements in {len(touched)} modules; manifest {options.manifest}")
    for line in added:
        print(f"  AFTER-CHECK: a rewrite added a whole-read finding: {line}")
    return 1 if added else 0


if __name__ == "__main__":
    raise SystemExit(main())
