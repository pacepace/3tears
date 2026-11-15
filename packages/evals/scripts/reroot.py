#!/usr/bin/env python3
"""Re-point a tree's uses of 3tears-evals' former public roots at the roots that replaced them.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is never installed. It is one file,
it needs only Python 3.14 and its standard library, and it never imports ``threetears``. So a host runs it at the
commit it pins, before it upgrades the package, with nothing installed::

    git -C <3tears clone> show <SHA>:packages/evals/scripts/reroot.py > reroot.py
    git -C <3tears clone> show <SHA>:packages/evals/api/reroot-map.txt > reroot-map.txt
    python3.14 reroot.py --map reroot-map.txt --to <pinned SHA> --dry-run .
    python3.14 reroot.py --map reroot-map.txt --to <pinned SHA> .

Python 3.14 is required because the package and its hosts use 3.14 syntax, and this parses with the running
interpreter. Python files are parsed with ``ast`` and rewritten at the exact source positions it reports.

**Idempotent.** A second run changes nothing. A half-rewritten tree, or a branch that merges new uses of old
spellings after the cut, converges in one pass: each occurrence is rewritten on its own, and a binding this script
made on an earlier run (``from threetears.evals import host as evals_host``) is reused, never added twice.

**The map** (``reroot-map.txt``) is UTF-8 text with LF line endings. Its header is the leading lines that start
with ``#``::

    # reroot-map 1
    # epoch <sha> <title>            one per epoch, oldest first
    # body-sha256 <hex>              SHA-256 of every byte after the header
    # <anything else>                a comment

Each body line is five tab-separated fields: ``epoch``, ``record``, ``old``, ``new``, ``note``. ``epoch`` is the
SHA of an epoch in the header; the line takes effect at that commit. The records:

- ``name``: a public spelling (``threetears.evals.ops.OpsHost``) to its new public spelling
  (``threetears.evals.host.OpsHost``); ``note`` is its home (``threetears.evals.ops.host.OpsHost``) or empty, and
  ``--home`` imports from the home.
- ``renamed``: as ``name``, and the name itself changes (``ROOT_EXTRAS`` to ``EXTRAS``); ``note`` says how its shape
  changed. Rewritten, uses included, and reported once so the uses are checked.
- ``demoted``, ``left``: a name that is no longer public; ``new`` is its home, ``note`` the replacement a host
  should use. Never rewritten (except with ``--home-for-left``); every use is reported.
- ``root``: a former root to the comma-separated roots its names went to. A string naming the bare root is
  resolved when it is paired with a name of that root (``seed_module=`` beside ``constant=``), else reported.
- ``module``: an internal module path to its new path, matched by dotted prefix in imports, strings and prose.

On load the script refuses a map whose digest does not match, and a map where a new value equals, or has as a dotted
prefix, an old value of a line at or before its epoch (a ``root`` line's old value is matched only whole, so it is
not part of that rule). That is what makes a re-run a no-op. ``--to <SHA>`` applies the epochs up to and including
that one, in order; the default is all.

**What it leaves to a person** is listed on every run, and the run exits 2: a use of a removed name (``file:line
old -> replacement``), a former root it cannot resolve (a bare ``import`` with no attribute read, ``import *``,
``getattr`` or ``__import__`` on one, an unpaired root string, a root in prose that went to several roots), a string
holding code, a generated file naming an old spelling (regenerate it instead), and a spelling that names both a
public name and a module. None of these is changed, so a re-run repeats the same list and redoes nothing.

**Never touched**: dot directories, ``site-packages``, ``node_modules``, ``CHANGELOG*``, ``MIGRATIONS.toml``, the map,
this script, ``tests/fixtures/reroot/``, ``--exclude`` paths, and any line carrying ``reroot: keep`` (in Python, a
``# reroot: keep`` comment). It does not format: run your formatter over the changed files afterwards.

**Exit codes**: 0, nothing to do; 3, rewrote and nothing is left; 2, a person must edit what it lists; 1, refused
(a bad map, a file that fails to parse before or after rewriting): nothing is written.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import io
import json
import os
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = "threetears.evals"
FORMAT = "reroot-map 1"
RECORDS = ("name", "renamed", "demoted", "left", "root", "module")
NAME_RECORDS = ("name", "renamed", "demoted", "left")
REMOVED_RECORDS = ("demoted", "left")
PRAGMA = "reroot: keep"
LINE_LENGTH = 120

EXIT_CLEAN = 0
EXIT_REFUSED = 1
EXIT_HAND_EDITS = 2
EXIT_REWROTE = 3

#: Kinds of site a person must act on; any of them makes the exit code 2.
HAND_EDIT_KINDS = ("removed", "unresolved", "ambiguous", "code-string", "generated", "renamed")

#: Text files rewritten in token mode, besides Python and Markdown.
TEXT_SUFFIXES = frozenset({".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".ts", ".rst", ".txt"})
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
PRUNED_DIRS = frozenset({"site-packages", "node_modules", "__pycache__"})

#: Calls whose string argument is a module path, not an attribute.
MODULE_CALLS = frozenset({"getLogger", "get_logger", "import_module", "find_spec"})
#: Calls whose string argument is an attribute target.
ATTRIBUTE_CALLS = frozenset({"patch", "setattr", "delattr"})

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
#: A dotted spelling under the package, in prose or a string. Run over string contents, comments and text files,
#: never over Python source.
TOKEN = re.compile(r"(?<![\w.])threetears\.evals(?:\." + _IDENT + r")*")
EXACT = re.compile(r"threetears\.evals(?:\." + _IDENT + r")*(?::" + _IDENT + r"(?:\." + _IDENT + r")*)?")
IDENTIFIER = re.compile(_IDENT)
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)")


class Refused(Exception):
    """The run cannot proceed; nothing is written."""


# --------------------------------------------------------------------------------------------------------------------
# The map


@dataclass(frozen=True)
class Epoch:
    sha: str
    title: str


@dataclass(frozen=True)
class Line:
    epoch: str
    record: str
    old: str
    new: str
    note: str
    lineno: int


@dataclass(frozen=True)
class RerootMap:
    epochs: tuple[Epoch, ...]
    lines: tuple[Line, ...]

    def upto(self, to: str | None) -> tuple[Epoch, ...]:
        if to is None:
            return self.epochs
        found = [i for i, epoch in enumerate(self.epochs) if epoch.sha.startswith(to)] if len(to) >= 4 else []
        if len(found) != 1:
            raise Refused(f"--to {to}: names {'no' if not found else 'more than one'} epoch of the map")
        return self.epochs[: found[0] + 1]


def body_digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def parse_map(text: str) -> RerootMap:
    """Read a map and check it, raising :class:`Refused` on any fault."""
    rows = text.split("\n")
    if text and not text.endswith("\n"):
        raise Refused("map: the last line has no newline")
    if not rows or rows[0].strip() != f"# {FORMAT}":
        raise Refused(f"map: the first line must be '# {FORMAT}'")
    epochs: list[Epoch] = []
    digest = None
    header = 0
    for row in rows:
        if not row.startswith("#"):
            break
        header += 1
        words = row[1:].strip().split(" ", 2)
        if words[0] == "epoch":
            if len(words) < 2 or not re.fullmatch(r"[0-9a-f]{7,64}", words[1]):
                raise Refused(f"map line {header}: an epoch needs a hex SHA")
            epochs.append(Epoch(words[1], words[2] if len(words) > 2 else ""))
        elif words[0] == "body-sha256":
            digest = words[1] if len(words) > 1 else ""
    if not epochs:
        raise Refused("map: no epoch in the header")
    if len({epoch.sha for epoch in epochs}) != len(epochs):
        raise Refused("map: an epoch is listed twice")
    body = "\n".join(rows[header:])
    if digest != body_digest(body):
        raise Refused("map: body-sha256 does not match the body; the map was edited or truncated")
    order = {epoch.sha: i for i, epoch in enumerate(epochs)}
    lines: list[Line] = []
    for lineno, row in enumerate(rows[header:-1], start=header + 1):
        fields = row.split("\t")
        if len(fields) != 5:
            raise Refused(f"map line {lineno}: expected 5 tab-separated fields, found {len(fields)}")
        line = Line(*fields, lineno=lineno)
        if line.epoch not in order:
            raise Refused(f"map line {lineno}: epoch {line.epoch} is not in the header")
        if line.record not in RECORDS:
            raise Refused(f"map line {lineno}: unknown record {line.record!r}")
        _check_spelling(line.old, lineno, under=PACKAGE)
        for new in _new_values(line):
            _check_spelling(new, lineno, under="threetears")
        if line.record in REMOVED_RECORDS and not line.note.strip():
            raise Refused(f"map line {lineno}: a {line.record} line needs its replacement in the note")
        lines.append(line)
    _check_lines(lines, order)
    return RerootMap(tuple(epochs), tuple(lines))


def _check_spelling(value: str, lineno: int, *, under: str) -> None:
    if not re.fullmatch(_IDENT + r"(?:\." + _IDENT + r")*", value) or not (
        value == under or value.startswith(under + ".")
    ):
        raise Refused(f"map line {lineno}: {value!r} is not a dotted spelling under {under}")


def _new_values(line: Line) -> list[str]:
    """Every spelling a line can write: its new value, or values, and a ``name`` line's home."""
    if line.record == "root":
        return line.new.split(",")
    values = [line.new]
    if line.record in ("name", "renamed") and line.note and _looks_dotted(line.note):
        values.append(line.note)
    return values


def _looks_dotted(text: str) -> bool:
    return re.fullmatch(_IDENT + r"(?:\." + _IDENT + r")+", text) is not None


def _class(record: str) -> str:
    return "name" if record in NAME_RECORDS else record


def _check_lines(lines: list[Line], order: dict[str, int]) -> None:
    seen: dict[tuple[str, str, str], int] = {}
    for line in lines:
        key = (line.epoch, _class(line.record), line.old)
        if key in seen:
            raise Refused(
                f"map line {line.lineno}: {line.old} already has a {_class(line.record)} line at line {seen[key]}"
            )
        seen[key] = line.lineno
    prefixed = [line for line in lines if line.record != "root"]
    for line in lines:
        for new in _new_values(line):
            for other in prefixed:
                if order[other.epoch] > order[line.epoch]:
                    continue
                if new == other.old or new.startswith(other.old + "."):
                    how = "equals" if new == other.old else "has as a dotted prefix"
                    raise Refused(
                        f"map line {line.lineno}: new value {new} {how} the old value of line {other.lineno}; "
                        "a second run would rewrite its own output"
                    )


@dataclass(frozen=True)
class Table:
    """One epoch's lines, keyed for lookup, with the run's targets."""

    names: dict[str, Line]
    modules: dict[str, Line]
    roots: dict[str, Line]
    home: bool = False
    home_for_left: bool = False
    prose: bool = True

    @classmethod
    def of(cls, lines: Sequence[Line], epoch: str, **targets: bool) -> Table:
        names = {line.old: line for line in lines if line.epoch == epoch and line.record in NAME_RECORDS}
        modules = {line.old: line for line in lines if line.epoch == epoch and line.record == "module"}
        roots = {line.old: line for line in lines if line.epoch == epoch and line.record == "root"}
        return cls(names, modules, roots, **targets)


# --------------------------------------------------------------------------------------------------------------------
# Resolving one spelling


@dataclass(frozen=True)
class Resolved:
    """What to do with one dotted spelling: ``rewrite`` to ``new``, report as ``removed``, ``unresolved`` or
    ``ambiguous``, or ``none`` (not the map's business)."""

    action: str
    new: str = ""
    why: str = ""
    record: str = ""


NONE = Resolved("none")


def resolve(table: Table, dotted: str, *, context: str, code: bool) -> Resolved:
    """Resolve a dotted spelling by its longest matching ``name`` or ``module`` prefix.

    :param context: ``"attr"`` for an attribute target (an import's name, a patch target, a chain), ``"module"`` for
        a module path (a logger name, ``import_module``), ``"plain"`` for a string or prose token that could be either.
    :param code: whether the spelling is code (an import, a read, a code string) rather than prose.
    """
    segments = dotted.split(".")
    name_k = next((k for k in range(len(segments), 2, -1) if ".".join(segments[:k]) in table.names), 0)
    module_k = next((k for k in range(len(segments), 2, -1) if ".".join(segments[:k]) in table.modules), 0)
    if context == "module":
        name_k = 0
    if name_k and module_k and name_k == module_k == len(segments) and context == "plain":
        return Resolved("ambiguous", why=f"{dotted} names both a public name and a module; rewrite it by hand")
    if module_k and module_k >= name_k and not (context == "attr" and name_k == module_k == len(segments)):
        line = table.modules[".".join(segments[:module_k])]
        return Resolved("rewrite", ".".join([line.new, *segments[module_k:]]), record="module")
    if name_k:
        line = table.names[".".join(segments[:name_k])]
        rest = segments[name_k:]
        if line.record in REMOVED_RECORDS:
            if table.home_for_left or (table.home and code):
                return Resolved("rewrite", ".".join([line.new, *rest]), record=line.record)
            return Resolved("removed", why=f"{line.old} -> {line.note}", record=line.record)
        target = line.new
        if table.home and code:
            if not _looks_dotted(line.note):
                return Resolved("unresolved", why=f"{line.old} has no home in the map for --home", record=line.record)
            target = line.note
        why = f"{line.old} -> {line.new}: {line.note}" if line.record == "renamed" else ""
        return Resolved("rewrite", ".".join([target, *rest]), why=why, record=line.record)
    if dotted in table.roots:
        targets = table.roots[dotted].new.split(",")
        if not code and len(targets) == 1:
            return Resolved("rewrite", targets[0], record="root")
        return Resolved("unresolved", why=f"{dotted} is a former root; its names went to {', '.join(targets)}")
    return NONE


# --------------------------------------------------------------------------------------------------------------------
# Results of one pass


@dataclass(frozen=True, order=True)
class Site:
    path: str
    line: int
    kind: str
    text: str


@dataclass
class Pending:
    """A site whose line is computed once every pass's edits are known: ``offset`` is in the text the pass read."""

    offset: int
    kind: str
    text: str


@dataclass
class Edit:
    start: int
    end: int
    text: str
    count: str


@dataclass
class Outcome:
    edits: list[Edit] = field(default_factory=list)
    sites: list[Pending] = field(default_factory=list)
    kept: list[Pending] = field(default_factory=list)

    def merge(self, other: Outcome, shift: int) -> None:
        self.edits += [Edit(e.start + shift, e.end + shift, e.text, e.count) for e in other.edits]
        self.sites += [Pending(p.offset + shift, p.kind, p.text) for p in other.sites]
        self.kept += [Pending(p.offset + shift, p.kind, p.text) for p in other.kept]


def apply_edits(text: str, edits: list[Edit]) -> tuple[str, list[Edit]]:
    """Apply non-overlapping edits; an edit inside a larger one is dropped (the larger carries that text)."""
    kept: list[Edit] = []
    for edit in sorted(edits, key=lambda e: (e.start, -e.end)):
        if kept and edit.start < kept[-1].end:
            if edit.end <= kept[-1].end:
                continue
            raise AssertionError(f"overlapping edits at {edit.start}")
        if edit.start == edit.end and not edit.text:
            continue
        kept.append(edit)
    parts: list[str] = []
    at = 0
    for edit in kept:
        parts += [text[at : edit.start], edit.text]
        at = edit.end
    parts.append(text[at:])
    return "".join(parts), kept


def shift_offset(offset: int, edits: list[Edit]) -> int:
    """Where an offset in the text before ``edits`` lands after them."""
    delta = 0
    for edit in edits:
        if edit.end <= offset and edit.start < offset:
            delta += len(edit.text) - (edit.end - edit.start)
        elif edit.start <= offset < edit.end:
            return edit.start + delta
    return offset + delta


class Lines:
    """Offsets of a text's lines, for converting ``ast`` and ``tokenize`` positions."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0]
        at = text.find("\n")
        while at != -1:
            self.starts.append(at + 1)
            at = text.find("\n", at + 1)

    def line_text(self, lineno: int) -> str:
        start = self.starts[lineno - 1]
        end = self.starts[lineno] - 1 if lineno < len(self.starts) else len(self.text)
        return self.text[start:end].rstrip("\r")

    def offset(self, lineno: int, col_bytes: int) -> int:
        """An ``ast`` position (UTF-8 byte column) as a character offset."""
        line = self.line_text(lineno)
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col_bytes].decode("utf-8", errors="ignore"))

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)

    def span(self, node: ast.AST) -> tuple[int, int]:
        start = self.offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end = self.offset(node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
        return start, end


# --------------------------------------------------------------------------------------------------------------------
# Token mode: prose in strings, comments and text files


def rewrite_tokens(
    table: Table, text: str, base: int, lines: Lines, kept_lines: set[int], out: Outcome, *, count: str
) -> None:
    """Rewrite every spelling in ``text`` (which starts at ``base`` in the file) by its longest prefix."""
    for match in TOKEN.finditer(text):
        offset = base + match.start()
        token = match.group()
        attribute = IDENTIFIER.match(text, match.end() + 1) if text[match.end() : match.end() + 1] == ":" else None
        resolved = resolve(table, f"{token}.{attribute.group()}", context="attr", code=False) if attribute else NONE
        if resolved.action == "rewrite" and resolved.record != "module":
            target, _, name = resolved.new.rpartition(".")
            token = f"{token}:{attribute.group()}"  # type: ignore[union-attr]
            resolved = Resolved("rewrite", f"{target}:{name}", resolved.why, resolved.record)
        elif resolved.action in ("removed", "unresolved", "ambiguous"):
            token = f"{token}:{attribute.group()}"  # type: ignore[union-attr]
        else:
            resolved = resolve(table, token, context="plain", code=False)
        if resolved.action == "none":
            continue
        if lines.line_of(offset) in kept_lines:
            out.kept.append(Pending(offset, "kept", token))
        elif resolved.action == "rewrite":
            out.edits.append(Edit(offset, offset + len(token), resolved.new, count))
        else:
            out.sites.append(Pending(offset, resolved.action, resolved.why))


def pragma_lines(text: str) -> set[int]:
    return {i for i, line in enumerate(text.split("\n"), start=1) if PRAGMA in line}


def rewrite_text(table: Table, text: str) -> Outcome:
    out = Outcome()
    if table.prose:
        rewrite_tokens(table, text, 0, Lines(text), pragma_lines(text), out, count="prose_tokens_rewritten")
    return out


def rewrite_markdown(table: Table, text: str, path: str) -> Outcome:
    """Prose in token mode; a ``python`` fence that parses is rewritten as code."""
    out = Outcome()
    rows = text.split("\n")
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row) + 1)
    i = 0
    prose_from = 0
    while i < len(rows):
        opening = FENCE.match(rows[i])
        if not opening:
            i += 1
            continue
        marker, info = opening.group(2), opening.group(3).lower()
        close = next(
            (j for j in range(i + 1, len(rows)) if rows[j].strip().startswith(marker[0] * len(marker))), len(rows)
        )
        if info in ("python", "py", "python3"):
            _prose(table, text, prose_from, offsets[i + 1], out)
            body = "\n".join(rows[i + 1 : close]) + "\n"
            try:
                inner = rewrite_python(table, body, path)
            except SyntaxError:
                inner = rewrite_text(table, body)
            out.merge(inner, offsets[i + 1])
            prose_from = offsets[close] if close < len(rows) else len(text)
        i = close + 1
    _prose(table, text, prose_from, len(text), out)
    return out


def _prose(table: Table, text: str, start: int, end: int, out: Outcome) -> None:
    if not table.prose or start >= end:
        return
    lines = Lines(text)
    rewrite_tokens(table, text[start:end], start, lines, pragma_lines(text), out, count="prose_tokens_rewritten")


# --------------------------------------------------------------------------------------------------------------------
# Python


Scope = ast.AST  # a Module, FunctionDef, AsyncFunctionDef or Lambda


@dataclass(frozen=True)
class Binder:
    """How a statement binds a local name: ``from`` (module, name), ``import-as`` (module), ``import-top``
    (module), or ``other``."""

    how: str
    module: str = ""
    name: str = ""
    node: ast.AST | None = None


@dataclass
class Entry:
    """One alias of an import statement as it will be written."""

    module: str
    name: str
    asname: str | None

    def text(self) -> str:
        return self.name if self.asname is None else f"{self.name} as {self.asname}"

    def key(self) -> tuple[str, str, str | None]:
        return (self.module, self.name, self.asname)


@dataclass
class Plan:
    """What happens to one import statement."""

    node: ast.Import | ast.ImportFrom
    entries: list[Entry]
    changed: bool = False
    removed: set[str] = field(default_factory=set)
    added: list[str] = field(default_factory=list)


def alias_for(module: str) -> str:
    """The local name a module binding is rewritten through: ``evals_host`` for ``threetears.evals.host``."""
    return module.removeprefix("threetears.").replace(".", "_")


def binding_statement(module: str) -> str:
    parent, _, last = module.rpartition(".")
    alias = alias_for(module)
    return f"from {parent} import {last}" if last == alias else f"from {parent} import {last} as {alias}"


class PythonPass:
    """One epoch over one Python text."""

    def __init__(self, table: Table, text: str, path: str) -> None:
        self.table = table
        self.text = text
        self.path = path
        self.tree = ast.parse(text, filename=path)
        self.lines = Lines(text)
        self.newline = "\r\n" if "\r\n" in text else "\n"
        self.out = Outcome()
        self.comments = self._comments()
        self.kept_lines = {row for row, comment in self.comments.items() if PRAGMA in comment[1]}
        self.parent: dict[ast.AST, ast.AST] = {}
        self.scope_of: dict[ast.AST, Scope] = {}
        self.binders: dict[Scope, dict[str, list[Binder]]] = {}
        self._index()
        self.plans: dict[ast.AST, Plan] = {}
        self.handled_strings: set[ast.AST] = set()

    # -- indexing ----------------------------------------------------------------------------------------------------

    def _comments(self) -> dict[int, tuple[int, str]]:
        found: dict[int, tuple[int, str]] = {}
        for token in tokenize.generate_tokens(io.StringIO(self.text).readline):
            if token.type == tokenize.COMMENT:
                found[token.start[0]] = (self.lines.starts[token.start[0] - 1] + token.start[1], token.string)
        return found

    def _index(self) -> None:
        stack: list[tuple[ast.AST, Scope]] = [(self.tree, self.tree)]
        while stack:
            node, scope = stack.pop()
            self.scope_of[node] = scope
            inner = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
                stack.append((child, inner))
            self._bind(node, scope, inner)

    def _bind(self, node: ast.AST, scope: Scope, inner: Scope) -> None:
        def add(where: Scope, name: str, binder: Binder) -> None:
            self.binders.setdefault(where, {}).setdefault(name, []).append(binder)

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    add(scope, alias.asname, Binder("import-as", alias.name, node=node))
                else:
                    add(scope, alias.name.split(".")[0], Binder("import-top", alias.name, node=node))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    module = "." * node.level + (node.module or "")
                    add(scope, alias.asname or alias.name, Binder("from", module, alias.name, node=node))
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            add(scope, node.id, Binder("other"))
        elif isinstance(node, ast.arg):
            add(scope, node.arg, Binder("other"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(scope, node.name, Binder("other"))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(scope, node.name, Binder("other"))

    def governing(self, node: ast.AST, name: str) -> Scope | None:
        """The scope whose binding of ``name`` a read at ``node`` sees."""
        scope = self.scope_of[node]
        while True:
            if name in self.binders.get(scope, {}):
                return scope
            if scope is self.tree:
                return None
            scope = self.scope_of[scope]

    def visible(self, scope: Scope, name: str) -> list[Binder] | None:
        while True:
            if name in self.binders.get(scope, {}):
                return self.binders[scope][name]
            if scope is self.tree:
                return None
            scope = self.scope_of[scope]

    def kept(self, node: ast.AST) -> bool:
        return any(row in self.kept_lines for row in range(node.lineno, node.end_lineno + 1))  # type: ignore[attr-defined]

    def site(self, node: ast.AST, kind: str, text: str) -> None:
        offset = self.lines.offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        if self.kept(node):
            self.out.kept.append(Pending(offset, "kept", text))
        else:
            self.out.sites.append(Pending(offset, kind, text))

    # -- the pass ----------------------------------------------------------------------------------------------------

    def run(self) -> Outcome:
        imports = sorted(
            (n for n in ast.walk(self.tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                self._plan_from(node)
            else:
                self._plan_import(node)
        self._module_bindings()
        self._chains()
        self._strings()
        self._comment_tokens()
        self._emit(imports)
        return self.out

    # -- imports -----------------------------------------------------------------------------------------------------

    def _plan_from(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or not (module == PACKAGE or module.startswith(PACKAGE + ".")):
            return
        if self.kept(node):
            self.site(node, "kept", f"from {module} import ...")
            return
        moved = resolve(self.table, module, context="module", code=True)
        residual = moved.new if moved.action == "rewrite" else module
        plan = Plan(node, [], changed=residual != module)
        for alias in node.names:
            qualified = f"{module}.{alias.name}"
            if alias.name == "*":
                if module in self.table.roots:
                    self.site(node, "unresolved", f"from {module} import *: name what it uses, then re-run")
                plan.entries.append(Entry(residual, "*", None))
                continue
            resolved = resolve(self.table, qualified, context="attr", code=True)
            if qualified in self.table.roots or resolved.action == "none" or resolved.record == "module":
                if resolved.record == "module" and qualified not in self.table.names:
                    parent, _, last = resolved.new.rpartition(".")
                    local = alias.asname or alias.name
                    plan.entries.append(Entry(parent, last, None if local == last else local))
                    plan.changed = True
                else:
                    plan.entries.append(Entry(residual, alias.name, alias.asname))
                continue
            if resolved.action != "rewrite":
                self.site(node, resolved.action, resolved.why)
                plan.entries.append(Entry(residual, alias.name, alias.asname))
                continue
            target, _, name = resolved.new.rpartition(".")
            asname = alias.asname
            if resolved.record == "renamed":
                self.site(node, "renamed", resolved.why)
            if asname is None and name != alias.name:
                self._rename_uses(node, alias.name, name)
            plan.entries.append(Entry(target, name, asname))
            plan.changed = True
        self.plans[node] = plan

    def _plan_import(self, node: ast.Import) -> None:
        names = [alias.name for alias in node.names]
        if not any(name == PACKAGE or name.startswith(PACKAGE + ".") for name in names):
            return
        if self.kept(node):
            self.site(node, "kept", f"import {', '.join(names)}")
            return
        plan = Plan(node, [])
        for alias in node.names:
            moved = resolve(self.table, alias.name, context="module", code=True)
            name = moved.new if moved.action == "rewrite" else alias.name
            plan.changed |= name != alias.name
            plan.entries.append(Entry("", name, alias.asname))
        self.plans[node] = plan

    def _rename_uses(self, statement: ast.ImportFrom, old: str, new: str) -> None:
        scope = self.scope_of[statement]
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id == old and self.governing(node, old) is scope:
                start, end = self.lines.span(node)
                self.out.edits.append(Edit(start, end, new, "code_names_rewritten"))

    # -- module bindings: ``from threetears.evals import ops`` then ``ops.X`` -----------------------------------------

    def _module_bindings(self) -> None:
        for scope, bound in self.binders.items():
            for local, binders in bound.items():
                roots = {self._bound_root(b) for b in binders}
                if not any(roots):
                    continue
                if len(roots) != 1 or None in roots:
                    first = next(b.node for b in binders if self._bound_root(b))
                    assert first is not None
                    self.site(first, "unresolved", f"{local} is bound to a former root and rebound; rewrite by hand")
                    continue
                (old_root,) = roots
                assert old_root is not None
                statements = [b.node for b in binders if b.node is not None]
                self._rewrite_reads(scope, local, old_root, statements)

    def _bound_root(self, binder: Binder) -> str | None:
        if binder.how == "from" and f"{binder.module}.{binder.name}" in self.table.roots:
            return f"{binder.module}.{binder.name}"
        if binder.how == "import-as" and binder.module in self.table.roots:
            return binder.module
        return None

    def _rewrite_reads(self, scope: Scope, local: str, old_root: str, statements: list[ast.AST]) -> None:
        anchor = min(statements, key=lambda n: (n.lineno, n.col_offset))  # type: ignore[attr-defined]
        if anchor not in self.plans or any(self.kept(s) for s in statements):
            return
        remaining = rewritten = 0
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, ast.Load)):
                continue
            if self.governing(node, local) is not scope:
                continue
            parent = self.parent.get(node)
            if not (isinstance(parent, ast.Attribute) and parent.value is node):
                remaining += 1
                if (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Name)
                    and parent.func.id
                    in (
                        "getattr",
                        "hasattr",
                    )
                ):
                    self.site(node, "unresolved", f"{parent.func.id}({local}, ...) reads former root {old_root}")
                continue
            if self.kept(parent):
                self.site(parent, "kept", f"{local}.{parent.attr}")
                remaining += 1
                continue
            qualified = f"{old_root}.{parent.attr}"
            resolved = (
                resolve(self.table, qualified, context="attr", code=True) if qualified in self.table.names else NONE
            )
            if resolved.action != "rewrite":
                if resolved.action != "none":
                    self.site(parent, resolved.action, resolved.why)
                remaining += 1
                continue
            target, _, name = resolved.new.rpartition(".")
            alias = alias_for(target)
            seen = self.visible(self.scope_of[node], alias)
            if seen is not None and not all(self._binds_module(b, target) for b in seen):
                self.site(
                    parent, "unresolved", f"{alias} is already bound to something else; rewrite {local}.{parent.attr}"
                )
                remaining += 1
                continue
            if seen is None:
                statement = binding_statement(target)
                if statement not in self.plans[anchor].added:
                    self.plans[anchor].added.append(statement)
            if resolved.record == "renamed":
                self.site(parent, "renamed", resolved.why)
            start, end = self.lines.span(parent)
            self.out.edits.append(Edit(start, end, f"{alias}.{name}", "code_names_rewritten"))
            rewritten += 1
        if rewritten and not remaining:
            for statement in statements:
                if statement in self.plans:
                    self.plans[statement].removed.add(local)

    @staticmethod
    def _binds_module(binder: Binder, module: str) -> bool:
        parent, _, last = module.rpartition(".")
        return (binder.how == "from" and binder.module == parent and binder.name == last) or (
            binder.how == "import-as" and binder.module == module
        )

    # -- chains: ``import threetears.evals.kernel`` then ``threetears.evals.kernel.X`` ------------------------------

    def _chains(self) -> None:
        tops = self.binders.get(self.tree, {}).get("threetears", [])
        if not tops or any(b.how != "import-top" for b in tops):
            return
        through: Counter[str] = Counter()
        left_through: Counter[str] = Counter()
        needed: list[str] = []
        heads = sorted(
            (n for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == "threetears"),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in heads:
            if not isinstance(node.ctx, ast.Load):
                continue
            chain: list[ast.AST] = [node]
            segments = ["threetears"]
            while isinstance(self.parent.get(chain[-1]), ast.Attribute) and self.parent[chain[-1]].value is chain[-1]:  # type: ignore[attr-defined]
                chain.append(self.parent[chain[-1]])
                segments.append(chain[-1].attr)  # type: ignore[attr-defined]
            dotted = ".".join(segments)
            if not dotted.startswith(PACKAGE + "."):
                continue
            roots = [root for root in self.table.roots if dotted == root or dotted.startswith(root + ".")]
            for root in roots:
                through[root] += 1
            resolved = resolve(self.table, dotted, context="attr", code=True)
            k = self._prefix_length(dotted, resolved)
            if resolved.action == "none" or self.kept(chain[-1]):
                if dotted in self.table.roots:
                    self.site(node, "unresolved", f"{dotted} is a former root used as an object; rewrite by hand")
                for root in roots:
                    left_through[root] += 1
                continue
            if resolved.action != "rewrite":
                self.site(node, resolved.action, resolved.why)
                for root in roots:
                    left_through[root] += 1
                continue
            prefix = chain[k - 1]
            start, end = self.lines.span(prefix)
            if self.text[start:end] != ".".join(segments[:k]):
                self.site(node, "unresolved", f"{dotted} is spelled across lines; rewrite by hand")
                for root in roots:
                    left_through[root] += 1
                continue
            new_prefix = ".".join(resolved.new.split(".")[: len(resolved.new.split(".")) - (len(segments) - k)])
            self.out.edits.append(Edit(start, end, new_prefix, "code_names_rewritten"))
            module = new_prefix if resolved.record == "module" else new_prefix.rpartition(".")[0]
            if module not in needed:
                needed.append(module)
        self._settle_bare_imports(through, left_through, needed)

    def _prefix_length(self, dotted: str, resolved: Resolved) -> int:
        """How many segments of ``dotted`` the resolution replaced."""
        segments = dotted.split(".")
        for k in range(len(segments), 2, -1):
            prefix = ".".join(segments[:k])
            if prefix in self.table.names or prefix in self.table.modules:
                return k
        return len(segments)

    def _settle_bare_imports(self, through: Counter[str], left_through: Counter[str], needed: list[str]) -> None:
        bare = [
            (plan, entry)
            for plan in self.plans.values()
            if isinstance(plan.node, ast.Import)
            for entry in plan.entries
            if entry.asname is None
        ]
        present = {entry.name for _, entry in bare}
        for plan, entry in bare:
            if entry.name in self.table.roots:
                if not through[entry.name]:
                    self.site(plan.node, "unresolved", f"import {entry.name} reads nothing from it; name what it needs")
                elif not left_through[entry.name]:
                    plan.removed.add(entry.name)
        anchors = [plan for plan, _ in bare if self.scope_of[plan.node] is self.tree] or [plan for plan, _ in bare]
        if not anchors:
            return
        anchor = min(anchors, key=lambda p: (p.node.lineno, p.node.col_offset))
        for module in needed:
            if module not in present:
                anchor.added.append(f"import {module}")
                present.add(module)

    # -- strings -----------------------------------------------------------------------------------------------------

    def _strings(self) -> None:
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Constant) and isinstance(node.value, str) and "threetears.evals" in node.value
            ):
                continue
            parent = self.parent.get(node)
            if isinstance(parent, ast.JoinedStr) or self._is_docstring(node, parent):
                self._prose_string(node)
                continue
            if EXACT.fullmatch(node.value):
                self._exact_string(node, parent)
            elif self._is_code(node.value):
                self._code_string(node)
            else:
                self._prose_string(node)

    def _is_docstring(self, node: ast.Constant, parent: ast.AST | None) -> bool:
        return isinstance(parent, ast.Expr)

    @staticmethod
    def _is_code(value: str) -> bool:
        if "\n" not in value and ";" not in value and not value.lstrip().startswith(("from ", "import ")):
            return False
        try:
            tree = ast.parse(value)
        except SyntaxError:
            return False
        return any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Call)) for n in ast.walk(tree))

    def _call_name(self, parent: ast.AST | None) -> str:
        if isinstance(parent, ast.keyword):
            parent = self.parent.get(parent)
        if not isinstance(parent, ast.Call):
            return ""
        func = parent.func
        return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""

    def _exact_string(self, node: ast.Constant, parent: ast.AST | None) -> None:
        value = node.value
        call = self._call_name(parent)
        if call == "__import__":
            if resolve(self.table, value, context="module", code=True).action != "none" or value in self.table.roots:
                self.site(node, "unresolved", f"__import__({value!r}) names a former spelling; use an import")
            return
        module, colon, attribute = value.partition(":")
        if colon:
            resolved = resolve(self.table, f"{module}.{attribute}", context="attr", code=True)
            if resolved.action == "rewrite" and resolved.record != "module":
                target, _, name = resolved.new.rpartition(".")
                new_value = f"{target}:{name}" if "." not in attribute else None
                if new_value is None:
                    head = resolve(self.table, f"{module}.{attribute.split('.')[0]}", context="attr", code=True)
                    target, _, name = head.new.rpartition(".")
                    new_value = f"{target}:{name}.{attribute.partition('.')[2]}"
                self._replace_in_string(node, value, new_value)
                return
            resolved = resolve(self.table, module, context="module", code=True)
            if resolved.action == "rewrite":
                self._replace_in_string(node, value, f"{resolved.new}:{attribute}")
            elif module in self.table.roots:
                self.site(node, "unresolved", f"{value!r}: {module} is a former root; rewrite by hand")
            return
        context = "module" if call in MODULE_CALLS else "attr" if call in ATTRIBUTE_CALLS else "plain"
        if value in self.table.roots and context != "module":
            self._paired_root(node, parent)
            return
        resolved = resolve(self.table, value, context=context, code=True)
        if resolved.action == "rewrite":
            if resolved.record == "renamed":
                self.site(node, "renamed", resolved.why)
            self._replace_in_string(node, value, resolved.new)
        elif resolved.action != "none":
            self.site(node, resolved.action, resolved.why)
        elif value in self.table.roots:
            self.site(
                node, "unresolved", f"{value!r} is a former root; its names went to {self.table.roots[value].new}"
            )

    def _paired_root(self, node: ast.Constant, parent: ast.AST | None) -> None:
        """A bare former root resolves when the call or tuple pairs it with the name it is read for."""
        root = node.value
        siblings: list[ast.AST] = []
        if isinstance(parent, ast.keyword):
            call = self.parent.get(parent)
            if isinstance(call, ast.Call):
                siblings = [*call.args, *(k.value for k in call.keywords)]
        elif isinstance(parent, (ast.Call, ast.Tuple, ast.List)):
            siblings = (
                [*parent.args, *(k.value for k in parent.keywords)] if isinstance(parent, ast.Call) else parent.elts
            )
        names = [
            s.value
            for s in siblings
            if s is not node
            and isinstance(s, ast.Constant)
            and isinstance(s.value, str)
            and IDENTIFIER.fullmatch(s.value)
        ]
        targets: set[str] = set()
        for name in names:
            line = self.table.names.get(f"{root}.{name}")
            if line is None:
                continue
            if line.record in REMOVED_RECORDS:
                self.site(node, "removed", f"{line.old} -> {line.note}")
                return
            target, _, new_name = line.new.rpartition(".")
            if new_name != name:
                self.site(
                    node, "unresolved", f"{root!r} is paired with {name!r}, renamed to {new_name}; rewrite by hand"
                )
                return
            targets.add(target)
        if len(targets) == 1:
            self._replace_in_string(node, root, targets.pop())
        else:
            self.site(
                node,
                "unresolved",
                f"{root!r} is a former root and nothing beside it names one of its names; its names went to "
                f"{self.table.roots[root].new}",
            )

    def _replace_in_string(self, node: ast.Constant, old: str, new: str) -> None:
        if self.kept(node):
            self.site(node, "kept", old)
            return
        start, end = self.lines.span(node)
        source = self.text[start:end]
        at = source.find(old)
        if at == -1 or source.find(old, at + 1) != -1:
            self.site(node, "unresolved", f"{old!r} is split across string literals; rewrite by hand")
            return
        self.out.edits.append(Edit(start + at, start + at + len(old), new, "strings_rewritten"))

    def _code_string(self, node: ast.Constant) -> None:
        if any(
            resolve(self.table, m.group(), context="plain", code=False).action != "none"
            for m in TOKEN.finditer(node.value)
        ):
            self.site(node, "code-string", "a string holding code names a former spelling; rewrite it by hand")

    def _prose_string(self, node: ast.Constant) -> None:
        if not self.table.prose:
            return
        start, end = self.lines.span(node)
        rewrite_tokens(
            self.table,
            self.text[start:end],
            start,
            self.lines,
            self.kept_lines,
            self.out,
            count="prose_tokens_rewritten",
        )

    def _comment_tokens(self) -> None:
        if not self.table.prose:
            return
        for offset, comment in self.comments.values():
            if PRAGMA not in comment:
                rewrite_tokens(self.table, comment, offset, self.lines, set(), self.out, count="prose_tokens_rewritten")

    # -- writing the import statements -------------------------------------------------------------------------------

    def _emit(self, imports: list[ast.Import | ast.ImportFrom]) -> None:
        present: dict[tuple[int, str], set[tuple[str, str, str | None]]] = {}
        for plan in self.plans.values():
            if not self._touched(plan):
                for entry in plan.entries:
                    present.setdefault(self._block(plan.node), set()).add(self._key(plan, entry))
        for node in imports:
            plan = self.plans.get(node)
            if plan is None or not self._touched(plan):
                continue
            seen = present.setdefault(self._block(node), set())
            kept: list[Entry] = []
            for entry in plan.entries:
                local = entry.asname or (entry.name.split(".")[0] if isinstance(node, ast.Import) else entry.name)
                if (entry.asname or entry.name) in plan.removed or (
                    isinstance(node, ast.Import) and entry.name in plan.removed
                ):
                    continue
                if local in plan.removed:
                    continue
                key = self._key(plan, entry)
                if key in seen and entry.name != "*":
                    continue
                seen.add(key)
                kept.append(entry)
            added = [s for s in plan.added if self._statement_key(s) not in seen]
            for statement in added:
                seen.add(self._statement_key(statement))
            self._write(node, kept, added)

    def _touched(self, plan: Plan) -> bool:
        return plan.changed or bool(plan.removed) or bool(plan.added)

    def _key(self, plan: Plan, entry: Entry) -> tuple[str, str, str | None]:
        return ("", entry.name, entry.asname) if isinstance(plan.node, ast.Import) else entry.key()

    @staticmethod
    def _statement_key(statement: str) -> tuple[str, str, str | None]:
        if statement.startswith("import "):
            return ("", statement[len("import ") :], None)
        words = statement.split()
        return (words[1], words[3], words[5] if len(words) > 5 else None)

    def _block(self, node: ast.AST) -> tuple[int, str]:
        parent = self.parent[node]
        for name, value in ast.iter_fields(parent):
            if isinstance(value, list) and any(item is node for item in value):
                return (id(parent), name)
        return (id(parent), "")

    def _write(self, node: ast.Import | ast.ImportFrom, entries: list[Entry], added: list[str]) -> None:
        start, end = self.lines.span(node)
        line_start = self.lines.starts[node.lineno - 1]
        indent = self.text[line_start:start]
        inline = bool(indent.strip())
        last = self.lines.line_text(node.end_lineno)  # type: ignore[arg-type]
        tail_start = end - self.lines.starts[node.end_lineno - 1]  # type: ignore[index]
        tail = last[tail_start:]
        inline |= bool(tail.strip()) and not tail.strip().startswith("#")
        comments = [
            (row, text)
            for row, (_, text) in sorted(self.comments.items())
            if node.lineno <= row <= node.end_lineno  # type: ignore[operator]
        ]
        lead = ""
        interior = []
        if comments and not inline:
            first_row = node.lineno if comments[0][0] == node.lineno else node.end_lineno
            lead = next((text for row, text in comments if row == first_row), "")
            interior = [text for row, text in comments if text != lead or row != first_row]
            if comments[-1][0] == node.end_lineno and tail.strip().startswith("#"):
                end = self.lines.starts[node.end_lineno - 1] + len(last)  # type: ignore[index]
        statements = self._render(node, entries)
        statements += [s.split("\n") for s in added]
        if inline:
            replacement = "; ".join(" ".join(part.strip() for part in s) for s in statements) or "pass"
            self.out.edits.append(Edit(start, end, replacement, "statements_rewritten"))
            return
        if not statements:
            if len(self._siblings(node)) == 1:
                statements = [["pass"]]
            else:
                full_end = (
                    self.lines.starts[node.end_lineno] if node.end_lineno < len(self.lines.starts) else len(self.text)
                )  # type: ignore[operator]
                text = "".join(f"{indent}{c}{self.newline}" for c in [lead, *interior] if c)
                self.out.edits.append(Edit(line_start, full_end, text, "statements_rewritten"))
                return
        if lead:
            first = statements[0]
            first[0] = f"{first[0]}  {lead}"
        rows = [c for c in interior] + [row for s in statements for row in s]
        replacement = (self.newline + indent).join(rows)
        self.out.edits.append(Edit(start, end, replacement, "statements_rewritten"))

    def _siblings(self, node: ast.AST) -> list[ast.AST]:
        parent = self.parent[node]
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list) and any(item is node for item in value):
                return value
        return [node]

    def _render(self, node: ast.Import | ast.ImportFrom, entries: list[Entry]) -> list[list[str]]:
        if isinstance(node, ast.Import):
            return [[f"import {entry.name}" + (f" as {entry.asname}" if entry.asname else "")] for entry in entries]
        groups: dict[str, list[Entry]] = {}
        for entry in entries:
            groups.setdefault(entry.module, []).append(entry)
        indent = node.col_offset
        rendered = []
        for module, members in groups.items():
            one = f"from {module} import {', '.join(m.text() for m in members)}"
            if indent + len(one) <= LINE_LENGTH or len(members) == 1:
                rendered.append([one])
            else:
                rendered.append([f"from {module} import (", *(f"    {m.text()}," for m in members), ")"])
        return rendered


def rewrite_python(table: Table, text: str, path: str) -> Outcome:
    """One epoch over a Python text. Raises ``SyntaxError`` when it does not parse."""
    return PythonPass(table, text, path).run()


# --------------------------------------------------------------------------------------------------------------------
# The tree


@dataclass
class Options:
    root: Path
    map_path: Path
    to: str | None = None
    dry_run: bool = False
    verbose: bool = False
    as_json: bool = False
    prose: bool = True
    home: bool = False
    home_for_left: bool = False
    exclude: tuple[str, ...] = ()


@dataclass
class Report:
    epochs: tuple[Epoch, ...] = ()
    counts: Counter[str] = field(default_factory=Counter)
    sites: list[Site] = field(default_factory=list)
    kept: list[Site] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    def exit_code(self) -> int:
        if self.refused:
            return EXIT_REFUSED
        if any(site.kind in HAND_EDIT_KINDS for site in self.sites):
            return EXIT_HAND_EDITS
        return EXIT_REWROTE if self.changed else EXIT_CLEAN


def excluded(relative: Path, options: Options, protected: set[Path], absolute: Path) -> bool:
    name = relative.name
    parts = relative.parts
    if name.startswith("CHANGELOG") or name == "MIGRATIONS.toml" or name == "reroot-map.txt":
        return True
    if absolute.resolve() in protected:
        return True
    if any(parts[i : i + 3] == ("tests", "fixtures", "reroot") for i in range(len(parts))):
        return True
    posix = relative.as_posix()
    return any(posix == e or posix.startswith(e.rstrip("/") + "/") for e in options.exclude)


def walk(options: Options) -> Iterator[Path]:
    if options.root.is_file():
        yield options.root
        return
    for directory, dirs, files in os.walk(options.root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in PRUNED_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in PYTHON_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES:
                yield path


def generated_header(text: str) -> str | None:
    """The line of a generated file's first ten that marks it, or ``None``."""
    for line in text.split("\n")[:10]:
        if "GENERATED" in line or "do not edit" in line.lower():
            return line.strip(" */#<>!-")
    return None


def one_pass(table: Table, text: str, path: Path) -> Outcome:
    if path.suffix in PYTHON_SUFFIXES:
        return rewrite_python(table, text, str(path))
    if path.suffix in MARKDOWN_SUFFIXES:
        return rewrite_markdown(table, text, str(path))
    return rewrite_text(table, text)


def process(rmap: RerootMap, options: Options) -> tuple[Report, dict[Path, str]]:
    report = Report(epochs=rmap.upto(options.to))
    targets = {"home": options.home, "home_for_left": options.home_for_left, "prose": options.prose}
    tables = [Table.of(rmap.lines, epoch.sha, **targets) for epoch in report.epochs]
    protected = {options.map_path.resolve(), Path(__file__).resolve()}
    texts: dict[Path, str] = {}
    for path in walk(options):
        relative = path.relative_to(options.root) if path != options.root else Path(path.name)
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            continue
        if "threetears" not in text:
            continue
        if excluded(relative, options, protected, path):
            if PACKAGE in text:
                report.skipped.append(f"{relative.as_posix()}: excluded")
                report.counts["files_skipped"] += 1
            continue
        if path.suffix in PYTHON_SUFFIXES:
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                report.refused.append(
                    f"{relative.as_posix()}:{error.lineno}: does not parse before rewriting: {error.msg}"
                )
                continue
        texts[path] = text
    if report.refused:
        return report, {}
    written: dict[Path, str] = {}
    for path, text in texts.items():
        relative = (path.relative_to(options.root) if path != options.root else Path(path.name)).as_posix()
        report.counts["files_scanned"] += 1
        header = generated_header(text)
        current = text
        renamed: list[Pending] = []
        broken = None
        for table in tables:
            outcome = one_pass(table, current, path)
            current, applied = apply_edits(current, outcome.edits)
            if applied and path.suffix in PYTHON_SUFFIXES:
                try:
                    ast.parse(current, filename=relative)
                except SyntaxError as error:
                    broken = f"{relative}:{error.lineno}: does not parse after rewriting: {error.msg}"
                    break
            for item in renamed:
                item.offset = shift_offset(item.offset, applied)
            renamed += [
                Pending(shift_offset(p.offset, applied), p.kind, p.text) for p in outcome.sites if p.kind == "renamed"
            ]
            if header is None:
                report.counts.update(edit.count for edit in applied)
            elif applied or outcome.sites:
                break
        if broken is not None:
            report.refused.append(broken)
            continue
        if header is not None:
            if current != text or renamed or any(one_pass(t, text, path).sites for t in tables):
                report.sites.append(Site(relative, 1, "generated", f"generated ({header}); regenerate it, do not edit"))
                report.counts["generated_files"] += 1
                report.counts["files_skipped"] += 1
                report.skipped.append(f"{relative}: generated")
            continue
        # What is left is read off the converged text, so a re-run lists the same sites at the same lines; a pass
        # that would still change it means the rewrite did not converge, which is refused rather than written.
        lines = Lines(current)
        found: list[Site] = [Site(relative, lines.line_of(p.offset), p.kind, p.text) for p in renamed]
        kept: list[Site] = []
        for table in tables:
            outcome = one_pass(table, current, path)
            if apply_edits(current, outcome.edits)[0] != current:
                report.refused.append(
                    f"{relative}: a second pass would change it again; the map or the script is wrong"
                )
                break
            found += [Site(relative, lines.line_of(p.offset), p.kind, p.text) for p in outcome.sites]
            kept += [Site(relative, lines.line_of(p.offset), "kept", p.text) for p in outcome.kept]
        report.sites += [site for i, site in enumerate(found) if site not in found[:i]]
        report.kept += [site for i, site in enumerate(kept) if site not in kept[:i]]
        if current != text:
            report.changed.append(relative)
            written[path] = current
    report.counts["files_changed"] = len(report.changed)
    for kind in ("removed", "unresolved", "ambiguous", "code-string", "renamed"):
        report.counts[kind.replace("-", "_") + ("_name_uses" if kind == "removed" else "")] = sum(
            1 for site in report.sites if site.kind == kind
        )
    report.counts["kept_lines"] = len(report.kept)
    report.sites.sort()
    report.kept.sort()
    if report.refused:
        return report, {}
    return report, written


COUNT_ORDER = (
    ("files_scanned", "files scanned"),
    ("files_changed", "files changed"),
    ("files_skipped", "files skipped (excluded or generated)"),
    ("statements_rewritten", "import statements rewritten"),
    ("code_names_rewritten", "names rewritten in code"),
    ("strings_rewritten", "strings rewritten"),
    ("prose_tokens_rewritten", "prose spellings rewritten"),
    ("removed_name_uses", "uses of a removed name"),
    ("unresolved", "former roots left unresolved"),
    ("ambiguous", "spellings naming a name and a module"),
    ("code_strings", "strings holding code"),
    ("generated_files", "generated files to regenerate"),
    ("renamed", "renamed names to check"),
    ("kept_lines", "pragma-kept sites"),
)


def counts_of(report: Report) -> dict[str, int]:
    counts = dict(report.counts)
    counts["code_strings"] = sum(1 for site in report.sites if site.kind == "code-string")
    return {key: counts.get(key, 0) for key, _ in COUNT_ORDER}


def render(report: Report, options: Options) -> str:
    code = report.exit_code()
    if options.as_json:
        return json.dumps(
            {
                "exit": code,
                "dry_run": options.dry_run,
                "epochs": [epoch.sha for epoch in report.epochs],
                "counts": counts_of(report),
                "hand_edits": [site.__dict__ for site in report.sites],
                "kept": [site.__dict__ for site in report.kept],
                "skipped": report.skipped,
                "changed": report.changed,
                "refused": report.refused,
            },
            indent=2,
        )
    out = [f"reroot: epochs applied: {', '.join(f'{e.sha[:12]} {e.title}'.strip() for e in report.epochs)}"]
    if report.refused:
        out.append("refused; nothing was written:")
        out += [f"  {line}" for line in report.refused]
        return "\n".join(out)
    counts = counts_of(report)
    width = max(len(label) for _, label in COUNT_ORDER)
    out += [f"  {label:<{width}}  {counts[key]}" for key, label in COUNT_ORDER]
    if report.sites:
        out.append("edit by hand (the script never changes these; re-run after):")
        out += [f"  {s.path}:{s.line}: {s.kind}: {s.text}" for s in report.sites]
    if options.verbose:
        out += ["changed:", *(f"  {path}" for path in report.changed)]
        out += ["skipped:", *(f"  {line}" for line in report.skipped)]
        out += ["kept by pragma:", *(f"  {s.path}:{s.line}: {s.text}" for s in report.kept)]
    if options.dry_run:
        out.append("dry run: nothing was written")
    elif report.changed:
        out.append("run your formatter over the changed files; this script emits canonical import lines only")
    return "\n".join(out)


def parse_args(argv: Sequence[str]) -> Options:
    parser = argparse.ArgumentParser(
        prog="reroot.py",
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        epilog=(
            "Needs only Python 3.14 and its standard library; it never imports threetears, so run it before "
            "upgrading the package: fetch this file and the map at the pinned commit with `git show "
            "<SHA>:packages/evals/scripts/reroot.py` and `git show <SHA>:packages/evals/api/reroot-map.txt`, then "
            "`python3.14 reroot.py --map reroot-map.txt --to <SHA> --dry-run .`. Exit codes: 0 nothing to do, "
            "3 rewrote, 2 edit by hand what it lists, 1 refused (nothing written)."
        ),
    )
    parser.add_argument("root", type=Path, help="the tree (or one file) to rewrite")
    parser.add_argument("--map", dest="map_path", type=Path, required=True, help="reroot-map.txt")
    parser.add_argument("--to", help="apply epochs up to and including this SHA (a prefix will do); default all")
    parser.add_argument("--dry-run", action="store_true", help="count and list; write nothing")
    parser.add_argument("--verbose", action="store_true", help="also list changed, skipped and pragma-kept files")
    parser.add_argument("--json", dest="as_json", action="store_true", help="print the report as JSON")
    parser.add_argument("--no-prose", dest="prose", action="store_false", help="skip docstrings, comments and text")
    parser.add_argument("--home", action="store_true", help="imports go to each name's home module (package src)")
    parser.add_argument(
        "--home-for-left", action="store_true", help="rewrite removed names to their home instead of reporting them"
    )
    parser.add_argument("--exclude", action="append", default=[], help="a path under the root to leave alone")
    args = parser.parse_args(argv)
    return Options(
        root=args.root,
        map_path=args.map_path,
        to=args.to,
        dry_run=args.dry_run,
        verbose=args.verbose,
        as_json=args.as_json,
        prose=args.prose,
        home=args.home,
        home_for_left=args.home_for_left,
        exclude=tuple(args.exclude),
    )


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    if sys.version_info < (3, 14):
        print("reroot.py needs Python 3.14: the package and its hosts use 3.14 syntax", file=sys.stderr)
        return EXIT_REFUSED
    try:
        rmap = parse_map(options.map_path.read_text(encoding="utf-8"))
        report, written = process(rmap, options)
    except Refused as refusal:
        print(f"reroot: refused: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    if not options.dry_run:
        for path, text in written.items():
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
    print(render(report, options))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
