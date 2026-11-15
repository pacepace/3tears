#!/usr/bin/env python3
"""Re-point a tree's uses of 3tears-evals' former public roots at the roots that replaced them.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is never installed. It is one file,
it needs only Python 3.14 and its standard library, and it never imports ``threetears``. So a host runs it at the
commit it pins, before it upgrades the package, with nothing installed::

    git -C <3tears clone> show <pin>:packages/evals/scripts/reroot.py > reroot.py
    git -C <3tears clone> show <pin>:packages/evals/api/reroot-map.txt > reroot-map.txt
    python3.14 reroot.py --map reroot-map.txt --dry-run .
    python3.14 reroot.py --map reroot-map.txt .

``<pin>`` is the 3tears commit the host pins. The map fetched there holds exactly the epochs at or before that commit,
so a run with no ``--to`` applies every epoch that governs the pin. ``--to`` takes only an epoch SHA from the map's
header (or a unique prefix of one, at least 4 characters): a release commit is not an epoch unless the header lists
it, and the map cannot say which epoch an arbitrary commit falls under.

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
not part of that rule). That is what makes a re-run a no-op. ``--to <epoch SHA>`` applies the header's epochs up to
and including that one, in order; the default is all.

**What it leaves to a person** is listed on every run, with ``file:line``, what was found and what to write, and the
run exits 2: a use of a removed name (``file:line old -> replacement``); a former root it cannot resolve (a bare
``import`` with no attribute read, ``import *``, a former-root binding used other than as ``binding.attr`` --
``getattr``, ``monkeypatch.setattr``, ``patch.object``, an argument, ``__all__`` -- ``__import__``, a read through an
alias of the package such as ``import threetears.evals as te`` then ``te.ops.X``, an unpaired root string, a root in
prose that went to several roots); a string holding code; a generated file naming an old spelling (regenerate it
instead); a spelling that names both a public name and a module; a renamed name whose new name is already bound in
the file; a symbolic link (never written through) or a file that is not text in its declared encoding, either of
which names the package. None of these is changed, so a re-run repeats the same list and redoes nothing. While one
remains, the run never exits 0 or 3.

**Never touched**: dot directories, ``site-packages``, ``node_modules``, ``CHANGELOG*``, ``MIGRATIONS.toml``, the map,
this script, ``tests/fixtures/reroot/``, ``--exclude`` paths, and any line carrying ``reroot: keep`` (in Python, a
``# reroot: keep`` comment). It does not format: run your formatter over the changed files afterwards.

**Writing.** Each file is written whole and atomically (a temporary file beside it, then a rename), keeping its line
endings, its encoding and a UTF-8 byte-order mark. If a write fails, the report says which files were written and
which were not, and the run exits 1; every written file is complete, so running it again finishes the job.

**Exit codes**: 0, nothing to do; 3, rewrote and nothing is left; 2, a person must edit what it lists; 1, refused
(a bad map, a file that fails to parse before or after rewriting: nothing is written), or a write failed (the report
lists what was and was not written).
"""

from __future__ import annotations

import argparse
import ast
import bisect
import contextlib
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
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
HAND_EDIT_KINDS = (
    "removed",
    "unresolved",
    "ambiguous",
    "code-string",
    "generated",
    "renamed",
    "collision",
    "linked",
    "undecodable",
)

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
            accepted = "; ".join(f"{epoch.sha} {epoch.title}".strip() for epoch in self.epochs)
            raise Refused(
                f"--to {to}: names {'no' if not found else 'more than one'} epoch of the map. --to takes an epoch SHA "
                f"from the map's header, or a unique prefix of one of at least 4 characters: {accepted}. A commit "
                "that is not an epoch (a release pin) is not accepted: fetch the map at your pin and run without "
                "--to, which applies every epoch at or before it"
            )
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
        self.expression_strings = self._expression_strings()

    # -- indexing ----------------------------------------------------------------------------------------------------

    def _comments(self) -> dict[int, tuple[int, str]]:
        found: dict[int, tuple[int, str]] = {}
        for token in tokenize.generate_tokens(io.StringIO(self.text).readline):
            if token.type == tokenize.COMMENT:
                found[token.start[0]] = (self.lines.starts[token.start[0] - 1] + token.start[1], token.string)
        return found

    def _expression_strings(self) -> dict[ast.Constant, str]:
        """String constants that are code naming a local: ``"type"`` for a string annotation or ``cast``'s first
        argument (an expression the type checker reads), ``"all"`` for an entry of ``__all__``."""
        found: dict[ast.Constant, str] = {}

        def mark(root: ast.AST | None, kind: str) -> None:
            for inner in ast.walk(root) if root is not None else ():
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    found.setdefault(inner, kind)

        for node in ast.walk(self.tree):
            if isinstance(node, ast.arg):
                mark(node.annotation, "type")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mark(node.returns, "type")
            elif isinstance(node, ast.AnnAssign):
                mark(node.annotation, "type")
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
                if called == "cast" and node.args:
                    mark(node.args[0], "type")
                if (
                    called in ("append", "extend")
                    and isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "__all__"
                ):
                    for argument in node.args:
                        mark(argument, "all")
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                mark(node.value, "all")  # type: ignore[attr-defined]
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
            add(scope, node.id, Binder("other", node=node))
        elif isinstance(node, ast.arg):
            add(scope, node.arg, Binder("other", node=node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(scope, node.name, Binder("other", node=node))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(scope, node.name, Binder("other", node=node))

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
        self._package_aliases()
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
            if asname is None and name != alias.name:
                clash = self._clash(name, target)
                if clash is not None:
                    self.site(
                        node,
                        "collision",
                        f"{qualified} is renamed to {resolved.new} ({resolved.why.partition(': ')[2]}), but {name} is "
                        f"already bound at line {clash}: write `from {target} import {name} as {alias.name}`, or "
                        f"rename the other {name}, then re-run",
                    )
                    plan.entries.append(Entry(residual, alias.name, alias.asname))
                    continue
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

    def _clash(self, name: str, module: str) -> int | None:
        """The line of a binding of ``name`` anywhere in the file other than an import of ``name`` from ``module``
        (an earlier run's, or a merge's), or ``None``."""
        for bound in self.binders.values():
            for binder in bound.get(name, []):
                if binder.how == "from" and binder.module == module and binder.name == name:
                    continue
                return binder.node.lineno if binder.node is not None else 1  # type: ignore[attr-defined]
        return None

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
                self.site(node, "unresolved", self._object_use(node, parent, local, old_root))
                continue
            if self.kept(parent):
                self.site(parent, "kept", f"{local}.{parent.attr}")
                remaining += 1
                continue
            target = self._read_target(node, parent, local, old_root, anchor)
            if target is None:
                remaining += 1
                continue
            start, end = self.lines.span(parent)
            self.out.edits.append(Edit(start, end, target, "code_names_rewritten"))
            rewritten += 1
        for constant, kind in self.expression_strings.items():
            if self.governing(constant, local) is not scope:
                continue
            done, left = self._expression_string(constant, kind, local, old_root, anchor)
            rewritten += done
            remaining += left
        if rewritten and not remaining:
            for statement in statements:
                if statement in self.plans:
                    self.plans[statement].removed.add(local)

    def _read_target(
        self, at: ast.AST, report_at: ast.AST, local: str, old_root: str, anchor: ast.AST, attr: str | None = None
    ) -> str | None:
        """What ``local.attr`` (a read of former root ``old_root``) becomes, adding the binding it needs; ``None``
        when it cannot be rewritten (the site is then listed, or the name is not the map's)."""
        attr = attr if attr is not None else report_at.attr  # type: ignore[attr-defined]
        qualified = f"{old_root}.{attr}"
        resolved = resolve(self.table, qualified, context="attr", code=True) if qualified in self.table.names else NONE
        if resolved.action != "rewrite":
            if resolved.action != "none":
                self.site(report_at, resolved.action, resolved.why)
            return None
        target, _, name = resolved.new.rpartition(".")
        alias = alias_for(target)
        seen = self.visible(self.scope_of[at], alias)
        if seen is not None and not all(self._binds_module(b, target) for b in seen):
            self.site(
                report_at,
                "unresolved",
                f"{alias} is already bound to something else; rewrite {local}.{attr} as {name} from {target}",
            )
            return None
        if seen is None:
            statement = binding_statement(target)
            if statement not in self.plans[anchor].added:
                self.plans[anchor].added.append(statement)
        if resolved.record == "renamed":
            self.site(report_at, "renamed", resolved.why)
        return f"{alias}.{name}"

    def _object_use(self, node: ast.Name, parent: ast.AST | None, local: str, old_root: str) -> str:
        """What to say about a former-root binding used as an object, not read as ``local.attr``."""
        went = self.table.roots[old_root].new.replace(",", ", ")
        if isinstance(parent, ast.keyword):
            parent = self.parent.get(parent)
        if isinstance(parent, ast.Call):
            func = parent.func
            called = ast.unparse(func)
            named = [
                a.value
                for a in [*parent.args, *(k.value for k in parent.keywords)]
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and IDENTIFIER.fullmatch(a.value)
            ]
            where = []
            for attr in named:
                line = self.table.names.get(f"{old_root}.{attr}")
                if line is not None and line.record not in REMOVED_RECORDS:
                    where.append(f"{attr} went to {line.new.rpartition('.')[0]}")
                elif line is not None:
                    where.append(f"{attr} was removed: {line.note}")
            detail = "; ".join(where) or f"its names went to {went}"
            return (
                f"{called}({local}, ...) passes former root {old_root} as an object; {detail}: pass the module the "
                "code reads it from instead"
            )
        return (
            f"{local} (former root {old_root}) is used as an object, not read as {local}.<name>; its names went to "
            f"{went}: name the module or the names this use needs"
        )

    def _expression_string(
        self, constant: ast.Constant, kind: str, local: str, old_root: str, anchor: ast.AST
    ) -> tuple[int, int]:
        """A string annotation or ``cast`` type naming ``local``, or an ``__all__`` entry exporting it: rewrite a
        type string's ``local.attr`` reads, list everything else. Returns (rewritten, left)."""
        value = constant.value
        if kind == "all":
            if value != local:
                return 0, 0
            self.site(
                constant,
                "unresolved",
                f"__all__ exports {local}, bound to former root {old_root}; export the names it is read for "
                f"(they went to {self.table.roots[old_root].new.replace(',', ', ')}) instead",
            )
            return 0, 1
        try:
            parsed = ast.parse(value.strip(), mode="eval")
        except SyntaxError:
            return 0, 0
        reads = [n for n in ast.walk(parsed) if isinstance(n, ast.Name) and n.id == local]
        if not reads:
            return 0, 0
        if self.kept(constant):
            self.site(constant, "kept", value)
            return 0, len(reads)
        start, end = self.lines.span(constant)
        source = self.text[start:end]
        lead = len(source) - len(source.lstrip("rRuUbBfF"))
        quote = 3 if source[lead : lead + 3] in ('"""', "'''") else 1
        body = source[lead + quote : len(source) - quote]
        indent = len(value) - len(value.lstrip())
        parent_of = {child: n for n in ast.walk(parsed) for child in ast.iter_child_nodes(n)}
        if body != value or "\n" in value:
            self.site(
                constant, "unresolved", f"{value!r} names former root {old_root} through {local}; rewrite it by hand"
            )
            return 0, len(reads)
        edits: list[Edit] = []
        left = 0
        for read in reads:
            attribute = parent_of.get(read)
            if not (isinstance(attribute, ast.Attribute) and attribute.value is read):
                self.site(
                    constant,
                    "unresolved",
                    f"{value!r} uses {local} (former root {old_root}) as an object; name the module it needs",
                )
                left += 1
                continue
            target = self._read_target(constant, constant, local, old_root, anchor, attr=attribute.attr)
            if target is None:
                left += 1
                continue
            first = Lines(value.strip())
            at = start + lead + quote + indent + first.offset(1, attribute.col_offset)
            stop = start + lead + quote + indent + first.offset(1, attribute.end_col_offset)  # type: ignore[arg-type]
            edits.append(Edit(at, stop, target, "strings_rewritten"))
        self.out.edits += edits
        return len(edits), left

    @staticmethod
    def _binds_module(binder: Binder, module: str) -> bool:
        parent, _, last = module.rpartition(".")
        return (binder.how == "from" and binder.module == parent and binder.name == last) or (
            binder.how == "import-as" and binder.module == module
        )

    # -- chains: ``import threetears.evals.kernel`` then ``threetears.evals.kernel.X`` ------------------------------

    def _chains(self) -> None:
        """Chains read through ``threetears``, grouped by the scope whose binding of it they see: a module's, or a
        function's own ``import threetears.evals.kernel``."""
        groups: dict[Scope | None, list[ast.Name]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id == "threetears" and isinstance(node.ctx, ast.Load):
                groups.setdefault(self.governing(node, "threetears"), []).append(node)
        for plan in self.plans.values():
            if isinstance(plan.node, ast.Import):
                groups.setdefault(self.scope_of[plan.node], [])
        for scope, heads in groups.items():
            self._chains_in(scope, sorted(heads, key=lambda n: (n.lineno, n.col_offset)))

    def _chain(self, node: ast.AST) -> tuple[list[ast.AST], list[str]]:
        """The attribute chain read off ``node`` (a name), and its segments."""
        chain: list[ast.AST] = [node]
        segments = [node.id]  # type: ignore[attr-defined]
        while isinstance(self.parent.get(chain[-1]), ast.Attribute) and self.parent[chain[-1]].value is chain[-1]:  # type: ignore[attr-defined]
            chain.append(self.parent[chain[-1]])
            segments.append(chain[-1].attr)  # type: ignore[attr-defined]
        return chain, segments

    def _names_old(self, dotted: str) -> str | None:
        """Why ``dotted`` (a spelling under the package read in code) is the map's business, or ``None``."""
        resolved = resolve(self.table, dotted, context="attr", code=True)
        if resolved.action == "rewrite":
            target, _, name = resolved.new.rpartition(".")
            return f"{dotted} is now {resolved.new}: write `from {target} import {name}` and use {name}"
        if resolved.action != "none":
            return resolved.why
        roots = [root for root in self.table.roots if dotted == root or dotted.startswith(root + ".")]
        if roots:
            root = max(roots, key=len)
            return f"{dotted} reads former root {root}; its names went to {self.table.roots[root].new}"
        return None

    def _chains_in(self, scope: Scope | None, heads: list[ast.Name]) -> None:
        binders = self.binders.get(scope, {}).get("threetears", []) if scope is not None else []
        bare = [
            plan
            for plan in self.plans.values()
            if isinstance(plan.node, ast.Import) and self.scope_of[plan.node] is scope
        ]
        imported = bool(binders) and all(b.how == "import-top" for b in binders) and bool(bare)
        through: Counter[str] = Counter()
        left_through: Counter[str] = Counter()
        needed: list[str] = []
        for node in heads:
            chain, segments = self._chain(node)
            dotted = ".".join(segments)
            if not dotted.startswith(PACKAGE + "."):
                continue
            if not imported:
                why = self._names_old(dotted)
                if why is not None:
                    bound = "rebound here" if binders else "not bound by an import here"
                    self.site(node, "unresolved", f"{why} (threetears is {bound}, so the chain is left)")
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
        if imported:
            self._settle_bare_imports(bare, through, left_through, needed)

    # -- reads through an alias of the package: ``import threetears.evals as te`` then ``te.ops.X`` ----------------

    def _package_aliases(self) -> None:
        for scope, bound in self.binders.items():
            for local, binders in bound.items():
                paths = {self._alias_path(b) for b in binders} - {None}
                if not paths:
                    continue
                (path, *_) = sorted(paths)  # type: ignore[type-var]
                for node in ast.walk(self.tree):
                    if not (isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, ast.Load)):
                        continue
                    if self.governing(node, local) is not scope:
                        continue
                    _, segments = self._chain(node)
                    dotted = ".".join([path, *segments[1:]])  # type: ignore[list-item]
                    if not dotted.startswith(PACKAGE + "."):
                        continue
                    why = self._names_old(dotted)
                    if why is not None:
                        self.site(node, "unresolved", f"{'.'.join(segments)} reads through an alias of {path}: {why}")

    @staticmethod
    def _alias_path(binder: Binder) -> str | None:
        """The package, or an ancestor of it, that an ``import ... as`` or ``from ... import`` binds a local to."""
        if binder.how == "import-as":
            path = binder.module
        elif binder.how == "from" and not binder.module.startswith("."):
            path = f"{binder.module}.{binder.name}"
        else:
            return None
        return path if path == PACKAGE or PACKAGE.startswith(path + ".") else None

    def _prefix_length(self, dotted: str, resolved: Resolved) -> int:
        """How many segments of ``dotted`` the resolution replaced."""
        segments = dotted.split(".")
        for k in range(len(segments), 2, -1):
            prefix = ".".join(segments[:k])
            if prefix in self.table.names or prefix in self.table.modules:
                return k
        return len(segments)

    def _settle_bare_imports(
        self, plans: list[Plan], through: Counter[str], left_through: Counter[str], needed: list[str]
    ) -> None:
        bare = [(plan, entry) for plan in plans for entry in plan.entries if entry.asname is None]
        present = {entry.name for _, entry in bare}
        for plan, entry in bare:
            if entry.name in self.table.roots:
                if not through[entry.name]:
                    self.site(plan.node, "unresolved", f"import {entry.name} reads nothing from it; name what it needs")
                elif not left_through[entry.name]:
                    plan.removed.add(entry.name)
        anchor = min(plans, key=lambda p: (p.node.lineno, p.node.col_offset))
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
        # A rewritten entry is dropped only when the same binding comes BEFORE it in its block (a merge left both):
        # the first binding in source order is the one every later use depends on, so it is always kept.
        present: dict[tuple[int, str], set[tuple[str, str, str | None]]] = {}
        for node in imports:
            plan = self.plans.get(node)
            if plan is None:
                continue
            seen = present.setdefault(self._block(node), set())
            if not self._touched(plan):
                seen.update(self._key(plan, entry) for entry in plan.entries)
                continue
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
            (row, self._carry(text))
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

    def _carry(self, comment: str) -> str:
        """A comment carried onto a rewritten import, its spellings rewritten as any other comment's are (the
        comment's own edit sits inside the statement's, so it would be dropped)."""
        if not self.table.prose or PRAGMA in comment:
            return comment
        out = Outcome()
        rewrite_tokens(self.table, comment, 0, Lines(comment), set(), out, count="prose_tokens_rewritten")
        return apply_edits(comment, out.edits)[0]

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
    written: list[str] = field(default_factory=list)
    not_written: list[str] = field(default_factory=list)

    def exit_code(self) -> int:
        if self.refused or self.not_written:
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


CANDIDATE_SUFFIXES = PYTHON_SUFFIXES | MARKDOWN_SUFFIXES | TEXT_SUFFIXES


def walk(options: Options) -> Iterator[Path]:
    """Every candidate file under the root, and every linked directory (yielded, never descended)."""
    if not options.root.is_dir():
        yield options.root
        return
    for directory, dirs, files in os.walk(options.root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in PRUNED_DIRS)
        for name in list(dirs):
            if (Path(directory) / name).is_symlink():
                dirs.remove(name)
                yield Path(directory) / name
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in CANDIDATE_SUFFIXES:
                yield path


def names_package(data: bytes) -> bool:
    """Whether bytes that are not text in their encoding still spell the package (ASCII-compatible or UTF-16)."""
    return any(PACKAGE.encode(codec) in data for codec in ("ascii", "utf-16-le", "utf-16-be"))


def linked_files_naming_package(path: Path) -> list[Path]:
    """The candidate files a link (to a file or a directory) reaches that name the package, resolved; read through
    the link, never written."""
    if path.is_dir():
        found = []
        for directory, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in PRUNED_DIRS]
            for name in files:
                inner = Path(directory) / name
                if inner.suffix in CANDIDATE_SUFFIXES and inner.is_file() and names_package(inner.read_bytes()):
                    found.append(inner.resolve())
        return found
    return [path.resolve()] if path.is_file() and names_package(path.read_bytes()) else []


def encoding_of(path: Path, data: bytes) -> str:
    """The codec a file is read and written back in: a Python file's declared one (PEP 263), else UTF-8; a
    byte-order mark gives ``utf-8-sig``, which drops it on reading and writes it back."""
    if path.suffix in PYTHON_SUFFIXES:
        return tokenize.detect_encoding(io.BytesIO(data).readline)[0]
    return "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"


def write_atomically(path: Path, data: bytes) -> None:
    """Replace ``path`` whole: a temporary file in its directory, then a rename, keeping its mode. A file the user
    cannot write is refused, never replaced."""
    if not os.access(path, os.W_OK):
        raise PermissionError(f"{path.name} is not writable")
    mode = stat.S_IMODE(path.stat().st_mode)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".reroot-tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


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


def process(rmap: RerootMap, options: Options) -> tuple[Report, dict[Path, tuple[str, str]]]:
    report = Report(epochs=rmap.upto(options.to))
    targets = {"home": options.home, "home_for_left": options.home_for_left, "prose": options.prose}
    tables = [Table.of(rmap.lines, epoch.sha, **targets) for epoch in report.epochs]
    protected = {options.map_path.resolve(), Path(__file__).resolve()}
    inside = options.root.resolve() if options.root.is_dir() else options.root.parent.resolve()
    texts: dict[Path, str] = {}
    encodings: dict[Path, str] = {}
    paths = list(walk(options))
    # A link to a file this run reaches at its own path is covered there; any other link is listed, never followed.
    covered = {path.resolve() for path in paths if not path.is_symlink() and path.resolve().is_relative_to(inside)}
    for path in paths:
        relative = path.relative_to(options.root) if path != options.root else Path(path.name)
        if excluded(relative, options, protected, path):
            if path.is_file() and not path.is_symlink() and names_package(path.read_bytes()):
                report.skipped.append(f"{relative.as_posix()}: excluded")
                report.counts["files_skipped"] += 1
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(inside):
            try:
                holds = bool(set(linked_files_naming_package(path)) - covered)
            except OSError as error:
                holds, why = True, f"; it cannot be read ({error.strerror or error})"
            else:
                why = ""
            if holds:
                target = os.readlink(path) if path.is_symlink() else str(path.resolve())
                report.sites.append(
                    Site(
                        relative.as_posix(),
                        1,
                        "linked",
                        f"a link to {target} that names {PACKAGE}{why}; the script never writes through a link or "
                        "outside its root: run it on the tree the link points into, or edit the target by hand",
                    )
                )
                report.counts["files_skipped"] += 1
            continue
        data = path.read_bytes()
        if b"threetears" not in data and not names_package(data):
            continue
        try:
            encoding = encoding_of(path, data)
            text = data.decode(encoding)
        except (SyntaxError, LookupError, UnicodeDecodeError) as error:
            if names_package(data):
                report.sites.append(
                    Site(
                        relative.as_posix(),
                        1,
                        "undecodable",
                        f"names {PACKAGE} but is not text in its declared encoding ({error}); the script reads "
                        "UTF-8 (a Python file: its PEP 263 coding line): convert it, or edit its spellings by hand",
                    )
                )
                report.counts["files_skipped"] += 1
            continue
        if "threetears" not in text:
            continue
        encodings[path] = encoding
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
    written: dict[Path, tuple[str, str]] = {}
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
            written[path] = (current, encodings[path])
    report.counts["files_changed"] = len(report.changed)
    for kind in ("removed", "unresolved", "ambiguous", "code-string", "renamed", "collision", "linked", "undecodable"):
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
    ("collision", "renamed names colliding with a binding"),
    ("linked", "links naming the package (never followed)"),
    ("undecodable", "files not text in their encoding"),
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
                "written": report.written,
                "not_written": report.not_written,
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
    if report.not_written:
        out.append(
            "a write failed; each file is written whole or not at all. Fix the cause and run it again to finish:"
        )
        out += [f"  written: {path}" for path in report.written]
        out += [f"  not written: {line}" for line in report.not_written]
    elif options.dry_run:
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
            "upgrading the package: fetch this file and the map at the commit you pin with `git show "
            "<pin>:packages/evals/scripts/reroot.py` and `git show <pin>:packages/evals/api/reroot-map.txt`, then "
            "`python3.14 reroot.py --map reroot-map.txt --dry-run .` and the same without --dry-run. The map "
            "fetched at your pin holds exactly the epochs at or before it, so no --to is needed. Exit codes: 0 "
            "nothing to do, 3 rewrote, 2 edit by hand what it lists, 1 refused (nothing written) or a write failed "
            "(the report lists what was and was not written; run it again to finish)."
        ),
    )
    parser.add_argument("root", type=Path, help="the tree (or one file) to rewrite")
    parser.add_argument("--map", dest="map_path", type=Path, required=True, help="reroot-map.txt")
    parser.add_argument(
        "--to",
        metavar="EPOCH_SHA",
        help="apply the map's epochs up to and including this one: an epoch SHA from the map's header, or a unique "
        "prefix of at least 4 characters; not a release pin. Default: every epoch in the map",
    )
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
        for path, (text, encoding) in written.items():
            relative = (path.relative_to(options.root) if path != options.root else Path(path.name)).as_posix()
            try:
                write_atomically(path, text.encode(encoding))
            except (OSError, UnicodeEncodeError) as error:
                report.not_written.append(f"{relative}: {error}")
            else:
                report.written.append(relative)
    print(render(report, options))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
