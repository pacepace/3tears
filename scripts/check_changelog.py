"""Refuse a change a consumer must act on that the CHANGELOG's ``## Unreleased`` section does not name.

The hub and reports read only the CHANGELOG when they re-pin to a 3tears line, so a change they must act on and the
CHANGELOG leaves out is one they miss. It happened: five merges into the 0.95.0 line changed agent-acl's schema, a
public error, and the retry helper every caller shares, and the Unreleased section named none of it (Critic R-6,
2026-11-26).

**What it holds.** Over a range of commits ``<base>..<head>``, three kinds of change, each read from git:

- **a new consumer-facing name**: a ``_public_api_consumers.txt`` entry at ``<head>`` (a name a caller outside this
  repo calls) whose name its module did not export at ``<base>``;
- **a changed consumer-facing name**: such an entry whose top-level definition (the ``def``, ``class`` or assignment,
  docstrings left out) differs between ``<base>`` and ``<head>``. An entry on a package ``__init__``'s re-export
  is not compared: the defining module's own entry is (the ledger lists both, as for ``get_registered_capabilities``);
- **a new migration**: a ``migrations/vNNN_*.py`` module at ``<head>`` that ``<base>`` lacks. A consumer whose own
  migrations create the same tables (the hub) repeats its steps.

Each must be named in the Unreleased section at ``<head>``: the name as a word (``WHOLE_RELOADS.gated`` names
``WHOLE_RELOADS``); a migration by its module name or as ``<package> vNNN`` (``agent-acl v007``, a span
``agent-acl v007-v011``, or a ``| v007 |`` row of a table headed ``| agent-acl |``).

**Unless a commit says why.** A commit in the range whose message carries a line
``Changelog-None: <name or migration module name>: <why a consumer has nothing to do>`` (a reason of at least 20
characters) excuses that one change.

**From where.** The range starts no earlier than :data:`CHECKED_FROM`, the commit that brought the CHANGELOG up to
date with what R-6 named. What landed before it was audited by hand then, not checked here: a release PR's range
(``develop..`` the line) covers the whole line's history, most of which predates this check. Once a release tag
contains that commit, every range's base is past it and it holds nothing.

**What it cannot see**, recorded rather than papered over: a change to a name a caller in this repo also calls
(``retry_serialization_conflicts``), since the ledger holds only names with no caller here; a change made inside a
private helper a ledger name calls; and whether the words that name a change say what to do.

Usage (CI runs it over a pull request's range; run it before merging into a line, the same way)::

    uv run --frozen python scripts/check_changelog.py <base>..<head>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from threetears.enforcement.release import BaselineUnreadableError, surface_at
from threetears.enforcement.release.versions import run_git

#: the commit whose CHANGELOG named every consumer action R-6 listed: a range never reaches back past it.
CHECKED_FROM = "4f21f7fed098352f2cb0005c7ec3fb2fbe0d684f"

LEDGER = "tests/enforcement/_public_api_consumers.txt"
CHANGELOG = "CHANGELOG.md"

_MIGRATION = re.compile(r"^(?P<package>packages/(?:agent/)?[^/]+)/src/.*/migrations/(?P<version>v\d+)_[^/]*\.py$")
_SOURCE_ROOT = re.compile(r"^(packages/(?:agent/)?[^/]+)/pyproject\.toml$")
_EXCUSE = re.compile(r"^Changelog-None:\s*(?P<what>[^:\s]+)\s*:\s*(?P<why>.+?)\s*$", re.MULTILINE)
_MIN_REASON = 20


class RangeUnreadableError(Exception):
    """the range, or a file in it, cannot be read in this clone; nothing was compared."""


@dataclass(frozen=True)
class Change:
    """one change a consumer must act on.

    :param name: what names it: a public name, or a migration's module name (``v007_namespaces_keyed_by_id``)
    :ptype name: str
    :param what: the sentence a finding shows
    :ptype what: str
    :param package: a migration's package label (``agent-acl``); empty for a name
    :ptype package: str
    :param version: a migration's number (7 for ``v007``); 0 for a name
    :ptype version: int
    """

    name: str
    what: str
    package: str = ""
    version: int = 0


def _git_text(root: Path, *args: str) -> str:
    """one git read that must succeed.

    :param root: repository root
    :ptype root: Path
    :param args: git arguments
    :ptype args: str
    :return: its standard output
    :rtype: str
    :raises RangeUnreadableError: if git fails
    """
    result = run_git(root, *args)
    if not result.ok:
        raise RangeUnreadableError(f"`git {' '.join(args)}` failed: {result.stderr or 'no output'}")
    return result.stdout.decode("utf-8", errors="replace")


def _show(root: Path, ref: str, path: str) -> str | None:
    """a file's text at *ref*, or ``None`` when *ref* has no such file.

    :param root: repository root
    :ptype root: Path
    :param ref: commit
    :ptype ref: str
    :param path: repo-relative path
    :ptype path: str
    :return: the text, or ``None``
    :rtype: str | None
    """
    result = run_git(root, "show", f"{ref}:{path}")
    return result.stdout.decode("utf-8", errors="replace") if result.ok else None


def unreleased_section(changelog: str) -> str:
    """the text under ``## Unreleased``, up to the next ``## `` heading.

    :param changelog: the CHANGELOG's text
    :ptype changelog: str
    :return: the section's text; empty when there is no such heading
    :rtype: str
    """
    matched = re.search(r"^## Unreleased[ \t]*\n(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL)
    return matched[1] if matched else ""


def ledger_entries(text: str) -> list[tuple[str, str]]:
    """the ledger's ``(module path, name)`` entries, as the caller gate reads them.

    :param text: ``_public_api_consumers.txt``
    :ptype text: str
    :return: the entries, in file order
    :rtype: list[tuple[str, str]]
    """
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        module, _, name = line.partition("#")[0].strip().rpartition(":")
        entries.append((module, name))
    return entries


def _without_docstrings(node: ast.AST) -> ast.AST:
    """*node* with every docstring dropped, so a reworded docstring is no change.

    :param node: a parsed definition (changed in place)
    :ptype node: ast.AST
    :return: the same node
    :rtype: ast.AST
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
            body = inner.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    inner.body = body[1:] or [ast.Pass()]
    return node


def definition(source: str, name: str) -> str | None:
    """the top-level definition of *name* in *source*, as a comparable dump, docstrings left out.

    :param source: module source
    :ptype source: str
    :param name: a top-level name
    :ptype name: str
    :return: the dump, or ``None`` when the module does not parse or does not define it
    :rtype: str | None
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        defines = (
            (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name == name)
            or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name)
            or (isinstance(node, ast.TypeAlias) and node.name.id == name)
        )
        if defines:
            return ast.dump(_without_docstrings(node))
    return None


def package_label(package: str) -> str:
    """how the CHANGELOG names a package: ``packages/agent/acl`` is ``agent-acl``, ``packages/core`` is ``core``.

    :param package: repo-relative package directory
    :ptype package: str
    :return: the label
    :rtype: str
    """
    return "-".join(PurePosixPath(package).parts[1:])


def _source_roots(root: Path, refs: tuple[str, ...]) -> list[str]:
    """every workspace member's ``src`` at any of *refs*.

    :param root: repository root
    :ptype root: Path
    :param refs: commits
    :ptype refs: tuple[str, ...]
    :return: repo-relative source roots
    :rtype: list[str]
    """
    found: set[str] = set()
    for ref in refs:
        for path in _git_text(root, "ls-tree", "-r", "--name-only", ref, "--", "packages").splitlines():
            matched = _SOURCE_ROOT.match(path)
            if matched:
                found.add(f"{matched[1]}/src")
    return sorted(found)


def _surfaces(root: Path, ref: str, source_roots: list[str]) -> dict[str, set[str]]:
    """module path -> exported names at *ref*, across every source root.

    :param root: repository root
    :ptype root: Path
    :param ref: commit
    :ptype ref: str
    :param source_roots: repo-relative source roots
    :ptype source_roots: list[str]
    :return: the surfaces
    :rtype: dict[str, set[str]]
    :raises RangeUnreadableError: if *ref*'s tree is not in this clone
    """
    modules: dict[str, set[str]] = {}
    for source_root in source_roots:
        try:
            modules.update(surface_at(root, ref, source_root).modules)
        except BaselineUnreadableError as exc:
            raise RangeUnreadableError(str(exc)) from None
    return modules


def consumer_changes(root: Path, base: str, head: str) -> list[Change]:
    """every change between *base* and *head* a consumer must act on.

    :param root: repository root
    :ptype root: Path
    :param base: the range's base commit
    :ptype base: str
    :param head: the range's head commit
    :ptype head: str
    :return: the changes, ordered
    :rtype: list[Change]
    :raises RangeUnreadableError: if either side cannot be read
    """
    changes: list[Change] = []
    ledger = _show(root, head, LEDGER)
    if ledger is not None:
        roots = _source_roots(root, (base, head))
        before = _surfaces(root, base, roots)
        now = _surfaces(root, head, roots)
        sources: dict[tuple[str, str], str | None] = {}
        for module, name in ledger_entries(ledger):
            if name not in now.get(module, set()):
                continue  # the caller gate holds a stale entry; nothing here to name
            if name not in before.get(module, set()):
                changes.append(Change(name, f"{module}:{name} is new, and a consumer outside this repo calls it"))
                continue
            if PurePosixPath(module).name == "__init__.py":
                continue
            for ref in (base, head):
                if (ref, module) not in sources:
                    sources[(ref, module)] = _show(root, ref, module)
            old, new = sources[(base, module)], sources[(head, module)]
            if old is not None and new is not None and definition(old, name) != definition(new, name):
                changes.append(Change(name, f"{module}:{name} changed, and a consumer outside this repo calls it"))
    added = _git_text(root, "diff", "--name-only", "--diff-filter=A", base, head, "--", "packages").splitlines()
    for path in sorted(added):
        matched = _MIGRATION.match(path)
        if matched:
            label = package_label(matched["package"])
            changes.append(
                Change(
                    PurePosixPath(path).stem,
                    f"{path} is a new migration of {label}",
                    package=label,
                    version=int(matched["version"][1:]),
                )
            )
    return changes


def excuses(root: Path, base: str, head: str) -> dict[str, str]:
    """``Changelog-None:`` lines in the range's commit messages: what each excuses, and why.

    :param root: repository root
    :ptype root: Path
    :param base: the range's base commit
    :ptype base: str
    :param head: the range's head commit
    :ptype head: str
    :return: excused name or migration file name -> the reason given
    :rtype: dict[str, str]
    """
    messages = _git_text(root, "log", "--format=%B%x00", f"{base}..{head}")
    return {matched["what"]: matched["why"] for matched in _EXCUSE.finditer(messages)}


def _word(word: str) -> str:
    """a pattern matching *word* as a word of its own (``-`` counts as part of a word, as in ``agent-acl``).

    :param word: literal text
    :ptype word: str
    :return: the pattern
    :rtype: str
    """
    return rf"(?<![\w-]){re.escape(word)}(?![\w-])"


def _migration_named(section: str, change: Change) -> bool:
    """whether *section* names a migration: its module name, ``<package> vNNN`` (``agent-acl v007``,
    ``agent_acl's v007``, or a span ``agent-acl v007-v011`` holding it), or a table row starting ``| vNNN |`` under a
    header whose first cell is the package.

    Never ``v007`` and the package anywhere in one paragraph: "coordination v007, mcp v003, agent_acl v006" would
    name agent-acl's v007.

    :param section: the Unreleased section
    :ptype section: str
    :param change: the migration
    :ptype change: Change
    :return: whether it is named
    :rtype: bool
    """
    if re.search(_word(change.name), section):
        return True
    package = "|".join(re.escape(spelling) for spelling in {change.package, change.package.replace("-", "_")})
    spans = rf"(?<![\w-])(?:{package})(?:'s)?\s+v(\d+)(?:\s*(?:-|\u2013|\.\.)\s*v(\d+))?(?![\w-])"
    for matched in re.finditer(spans, section):
        first = int(matched[1])
        last = int(matched[2]) if matched[2] else first
        if first <= change.version <= last:
            return True
    for table in re.findall(r"(?:^\|.*\n?)+", section, re.MULTILINE):
        rows = [[cell.strip().strip("`") for cell in row.strip().strip("|").split("|")] for row in table.splitlines()]
        if rows and rows[0][0] in {change.package, change.package.replace("-", "_")}:
            if any(re.fullmatch(rf"v0*{change.version}", row[0]) for row in rows[1:]):
                return True
    return False


def named(section: str, change: Change) -> bool:
    """whether the Unreleased section names *change*.

    :param section: the Unreleased section
    :ptype section: str
    :param change: the change
    :ptype change: Change
    :return: whether it is named
    :rtype: bool
    """
    if change.package:
        return _migration_named(section, change)
    return re.search(_word(change.name), section) is not None


def effective_base(root: Path, base: str, head: str, checked_from: str) -> str:
    """*base*, or *checked_from* when that commit lies between *base* and *head*.

    :param root: repository root
    :ptype root: Path
    :param base: the range's base
    :ptype base: str
    :param head: the range's head
    :ptype head: str
    :param checked_from: the commit before which nothing is checked
    :ptype checked_from: str
    :return: the commit the check compares *head* against
    :rtype: str
    """
    between = (
        run_git(root, "merge-base", "--is-ancestor", base, checked_from).ok
        and run_git(root, "merge-base", "--is-ancestor", checked_from, head).ok
    )
    return checked_from if between else base


def findings(root: Path, base: str, head: str, checked_from: str = CHECKED_FROM) -> list[str]:
    """every change in ``base..head`` a consumer must act on that the Unreleased section does not name.

    :param root: repository root
    :ptype root: Path
    :param base: the range's base
    :ptype base: str
    :param head: the range's head
    :ptype head: str
    :param checked_from: the commit before which nothing is checked
    :ptype checked_from: str
    :return: one line per unnamed change, and per excuse too short to be a reason
    :rtype: list[str]
    :raises RangeUnreadableError: if the range cannot be read
    """
    base = effective_base(root, _git_text(root, "rev-parse", base).strip(), head, checked_from)
    section = unreleased_section(_show(root, head, CHANGELOG) or "")
    given = excuses(root, base, head)
    problems: list[str] = []
    for change in consumer_changes(root, base, head):
        reason = given.get(change.name)
        if reason is not None:
            if len(reason) < _MIN_REASON:
                problems.append(f"  {change.name}: the Changelog-None reason {reason!r} says too little")
            continue
        if not named(section, change):
            problems.append(f"  {change.what}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """run the check over one range; exit 1 with what is missing.

    :param argv: arguments (default: ``sys.argv[1:]``)
    :ptype argv: list[str] | None
    :return: exit status
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("range", help="<base>..<head>")
    parser.add_argument("--checked-from", default=CHECKED_FROM, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base, dots, head = args.range.partition("..")
    if not dots or not base or not head:
        parser.error("the range is <base>..<head>")
    root = Path(_git_text(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    try:
        problems = findings(root, base, head, args.checked_from)
    except RangeUnreadableError as exc:
        print(f"check_changelog: {exc}. Nothing was compared (a CI checkout needs fetch-depth: 0).", file=sys.stderr)
        return 1
    if problems:
        print(
            f"{len(problems)} change(s) a consumer must act on that {CHANGELOG}'s ## Unreleased section does not "
            f"name. Add an entry saying what a consumer does, or, when there is nothing to do, say why in a commit "
            f"of the range: `Changelog-None: <name>: <why>`.\n" + "\n".join(problems),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
