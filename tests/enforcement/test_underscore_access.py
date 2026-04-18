"""enforcement test for leading-underscore access conventions.

the ``_name`` prefix in python is a **stability contract**, not merely
a module-private scope marker. it declares: "this is implementation
detail; i reserve the right to change it; do not bind to it." that
contract must be honored across every axis:

- shape A: cross-module private import
  ``from pkg.sub.module import _helper`` where the importer is not in
  the same top-level package as the defining module.
- shape B: cross-class protected attribute access
  ``obj._attr`` where ``obj`` is not ``self`` / ``cls``. delegated to
  ``ruff check --select SLF001`` because ruff already does the scope
  analysis correctly and reimplementing it is a tar pit.
- shape C: missing ``__all__`` on a module with public exports
  any non-trivial ``src/`` module that defines at least one public
  top-level name without an explicit ``__all__``.
- shape D: subclass shadows a base-class private name
  ``class Sub(Base): _attr = ...`` where ``Base`` declares ``_attr``
  as implementation detail. the ``_`` says "do not touch" -- a
  subclass overriding it is binding to the base's internal contract.
  covers both attribute redefinitions and method overrides.
- shape E: ``__all__`` entry is a private name
  ``__all__ = ["_helper"]`` -- contradictory: the underscore says
  "private" and the ``__all__`` entry says "public." the two
  declarations disagree about the name's contract.

run mode is controlled by the ``UNDERSCORE_AUDIT_MODE`` env var:

- ``report`` -- prints violations, exits 0. used during cleanup.
- ``strict`` (default, post-cleanup) -- fails on any unexplained violation.

exemptions live in ``tests/enforcement/_underscore_exemptions.txt``.
every entry requires a preceding ``# rationale: <reason>`` line. the
parser enforces this contract.

:see: docs/hygiene-task-01-underscore-audit.md -- task shard.

CANONICAL SOURCE: this file is maintained in the ``14-eng-ai-bot``
repo under ``tests/enforcement/test_underscore_access.py`` and
vendored verbatim into the sibling repos (``3tears``,
``14-eng-ai-bot-agents``, ``14-eng-ai-bot-agent-admin``). updates
MUST be applied to every copy in the same commit; there is no
shared-package abstraction yet. if four-way sync becomes painful,
extract to a shared dev-dep package; until then keep the copies
identical.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


# ------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------

MODE_REPORT = "report"
MODE_STRICT = "strict"
# default is ``strict`` post-cleanup (phase-4 of the hygiene-task-01
# shard landed). any new violation fails CI. see
# docs/hygiene-task-01-underscore-audit.md for the shape catalogue
# and ``_underscore_exemptions.txt`` for the narrow allow-list of
# rationale-backed carve-outs.
_DEFAULT_MODE = MODE_STRICT
_TIMEOUT_SECONDS = 60


# files / directories never walked for source-scanning passes
_SKIP_DIRS = frozenset({
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".git",
    "dist",
    "build",
    ".eggs",
    "_build",
    # cookiecutter template directories: the path literally contains
    # '{{cookiecutter.x}}' and the files inside are jinja2 templates,
    # not valid python source.
})


def _is_cookiecutter_template(path: Path) -> bool:
    """detect a path that lives inside a cookiecutter template tree.

    cookiecutter directories have components containing '{{' / '}}'
    which are jinja2 placeholders, not valid python; walking into
    them produces false-positive syntax errors and spurious hits.

    :param path: path to classify
    :ptype path: Path
    :return: whether any path component contains jinja2 template markers
    :rtype: bool
    """
    result = any(("{{" in part or "}}" in part) for part in path.parts)
    return result

# file-level markers that opt a module out of shape-C (missing __all__)
# detection. these are modules that are intentionally "no public surface"
# (pure script entrypoints, runtime __main__ shims, test conftests).
_SHAPE_C_SKIP_BASENAMES = frozenset({
    "conftest.py",
    "__main__.py",
    "_version.py",
})


# ------------------------------------------------------------------------
# public dataclasses (exported for meta-tests)
# ------------------------------------------------------------------------


@dataclass(frozen=True)
class ExemptionEntry:
    """one line of the exemptions file, parsed.

    :ivar file: source-relative path from the entry line (as written)
    :ivar line: line number within that file
    :ivar symbol: underscore-prefixed symbol being exempted
    :ivar rationale: human-readable justification from the preceding line
    """

    file: str
    line: int
    symbol: str
    rationale: str


@dataclass(frozen=True)
class Violation:
    """one detected violation, uniformly described across shapes."""

    shape: str
    file: Path
    line: int
    symbol: str
    detail: str

    def format(self, repo_root: Path) -> str:
        """format a violation as ``{rel}:{line}:{symbol}  -- {detail}``.

        :param repo_root: repo-root anchor for relative-path rendering
        :ptype repo_root: Path
        :return: single-line human-readable string
        :rtype: str
        """
        try:
            rel = self.file.relative_to(repo_root)
        except ValueError:
            rel = self.file
        result = f"[{self.shape}] {rel}:{self.line}:{self.symbol}  -- {self.detail}"
        return result


# ------------------------------------------------------------------------
# public helpers (exported for meta-tests)
# ------------------------------------------------------------------------


def find_src_roots(repo_root: Path) -> tuple[Path, ...]:
    """locate all ``src/`` directories under ``repo_root``.

    handles two layouts:

    - single-project: one ``src/`` at the root.
    - monorepo: ``packages/*/src/`` (threetears style).

    :param repo_root: repo root path to search
    :ptype repo_root: Path
    :return: tuple of discovered src roots, sorted for stable ordering
    :rtype: tuple[Path, ...]
    """
    candidates: set[Path] = set()
    for candidate in repo_root.rglob("src"):
        if not candidate.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in candidate.parts):
            continue
        if _is_cookiecutter_template(candidate):
            continue
        candidates.add(candidate)
    result = tuple(sorted(candidates))
    return result


def package_id(path: Path, src_roots: Iterable[Path]) -> tuple[str, ...]:
    """compute package identity for a source file.

    identity = ``(src_root_str, top_level_package_name)``. two files
    share a package iff they share an identity. returns ``()`` when the
    file sits outside every provided src root.

    :param path: source file path
    :ptype path: Path
    :param src_roots: candidate src-root directories
    :ptype src_roots: Iterable[Path]
    :return: identity tuple, or ``()`` if outside any src root
    :rtype: tuple[str, ...]
    """
    resolved = path.resolve()
    result: tuple[str, ...] = ()
    for root in src_roots:
        root_resolved = root.resolve()
        try:
            rel = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        parts = rel.parts
        if not parts:
            continue
        result = (str(root_resolved), parts[0])
        break
    return result


def same_package(a: Path, b: Path, src_roots: Iterable[Path]) -> bool:
    """true iff two source paths share a package identity under any src root.

    :param a: first source file
    :ptype a: Path
    :param b: second source file
    :ptype b: Path
    :param src_roots: candidate src-root directories
    :ptype src_roots: Iterable[Path]
    :return: whether both files share a (non-empty) package identity
    :rtype: bool
    """
    roots = tuple(src_roots)
    id_a = package_id(a, roots)
    id_b = package_id(b, roots)
    result = id_a != () and id_a == id_b
    return result


_ENTRY_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>[^:]+):(?P<symbol>[A-Za-z_][A-Za-z_0-9]*)\s*$"
)


def parse_exemptions(path: Path) -> list[ExemptionEntry]:
    """parse an exemptions file into a list of ExemptionEntry records.

    every entry line must be preceded by a line matching
    ``# rationale: <non-empty reason>``. blank lines are skipped.
    non-rationale comment lines (``# ...`` without the ``rationale:``
    tag) are ignored.

    :param path: path to exemptions file
    :ptype path: Path
    :return: parsed entries in file order
    :rtype: list[ExemptionEntry]
    :raises ValueError: malformed file (missing rationale, bad entry shape,
        non-integer line number)
    """
    entries: list[ExemptionEntry] = []
    if not path.exists():
        return entries
    text = path.read_text(encoding="utf-8")
    pending_rationale: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped.lower().startswith("rationale:"):
                rationale = stripped[len("rationale:"):].strip()
                if not rationale:
                    raise ValueError(
                        f"{path}:{lineno}: '# rationale:' must be followed by a "
                        f"non-empty reason"
                    )
                pending_rationale = rationale
            # non-rationale comment lines are ignored (no state change)
            continue
        match = _ENTRY_RE.match(line)
        if match is None:
            raise ValueError(
                f"{path}:{lineno}: malformed entry; expected 'file:line:symbol' "
                f"triple, got {line!r}"
            )
        if pending_rationale is None:
            raise ValueError(
                f"{path}:{lineno}: entry has no preceding '# rationale: ...' line"
            )
        line_field = match.group("line")
        try:
            line_int = int(line_field)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{lineno}: line number must be an integer, got "
                f"{line_field!r}"
            ) from exc
        entry = ExemptionEntry(
            file=match.group("file"),
            line=line_int,
            symbol=match.group("symbol"),
            rationale=pending_rationale,
        )
        entries.append(entry)
        pending_rationale = None
    return entries


# ------------------------------------------------------------------------
# shape walkers
# ------------------------------------------------------------------------


def _iter_python_files(root: Path) -> Iterable[Path]:
    """yield ``.py`` files under root, skipping cache / venv directories.

    :param root: directory to walk
    :ptype root: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    for path in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if _is_cookiecutter_template(path):
            continue
        yield path


def _parse_file(path: Path) -> ast.Module | None:
    """parse a python file, returning None on syntax error.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module AST or None
    :rtype: ast.Module | None
    """
    result: ast.Module | None
    try:
        source = path.read_text(encoding="utf-8")
        result = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        result = None
    return result


def _is_private_name(name: str) -> bool:
    """true iff name is a single-leading-underscore identifier.

    excludes dunders (``__foo__``), name-mangled (``__foo``), trailing
    underscore (``id_``), and the throwaway ``_``.

    :param name: identifier to classify
    :ptype name: str
    :return: whether name is a "private, module-internal" identifier
    :rtype: bool
    """
    result = (
        name.startswith("_")
        and not name.startswith("__")
        and name != "_"
        and not name.endswith("_")
    )
    return result


def _resolve_module_to_file(
    module: str, src_roots: Iterable[Path]
) -> Path | None:
    """find the source file that defines a fully-qualified module name.

    :param module: dotted module path, e.g. ``threetears.core.backends.x``
    :ptype module: str
    :param src_roots: candidate src roots
    :ptype src_roots: Iterable[Path]
    :return: path to the ``.py`` file, or None if unresolved
    :rtype: Path | None
    """
    parts = module.split(".")
    result: Path | None = None
    for root in src_roots:
        file_candidate = root.joinpath(*parts).with_suffix(".py")
        if file_candidate.is_file():
            result = file_candidate
            break
        pkg_candidate = root.joinpath(*parts, "__init__.py")
        if pkg_candidate.is_file():
            result = pkg_candidate
            break
    return result


def _shape_a_violations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every source file for cross-module private imports.

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (for error formatting)
    :ptype repo_root: Path
    :return: list of violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for importer in _iter_python_files(root):
            tree = _parse_file(importer)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module
                if module is None:
                    continue
                if node.level and node.level > 0:
                    # relative imports are always intra-package; skip.
                    continue
                private_names = [
                    alias for alias in node.names
                    if _is_private_name(alias.name)
                ]
                if not private_names:
                    continue
                defining_file = _resolve_module_to_file(module, src_roots)
                if defining_file is None:
                    # cannot resolve (probably third-party) -- skip.
                    continue
                if same_package(importer, defining_file, src_roots):
                    continue
                for alias in private_names:
                    detail = (
                        f"imports private name '{alias.name}' from "
                        f"'{module}' across package boundary"
                    )
                    violations.append(
                        Violation(
                            shape="A",
                            file=importer,
                            line=node.lineno,
                            symbol=alias.name,
                            detail=detail,
                        )
                    )
    _ = repo_root  # retained for potential future context inclusion
    return violations


def _shape_c_violations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every source file for missing ``__all__`` with public surface.

    a module is flagged iff it:

    - is not ``__init__.py`` with zero top-level names (empty package)
    - is not in the _SHAPE_C_SKIP_BASENAMES set (conftest, __main__, etc.)
    - defines at least one public top-level name (class, func, assign)
    - does not define ``__all__``

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (for error formatting)
    :ptype repo_root: Path
    :return: list of violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for module_path in _iter_python_files(root):
            if module_path.name in _SHAPE_C_SKIP_BASENAMES:
                continue
            tree = _parse_file(module_path)
            if tree is None:
                continue
            has_all = False
            public_names: list[tuple[str, int]] = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "__all__":
                                has_all = True
                            elif not target.id.startswith("_"):
                                public_names.append((target.id, node.lineno))
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                    if isinstance(target, ast.Name):
                        if target.id == "__all__":
                            has_all = True
                        elif not target.id.startswith("_"):
                            public_names.append((target.id, node.lineno))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        public_names.append((node.name, node.lineno))
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        public_names.append((node.name, node.lineno))
            if has_all:
                continue
            if not public_names:
                continue
            first_name, first_line = public_names[0]
            detail = (
                f"module defines {len(public_names)} public name(s) but no "
                f"__all__; first is '{first_name}'"
            )
            violations.append(
                Violation(
                    shape="C",
                    file=module_path,
                    line=first_line,
                    symbol="__all__",
                    detail=detail,
                )
            )
    _ = repo_root
    return violations


def _collect_class_private_attrs(
    src_roots: tuple[Path, ...],
) -> dict[str, dict[str, tuple[Path, int]]]:
    """build a map of ``class_name`` -> ``{_private_name -> (file, line)}``.

    keyed by bare class name (last textual segment). this is lossy for
    aliased imports (``from x import Base as B``) but captures the
    overwhelming common case of "subclass shares the base's visible
    name." two distinct ``Base`` classes in different modules merge
    into one entry here -- downstream shape-D detection may over-flag
    for that collision pattern, which is acceptable: class-name
    collisions across a codebase are themselves a smell worth
    surfacing.

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :return: map ``class_name`` -> ``{_private_name -> (file, line)}``
    :rtype: dict[str, dict[str, tuple[Path, int]]]
    """
    result: dict[str, dict[str, tuple[Path, int]]] = {}
    for root in src_roots:
        for file in _iter_python_files(root):
            tree = _parse_file(file)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                entries = result.setdefault(node.name, {})
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _is_private_name(item.name):
                            entries[item.name] = (file, item.lineno)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and _is_private_name(
                                target.id
                            ):
                                entries[target.id] = (file, item.lineno)
                    elif isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name) and _is_private_name(
                            item.target.id
                        ):
                            entries[item.target.id] = (file, item.lineno)
    return result


def _shape_d_violations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every class for subclass shadowing of a base class private name.

    a class ``Sub`` with base ``Base`` violates shape D iff it declares
    any class-body ``_name`` (attribute assignment or method) that
    ``Base`` also declares. the underscore on ``Base._name`` is a
    stability contract; ``Sub`` redefining that name binds to the
    contract and breaks when the base refactors.

    resolution is textual -- base class ``node.bases`` are treated as
    last-segment names (``pkg.mod.Base`` -> ``"Base"``). indirect
    inheritance (generics, metaclasses) and aliased imports are not
    resolved; those false negatives are accepted because the common
    case (direct subclass in the same repo) is what causes this shard
    to exist.

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for potential future context)
    :ptype repo_root: Path
    :return: list of violations
    :rtype: list[Violation]
    """
    class_privates = _collect_class_private_attrs(src_roots)
    violations: list[Violation] = []
    for root in src_roots:
        for file in _iter_python_files(root):
            tree = _parse_file(file)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names: list[str] = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                if not base_names:
                    continue
                sub_privates: list[tuple[str, int]] = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _is_private_name(item.name):
                            sub_privates.append((item.name, item.lineno))
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and _is_private_name(
                                target.id
                            ):
                                sub_privates.append((target.id, item.lineno))
                    elif isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name) and _is_private_name(
                            item.target.id
                        ):
                            sub_privates.append((item.target.id, item.lineno))
                for base_name in base_names:
                    if base_name == node.name:
                        continue
                    base_entries = class_privates.get(base_name, {})
                    for name, line in sub_privates:
                        if name not in base_entries:
                            continue
                        defining_file, defining_line = base_entries[name]
                        if defining_file == file and defining_line == line:
                            continue
                        try:
                            rel_def = defining_file.relative_to(repo_root)
                        except ValueError:
                            rel_def = defining_file
                        detail = (
                            f"class '{node.name}' shadows private '{name}' "
                            f"declared on base class '{base_name}' at "
                            f"{rel_def}:{defining_line}; underscore names are "
                            f"implementation detail of the defining class "
                            f"and must not be overridden"
                        )
                        violations.append(
                            Violation(
                                shape="D",
                                file=file,
                                line=line,
                                symbol=name,
                                detail=detail,
                            )
                        )
    return violations


def _shape_e_violations(
    src_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every ``__all__`` for underscore-prefixed entries.

    ``__all__ = ["_helper"]`` is a contradictory declaration: the
    underscore says "implementation detail, private" while the
    ``__all__`` listing says "this is the module's public api." a
    module that genuinely wants to export a name should drop the
    underscore; a module that wants the name to stay private should
    not list it.

    only literal list/tuple/set forms are inspected. computed shapes
    (``__all__ = list(generated())``, ``__all__ += [...]``) are
    skipped because static analysis cannot resolve them; cases in
    practice are rare.

    :param src_roots: all src roots in the repo
    :ptype src_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for potential future context)
    :ptype repo_root: Path
    :return: list of violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in src_roots:
        for file in _iter_python_files(root):
            tree = _parse_file(file)
            if tree is None:
                continue
            for node in tree.body:
                value: ast.expr | None = None
                target_line: int = 0
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            value = node.value
                            target_line = node.lineno
                            break
                elif isinstance(node, ast.AnnAssign):
                    if (
                        isinstance(node.target, ast.Name)
                        and node.target.id == "__all__"
                    ):
                        value = node.value
                        target_line = node.lineno
                if value is None:
                    continue
                if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    continue
                for elt in value.elts:
                    if not isinstance(elt, ast.Constant):
                        continue
                    if not isinstance(elt.value, str):
                        continue
                    name = elt.value
                    if not _is_private_name(name):
                        continue
                    detail = (
                        f"__all__ lists private name '{name}'; underscore "
                        f"prefix declares implementation detail, __all__ "
                        f"declares public api -- the two declarations "
                        f"contradict. either drop the underscore (name is "
                        f"public) or remove from __all__ (name is private)"
                    )
                    violations.append(
                        Violation(
                            shape="E",
                            file=file,
                            line=elt.lineno if hasattr(elt, "lineno") else target_line,
                            symbol=name,
                            detail=detail,
                        )
                    )
    _ = repo_root
    return violations


def _shape_b_violations(
    repo_root: Path,
    src_roots: tuple[Path, ...],
) -> list[Violation]:
    """delegate cross-class protected access to ``ruff check --select SLF001``.

    ruff owns the scope analysis (``obj._attr`` is only a violation when
    ``obj`` is not ``self`` / ``cls``) and re-implementing it badly wastes
    time. ruff is always present in each repo's dev-deps.

    :param repo_root: repo root to lint
    :ptype repo_root: Path
    :param src_roots: src-roots to check (ruff is given the repo root but
        filters to these)
    :ptype src_roots: tuple[Path, ...]
    :return: list of violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if not src_roots:
        return violations
    args = [
        "ruff", "check",
        "--select", "SLF001",
        "--output-format", "concise",
        "--no-fix",
        "--force-exclude",
    ]
    args.extend(str(r) for r in src_roots)
    try:
        completed = subprocess.run(
            args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        pytest.skip("ruff not available on PATH; shape-B detection needs ruff")
    except subprocess.TimeoutExpired:
        pytest.fail(f"ruff check timed out after {_TIMEOUT_SECONDS}s")
    output = completed.stdout.splitlines()
    # concise format: path:line:col: CODE message
    pattern = re.compile(
        r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+SLF001\s+(?P<detail>.+)$"
    )
    for raw in output:
        match = pattern.match(raw)
        if match is None:
            continue
        file_path = Path(match.group("file"))
        if not file_path.is_absolute():
            file_path = (repo_root / file_path).resolve()
        # extract _name from ruff's standard message
        # "Private member accessed: `_name`"
        detail_msg = match.group("detail")
        symbol_match = re.search(r"`(_[A-Za-z_0-9]+)`", detail_msg)
        symbol = symbol_match.group(1) if symbol_match else "_<unknown>"
        violations.append(
            Violation(
                shape="B",
                file=file_path,
                line=int(match.group("line")),
                symbol=symbol,
                detail=detail_msg,
            )
        )
    return violations


# ------------------------------------------------------------------------
# exemptions application
# ------------------------------------------------------------------------


def _apply_exemptions(
    violations: list[Violation],
    exemptions: list[ExemptionEntry],
    repo_root: Path,
) -> list[Violation]:
    """filter violations against exemptions; exact match on (file, line, symbol).

    file comparison is done on the repo-relative path (as written in
    the exemptions file) against the violation's repo-relative path.

    :param violations: raw violations from the walkers
    :ptype violations: list[Violation]
    :param exemptions: parsed exemption entries
    :ptype exemptions: list[ExemptionEntry]
    :param repo_root: repo root for relative-path computation
    :ptype repo_root: Path
    :return: violations that did not match any exemption
    :rtype: list[Violation]
    """
    result: list[Violation] = []
    exemption_keys = {
        (entry.file, entry.line, entry.symbol) for entry in exemptions
    }
    for violation in violations:
        try:
            rel = str(violation.file.relative_to(repo_root))
        except ValueError:
            rel = str(violation.file)
        key = (rel, violation.line, violation.symbol)
        if key in exemption_keys:
            continue
        result.append(violation)
    return result


# ------------------------------------------------------------------------
# repo-anchoring
# ------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path:
    """walk upward from ``start`` to the nearest directory containing pyproject.toml.

    :param start: path to start search
    :ptype start: Path
    :return: repo root
    :rtype: Path
    :raises RuntimeError: if no pyproject.toml ancestor is found
    """
    current = start.resolve()
    result: Path | None = None
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            result = candidate
            break
    if result is None:
        raise RuntimeError(f"no pyproject.toml ancestor found above {start}")
    return result


_REPO_ROOT = _find_repo_root(Path(__file__))
_EXEMPTIONS_PATH = Path(__file__).parent / "_underscore_exemptions.txt"


# ------------------------------------------------------------------------
# pytest entry points
# ------------------------------------------------------------------------


def _collect_all_violations() -> tuple[list[Violation], tuple[Path, ...]]:
    """run all three shape walkers and return aggregated violations.

    :return: (violations, src_roots_discovered)
    :rtype: tuple[list[Violation], tuple[Path, ...]]
    """
    src_roots = find_src_roots(_REPO_ROOT)
    shape_a = _shape_a_violations(src_roots, _REPO_ROOT)
    shape_b = _shape_b_violations(_REPO_ROOT, src_roots)
    shape_c = _shape_c_violations(src_roots, _REPO_ROOT)
    shape_d = _shape_d_violations(src_roots, _REPO_ROOT)
    shape_e = _shape_e_violations(src_roots, _REPO_ROOT)
    all_violations = shape_a + shape_b + shape_c + shape_d + shape_e
    return all_violations, src_roots


def _mode() -> str:
    """resolve the audit mode from the env var, defaulting to strict.

    :return: normalized mode (``report`` or ``strict``)
    :rtype: str
    :raises ValueError: if the env var holds an unknown mode
    """
    raw = os.environ.get("UNDERSCORE_AUDIT_MODE", _DEFAULT_MODE).strip().lower()
    if raw not in (MODE_REPORT, MODE_STRICT):
        raise ValueError(
            f"UNDERSCORE_AUDIT_MODE must be '{MODE_REPORT}' or "
            f"'{MODE_STRICT}', got {raw!r}"
        )
    return raw


class TestUnderscoreAccess:
    """aggregate test: three shapes, one assertion, exemptions applied."""

    def test_no_underscore_violations(self) -> None:
        """every private access crosses a public API boundary or is exempted."""
        exemptions = parse_exemptions(_EXEMPTIONS_PATH)
        raw_violations, src_roots = _collect_all_violations()
        filtered = _apply_exemptions(raw_violations, exemptions, _REPO_ROOT)
        mode = _mode()
        by_shape: dict[str, list[Violation]] = {
            "A": [], "B": [], "C": [], "D": [], "E": []
        }
        for violation in filtered:
            by_shape.setdefault(violation.shape, []).append(violation)
        lines: list[str] = [
            f"repo_root: {_REPO_ROOT}",
            f"src_roots: {[str(r) for r in src_roots]}",
            f"mode: {mode}",
            f"exemptions_loaded: {len(exemptions)}",
            f"raw_violations: {len(raw_violations)}",
            f"after_exemptions: {len(filtered)}",
            f"  shape-A (cross-module private import): {len(by_shape['A'])}",
            f"  shape-B (cross-class protected access): {len(by_shape['B'])}",
            f"  shape-C (missing __all__): {len(by_shape['C'])}",
            f"  shape-D (subclass shadows base private): {len(by_shape['D'])}",
            f"  shape-E (__all__ contains private name): {len(by_shape['E'])}",
        ]
        for violation in sorted(
            filtered, key=lambda v: (v.shape, str(v.file), v.line, v.symbol)
        ):
            lines.append(violation.format(_REPO_ROOT))
        report = "\n".join(lines)
        # always print so ``-s`` surfaces the inventory even in strict mode
        print(report, file=sys.stderr)
        if mode == MODE_REPORT:
            return
        if filtered:
            pytest.fail(
                f"underscore-access enforcement found {len(filtered)} "
                f"violation(s):\n{report}"
            )
