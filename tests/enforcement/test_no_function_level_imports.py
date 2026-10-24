"""
enforcement: the serving packages import nothing inside a function.

An import inside a function runs the first time that function is called, which in a serving process is
in the middle of a request: the import's module loading, and every module it pulls in, hold the event loop
(a first collection read loaded fourteen OpenTelemetry modules on the loop of a report pod). Owner ruling,
2026-10-24: lazy imports are banned in serving code. A module imports what it needs at its top, an optional
extra's module under ``try/except ImportError`` there, and an import cycle is fixed by restructuring the
modules, not by deferring one side.

**What it holds.** Every ``import`` and ``from ... import`` statement inside a function or method body
(nested ones included) in every package of the workspace (``packages/*/src``, ``packages/agent/*/src``), so a
package added later is held the day it exists. A module that runs on no serving path -- the enforcement
package, which runs in test suites and CI; the testing extra's fixtures; telemetry start-up -- is named in the
ledger with why. An import under ``if
TYPE_CHECKING:`` runs nowhere and is not counted. A module-level ``__getattr__`` resolving a name through
``importlib`` holds no import statement and is not counted.

**Exemptions** live in ``_function_level_import_exemptions.txt``, one ``<path>::<function>  # <reason>``
per line, where ``<path>`` may end in ``/*`` for a whole directory. A reason is at least thirty
characters; an entry the scan no longer needs is stale and fails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_function_level_import_exemptions.txt"
_MIN_REASON = 30

#: every workspace member's source tree: the ruling covers all serving code, and a module that serves nothing
#: is exempted in the ledger by name, with why
_SERVING = tuple(
    sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)


@dataclass(frozen=True)
class _Found:
    """one import statement inside a function."""

    path: str
    function: str
    line: int
    module: str


def _is_type_checking(test: ast.expr) -> bool:
    """whether an ``if`` tests ``TYPE_CHECKING``, under which nothing runs."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


class _Walker(ast.NodeVisitor):
    """collects every import statement some enclosing scope of which is a function."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.scopes: list[tuple[str, bool]] = []
        self.found: list[_Found] = []

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, function: bool) -> None:
        self.scopes.append((node.name, function))
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, True)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, True)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, False)

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def _imported(self, node: ast.Import | ast.ImportFrom, module: str) -> None:
        if any(function for _, function in self.scopes):
            self.found.append(_Found(self.path, ".".join(name for name, _ in self.scopes), node.lineno, module))

    def visit_Import(self, node: ast.Import) -> None:
        self._imported(node, ", ".join(alias.name for alias in node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._imported(node, node.module or ".")


def _scan(repo_root: Path, roots: tuple[str, ...]) -> list[_Found]:
    """every function-level import under ``roots``.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :param roots: the source roots, relative to it
    :ptype roots: tuple[str, ...]
    :return: the imports found
    :rtype: list[_Found]
    """
    found: list[_Found] = []
    for root in roots:
        for file in sorted((repo_root / root).rglob("*.py")):
            relative = file.relative_to(repo_root).as_posix()
            walker = _Walker(relative)
            walker.visit(ast.parse(file.read_text(encoding="utf-8"), filename=relative))
            found.extend(walker.found)
    return found


def _ledger(path: Path) -> dict[str, str]:
    """the exemptions, ``<path>::<function>`` (or ``<dir>/*``) to its reason.

    :param path: the ledger
    :ptype path: Path
    :return: the entries
    :rtype: dict[str, str]
    """
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = line.partition("#")
        entries[key.strip()] = reason.strip()
    return entries


def _exempted_by(found: _Found, ledger: dict[str, str]) -> str | None:
    """the ledger entry that exempts ``found``, or ``None``."""
    exact = f"{found.path}::{found.function}"
    if exact in ledger:
        return exact
    for key in ledger:
        if key.endswith("/*") and found.path.startswith(key[:-1]):
            return key
    return None


def _violations(repo_root: Path, roots: tuple[str, ...], ledger_path: Path) -> list[str]:
    """every unexempted function-level import, every reasonless entry, and every stale one.

    :param repo_root: the repository, or a planted tree
    :ptype repo_root: Path
    :param roots: the source roots
    :ptype roots: tuple[str, ...]
    :param ledger_path: the exemptions
    :ptype ledger_path: Path
    :return: one line per problem
    :rtype: list[str]
    """
    ledger = _ledger(ledger_path)
    problems: list[str] = []
    used: set[str] = set()
    for found in _scan(repo_root, roots):
        entry = _exempted_by(found, ledger)
        if entry is None:
            problems.append(f"{found.path}:{found.line} {found.function} imports {found.module} inside a function")
        else:
            used.add(entry)
    for key, reason in sorted(ledger.items()):
        if len(reason) < _MIN_REASON:
            problems.append(f"exemption {key} gives no reason of {_MIN_REASON}+ characters")
        if key not in used:
            problems.append(f"exemption {key} exempts nothing the scan finds: remove it")
    return problems


class TestTheServingPackages:
    def test_the_scan_reads_the_serving_packages(self) -> None:
        files = [file for root in _SERVING for file in (_REPO_ROOT / root).rglob("*.py")]
        assert len(_SERVING) > 25, _SERVING
        assert len(files) > 1000, f"only {len(files)} files under the serving packages"

    def test_no_serving_module_imports_inside_a_function(self) -> None:
        problems = _violations(_REPO_ROOT, _SERVING, _LEDGER)
        assert not problems, f"{len(problems)} function-level import problem(s):\n" + "\n".join(problems)


class TestTheWalker:
    def _plant(self, tmp_path: Path, source: str) -> Path:
        package = tmp_path / "packages" / "planted" / "src" / "threetears" / "planted"
        package.mkdir(parents=True)
        (package / "module.py").write_text(source, encoding="utf-8")
        return tmp_path

    def test_an_import_in_a_function_method_or_nested_block_is_found(self, tmp_path: Path) -> None:
        root = self._plant(
            tmp_path,
            "import os\n\n\n"
            "def f():\n    import json\n\n\n"
            "class C:\n    async def m(self):\n        try:\n            from a import b\n        except ImportError:\n"
            "            pass\n\n\n"
            "def g():\n    def inner():\n        if True:\n            import zlib\n",
        )
        found = [(f.function, f.module) for f in _scan(root, ("packages/planted/src",))]
        assert found == [("f", "json"), ("C.m", "a"), ("g.inner", "zlib")]

    def test_a_module_level_import_and_a_type_checking_one_are_not_found(self, tmp_path: Path) -> None:
        root = self._plant(
            tmp_path,
            "from typing import TYPE_CHECKING\n\ntry:\n    import yaml\nexcept ImportError:\n    yaml = None\n\n\n"
            "def f():\n    if TYPE_CHECKING:\n        from a import b\n",
        )
        assert _scan(root, ("packages/planted/src",)) == []

    def test_an_exemption_needs_a_reason_and_goes_stale(self, tmp_path: Path) -> None:
        root = self._plant(tmp_path, "def f():\n    import json\n")
        ledger = tmp_path / "ledger.txt"
        ledger.write_text(
            "packages/planted/src/threetears/planted/module.py::f  # short\n"
            "packages/planted/src/threetears/planted/gone.py::g  # a reason long enough to be a reason at all\n",
            encoding="utf-8",
        )
        problems = _violations(root, ("packages/planted/src",), ledger)
        assert any("gives no reason" in problem for problem in problems)
        assert any("gone.py::g exempts nothing" in problem for problem in problems)
