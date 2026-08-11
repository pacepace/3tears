"""The leaf depends downward only, and the httpx module stays opt-in.

Two rules that are cheap to hold and silent to break, so they get mechanical
pins rather than review attention:

- **nothing in the package imports upward** (search-spec.md §2, SR-L7): not
  ``threetears.core``, not ``threetears.agent.*``, not langchain, not NATS.
  The whole reason a consumer can take this leaf on a Pi is that the leaf's
  dependency floor is the floor D24 declares.
- **``standalone`` is imported by nothing in the package at module level**
  (§3.8). httpx rides the ``[standalone]`` extra, so a module-level import
  anywhere else would make an extra mandatory -- and it would do so silently,
  because a development environment has httpx installed.

Both are checked by reading source rather than by importing, because an import
check can only prove the modules that happened to load are clean.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import threetears.search.contracts as contracts

_PACKAGE_ROOT = Path(contracts.__file__).resolve().parents[1]

#: import roots this leaf must never reach for. ``httpx`` is absent on
#: purpose: exactly one module may import it, and that is the next pin.
_FORBIDDEN_ROOTS = ("threetears.core", "threetears.agent", "langchain", "nats", "trafilatura")


def _modules() -> list[Path]:
    """Every source module in the package.

    :return: the module paths, sorted
    :rtype: list[Path]
    """
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def _module_level_imports(source: Path) -> set[str]:
    """Dotted names ``source`` imports at module level.

    Nested imports inside a function are excluded: a lazily-imported optional
    dependency is the sanctioned pattern, and it is only the module-level ones
    that make an extra mandatory.

    :param source: the module to read
    :ptype source: Path
    :return: the imported dotted names
    :rtype: set[str]
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_the_package_has_modules_to_check() -> None:
    """A pin over an empty set passes for the wrong reason."""
    assert len(_modules()) >= 8


@pytest.mark.parametrize("source", _modules(), ids=lambda path: path.name)
def test_no_module_imports_upward(source: Path) -> None:
    """SR-L7: the leaf depends downward only, at every depth."""
    offenders = [
        name
        for name in _module_level_imports(source)
        for root in _FORBIDDEN_ROOTS
        if name == root or name.startswith(f"{root}.")
    ]
    assert offenders == [], f"{source.name} imports {offenders}"


def test_only_the_standalone_module_imports_httpx() -> None:
    """§3.8: httpx is an extra, and one module's business."""
    importers = sorted(
        source.relative_to(_PACKAGE_ROOT).as_posix()
        for source in _modules()
        if any(name == "httpx" or name.startswith("httpx.") for name in _module_level_imports(source))
    )

    assert importers == ["standalone.py"]


def test_nothing_imports_standalone_at_module_level() -> None:
    """A host chooses the bare transport; the package never chooses it for it."""
    importers = sorted(
        source.relative_to(_PACKAGE_ROOT).as_posix()
        for source in _modules()
        if source.name != "standalone.py"
        and any(name.endswith("search.standalone") for name in _module_level_imports(source))
    )

    assert importers == []


def test_importing_the_working_layers_pulls_no_http_library() -> None:
    """The proof the source check cannot give: nothing transitively drags httpx.

    Adapters, Call and Bind are the whole embedded path, and a host injecting
    its own transport must be able to import all three without httpx or
    trafilatura installed at all.
    """
    probe = (
        "import sys; "
        "import threetears.search.adapters.searxng, threetears.search.call, threetears.search.bind; "
        "banned = [m for m in sys.modules if m.startswith(('httpx', 'trafilatura', 'threetears.core', "
        "'threetears.agent', 'langchain', 'nats'))]; "
        "sys.exit(repr(banned) if banned else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"the working layers pulled banned modules: {result.stderr}"


def test_the_conformance_suite_pulls_no_test_framework() -> None:
    """SR-L6: ``testing/`` is importable by a consumer without adding pytest."""
    probe = (
        "import sys; import threetears.search.testing; "
        "banned = [m for m in sys.modules if m.startswith(('pytest', '_pytest', 'httpx'))]; "
        "sys.exit(repr(banned) if banned else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"the testing helpers pulled banned modules: {result.stderr}"
