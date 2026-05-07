"""Enforcement -- ban ``DATETIME_TYPE`` from ``Column(...)`` declarations.

collections-task-05 eliminated the ``DATETIME_TYPE`` /
``TIMESTAMP`` (timezone-naive) column-type tag from the platform.
Every datetime column is now ``DATETIMETZ_TYPE`` / ``TIMESTAMPTZ``
(timezone-aware UTC).

The constant has been deleted from
``threetears.core.collections.schema_backed`` so any source that
still imports it fails at module load with ``ImportError``. This
test is the structural belt-and-suspenders: it walks the AST of
every package source file and fails if a ``Column(..., DATETIME_TYPE, ...)``
literal slips back in (e.g. via copy-paste from an external
example or a downstream consumer that defines its own
``DATETIME_TYPE = "datetime"`` and uses it locally).

The test runs in <1s; it is AST-only, deterministic, and
side-effect-free.

Why this enforcement matters: asyncpg's TIMESTAMPTZ codec calls
``obj.astimezone(utc)`` on every bound parameter. A naive
datetime that the platform thinks is "TIMESTAMP" gets shifted by
the host's local-tz offset on non-UTC dev machines, silently
corrupting wire values. Before collections-task-05 the platform
relied on a global ``data[k] = v.replace(tzinfo=None)`` strip in
``BaseCollection.save_entity`` to bridge aware-UTC application
code to naive-TIMESTAMP storage; removing the strip and the
``DATETIME_TYPE`` column tag in the same task means we cannot
silently regress. This test is the regression brake.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

__all__: list[str] = []


# resolves to the 3tears repo root: this file lives at
# packages/core/tests/enforcement/, four levels under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# package source roots to walk for Column(...) declarations.
#
# Walks every package with a ``src/`` tree. The original walk skipped
# channels / langgraph / models / nats / observe / registry / enforcement
# -- a coverage gap that let the registry's
# ``HeartbeatCollection.save_entity`` retain a tzinfo-strip well after
# collections-task-05 phase D was supposed to have removed every such
# leak. Closing the coverage gap here so the AST guard runs against
# the entire 3tears workspace.
_PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "agent" / "acl" / "src",
    _REPO_ROOT / "packages" / "agent" / "audit" / "src",
    _REPO_ROOT / "packages" / "agent" / "memory" / "src",
    _REPO_ROOT / "packages" / "agent" / "tools" / "src",
    _REPO_ROOT / "packages" / "agent" / "workspace" / "src",
    _REPO_ROOT / "packages" / "channels" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "enforcement" / "src",
    _REPO_ROOT / "packages" / "epoch" / "src",
    _REPO_ROOT / "packages" / "langgraph" / "src",
    _REPO_ROOT / "packages" / "mcp" / "src",
    _REPO_ROOT / "packages" / "models" / "src",
    _REPO_ROOT / "packages" / "nats" / "src",
    _REPO_ROOT / "packages" / "observe" / "src",
    _REPO_ROOT / "packages" / "registry" / "src",
]


def _walk_source_files(root: Path) -> list[Path]:
    """return every .py file under ``root`` (recursive)."""
    return [p for p in root.rglob("*.py") if p.is_file()]


def _column_uses_datetime_type(call: ast.Call) -> bool:
    """return ``True`` iff ``call`` is ``Column(..., DATETIME_TYPE, ...)``.

    matches:
      - ``Column("foo", DATETIME_TYPE)``
      - ``Column("foo", DATETIME_TYPE, immutable=True)``
      - ``Column(name="foo", column_type=DATETIME_TYPE)``
    """
    func = call.func
    if isinstance(func, ast.Name) and func.id != "Column":
        return False
    if isinstance(func, ast.Attribute) and func.attr != "Column":
        return False
    if not isinstance(func, (ast.Name, ast.Attribute)):
        return False
    for arg in call.args:
        if isinstance(arg, ast.Name) and arg.id == "DATETIME_TYPE":
            return True
    for kw in call.keywords:
        value = kw.value
        if isinstance(value, ast.Name) and value.id == "DATETIME_TYPE":
            return True
    return False


def _collect_violations(path: Path) -> list[tuple[Path, int]]:
    """return ``[(path, lineno), ...]`` for every Column(...DATETIME_TYPE) call."""
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    violations: list[tuple[Path, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _column_uses_datetime_type(node):
            violations.append((path, node.lineno))
    return violations


@pytest.mark.parametrize("src_root", _PACKAGE_SRC_ROOTS)
def test_no_datetime_type_in_column_declarations(src_root: Path) -> None:
    """Every Column(...) declaration uses DATETIMETZ_TYPE, not DATETIME_TYPE."""
    if not src_root.is_dir():
        pytest.skip(f"package src root absent: {src_root}")
    violations: list[tuple[Path, int]] = []
    for source_file in _walk_source_files(src_root):
        violations.extend(_collect_violations(source_file))
    if violations:
        formatted = "\n".join(
            f"  {path.relative_to(_REPO_ROOT)}:{lineno}"
            for path, lineno in violations
        )
        pytest.fail(
            "DATETIME_TYPE was eliminated by collections-task-05; use "
            "DATETIMETZ_TYPE instead. The following Column declarations "
            "still reference DATETIME_TYPE:\n" + formatted,
        )
