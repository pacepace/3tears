"""Only the modules named here import core at module level, and nothing imports upward.

The base install leaves out core, the NATS client and the agent layer, so a module-level import
of any of them anywhere but the two named ``core`` modules makes an extra mandatory. It would do
so silently: a development environment has all of them installed, so every test still passes.

Checked by reading source rather than by importing, because an import check can only prove the
modules that happened to load are clean. A module-level import here is one that runs when the
module is imported: at the top, or inside a module-level ``try``/``if``/``with`` or class body.
An import under ``if TYPE_CHECKING:`` costs nothing at import and is not counted. An import
inside a function is not counted here either, because it is not allowed at all: serving code
imports what it needs at its top, an optional extra's module under a guard there (owner ruling,
2026-10-24, held family-wide by ``tests/enforcement/test_no_function_level_imports.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "threetears" / "iam"

#: the roots the base install does not carry
_FORBIDDEN_ROOTS = (
    "threetears.core",
    "threetears.agent",
    "threetears.nats",
    "sqlalchemy",
    "asyncpg",
    "nats",
    "nkeys",
)

#: the modules that import ``threetears.core`` at module level, behind the guard that names the
#: ``core`` extra. ``dpop`` and ``stores.attempt_limiter`` need core to import at all; ``totp``
#: imports only its sealing there, and the rest of it runs on the base install.
_CORE_EXTRA_MODULES = frozenset({"dpop.py", "stores/attempt_limiter.py", "totp.py"})


def _modules() -> list[Path]:
    """Every source module in the package.

    :return: the module paths, sorted
    :rtype: list[Path]
    """
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def _is_type_checking(test: ast.expr) -> bool:
    """Whether an ``if`` test is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``.

    :param test: the ``if`` statement's test
    :ptype test: ast.expr
    :return: whether the branch is for type checkers only
    :rtype: bool
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _import_time_imports(source: Path) -> set[str]:
    """Dotted names ``source`` imports when it is imported.

    :param source: the module to read
    :ptype source: Path
    :return: the imported dotted names
    :rtype: set[str]
    """
    names: set[str] = set()

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    names.add(node.module)
            elif isinstance(node, ast.If):
                if not _is_type_checking(node.test):
                    visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, (ast.With, ast.ClassDef)):
                visit(node.body)

    visit(ast.parse(source.read_text(encoding="utf-8")).body)
    return names


def _forbidden(names: set[str]) -> list[str]:
    """The names under a forbidden root.

    :param names: dotted import names
    :ptype names: set[str]
    :return: those the base install does not carry, sorted
    :rtype: list[str]
    """
    return sorted(name for name in names for root in _FORBIDDEN_ROOTS if name == root or name.startswith(f"{root}."))


def test_the_package_has_modules_to_check() -> None:
    """A pin over an empty set passes for the wrong reason."""
    relative = {source.relative_to(_PACKAGE_ROOT).as_posix() for source in _modules()}
    assert len(relative) >= 20
    assert _CORE_EXTRA_MODULES <= relative, sorted(_CORE_EXTRA_MODULES - relative)


@pytest.mark.parametrize("source", _modules(), ids=lambda path: path.relative_to(_PACKAGE_ROOT).as_posix())
def test_no_base_module_imports_what_the_base_install_leaves_out(source: Path) -> None:
    relative = source.relative_to(_PACKAGE_ROOT).as_posix()
    offenders = _forbidden(_import_time_imports(source))
    if relative in _CORE_EXTRA_MODULES:
        offenders = [name for name in offenders if not name.startswith("threetears.core")]
    assert offenders == [], f"{relative} imports {offenders} at import time"


@pytest.mark.parametrize("relative", sorted(_CORE_EXTRA_MODULES))
def test_the_core_extra_modules_do_import_core(relative: str) -> None:
    """The exemption stays earned: a module that stopped needing core leaves the list."""
    names = _import_time_imports(_PACKAGE_ROOT / relative)
    assert any(name.startswith("threetears.core") for name in names), relative


def test_the_walker_sees_through_a_guard_and_skips_type_checking(tmp_path: Path) -> None:
    """Guard the guard: a ``try``-wrapped import counts, a ``TYPE_CHECKING`` one and a deferred one do not."""
    source = tmp_path / "probe.py"
    source.write_text(
        "from typing import TYPE_CHECKING\n"
        "try:\n"
        "    import sqlalchemy\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "if TYPE_CHECKING:\n"
        "    import asyncpg\n"
        "def later():\n"
        "    import nats\n",
        encoding="utf-8",
    )
    assert _forbidden(_import_time_imports(source)) == ["sqlalchemy"]
