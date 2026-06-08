"""lazy-surface consistency tests for the core package __init__.

pins the three-way agreement between ``__all__``, the ``_LAZY`` map,
and the ``TYPE_CHECKING`` import block (the decision record is
docs/separate-concerns-decisions.md), plus the import-cost win: a bare
``import threetears.core`` must not load the heavy backend stack.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import threetears.core as core


def _type_checking_names(init_path: Path) -> set[str]:
    """collect names imported inside ``if TYPE_CHECKING:`` blocks.

    :param init_path: path to the package ``__init__.py``
    :ptype init_path: Path
    :return: exported names (aliases respected) from the TC block
    :rtype: set[str]
    """
    tree = ast.parse(init_path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not is_tc:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    names.add(alias.asname or alias.name)
    return names


class TestLazySurfaceConsistency:
    def test_all_is_subset_of_lazy(self) -> None:
        assert set(core.__all__) <= set(core._LAZY)

    def test_type_checking_block_matches_lazy(self) -> None:
        init_path = Path(core.__file__)
        assert _type_checking_names(init_path) == set(core._LAZY)

    def test_every_public_name_resolves(self) -> None:
        for name in core.__all__:
            assert getattr(core, name) is not None

    def test_unknown_attribute_raises(self) -> None:
        try:
            core.definitely_not_an_attribute
        except AttributeError as exc:
            assert "definitely_not_an_attribute" in str(exc)
        else:
            raise AssertionError("expected AttributeError")

    def test_dir_includes_lazy_names(self) -> None:
        listing = dir(core)
        assert "DataStore" in listing
        assert "BaseCollection" in listing


class TestImportCost:
    def test_bare_core_import_does_not_load_backends(self) -> None:
        """``import threetears.core`` must not pull sqlalchemy/asyncpg/nats."""
        probe = (
            "import json, sys\n"
            "import threetears.core\n"
            "prefixes = ('sqlalchemy', 'asyncpg', 'nats', 'threetears.nats')\n"
            "loaded = sorted(n for n in sys.modules"
            " if any(n == p or n.startswith(p + '.') for p in prefixes))\n"
            "print(json.dumps(loaded))\n"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        loaded = json.loads(result.stdout.strip())
        assert loaded == [], f"bare core import loaded heavy modules: {loaded}"
