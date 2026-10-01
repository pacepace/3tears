"""lazy-surface consistency tests for the agent-identity package __init__.

pins the three-way agreement between ``__all__``, the lazily-resolved names ``dir()`` advertises, and
the ``TYPE_CHECKING`` import block, plus the import-cost win: importing the
pure types module must not load the data stack.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import threetears.agent.identity as identity


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


def _lazy_names(package: ModuleType) -> set[str]:
    """names the package resolves on first access rather than binding at import.

    executes the package ``__init__`` into a fresh module object that is never
    registered in ``sys.modules``, so no other test's attribute access has
    materialized a lazy name into it yet. the package's ``__dir__`` advertises
    every lazily-resolvable name, so whatever ``dir()`` reports that the fresh
    namespace does not yet hold is exactly the lazy surface.

    :param package: the imported package whose lazy surface to read
    :ptype package: ModuleType
    :return: the names ``__getattr__`` resolves on first access
    :rtype: set[str]
    """
    spec = importlib.util.find_spec(package.__name__)
    assert spec is not None
    assert spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    return set(dir(fresh)) - set(vars(fresh))


class TestLazySurfaceConsistency:
    def test_all_is_subset_of_lazy(self) -> None:
        lazy = _lazy_names(identity)
        assert lazy, "the package advertises no lazily-resolved names; the probe is vacuous"
        assert set(identity.__all__) <= lazy

    def test_type_checking_block_matches_lazy(self) -> None:
        init_path = Path(identity.__file__)
        type_checking = _type_checking_names(init_path)
        lazy = _lazy_names(identity)
        assert type_checking, "no TYPE_CHECKING imports found; the probe is vacuous"
        assert lazy, "the package advertises no lazily-resolved names; the probe is vacuous"
        assert type_checking == lazy

    def test_every_public_name_resolves(self) -> None:
        for name in identity.__all__:
            assert getattr(identity, name) is not None

    def test_unknown_attribute_raises(self) -> None:
        try:
            identity.definitely_not_an_attribute
        except AttributeError as exc:
            assert "definitely_not_an_attribute" in str(exc)
        else:
            raise AssertionError("expected AttributeError")


class TestImportCost:
    def test_types_import_does_not_load_data_stack(self) -> None:
        """importing the pure types module stays light."""
        probe = (
            "import json, sys\n"
            "import threetears.agent.identity.types\n"
            "prefixes = ('threetears.core', 'langgraph', 'sqlalchemy',"
            " 'asyncpg', 'pgvector', 'nats', 'threetears.nats')\n"
            "loaded = sorted(n for n in sys.modules"
            " if any(n == p or n.startswith(p + '.') for p in prefixes))\n"
            "print(json.dumps(loaded))\n"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        loaded = json.loads(result.stdout.strip())
        assert loaded == [], f"types import loaded the data stack: {loaded}"
