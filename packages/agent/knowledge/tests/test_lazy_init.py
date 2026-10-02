"""lazy-surface consistency tests for the agent-knowledge package __init__.

pins the three-way agreement between ``__all__``, the lazily-resolved names ``dir()`` advertises, and the
``TYPE_CHECKING`` import block, plus the import-cost win: importing the package
namespace must not eagerly load the data / framework stack.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import threetears.agent.knowledge as knowledge


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
        lazy = _lazy_names(knowledge)
        assert lazy, "the package advertises no lazily-resolved names; the probe is vacuous"
        assert set(knowledge.__all__) <= lazy

    def test_type_checking_block_matches_lazy(self) -> None:
        init_path = Path(knowledge.__file__)
        type_checking = _type_checking_names(init_path)
        lazy = _lazy_names(knowledge)
        assert type_checking, "no TYPE_CHECKING imports found; the probe is vacuous"
        assert lazy, "the package advertises no lazily-resolved names; the probe is vacuous"
        assert type_checking == lazy

    def test_every_public_name_resolves(self) -> None:
        for name in knowledge.__all__:
            assert getattr(knowledge, name) is not None

    def test_unknown_attribute_raises(self) -> None:
        try:
            knowledge.definitely_not_an_attribute
        except AttributeError as exc:
            assert "definitely_not_an_attribute" in str(exc)
        else:
            raise AssertionError("expected AttributeError")


class TestImportCost:
    def test_package_import_does_not_load_the_stack(self) -> None:
        """importing the package namespace stays light (PEP 562 laziness)."""
        probe = (
            "import json, sys\n"
            "import threetears.agent.knowledge\n"
            "prefixes = ('threetears.core', 'threetears.knowledge', 'threetears.agent.acl',"
            " 'langchain', 'langgraph', 'sqlalchemy', 'pgvector')\n"
            "loaded = sorted(n for n in sys.modules"
            " if any(n == p or n.startswith(p + '.') for p in prefixes))\n"
            "print(json.dumps(loaded))\n"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        loaded = json.loads(result.stdout.strip())
        assert loaded == [], f"package import eagerly loaded the stack: {loaded}"
