"""lazy-surface consistency tests for the agent-knowledge package __init__.

pins the three-way agreement between ``__all__``, the ``_LAZY`` map, and the
``TYPE_CHECKING`` import block, plus the import-cost win: importing the package
namespace must not eagerly load the data / framework stack.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

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


class TestLazySurfaceConsistency:
    def test_all_is_subset_of_lazy(self) -> None:
        assert set(knowledge.__all__) <= set(knowledge._LAZY)

    def test_type_checking_block_matches_lazy(self) -> None:
        init_path = Path(knowledge.__file__)
        assert _type_checking_names(init_path) == set(knowledge._LAZY)

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
