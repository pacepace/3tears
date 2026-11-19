"""No 3tears source runs an Acero plan for a table: no ``Table.join``, ``join_asof``, ``group_by``, nor ``filter`` given
an expression (``pc.field``), nor ``pyarrow.acero`` itself.

Each answers ``DeclarationToTable``'s table, which is whatever the plan's table sink holds when the plan reports itself
finished; a plan run on threads can answer a null table with an OK status, and pyarrow's ``Table`` over it segfaults on
first use. Reports' loop_timing run of 2026-11-18 04:40 died so in its ``follower_parts._named_rows``' left semi join
(SG1: ``KERN_INVALID_ADDRESS at 0x8``, the schema read off a null table; :mod:`threetears.core.arrow_keys` has
the whole account), and 3tears' own ``DuckDBBackend.storage_arrow`` kept the last row of each key by a ``group_by``.
A key is matched by :mod:`~threetears.core.arrow_keys` (Arrow's hash kernels) instead, and a table filtered by
a mask array, which runs the ``filter`` kernel. Every package's ``src`` is read; tests are not (a test may hold an
Acero answer up as the reference).

A call counts, and so does a method handed to one (``asyncio.to_thread(rows.join, ...)``); a field of the same name
(a spec's ``group_by``) does not. ``join`` is a common name, so it counts only given a keyword only ``Table.join``
takes (``keys=``, ``join_type=``). Any ``.field(`` of ``pyarrow.compute`` is refused outright, as it exists only to
build those expressions.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_PACKAGES: Final = Path(__file__).resolve().parents[2] / "packages"

#: method names only an Arrow table (or Acero) answers to
_ACERO_METHODS: Final = frozenset({"group_by", "join_asof"})

#: ``join``'s keywords only ``Table.join`` takes
_JOIN_KEYWORDS: Final = frozenset({"keys", "join_type", "right_keys", "coalesce_keys"})

#: the names ``pyarrow.compute`` goes by in this code
_COMPUTE: Final = frozenset({"pc", "pyarrow.compute", "pa.compute"})


class _Acero(ast.NodeVisitor):
    """the places a module runs (or hands off) an Acero plan."""

    def __init__(self) -> None:
        self.found: list[tuple[int, str]] = []

    def _found(self, node: ast.AST) -> None:
        self.found.append((getattr(node, "lineno", 0), ast.unparse(node)))

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast's visitor name
        if node.attr == "field" and ast.unparse(node.value) in _COMPUTE:
            self._found(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast's visitor name
        named = {keyword.arg for keyword in node.keywords}
        called = node.func.attr if isinstance(node.func, ast.Attribute) else None
        handed = {arg.attr for arg in node.args if isinstance(arg, ast.Attribute)}
        # ``rows.group_by(key)``, or ``to_thread(rows.group_by, key)``: a field of that name (a spec's) is no call
        acero = called in _ACERO_METHODS or bool(handed & _ACERO_METHODS)
        # ``rows.join(other, keys=...)``, or ``to_thread(rows.join, other, keys=...)``
        table_join = (called == "join" or "join" in handed) and bool(named & _JOIN_KEYWORDS)
        if acero or table_join:
            self._found(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast's visitor name
        if any(alias.name.startswith("pyarrow.acero") for alias in node.names):
            self._found(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast's visitor name
        module = node.module or ""
        if module.startswith("pyarrow.acero") or (module == "pyarrow" and any(a.name == "acero" for a in node.names)):
            self._found(node)


def acero_sites(source: str) -> list[tuple[int, str]]:
    """the lines of ``source`` that run an Acero plan for a table, and what they say.

    :param source: a module's text
    :ptype source: str
    :return: (line, the expression) for each
    :rtype: list[tuple[int, str]]
    """
    visitor = _Acero()
    visitor.visit(ast.parse(source))
    return visitor.found


def test_the_rule_sees_each_shape_the_code_had() -> None:
    """each shape reports' and 3tears' code held before SG1 (and the import) is seen; a mask filter and a string join
    are not."""
    seen = [
        'keys.join(named, keys=list(key), join_type="left semi")',
        'asyncio.to_thread(held_keys.join, marked, keys=key, join_type="left outer")',
        "rows.group_by(list(key)).aggregate([])",
        'indexed.group_by(key, use_threads=False).aggregate([("__row", "max")])',
        "states.filter(pc.is_valid(pc.field(STATE_COLUMN)))",
        "left.join_asof(right, on='t', by='k', tolerance=1)",
        "import pyarrow.acero",
        "from pyarrow import acero",
    ]
    for source in seen:
        assert acero_sites(source), source
    unseen = (
        '", ".join(names)',
        "states.filter(pc.is_valid(states[STATE_COLUMN]))",
        "sep.join(parts)",
        "named = [shown.group_by, shown.title]",
    )
    for source in unseen:
        assert not acero_sites(source), source


def test_no_3tears_module_runs_an_acero_plan_for_a_table() -> None:
    found = {
        str(path.relative_to(_PACKAGES)): sites
        for path in sorted(_PACKAGES.glob("**/src/**/*.py"))
        if (sites := acero_sites(path.read_text(encoding="utf-8")))
    }
    assert not found, f"Acero plans for tables (match keys with threetears.core.arrow_keys): {found}"
