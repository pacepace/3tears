"""The no-whole-reads walk's cost grows with the source it reads, never with its calls times the class they sit in.

Profiled 2026-11-15: the walk re-walked a class's whole body once per method call judged in it (to learn the receiver's
class), re-walked every module whole about fifteen times, and scanned every module path once per import it looked up.
On the hub the gate took 81-89 s alone and hung xdist under ``-n 10``; 3tears' two gates took ~260 s each. These tests
count the nodes :func:`ast.walk` hands the walk, so they hold on any machine at any load: on the old walk the count
per source node grew with the class's size (a class four times the size cost about four times as much per node).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.enforcement.common import remembered_walks, walk_tree
from threetears.enforcement.no_whole_reads import NoWholeReadsConfig, no_whole_reads_findings
from threetears.enforcement.no_whole_reads.walker import ModulesByPath


def _module(methods: int) -> str:
    """a module with one class of ``methods`` methods, each making calls on attributes and locals of known classes.

    :param methods: how many methods the class has
    :ptype methods: int
    :return: the module's source
    :rtype: str
    """
    lines = [
        "class Tool:",
        "    def run(self, value):",
        "        return value",
        "",
        "",
        "class Holder:",
        "    def __init__(self, tool: Tool) -> None:",
        "        self._tool = tool",
        "        self._other = Tool()",
    ]
    for index in range(methods):
        lines += [
            f"    def method_{index}(self, value):",
            "        local = Tool()",
            "        local.run(value)",
            "        self._tool.run(value)",
            "        self._other.run(value)",
            "        self._tool.run(local.run(value))",
            "        return self._other.run(value)",
        ]
    return "\n".join(lines) + "\n"


def _visits(tmp_path: Path, methods: int, monkeypatch: pytest.MonkeyPatch) -> float:
    """the nodes :func:`ast.walk` yields to the gate over a module of ``methods`` methods, per node of the module.

    :param tmp_path: a fresh directory
    :ptype tmp_path: Path
    :param methods: the class's methods
    :ptype methods: int
    :param monkeypatch: patches ``ast.walk`` to count
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: walked nodes per source node
    :rtype: float
    """
    root = tmp_path / f"m{methods}"
    src = root / "src"
    src.mkdir(parents=True)
    source = _module(methods)
    (src / "holder.py").write_text(source, encoding="utf-8")
    (root / "_no_whole_reads_exemptions.txt").write_text("", encoding="utf-8")
    config = NoWholeReadsConfig(repo_root=root, src_roots=(src,), ledger_dir=root, dependency_schema=False)
    size = sum(1 for _ in ast.walk(ast.parse(source)))
    walked = 0
    real = ast.walk

    def counting(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal walked
        for found in real(node):
            walked += 1
            yield found

    with monkeypatch.context() as patched:
        patched.setattr(ast, "walk", counting)
        no_whole_reads_findings(config)
    return walked / size


def test_the_walk_costs_the_same_per_node_however_big_the_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a class four times the size costs the walk no more per node: each body is read once, not once per call in it.

    :param tmp_path: a fresh directory
    :ptype tmp_path: Path
    :param monkeypatch: patches ``ast.walk`` to count
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    small = _visits(tmp_path, 20, monkeypatch)
    large = _visits(tmp_path, 80, monkeypatch)
    assert large < small * 1.25, f"walked {small:.0f} nodes per source node at 20 methods, {large:.0f} at 80"


def test_a_kept_walk_is_the_walk() -> None:
    """inside a scope a module's, class's or function's walk is kept and is ``ast.walk``'s, node for node; an
    expression's is walked afresh (a walker rewrites its copies in place); outside a scope nothing is kept.

    :return: nothing
    :rtype: None
    """
    tree = ast.parse("class A:\n    def f(self):\n        return g(1)\n")
    owner = tree.body[0]
    assert isinstance(owner, ast.ClassDef)
    function = owner.body[0]
    expression = ast.parse("g(1)", mode="eval").body
    assert not isinstance(walk_tree(tree), tuple)
    with remembered_walks():
        for node in (tree, owner, function):
            first = walk_tree(node)
            assert isinstance(first, tuple)
            assert first == tuple(ast.walk(node))
            assert walk_tree(node) is first
        assert not isinstance(walk_tree(expression), tuple)
        with remembered_walks():
            assert walk_tree(tree) is walk_tree(tree)
    assert not isinstance(walk_tree(tree), tuple)


@pytest.mark.parametrize(
    "module",
    ["pkg.tables", "tables", "pkg", "a.pkg.tables", "pkg.__init__", "other", "kg.tables", "src.pkg.tables"],
)
def test_a_module_found_by_its_index_is_the_one_a_scan_finds(module: str) -> None:
    """:class:`ModulesByPath` answers what scanning every path for the name answers: the first path, in order, that is
    the module (``<name>.py``) or its package (``<name>/__init__.py``), at the root or under any directory.

    :param module: the dotted name looked up
    :ptype module: str
    :return: nothing
    :rtype: None
    """
    entries = {
        "src/pkg/tables.py": {"which": "src tables"},
        "lib/pkg/tables.py": {"which": "lib tables"},
        "src/pkg/__init__.py": {"which": "src package"},
        "tables.py": {"which": "top tables"},
        "a/pkg/tables.py": {"which": "a tables"},
        "README.md": {"which": "not a module"},
    }
    base = module.replace(".", "/")
    scanned = next(
        (
            path
            for path in entries
            if path.endswith((f"/{base}.py", f"/{base}/__init__.py")) or path in {f"{base}.py", f"{base}/__init__.py"}
        ),
        None,
    )
    assert ModulesByPath(entries).paths.get(base) == scanned
