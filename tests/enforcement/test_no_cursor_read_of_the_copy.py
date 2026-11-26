"""No serving module reads the copy on a cursor of the serving process: every read of it is a query worker's.

**Why** (``design-copy-worker-backend.md``, decision 4; chunk 3T-8). The serving process read its copy with DuckDB on
its own threads: the "engine slots" (``shared_arrow.SharedArrowReader`` and its ``ReadSlot``, ``held_slot``,
``riding``, ``_Admission``), the follower's cursor reads (``copy_reads.CursorReads`` over the in-process copy's
snapshot), and the in-process DuckDB itself as a follower's backend (``cache.duckdb.DuckDBBackend``, typed so in
``copy_follower``). CPU beside the event loop, and three slots leaked in reports' replay 9 and starved every read.
They are deleted: the copy is published as shared Arrow and read in query workers
(``collections.copy_queries.CopyQueries``: ``lookup``, ``run_job``, ``current``). This rule keeps them deleted.

**What it refuses**, in every package's serving source (``serving_code_support``: tests, the testing helpers, the rule
engine and the evals harness are not serving code):

- an import of a cursor read of the copy from shared Arrow (``SharedArrowReader``, ``SharedRead``, ``ReadSlot``,
  ``held_slot``, ``riding``, ``_Admission``) or from the follower's reads (``CursorReads``, ``CursorCopy``), or an
  attribute of either module by those names;
- in the copy's own modules (``collections/copy_*.py``, ``shared_copy.py``, ``volume_fulls.py``), an import of
  ``DuckDBBackend``: the in-process copy, whose reads are its cursor's, is no follower's backend.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tests.enforcement.serving_code_support import NON_SERVING_ROOTS, REPO_ROOT, SRC_ROOTS

#: each module's names that are a cursor read of the copy in the serving process
_CURSOR_READS: Final = {
    "threetears.core.shared_arrow": frozenset(
        {"SharedArrowReader", "SharedRead", "ReadSlot", "held_slot", "riding", "_Admission"}
    ),
    "threetears.core.collections.copy_reads": frozenset({"CursorReads", "CursorCopy"}),
}

#: the in-process copy, by its module and name: refused in the copy's own modules
_IN_PROCESS_COPY: Final = ("threetears.core.cache.duckdb", "DuckDBBackend")

#: the copy's own modules besides ``collections/copy_*.py``
_COPY_MODULE_NAMES: Final = frozenset({"shared_copy.py", "volume_fulls.py"})


class _CursorReadImports(ast.NodeVisitor):
    """the places a module imports a cursor read of the copy (and, in a copy module, the in-process copy)."""

    def __init__(self, *, copy_module: bool) -> None:
        self.found: list[tuple[int, str]] = []
        self._copy_module = copy_module
        #: the names this module binds to a refused module (``import threetears.core.shared_arrow as sa``)
        self._bound: dict[str, frozenset[str]] = {}

    def _found(self, node: ast.AST) -> None:
        self.found.append((getattr(node, "lineno", 0), ast.unparse(node)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast's visitor name
        module = node.module or ""
        names = {alias.name for alias in node.names}
        refused = _CURSOR_READS.get(module, frozenset())
        if names & refused:
            self._found(node)
        if self._copy_module and module == _IN_PROCESS_COPY[0] and _IN_PROCESS_COPY[1] in names:
            self._found(node)
        # ``from threetears.core import shared_arrow``: its attributes are watched below
        for alias in node.names:
            dotted = f"{module}.{alias.name}"
            if dotted in _CURSOR_READS:
                self._bound[alias.asname or alias.name] = _CURSOR_READS[dotted]
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast's visitor name
        for alias in node.names:
            if alias.name in _CURSOR_READS:
                self._bound[alias.asname or alias.name] = _CURSOR_READS[alias.name]
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast's visitor name
        base = ast.unparse(node.value)
        if node.attr in self._bound.get(base, frozenset()):
            self._found(node)
        self.generic_visit(node)


def cursor_read_imports(source: str, *, copy_module: bool = False) -> list[tuple[int, str]]:
    """the lines of ``source`` that import a cursor read of the copy, and what they say.

    :param source: a module's text
    :ptype source: str
    :param copy_module: whether it is one of the copy's own modules (the in-process copy refused there too)
    :ptype copy_module: bool
    :return: (line, the statement) for each
    :rtype: list[tuple[int, str]]
    """
    visitor = _CursorReadImports(copy_module=copy_module)
    visitor.visit(ast.parse(source))
    return visitor.found


def _serving_modules() -> list[Path]:
    """every serving module of every package (``serving_code_support``)."""
    return sorted(
        path
        for root in SRC_ROOTS
        for path in root.glob("**/*.py")
        if not any(path.is_relative_to(skipped) for skipped in NON_SERVING_ROOTS)
    )


def _is_copy_module(path: Path) -> bool:
    """whether ``path`` is one of the copy's own modules."""
    if path.parent.name != "collections" or path.parent.parent.name != "core":
        return False
    return path.name.startswith("copy_") or path.name in _COPY_MODULE_NAMES


def test_the_rule_sees_each_shape_the_code_had() -> None:
    """each import the follower and the shared Arrow store's readers held is seen; the copy's worker reads are not."""
    seen = [
        # copy_follower.py:221 before 3T-7
        ("from threetears.core.shared_arrow import ReadSlot, held_slot, riding", False),
        # copy_follower.py:174 before 3T-8
        (
            "from threetears.core.collections.copy_reads import CopyReads, CursorReads, PinnedCopy, QueriedReads",
            False,
        ),
        ("from threetears.core.shared_arrow import SharedArrowReader, SharedArrowStore", False),
        ("import threetears.core.shared_arrow as sa\nsa.held_slot()", False),
        ("from threetears.core import shared_arrow\nshared_arrow.riding(None)", False),
        # copy_follower.py:223 before 3T-8, under TYPE_CHECKING
        ("if TYPE_CHECKING:\n    from threetears.core.cache.duckdb import DuckDBBackend", True),
    ]
    for source, copy_module in seen:
        assert cursor_read_imports(source, copy_module=copy_module), source
    unseen = [
        ("from threetears.core.collections.copy_reads import CopyReads, PinnedCopy, QueriedReads", False),
        ("from threetears.core.shared_arrow import SharedArrowPool, SharedArrowStore, read_version", False),
        ("from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange", True),
        # the in-process DuckDB L1 stays a cache backend outside the copy's modules
        ("from threetears.core.cache.duckdb import DuckDBBackend", False),
        ("import threetears.core.shared_arrow as sa\nsa.SharedArrowStore", False),
    ]
    for source, copy_module in unseen:
        assert not cursor_read_imports(source, copy_module=copy_module), source


def test_the_scan_reaches_the_copy_modules() -> None:
    """non-vacuity: the serving modules walked hold the follower and its reads, each known as a copy module."""
    modules = {path.relative_to(REPO_ROOT).as_posix(): path for path in _serving_modules()}
    for name in ("copy_follower.py", "copy_reads.py", "copy_checkpoint.py", "shared_copy.py", "volume_fulls.py"):
        path = modules.get(f"packages/core/src/threetears/core/collections/{name}")
        assert path is not None and _is_copy_module(path), name
    assert "packages/core/src/threetears/core/shared_arrow.py" in modules
    assert not any(name.startswith("packages/core/src/threetears/core/testing/") for name in modules)


def test_no_serving_module_imports_a_cursor_read_of_the_copy() -> None:
    found = {
        path.relative_to(REPO_ROOT).as_posix(): sites
        for path in _serving_modules()
        if (sites := cursor_read_imports(path.read_text(encoding="utf-8"), copy_module=_is_copy_module(path)))
    }
    assert not found, (
        "a serving module reads the copy on a cursor of the serving process (read it in a query worker: "
        f"collections.copy_queries.CopyQueries lookup, run_job, current): {found}"
    )
