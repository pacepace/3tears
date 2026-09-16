"""Enforcement -- a collection that caches absences writes L3 only through the framework.

A recorded absence stops answering because every committed write advances the table's write
generation, and only two paths do that: ``BaseCollection.save_entity`` and
``BaseCollection.l2_cas_mutate``. A collection that opts into ``negative_cache_max_age`` and then
writes L3 some other way -- its own SQL through ``l3_pool``, or filling L2 directly through
``_save_to_l2`` -- commits a row that every recorded absence keeps hiding until the max age
lapses. The record says this absolutely; this walker is what makes it true.

**Two shapes are permitted, and the list is closed.** A marker only ever claims a key is ABSENT,
so a write that cannot make a key present cannot make a marker wrong:

- ``delete``, the framework's own removal path;
- an expired-row sweep, which removes rows every tier already reads as absent
  (``CoordinationCollection.sweep_expired``).

If this check ever fires on one of those, encode the shape here. Do NOT widen the check to fit
whatever the code happens to do -- the reason the exception is safe is the reason it is narrow.

AST-only, well under the 15s budget, no dynamic execution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__: list[str] = []

# this file lives at packages/core/tests/enforcement/, four levels under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

#: every package that may declare a collection. Discovered rather than listed, so a new package
#: is covered the day it exists: the sibling alignment gate hard-coded five roots, three of which
#: matched no directory and were skipped in silence.
_SRC_GLOBS = ("packages/*/src", "packages/*/*/src")

#: the opt-in this check keys on.
_OPT_IN = "negative_cache_max_age"

#: writing L3 outside the framework's own paths. ``save_to_store`` and ``delete_from_store`` are
#: the framework's seams (``save_entity`` calls the first and advances the generation after it);
#: overriding them is how a collection reaches its own store, and that is fine. What is not fine
#: is a method of the collection's own that writes and then fills L2 itself.
_L2_WRITE_HELPERS = frozenset({"_save_to_l2", "_delete_from_l2"})

#: methods whose whole purpose is removal, where a marker cannot be made wrong.
_REMOVAL_METHODS = frozenset({"delete", "delete_from_store", "sweep_expired", "sweep_expired_if_due"})


@dataclass(frozen=True)
class Violation:
    """one collection reaching L2 or L3 from a method of its own."""

    cls: str
    method: str
    source: Path
    lineno: int


def _source_roots() -> list[Path]:
    """every package source root in the repo.

    :return: the roots, sorted
    :rtype: list[Path]
    """
    roots: set[Path] = set()
    for pattern in _SRC_GLOBS:
        roots.update(path for path in _REPO_ROOT.glob(pattern) if path.is_dir())
    return sorted(roots)


def _opts_into_negative_caching(node: ast.ClassDef) -> bool:
    """whether a class body sets ``negative_cache_max_age`` to anything but ``None``.

    :param node: the class to inspect
    :ptype node: ast.ClassDef
    :return: whether it opts in
    :rtype: bool
    """
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign | ast.Assign):
            continue
        targets = [statement.target] if isinstance(statement, ast.AnnAssign) else list(statement.targets)
        if not any(isinstance(t, ast.Name) and t.id == _OPT_IN for t in targets):
            continue
        value = statement.value
        if value is None:
            continue
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        return True
    return False


def _l2_writes_in(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """line numbers where a method calls an L2 write helper on ``self``.

    :param method: the method to inspect
    :ptype method: ast.FunctionDef | ast.AsyncFunctionDef
    :return: the line numbers
    :rtype: list[int]
    """
    return [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _L2_WRITE_HELPERS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]


def collect_violations(roots: list[Path]) -> list[Violation]:
    """every negative-caching collection that fills L2 from a method of its own.

    :param roots: package source roots to walk
    :ptype roots: list[Path]
    :return: the violations
    :rtype: list[Violation]
    """
    out: list[Violation] = []
    for root in roots:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except OSError, SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or not _opts_into_negative_caching(node):
                    continue
                for method in node.body:
                    if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                        continue
                    if method.name in _REMOVAL_METHODS:
                        continue
                    for lineno in _l2_writes_in(method):
                        out.append(Violation(cls=node.name, method=method.name, source=path, lineno=lineno))
    return out


def test_a_negative_caching_collection_does_not_fill_l2_itself() -> None:
    """the generation-advance invariant, made structural.

    :return: nothing
    :rtype: None
    :raises AssertionError: when a negative-caching collection writes L2 from its own method
    """
    roots = _source_roots()
    assert roots, "no package source roots found; the walker would pass by checking nothing"
    violations = collect_violations(roots)
    formatted = "\n  ".join(
        f"{v.source}:{v.lineno}: {v.cls}.{v.method} writes L2 directly, but {v.cls} caches absences, "
        f"so a row it commits stays hidden by every recorded absence. Write through save_entity or "
        f"l2_cas_mutate."
        for v in violations
    )
    assert not violations, f"negative-caching collections writing L2 themselves:\n  {formatted}"


def test_the_walker_finds_a_planted_violation() -> None:
    """the check can fail -- otherwise it is a test that only ever passes.

    :return: nothing
    :rtype: None
    :raises AssertionError: when the walker misses a planted violation
    """
    planted = ast.parse(
        "class Denylist:\n"
        "    negative_cache_max_age = 60\n"
        "\n"
        "    async def record(self):\n"
        "        await self.l3_pool.execute('INSERT ...')\n"
        "        await self._save_to_l2(key, row)\n"
    )
    found = [
        method.name
        for node in ast.walk(planted)
        if isinstance(node, ast.ClassDef) and _opts_into_negative_caching(node)
        for method in node.body
        if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef) and _l2_writes_in(method)
    ]
    assert found == ["record"], "the walker did not see a collection filling L2 from its own method"


def test_a_removal_method_is_permitted() -> None:
    """the two permitted shapes stay permitted, and for the stated reason.

    :return: nothing
    :rtype: None
    :raises AssertionError: when a removal method is treated as a violation
    """
    assert "delete" in _REMOVAL_METHODS
    assert "sweep_expired" in _REMOVAL_METHODS
    assert "record_revocation" not in _REMOVAL_METHODS
