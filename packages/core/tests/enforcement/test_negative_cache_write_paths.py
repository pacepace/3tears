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

#: filling L2 from a method of the collection's own. ``save_to_store`` and ``delete_from_store``
#: are the framework's seams (``save_entity`` calls the first and advances the generation after
#: it); overriding them is how a collection reaches its own store, and that is fine.
_L2_WRITE_HELPERS = frozenset({"_save_to_l2", "_delete_from_l2"})

#: reaching the durable tier directly. This is the shape the rule names FIRST -- a method doing
#: ``await self.l3_pool.execute("INSERT ...")`` commits a row that every recorded absence keeps
#: hiding, and it need never touch L2 to do it.
_L3_POOLS = frozenset({"l3_pool", "required_l3_pool"})
_L3_WRITE_METHODS = frozenset(
    {"execute", "executemany", "fetch", "fetchrow", "fetchval", "upsert", "copy_records_to_table"}
)

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


def _declares_opt_in(node: ast.ClassDef) -> bool:
    """whether a class body itself sets ``negative_cache_max_age`` to anything but ``None``.

    :param node: the class to inspect
    :ptype node: ast.ClassDef
    :return: whether this class body opts in
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


def _opted_in_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """every class in a module that caches absences, including through a base class in the file.

    The opt-in is a ClassVar, so a subclass inherits it: reading only the class's own body meant a
    subclass could opt in while the write that breaks the guarantee sat on the base it inherits.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: class name -> the class, for every class that opts in itself or through a base
    :rtype: dict[str, ast.ClassDef]
    """
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    declared = {name for name, node in classes.items() if _declares_opt_in(node)}
    opted: dict[str, ast.ClassDef] = {}
    for name, node in classes.items():
        seen: set[str] = set()
        pending = [name]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in declared:
                opted[name] = node
                break
            base_node = classes.get(current)
            if base_node is None:
                continue
            pending.extend(base.id for base in base_node.bases if isinstance(base, ast.Name))
    return opted


def _methods_of(node: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """the methods a class has, its in-file base classes included.

    :param node: the class to collect from
    :ptype node: ast.ClassDef
    :param classes: every class in the file, by name
    :ptype classes: dict[str, ast.ClassDef]
    :return: the methods
    :rtype: list[ast.FunctionDef | ast.AsyncFunctionDef]
    """
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    seen: set[str] = set()
    pending = [node.name]
    while pending:
        current = pending.pop()
        if current in seen or current not in classes:
            continue
        seen.add(current)
        body = classes[current]
        out.extend(item for item in body.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef))
        pending.extend(base.id for base in body.bases if isinstance(base, ast.Name))
    return out


def _unframework_writes_in(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """line numbers where a method writes a tier itself instead of through the framework.

    Two shapes, both named by the rule: filling L2 from the collection's own method, and reaching
    the durable tier directly through ``self.l3_pool``.

    :param method: the method to inspect
    :ptype method: ast.FunctionDef | ast.AsyncFunctionDef
    :return: the line numbers
    :rtype: list[int]
    """
    out: list[int] = []
    # a local alias evades an attribute-only match: ``store = self.l3_pool`` then
    # ``store.execute(...)`` is the same write with one more line. The shipped sweep is written
    # exactly that way, so this is the spelling a violator would reach for first.
    aliases = {
        target.id
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr in _L3_POOLS
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    }
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target_node = node.func.value
        # self._save_to_l2(...) / self._delete_from_l2(...)
        if node.func.attr in _L2_WRITE_HELPERS and isinstance(target_node, ast.Name) and target_node.id == "self":
            out.append(node.lineno)
            continue
        if node.func.attr not in _L3_WRITE_METHODS:
            continue
        # self.l3_pool.execute(...)
        if (
            isinstance(target_node, ast.Attribute)
            and target_node.attr in _L3_POOLS
            and isinstance(target_node.value, ast.Name)
            and target_node.value.id == "self"
        ):
            out.append(node.lineno)
            continue
        # store = self.l3_pool; store.execute(...)
        if isinstance(target_node, ast.Name) and target_node.id in aliases:
            out.append(node.lineno)
    return out


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
            classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
            for name, node in _opted_in_classes(tree).items():
                for method in _methods_of(node, classes):
                    if method.name in _REMOVAL_METHODS:
                        continue
                    for lineno in _unframework_writes_in(method):
                        out.append(Violation(cls=name, method=method.name, source=path, lineno=lineno))
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


def _planted(source: str) -> list[str]:
    """the methods the walker flags in one planted module.

    :param source: the module source to plant
    :ptype source: str
    :return: the flagged method names
    :rtype: list[str]
    """
    tree = ast.parse(source)
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return [
        method.name
        for node in _opted_in_classes(tree).values()
        for method in _methods_of(node, classes)
        if method.name not in _REMOVAL_METHODS and _unframework_writes_in(method)
    ]


def test_the_walker_finds_a_direct_l3_write() -> None:
    """the shape the rule names FIRST, planted on its own.

    Asserted without an L2 write beside it: a fixture carrying both passes on the L2 line alone,
    which is how this half of the walker read as covered while catching nothing.

    :return: nothing
    :rtype: None
    :raises AssertionError: when the walker misses a direct durable write
    """
    found = _planted(
        "class Denylist:\n"
        "    negative_cache_max_age = 60\n"
        "\n"
        "    async def record(self):\n"
        "        await self.l3_pool.execute('INSERT ...')\n"
    )
    assert found == ["record"], "a negative-caching collection writing L3 with its own SQL passed the gate"


def test_the_walker_finds_an_l2_fill() -> None:
    """the other shape, also on its own.

    :return: nothing
    :rtype: None
    :raises AssertionError: when the walker misses a collection filling L2 itself
    """
    found = _planted(
        "class Denylist:\n"
        "    negative_cache_max_age = 60\n"
        "\n"
        "    async def record(self):\n"
        "        await self._save_to_l2(key, row)\n"
    )
    assert found == ["record"], "a negative-caching collection filling L2 itself passed the gate"


def test_the_walker_follows_the_opt_in_through_a_base_class() -> None:
    """the opt-in is inherited, so the write that breaks it may sit on the base.

    :return: nothing
    :rtype: None
    :raises AssertionError: when the walker checks only the class's own body
    """
    found = _planted(
        "class Base:\n"
        "    async def record(self):\n"
        "        await self.l3_pool.execute('INSERT ...')\n"
        "\n"
        "\n"
        "class Denylist(Base):\n"
        "    negative_cache_max_age = 60\n"
    )
    assert found == ["record"], "an inherited write escaped the gate"


def test_a_collection_that_does_not_opt_in_is_not_examined() -> None:
    """the check is scoped to the opt-in, not to every collection in the repo.

    :return: nothing
    :rtype: None
    :raises AssertionError: when a collection without the opt-in is flagged
    """
    found = _planted(
        "class Plain:\n"
        "    negative_cache_max_age = None\n"
        "\n"
        "    async def record(self):\n"
        "        await self.l3_pool.execute('INSERT ...')\n"
    )
    assert found == [], "a collection that caches no absences was flagged"


def test_the_walker_sees_through_a_local_alias() -> None:
    """the spelling the shipped sweep uses, and the one a violator reaches for first.

    :return: nothing
    :rtype: None
    :raises AssertionError: when an aliased durable write escapes the walker
    """
    found = _planted(
        "class Denylist:\n"
        "    negative_cache_max_age = 60\n"
        "\n"
        "    async def record(self):\n"
        "        store = self.l3_pool\n"
        "        await store.execute('INSERT ...')\n"
    )
    assert found == ["record"], "a durable write through a local alias escaped the gate"


def test_a_removal_method_is_permitted() -> None:
    """the two permitted shapes stay permitted, and for the stated reason.

    :return: nothing
    :rtype: None
    :raises AssertionError: when a removal method is treated as a violation
    """
    assert "delete" in _REMOVAL_METHODS
    assert "sweep_expired" in _REMOVAL_METHODS
    assert "record_revocation" not in _REMOVAL_METHODS
