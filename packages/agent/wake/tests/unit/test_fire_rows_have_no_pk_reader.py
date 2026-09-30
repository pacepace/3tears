"""No production code reads a ``wake_fires`` row by primary key.

Every ``wake_fires`` UPDATE (``link_started_conversation``, ``finalize_success``,
``finalize_failed``, ``reap_stale_dispatching``) bypasses ``save_entity`` and evicts
nothing. That is correct only while no reader is served a cached fire row, which is
the claim the ``threetears.agent.wake.collections`` module docstring makes under
"Fire rows are the exception". A by-pk read of a fire (``get``, ``ensure``,
``reload_entity``, the sync cache reads, a subscript, ``l2_cas_mutate``) would be
served whatever L1 or L2 last held, and those UPDATEs would leave it stale with
nothing to evict it. This test fails on the first such reader, so the eviction is
added when the reader is.

It is a static walk over every package's ``src`` tree. It cannot know types, so a
receiver counts as a fire collection when its name says so (``fires``,
``fire_collection``, ``self._fires``) or when the call is ``self.<reader>`` inside
:class:`WakeFireCollection` itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]

# every BaseCollection entry point that answers a row by primary key from a cache tier
_PK_READERS = frozenset(
    {"get", "ensure", "reload_entity", "get_field_sync", "get_row_sync", "l2_cas_mutate"},
)

_FIRE_COLLECTION_CLASS = "WakeFireCollection"

_DOCSTRING_POINTER = (
    "a wake_fires row is read by primary key; the fire UPDATEs in "
    "threetears/agent/wake/collections.py evict nothing, on the grounds that no code does this "
    "(module docstring, 'Fire rows are the exception'). Route every wake_fires UPDATE through "
    "WakeFireCollection.bypassing_write before adding this reader"
)


def _receiver_name(node: ast.expr) -> str | None:
    """the last identifier of a call's receiver: ``fires`` for ``fires``, ``_fires`` for ``self._fires``.

    :param node: the receiver expression
    :ptype node: ast.expr
    :return: the identifier, or ``None`` for any other expression
    :rtype: str | None
    """
    name: str | None = None
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    return name


def _is_fire_receiver(node: ast.expr) -> bool:
    """whether the receiver's name marks it as a fire collection.

    :param node: the receiver expression
    :ptype node: ast.expr
    :return: ``True`` for ``fires``, ``fire_collection``, ``self._fires`` and the like
    :rtype: bool
    """
    name = _receiver_name(node)
    return name is not None and "fire" in name.lower() and name != "self"


class _FireReadFinder(ast.NodeVisitor):
    """collect by-pk reads of a fire collection, and every fire-collection call seen."""

    def __init__(self) -> None:
        """start with nothing found."""
        self.offences: list[tuple[int, str]] = []
        self.fire_calls = 0
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """track the enclosing class, so ``self.get`` inside WakeFireCollection counts.

        :param node: the class
        :ptype node: ast.ClassDef
        """
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """flag ``<fire receiver>.<pk reader>(...)`` and ``self.<pk reader>`` in WakeFireCollection.

        :param node: the call
        :ptype node: ast.Call
        """
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = func.value
            in_fire_class = bool(self._class_stack) and self._class_stack[-1] == _FIRE_COLLECTION_CLASS
            is_self = isinstance(receiver, ast.Name) and receiver.id == "self"
            if _is_fire_receiver(receiver) or (in_fire_class and is_self):
                self.fire_calls += 1
                if func.attr in _PK_READERS:
                    self.offences.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """flag ``<fire receiver>[key]``, BaseCollection's by-pk ``__getitem__``.

        :param node: the subscript
        :ptype node: ast.Subscript
        """
        if isinstance(node.ctx, ast.Load) and _is_fire_receiver(node.value):
            self.offences.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)


def _find(source: str) -> _FireReadFinder:
    """walk one module's source.

    :param source: python source
    :ptype source: str
    :return: the finder, after the walk
    :rtype: _FireReadFinder
    """
    finder = _FireReadFinder()
    finder.visit(ast.parse(source))
    return finder


def _src_files() -> list[Path]:
    """every python file under a package's ``src`` tree.

    :return: the files
    :rtype: list[Path]
    """
    return sorted(path for path in (_REPO_ROOT / "packages").rglob("src/**/*.py") if "/tests/" not in path.as_posix())


def test_the_walker_flags_a_by_pk_fire_read() -> None:
    """positive control: each shape of by-pk fire read is found."""
    source = (
        "class WakeFireCollection:\n"
        "    async def m(self):\n"
        "        return await self.get((c, f))\n"
        "async def f(fires, holder):\n"
        "    await fires.get((c, f))\n"
        "    await holder._fires.ensure((c, f))\n"
        "    return fires[(c, f)]\n"
    )
    assert len(_find(source).offences) == 4


def test_the_walker_leaves_scans_and_writes_alone() -> None:
    """negative control: the fire collection's L3 scans and targeted writes are not reads by pk."""
    source = (
        "async def f(fires):\n"
        "    await fires.latest_for_schedule(c, s)\n"
        "    await fires.finalize_success(c, f)\n"
        "    await fires.list_for_conversation(c)\n"
    )
    finder = _find(source)
    assert finder.offences == []
    assert finder.fire_calls == 3


def test_no_production_code_reads_a_fire_row_by_primary_key() -> None:
    """the claim the fire UPDATEs' missing eviction rests on holds across every package."""
    files = _src_files()
    assert files, f"no source files found under {_REPO_ROOT / 'packages'}"
    offences: list[str] = []
    fire_calls = 0
    for path in files:
        finder = _find(path.read_text(encoding="utf-8"))
        fire_calls += finder.fire_calls
        offences.extend(f"{path.relative_to(_REPO_ROOT)}:{line}: {text}" for line, text in finder.offences)
    # non-vacuity: the walk recognised the fire collection's real callers (tick, webhook, dispatch)
    assert fire_calls >= 5, f"the walk saw only {fire_calls} fire-collection calls; the receiver heuristic is broken"
    assert not offences, _DOCSTRING_POINTER + ":\n" + "\n".join(offences)
