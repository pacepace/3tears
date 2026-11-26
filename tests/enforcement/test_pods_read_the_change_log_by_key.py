"""Only the change log's owner reads it by sequence; every other caller reads it by the table's keys.

A message get by sequence (``NatsChangeLog.at``, ``next_after``: ``$JS.API.STREAM.MSG.GET``) carries its subject in
the request body, where no grant can narrow it, so ``threetears.nats.subject_permissions`` grants no pod that verb on
the shared epoch change log (a body-keyed read reaches every tenant's keys). A pod asking so is refused and waits out
its deadline on every try: the beside-keeper's scrub did (fixed 2026-11-24 by reading ``changes_after``), and so did a
writer's delta-file recreate through ``change`` (a local replay, 2026-11-24). A pod reads by key: ``records_after``
(a named consumer whose filter rides in its create subject) or ``last`` (a direct get naming the key's subject).

Every ``.at(`` on a change log (a receiver spelled with ``log``), every ``.next_after(`` and every ``.epoch_at(`` in a
package's ``src`` must sit in a function :data:`_OWNER_ONLY` names, each with why only the log's owner (the hub)
reaches it. Tests are not read (a test reads a fake log by sequence to look at what a write appended).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_REPO: Final = Path(__file__).resolve().parents[2]
_PACKAGES: Final = _REPO / "packages"

#: (path under packages/, enclosing ``Class.function``) -> why only the change log's owner runs it
_OWNER_ONLY: Final = {
    (
        "epoch/src/threetears/epoch/generation.py",
        "EpochGenerationSource.epoch_at",
    ): "the owner's scrub read, called only from LogKeeper._epochs_between",
    (
        "core/src/threetears/core/collections/log_keeper.py",
        "LogKeeper._epochs_between",
    ): "runs only under keeps_log=True, the hub's keeper; a keeper beside the log scrubs by changes_after",
    (
        "core/src/threetears/core/collections/quiet_cleanup.py",
        "log_window_floor",
    ): "the hub's broker's own bound on the cleanup floor (aibots.hub.broker.proxy), over the log it writes; no pod "
    "calls it, a pod's QuietCleanup is handed its floor",
}

#: methods that read the log by sequence whatever their receiver
_BY_SEQUENCE: Final = frozenset({"next_after", "epoch_at"})


class _BySequence(ast.NodeVisitor):
    """each by-sequence read of the change log, with the ``Class.function`` it sits in."""

    def __init__(self) -> None:
        self.found: list[tuple[str, int, str]] = []
        self._scope: list[str] = []

    def _enter(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast's visitor name
        self._enter(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast's visitor name
        self._enter(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast's visitor name
        self._enter(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast's visitor name
        func = node.func
        if isinstance(func, ast.Attribute):
            on_a_log = func.attr == "at" and "log" in ast.unparse(func.value).lower()
            if on_a_log or func.attr in _BY_SEQUENCE:
                self.found.append((".".join(self._scope[-2:]), node.lineno, ast.unparse(node)))
        self.generic_visit(node)


def by_sequence_reads(source: str) -> list[tuple[str, int, str]]:
    """the by-sequence reads of the change log in ``source``: (``Class.function``, line, the call).

    :param source: a module's text
    :ptype source: str
    :return: each read
    :rtype: list[tuple[str, int, str]]
    """
    visitor = _BySequence()
    visitor.visit(ast.parse(source))
    return visitor.found


def test_the_rule_sees_each_shape_the_code_had() -> None:
    """each shape a pod path held before 2026-11-24 is seen; a keyed read and another type's ``at`` are not."""
    seen = (
        "class R:\n    async def change(self, t, s):\n        return await self._log().at(s)",
        "class K:\n    async def scrub(self):\n        return await self._log.epoch_at(3)",
        "class R:\n    async def _next(self, k, s):\n        return await log.next_after(k, s)",
        "async def f(bus):\n    return await bus.change_log('b').at(4)",
    )
    for source in seen:
        assert by_sequence_reads(source), source
    unseen = (
        "class R:\n    async def change(self, t, s):\n        return await self._log().records_after(t, s - 1, s)",
        "class R:\n    async def newest(self, k):\n        return await log.last(k)",
        "AnalystSkill.at('skills/control')",
        "row.at(0.95)",
    )
    for source in unseen:
        assert not by_sequence_reads(source), source


def test_only_the_change_logs_owner_reads_it_by_sequence() -> None:
    found = [
        f"{relative}:{line} in {where}: {call}"
        for path in sorted(_PACKAGES.glob("**/src/**/*.py"))
        for relative in (str(path.relative_to(_PACKAGES)),)
        for where, line, call in by_sequence_reads(path.read_text(encoding="utf-8"))
        if (relative, where) not in _OWNER_ONLY
    ]
    assert not found, (
        "a by-sequence read of the change log outside the owner's own functions: no pod is granted the message get, "
        f"so read by the table's keys (records_after / last), or add the owner-only function with why: {found}"
    )


def test_every_owner_only_site_still_reads_by_sequence() -> None:
    """an entry whose function no longer reads by sequence (renamed, moved, fixed) is dropped, so the list only
    shrinks."""
    held = {
        (str(path.relative_to(_PACKAGES)), where)
        for path in sorted(_PACKAGES.glob("**/src/**/*.py"))
        for where, _, _ in by_sequence_reads(path.read_text(encoding="utf-8"))
    }
    assert set(_OWNER_ONLY) <= held, f"owner-only entries that read nothing by sequence: {set(_OWNER_ONLY) - held}"
