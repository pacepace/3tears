"""The change-note keeper reads its whole coverage only to put a lost change log back.

``ChangeNoteKeeper._count_since``, ``._estimate`` and ``_StreamedChanges._stream`` read every note of the coverage
(the current and the previous ring tables) since a cut: what a lost log held is exactly those notes, so a restore
reads them all (declared ``whole_read(kind="cold_start")``, Pace 2026-11-14). These hold that nothing else reaches
them: each is called from one place, and the chain ends at ``_restore``, which ``run_pass`` calls only when the log is
lost (no state while the ring names a log) or a restore of the ring's lost log was declared and never labelled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import threetears.core.collections.change_notes as change_notes

_SOURCE = Path(change_notes.__file__)


def _callers() -> dict[str, set[str]]:
    """each attribute or name called in the module, to the qualified functions calling it."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    callers: dict[str, set[str]] = {}

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, f"{scope}.{child.name}" if scope else child.name)
                continue
            if isinstance(child, ast.Call):
                func = child.func
                name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
                if name:
                    callers.setdefault(name, set()).add(scope)
            walk(child, scope)

    walk(tree, "")
    return callers


def _guards_of_restore_calls() -> list[tuple[str, ...]]:
    """the conditions guarding each ``self._restore(...)`` call in ``run_pass``, outermost first, as source text
    (an else branch's condition prefixed ``not``)."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    run_pass = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_pass"
    )
    found: list[tuple[str, ...]] = []

    def walk(statements: list[ast.stmt], guards: tuple[str, ...]) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk(statement.body, (*guards, ast.unparse(statement.test)))
                walk(statement.orelse, (*guards, f"not {ast.unparse(statement.test)}"))
                continue
            for call in ast.walk(statement):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_restore":
                    found.append(guards)

    walk(run_pass.body, ())
    return found


def test_the_coverage_is_read_whole_only_by_a_restore() -> None:
    """Flips if ``_count_since``, ``_estimate`` or ``_stream`` (the coverage's whole reads) is reached from anything
    but a restore: each is called from one place, the chain ends at ``_restore``, and ``run_pass`` calls ``_restore``
    only when the log is lost or a declared restore of the ring's lost log was never labelled."""
    callers = _callers()
    assert callers["_count_since"] == {"ChangeNoteKeeper._restore"}
    assert callers["_estimate"] == {"ChangeNoteKeeper._fitting"}
    assert callers["_fitting"] == {"ChangeNoteKeeper._restore"}
    assert callers["_stream"] == {"_StreamedChanges.__aiter__"}
    assert callers["_StreamedChanges"] == {"ChangeNoteKeeper._changes_since"}
    assert callers["_changes_since"] == {"ChangeNoteKeeper._restore"}
    assert callers["_restore"] == {"ChangeNoteKeeper.run_pass"}
    assert sorted(_guards_of_restore_calls()) == sorted(
        [
            ("state is None", "not ring.log_incarnation is None"),
            ("state.restoring is not None and state.restoring == ring.log_incarnation and (state.continues is None)",),
        ]
    )


def _empty_slot_calls() -> list[tuple[str, str]]:
    """each ``self.empty_slot(...)`` call in ``run_pass``: its argument, and the value that argument was bound to."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    run_pass = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_pass"
    )
    bound = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(run_pass)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    return [
        (ast.unparse(call.args[0]), bound.get(ast.unparse(call.args[0]), ""))
        for call in ast.walk(run_pass)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "empty_slot"
    ]


def test_a_ring_slot_is_emptied_whole_only_when_idle() -> None:
    """Flips if ``_delete_band`` or ``_delete_past`` (a ring slot's whole emptying) is reached from anything but
    ``empty_slot``, or the keeper's pass empties any slot but the idle one (the oldest, two switches back)."""
    callers = _callers()
    assert callers["_delete_band"] == {"ChangeNoteKeeper.empty_slot"}
    assert callers["_delete_past"] == {"ChangeNoteKeeper.empty_slot"}
    assert callers["empty_slot"] == {"ChangeNoteKeeper.run_pass"}
    assert _empty_slot_calls() == [("idle", "ring.oldest")]
