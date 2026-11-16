"""The change-note keeper reads its whole coverage only to put a lost change log back.

``ChangeNoteKeeper._count_since``, ``._estimate`` and ``_StreamedChanges._stream`` (its keyed notes in ``_folded``) read every note of the coverage
(the current and the previous ring tables) since a cut: what a lost log held is exactly those notes, so a restore
reads them all (declared ``whole_read(kind="cold_start")``, Pace 2026-11-14). These hold that nothing else reaches
them: each is called from one place, and the chain ends at ``_restore``, which ``run_pass`` calls only when the log is
lost (no state while the ring names a log) or a restore of the ring's lost log was declared and never labelled, or at
``_put_back``, a ring beside the log's owner putting its own notes back, which ``run_pass`` calls only for such a ring
and only when the log is a restore of the one it last saw (RC5, 2026-11-16).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest

import threetears.core.collections.change_note_keeper as change_note_keeper

_SOURCE = Path(change_note_keeper.__file__)


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


def _guards_of_restore_calls(method: str = "_restore") -> list[tuple[str, ...]]:
    """the conditions guarding each ``self.<method>(...)`` call in ``run_pass``, outermost first, as source text
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
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == method:
                    found.append(guards)

    walk(run_pass.body, ())
    return found


def test_the_coverage_is_read_whole_only_by_a_restore() -> None:
    """Flips if ``_count_since``, ``_estimate`` or ``_stream`` (the coverage's whole reads) is reached from anything
    but a restore: each is called only from the restore's own steps, the chain ends at ``_restore``, and ``run_pass`` calls ``_restore``
    only when the log is lost or a declared restore of the ring's lost log was never labelled."""
    callers = _callers()
    # the count sizes the coverage before the stream (_restore) or beside it (_sized); a ring beside sizes its own
    assert callers["_count_since"] == {
        "ChangeNoteKeeper._restore",
        "ChangeNoteKeeper._sized",
        "ChangeNoteKeeper._put_back",
    }
    # the estimate sizes the whole coverage (_fitting, or _sized beside the stream, or a ring beside's) or a cut of it
    assert callers["_estimate"] == {
        "ChangeNoteKeeper._fitting",
        "ChangeNoteKeeper._sized",
        "ChangeNoteKeeper._cut",
        "ChangeNoteKeeper._put_back",
    }
    assert callers["_fitting"] == {"ChangeNoteKeeper._restore"}
    assert callers["_sized"] == {"ChangeNoteKeeper._restore"}
    assert callers["_cut"] == {"ChangeNoteKeeper._fitting", "ChangeNoteKeeper._restore"}
    assert callers["_stream"] == {"_StreamedChanges.__aiter__", "_StreamedChanges.stream"}
    # the keyed notes' COPY, folded in the restore's worker process (L1, 2026-11-15)
    assert callers["_folded"] == {"_StreamedChanges._stream"}
    assert callers["stream"] == {"_Gated._gate"}
    assert callers["_Gated"] == {"ChangeNoteKeeper._restore"}
    assert callers["_StreamedChanges"] == {"ChangeNoteKeeper._changes_since"}
    assert callers["_changes_since"] == {"ChangeNoteKeeper._restore", "ChangeNoteKeeper._put_back"}
    assert callers["_restore"] == {"ChangeNoteKeeper.run_pass"}
    assert callers["_put_back"] == {"ChangeNoteKeeper.run_pass"}
    assert _guards_of_restore_calls("_put_back") == [
        (
            "not self._keeps_log",
            "lost is not None and state.incarnation != lost and (lost in (state.restoring, state.continues))",
        )
    ]
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
    """Flips if ``_delete_band``, ``_delete_past`` or ``_vacuum_emptied`` (a ring slot's whole emptying, and its VACUUM on
    Postgres) is reached from anything but ``empty_slot``, or the keeper's pass empties any slot but the idle one (the oldest, two switches back)."""
    callers = _callers()
    assert callers["_delete_band"] == {"ChangeNoteKeeper.empty_slot"}
    assert callers["_delete_past"] == {"ChangeNoteKeeper.empty_slot"}
    assert callers["_vacuum_emptied"] == {"ChangeNoteKeeper.empty_slot"}
    assert callers["empty_slot"] == {"ChangeNoteKeeper.run_pass"}
    assert _empty_slot_calls() == [("idle", "ring.oldest")]


class _Pool:
    """a pool answering an emptying of an already empty slot, every statement it is sent recorded in order."""

    def __init__(self, version: str, *, vacuum_fails: bool = False) -> None:
        self.version = version
        self.vacuum_fails = vacuum_fails
        self.sent: list[str] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append(query)
        if query.startswith("SELECT version()"):
            return [{"v": self.version}]
        if query.startswith("SELECT count(*) AS notes"):
            return [{"notes": 0}]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append(query)
        if query.startswith("VACUUM") and self.vacuum_fails:
            raise RuntimeError("the vacuum failed")
        return "VACUUM" if query.startswith("VACUUM") else "DELETE 0"

    def acquire(self) -> Any:
        raise AssertionError("an emptying shares no session")


def _keeper(pool: _Pool) -> change_note_keeper.ChangeNoteKeeper:
    return change_note_keeper.ChangeNoteKeeper(pool, notes_schema="ring", log=cast(Any, object()))


@pytest.mark.parametrize(
    ("version", "vacuums"), [("PostgreSQL 16.4", True), ("PostgreSQL 11.2-YB-2026.1.0.0-b0", False)]
)
async def test_an_emptied_slot_is_reported_empty_only_after_its_vacuum_on_postgres(version: str, vacuums: bool) -> None:
    """Flips if empty_slot reports a slot empty (what run_pass marks ``emptied`` on, the switch's guard) before its
    VACUUM returned on Postgres, reports it empty when the VACUUM failed, or sends a VACUUM to YugabyteDB (Pace
    2026-11-15: a reused slot's dead notes are vacuumed on Postgres; on YugabyteDB a VACUUM does nothing)."""
    pool = _Pool(version)
    assert await _keeper(pool).empty_slot(1) == (True, 0)
    vacuums_sent = [query for query in pool.sent if query.startswith("VACUUM")]
    if vacuums:
        assert vacuums_sent == ['VACUUM "ring"."threetears_change_notes_1"'], pool.sent
        assert pool.sent[-1] == vacuums_sent[0], pool.sent
        failing = _Pool(version, vacuum_fails=True)
        with pytest.raises(RuntimeError, match="the vacuum failed"):
            await _keeper(failing).empty_slot(1)
    else:
        assert vacuums_sent == [], pool.sent


def test_the_ring_is_marked_emptied_only_on_empty_slot_s_report() -> None:
    """Flips if run_pass marks the ring row ``emptied`` (the switch's guard) anywhere but under empty_slot's own
    report that the slot is empty, which on Postgres comes only after its VACUUM returned."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    run_pass = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_pass"
    )
    marking = [
        node
        for node in ast.walk(run_pass)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and any(keyword.arg == "emptied" for keyword in call.keywords)
            for statement in node.body
            for call in ast.walk(statement)
        )
    ]
    assert [ast.unparse(node.test) for node in marking] == ["not ring.emptied", "done"]
    inner = marking[1]
    assert ast.unparse(inner.body[0]) == "ring = await self._update(ring, emptied=True)"
    assigned = [
        ast.unparse(node.value)
        for node in ast.walk(marking[0])
        if isinstance(node, ast.Assign) and "done" in ast.unparse(node.targets[0])
    ]
    assert assigned == ["await self.empty_slot(idle)"]
