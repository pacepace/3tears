"""A copy follower keeps each table's working state in one record, replaced whole by one owner.

**Why** (boundary review R-4, 2026-11-26). ``CopyFollower`` kept what each table was doing (following by key, a whole
read due, waiting for the writer's full, waiting for a delta file, held past its cap) across nine maps keyed by table
(``_heard``, ``_heard_counts``, ``_whole``, ``_partitions``, ``_cleared``, ``_full_after``, ``_file_waits``,
``_held_counts``, ``_whole_left_out``), and every transition had to update its own subset of them by hand; nothing
checked that a new one touched the right maps. They are one frozen record a table (``_TableState``) in one
copy-on-write map (``_states``): a transition builds the next record and puts it in one assignment, so a table's
state changes in one replacement and a reader (the status, a coroutine across an await, a thread) sees the old record
or the new one, never a mix. This rule keeps it so.

**What it refuses**, in ``copy_follower.py``:

- any of the nine maps back as an attribute of the follower (``self._whole``, ...);
- an assignment to ``self._states`` anywhere but the follower's ``__init__``, ``_put`` (one table's record) and
  ``_let_go_of_left_out`` (every record at an apply): the owner's state is replaced through them alone;
- a ``_TableState`` that is not a frozen dataclass, or one compared by value (``eq=False``: it holds Arrow batches,
  whose ``==`` compares every row).

Why a source rule and not a test of the follower: the record is the follower's own; the repo's tests reach an object
through its front door (``CLAUDE.md``, "Leading underscores are a stability contract"), and no front door shows how
many assignments a transition took.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOLLOWER: Final = _REPO_ROOT / "packages/core/src/threetears/core/collections/copy_follower.py"

#: the per-table maps the record replaced (R-4's members)
_RETIRED: Final = frozenset(
    {
        "_heard",
        "_heard_counts",
        "_whole",
        "_partitions",
        "_cleared",
        "_full_after",
        "_file_waits",
        "_held_counts",
        "_whole_left_out",
    }
)

#: the follower's methods that may assign its states
_STATE_WRITERS: Final = frozenset({"__init__", "_put", "_let_go_of_left_out"})


def _self_attribute(node: ast.AST) -> str | None:
    """the attribute name of ``self.<name>``, else None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def state_record_findings(source: str) -> list[str]:
    """what in ``source`` (a copy follower's module) breaks the one-record rule.

    :param source: the module's text
    :ptype source: str
    :return: one line a finding
    :rtype: list[str]
    """
    tree = ast.parse(source)
    found: list[str] = []
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    follower = classes.get("CopyFollower")
    record = classes.get("_TableState")
    if follower is None or record is None:
        return ["no CopyFollower or no _TableState in the module"]
    frozen = False
    for decorator in record.decorator_list:
        if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "dataclass":
            given = {keyword.arg: keyword.value for keyword in decorator.keywords}
            frozen_kw, eq_kw = given.get("frozen"), given.get("eq")
            frozen = isinstance(frozen_kw, ast.Constant) and frozen_kw.value is True
            if not (isinstance(eq_kw, ast.Constant) and eq_kw.value is False):
                found.append(f"{record.lineno}: _TableState is compared by value (give it eq=False)")
    if not frozen:
        found.append(f"{record.lineno}: _TableState is not a frozen dataclass")
    for method in follower.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(method):
            name = _self_attribute(node)
            if name in _RETIRED:
                found.append(f"{node.lineno}: CopyFollower.{method.name} uses self.{name}, a retired per-table map")
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                for each in ast.walk(target):
                    if _self_attribute(each) == "_states" and method.name not in _STATE_WRITERS:
                        found.append(
                            f"{node.lineno}: CopyFollower.{method.name} assigns self._states; put a record (_put)"
                        )
    return found


_GOOD: Final = """
@dataclass(frozen=True, slots=True, eq=False)
class _TableState:
    heard: tuple = ()

class CopyFollower:
    def __init__(self):
        self._states = {}
    def _put(self, table, state):
        self._states = {**self._states, table: state}
    def _hear(self, table, batch):
        self._put(table, self._states[table])
"""


def test_the_rule_sees_each_break() -> None:
    """non-vacuity: each shape the rule refuses is found, and the shape it allows is not."""
    assert not state_record_findings(_GOOD)
    broken = {
        "a retired map": _GOOD.replace("self._put(table, self._states[table])", "self._whole = {}"),
        "a state written outside _put": _GOOD.replace(
            "self._put(table, self._states[table])", "self._states = {table: batch}"
        ),
        "a record not frozen": _GOOD.replace("frozen=True, ", ""),
        "a record compared by value": _GOOD.replace(", eq=False", ""),
    }
    for why, source in broken.items():
        assert state_record_findings(source), why


def test_the_follower_keeps_each_tables_state_in_one_record() -> None:
    source = _FOLLOWER.read_text(encoding="utf-8")
    assert source.count("self._put(") > 20, "the scan found the follower without its record's writes"
    found = state_record_findings(source)
    assert not found, "a copy follower's table state is spread or written outside its one record:\n" + "\n".join(found)
