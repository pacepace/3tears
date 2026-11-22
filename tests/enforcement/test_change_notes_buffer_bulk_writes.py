"""enforcement: a noted table's row trigger never inserts into the ring a row while its writer buffers the notes, and the
speed budget that proves a bulk write stays cheap is kept.

On YugabyteDB a note inserted into the ring a row is a distributed write a row: the cost of a bulk write's notes
(docs/writing-to-l3-on-yugabytedb.md: a 1000-row copy 29 ms bare, 471 ms with a ring insert a note, 103 ms buffered).
So:

- every keyed note the trigger function writes goes into the connection's buffer while the transaction is flagged,
  and into the ring only otherwise: a new note branch that inserts into the ring unconditionally fails here;
- the live speed budget (``test_a_pod_tables_buffered_copy_and_upsert_stay_within_the_budget``) is there, runs on
  YugabyteDB among the timing tests, and allows at most 5x the bare write: raising it to let a slower path through
  fails here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from threetears.core.ddl.change_notes import NOTE_BUFFER_SETTING, NOTE_BUFFER_TABLE, trigger_function_body
from threetears.core.tenancy import TENANT_BY_SCHEMA

_REPO = Path(__file__).resolve().parents[2]
_BUDGET_TEST = _REPO / "packages/core/tests/integration/test_note_buffer_live.py"
_BUDGET_NAME = "test_a_pod_tables_buffered_copy_and_upsert_stay_within_the_budget"
#: the most the budget may allow (the measured buffered write is 3.4-3.5x; the ring insert a note 6.9-16x)
_BUDGET_MOST = 5.0

#: a keyed note into the ring, as the trigger writes it
_KEYED_PUT = re.compile(r"PERFORM \S+\"threetears_change_note_put\"\(TG_TABLE_NAME, ARRAY\[")


def test_every_keyed_note_waits_in_the_buffer_while_the_writer_buffers() -> None:
    """Flips if a noted table's trigger writes a keyed note into the ring without first asking whether its writer
    buffers the transaction's notes: that note would be a distributed write a row again."""
    for columns in ((), ("state",)):
        body = trigger_function_body(
            notes_schema="platform", keys=("race", "geo"), columns=columns, tenancy=TENANT_BY_SCHEMA
        )
        lines = body.splitlines()
        puts = [index for index, line in enumerate(lines) if _KEYED_PUT.search(line)]
        assert len(puts) == 2, body  # the new row's note and the old row's
        for index in puts:
            # each sits in the ELSE of the buffer's IF, whose branch inserts the same note into the buffer
            assert lines[index - 1].strip() == "ELSE", body
            assert NOTE_BUFFER_TABLE in lines[index - 2], body
            assert f"current_setting('{NOTE_BUFFER_SETTING}', true) = '1'" in lines[index - 3], body


def test_the_bulk_write_speed_budget_is_kept() -> None:
    """Flips if the live budget test is gone, no longer runs among YugabyteDB's timing tests, or allows more than
    _BUDGET_MOST times the bare write."""
    tree = ast.parse(_BUDGET_TEST.read_text())
    tests = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
    assert _BUDGET_NAME in tests, f"{_BUDGET_TEST} lost {_BUDGET_NAME}"
    marks = {ast.unparse(decorator) for decorator in tests[_BUDGET_NAME].decorator_list}
    assert {"pytest.mark.loop_timing", "pytest.mark.yugabyte_only"} <= marks, marks
    budget = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_WRITE_BUDGET" for target in _targets(node))
    )
    assert budget is not None
    value = ast.literal_eval(budget)
    assert value <= _BUDGET_MOST, f"the bulk write budget is {value}x, over {_BUDGET_MOST}x"


def _targets(node: ast.AnnAssign | ast.Assign) -> list[ast.expr]:
    return [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
