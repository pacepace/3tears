"""Unit: a copy is cut by what its connection carries, and a whole-marked write is noted by one trigger.

A copy cut at a thousand rows cost two and a half times one ``COPY`` of the rows on YugabyteDB: a direct
connection now streams one ``COPY``, the rail sizes each request to the cap its hub announces, and a hub
that announces none keeps the fixed batches. In a transaction marked whole the row trigger's ``WHEN`` skips
every row (no function call a row) and a statement trigger writes the one note.
"""

from __future__ import annotations

from typing import Any

from threetears.core.backends.protocol import copy_bounds
from threetears.core.ddl.change_notes import (
    NOTE_TRIGGER,
    NOTE_TRIGGERS,
    NOTE_TRUNCATE_TRIGGER,
    NOTE_WHOLE_SETTING,
    NOTE_WHOLE_TRIGGER,
    note_trigger_sql,
)


class _Streaming:
    """asyncpg's shape: a copy streams, nothing measures its request."""


class _Rail:
    def __init__(self, budget: int | None) -> None:
        self._budget = budget

    def copy_budget_bytes(self) -> int | None:
        return self._budget

    def copy_request_bytes(self, table: str, columns: Any, records: Any) -> int:
        return 0


def test_a_streaming_connection_copies_every_row_at_once() -> None:
    """Flips if a direct copy is cut into batches nothing on its path needs."""
    assert copy_bounds(_Streaming(), 50_000) == (50_000, None)


def test_the_rail_sizes_a_copy_to_its_hubs_cap() -> None:
    """Flips if a copy on the rail is cut by rows once the hub has announced its cap."""
    assert copy_bounds(_Rail(4 * 1024 * 1024), 50_000) == (50_000, 4 * 1024 * 1024)


def test_the_rail_of_a_hub_that_announces_no_cap_keeps_the_callers_batches() -> None:
    """Flips if a copy to an older hub is sent in requests it never sized for."""
    assert copy_bounds(_Rail(None), 50_000) == (None, None)


def test_a_noted_table_gets_every_trigger_and_the_whole_ones_complement_the_row_one() -> None:
    """Flips if a whole-marked write is noted by both triggers, by neither, or by a function call a row."""
    statements = note_trigger_sql(notes_schema="notes", table_schema="s", table="t")
    assert tuple(statements) == NOTE_TRIGGERS
    row, whole = statements[NOTE_TRIGGER], statements[NOTE_WHOLE_TRIGGER]
    setting = f"current_setting('{NOTE_WHOLE_SETTING}', true)"
    assert f"FOR EACH ROW WHEN ({setting} IS DISTINCT FROM '1')" in row
    assert f"FOR EACH STATEMENT WHEN ({setting} = '1')" in whole
    assert "AFTER INSERT OR UPDATE OR DELETE" in whole
    assert "AFTER TRUNCATE" in statements[NOTE_TRUNCATE_TRIGGER]
