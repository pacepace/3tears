"""The note ring's tables carry an index led by the table and then the time, covering what the reconciler reads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from threetears.core.collections.change_notes import noted_rows_since
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, change_notes_ddl, note_table


class _Recording:
    """a pool that records each statement it is asked and answers none."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        self.asked.append(query)
        return []


def test_every_note_table_has_a_covering_table_and_time_index() -> None:
    """Flips if a ring table loses the index noted_rows_since is answered from, or it stops carrying what it reads."""
    statements = change_notes_ddl("platform")
    for index in range(NOTE_RING_SIZE):
        wanted = (
            f'CREATE INDEX IF NOT EXISTS "{note_table(index)}_table" ON "platform"."{note_table(index)}" '
            "(table_name, noted_at) INCLUDE (key_text, keys, columns, tenant)"
        )
        assert wanted in statements


async def test_the_ask_is_one_statement_reading_the_ring_and_both_covered_tables() -> None:
    """Flips if noted_rows_since goes back to a statement for the ring row and another for the notes."""
    pool = _Recording()
    await noted_rows_since(pool, notes_schema="platform", table="results", since=datetime.now(UTC), cap=10)
    (sql,) = pool.asked
    assert sql.count("SELECT") >= NOTE_RING_SIZE + 2
    assert "threetears_change_note_ring" in sql
    for index in range(NOTE_RING_SIZE):
        assert note_table(index) in sql
