"""The note ring's tables carry an index led by the table and then the time, covering what the reconciler reads."""

from __future__ import annotations

from threetears.core.collections.change_notes import _noted_rows_sql
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, change_notes_ddl, note_table


def test_every_note_table_has_a_covering_table_and_time_index() -> None:
    """Flips if a ring table loses the index noted_rows_since is answered from, or it stops carrying what it reads."""
    statements = change_notes_ddl("platform")
    for index in range(NOTE_RING_SIZE):
        wanted = (
            f'CREATE INDEX IF NOT EXISTS "{note_table(index)}_table" ON "platform"."{note_table(index)}" '
            "(table_name, noted_at) INCLUDE (key_text, keys, columns, tenant)"
        )
        assert wanted in statements


def test_the_ask_is_one_statement_reading_the_ring_and_both_covered_tables() -> None:
    """Flips if noted_rows_since goes back to a statement for the ring row and another for the notes."""
    sql = _noted_rows_sql('"platform"')
    assert sql.count("SELECT") >= NOTE_RING_SIZE + 2
    assert "threetears_change_note_ring" in sql
    for index in range(NOTE_RING_SIZE):
        assert note_table(index) in sql
