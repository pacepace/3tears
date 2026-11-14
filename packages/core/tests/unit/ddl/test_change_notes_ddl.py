"""the change-note ring's DDL, as ``_install_change_notes`` runs it."""

from __future__ import annotations

from datetime import timedelta

from threetears.core.ddl.change_notes import change_notes_ddl
from threetears.core.ddl.index_health import ddl_body


def test_install_change_notes_rebuilds_no_unique_index() -> None:
    """the proof of ``_install_change_notes``' declared ``schema_change`` call into ``rebuild_unusable_indexes``: it
    hands the repair the ring's own statements as the definitions, and a declared statement that is not UNIQUE needs
    no duplicate-key check, the only read of a table's rows the repair makes. Every ring index is a plain
    ``CREATE INDEX``, so the install never reads a ring table whole; a unique ring index would flip this."""
    statements = change_notes_ddl("notes_schema", fence=timedelta(seconds=5))
    # each read past the ``-- purpose:`` lines declaring its index, as the repair reads it
    bodies = [ddl_body(s).upper() for s in statements]
    indexes = [s for s in bodies if s.startswith(("CREATE INDEX", "CREATE UNIQUE INDEX"))]
    assert indexes, "the ring declares its indexes"
    assert not [s for s in indexes if s.startswith("CREATE UNIQUE ")]
