"""The run-marker and moves tables' declarations name the tables and keys their statements write.

``RUN_MARKERS_SCHEMA`` and ``TENANT_MOVES_SCHEMA`` spell their names and keys where they are declared, so the
no-whole-reads walk, which reads a declaration only from its own module, keys every read of them; the statements that
write and read the tables name them by ``RUN_MARKERS_TABLE`` / ``TENANT_MOVES_TABLE`` and their column constants. Each
test flips when the two spellings part.
"""

from __future__ import annotations

from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.table_restore import StampedTable, moves_relation
from threetears.core.collections.tenant_moves import TENANT_MOVES_SCHEMA
from threetears.core.ddl.run_markers import RUN_MARKER_COLUMNS, RUN_MARKERS_TABLE
from threetears.core.ddl.tenant_moves import (
    TENANT_MOVE_COLUMNS,
    TENANT_MOVES_KEY,
    TENANT_MOVES_TABLE,
)
from threetears.core.tenancy import Tenancy


def test_the_run_marker_declaration_names_the_table_and_key_its_statements_write() -> None:
    """Flips if the run-marker table's declaration and its statements' constants name another table or key."""
    assert RUN_MARKERS_SCHEMA.name == RUN_MARKERS_TABLE
    assert tuple(RUN_MARKERS_SCHEMA.pk_columns) == RUN_MARKER_COLUMNS
    assert tuple(column.name for column in RUN_MARKERS_SCHEMA.columns) == RUN_MARKER_COLUMNS


def test_the_moves_declaration_names_the_table_and_key_its_statements_write() -> None:
    """Flips if the moves table's declaration and its statements' constants name another table or key."""
    assert TENANT_MOVES_SCHEMA.name == TENANT_MOVES_TABLE
    assert tuple(TENANT_MOVES_SCHEMA.pk_columns) == TENANT_MOVES_KEY
    assert tuple(column.name for column in TENANT_MOVES_SCHEMA.columns) == TENANT_MOVE_COLUMNS


def test_a_restore_reads_a_stamped_tables_moves_from_the_moves_table_in_its_schema() -> None:
    """Flips if ``moves_relation`` names another table than the moves table its statements write, in the moved table's
    schema."""
    table = StampedTable(
        schema="owner",
        table="runs",
        bucket="epoch_bucket",
        buckets=16,
        keys=("id",),
        instants=frozenset(),
        tenancy=Tenancy("undeclared"),
    )
    assert moves_relation(table) == f"owner.{TENANT_MOVES_TABLE}"
