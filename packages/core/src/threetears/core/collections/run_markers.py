"""the run-marker table's declaration: one row a whole run of a versioned table deleted (:mod:`~threetears.core.ddl.run_markers`).

A consumer that declares versioned tables (``TableSchema(row_epoch=row_versions(...))``) and deletes whole runs of them
(``delete_prefixes``) declares this table beside them, in the same schema, so it is created with them: its DDL is
``RUN_MARKERS_SCHEMA.create_statements(yugabyte=...)``. Its rows are written by the store running the delete (the hub's
broker on the L3 rail, 3tears' ``SqlL3Backend`` on a direct connection), never by a collection of its own.

**Its key**: ``(table_name, scope, prefix, epoch)``, every marker of a run of a table together, oldest first (on
YugabyteDB the first column hashed, the rest ascending). A read of "is this run gone, and since which epoch" is one
probe of the key's first three columns; the restore of one scope finds its markers by ``(table_name, scope)``. The
table holds a row a run deleted (23 on the night of 2026-11-03), so one tablet a table is no hot spot.
"""

from __future__ import annotations

from typing import Final

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA

__all__ = ["RUN_MARKERS_SCHEMA"]

#: the run-marker table: the table a run of was deleted, the run's scope and key prefix as text, the delete's epoch
RUN_MARKERS_SCHEMA: Final = TableSchema(
    # the name and key spelled here, not by RUN_MARKERS_TABLE and its column constant: the no-whole-reads walk reads a
    # declaration's names only from its own module, and judges every read of the table by this key
    # (test_run_markers_schema holds them equal)
    name="threetears_run_markers",
    primary_key=("table_name", "scope", "prefix", "epoch"),
    columns=[
        Column("table_name", STRING_TYPE),
        Column("scope", STRING_TYPE),
        Column("prefix", STRING_TYPE),
        Column("epoch", BIGINT_TYPE),
    ],
    # it lives among the owner's versioned tables, in the owner's own schema, so its tenancy is that schema
    tenancy=TENANT_BY_SCHEMA,
    l1_eviction=Declared(Policy.EVICTABLE, "a run's markers are read by key from L3; a cache of them may be dropped"),
)
