"""the cleanup progress table: how far quiet-time cleanup has cleared each scope of each versioned table.

Append-only L3 (``design-append-only-l3.md``, decision 5; chunk 6). Cleanup (:mod:`~threetears.core.collections.
quiet_cleanup`) visits every version of a versioned table once, a scope at a time, in epoch order, through the table's
``(scope, epoch, key...)`` index. Each version it visits clears the older versions of its key (and itself, for a
"gone" marker); a key's newest live version stays and is never visited again. So cleanup must remember where it got
to, or each round would read the scope's surviving versions again from its start: a whole read in batches. This table
is that memory: one row a ``(table, scope)``, the epoch below which every version of the scope was visited.

**Its key**: ``(table_name, scope)``; a round reads one table's rows by ``table_name`` (one row a scope, ~60 for reports'
states). **Its rows** are written by cleanup alone, one upsert a scope finished, only while nothing is loading. They
are cleanup's own bookkeeping, never data: a restore needs none of them, and a lost row costs one revisit of a scope.

A consumer that declares versioned tables declares this table beside them, in the same schema (as the run-marker
table, :data:`~threetears.core.collections.run_markers.RUN_MARKERS_SCHEMA`): its DDL is
``CLEANUP_PROGRESS_SCHEMA.create_statements(yugabyte=...)``.
"""

from __future__ import annotations

from typing import Final

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA

__all__ = ["CLEANUP_PROGRESS_SCHEMA", "CLEANUP_PROGRESS_TABLE"]

#: the cleanup progress table, in the same schema as the versioned tables it tracks
CLEANUP_PROGRESS_TABLE: Final[str] = "threetears_cleanup_progress"

#: one row a (table, scope): the epoch below which cleanup visited every version of the scope
CLEANUP_PROGRESS_SCHEMA: Final = TableSchema(
    name=CLEANUP_PROGRESS_TABLE,
    primary_key=("table_name", "scope"),
    columns=[
        Column("table_name", STRING_TYPE),
        Column("scope", STRING_TYPE),
        Column("cleaned_to", BIGINT_TYPE),
    ],
    # it lives among the owner's versioned tables, in the owner's own schema, so its tenancy is that schema
    tenancy=TENANT_BY_SCHEMA,
    l1_eviction=Declared(Policy.EVICTABLE, "cleanup reads its progress by key from L3; a cache of it may be dropped"),
)
