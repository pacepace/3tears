"""migration-yugabyte-safety enforcement domain — thin pytest adapter.

every alembic-style migration file in a 3tears-ecosystem repo must
ship yugabyte-safe SQL: no DDL/DML mixing inside ``DO`` blocks (M-1),
backfill ``UPDATE`` statements must carry a replay-guard predicate
(M-2), bare ``CREATE TABLE`` / ``CREATE INDEX`` / ``ADD COLUMN``
without ``IF NOT EXISTS`` are non-idempotent (M-3), composite-PK adds
must preserve a ``UNIQUE`` index on the legacy id column for inbound
FK pinning (M-4), and ``TRUNCATE TABLE`` is forbidden inside
migrations (M-5). the underlying walker logic lives in
:mod:`threetears.core.data.migrations.enforcement` so every consumer
exercises the same evaluator.

per-repo configuration goes through :class:`MigrationYugabyteConfig`;
:func:`run_migration_enforcement` is the pytest-friendly entry point
that orchestrates the walker, applies exemptions, emits the report,
and fails in strict mode.
"""

from threetears.enforcement.migration_yugabyte_safety.config import (
    MigrationYugabyteConfig,
)
from threetears.enforcement.migration_yugabyte_safety.runner import (
    run_migration_enforcement,
)
from threetears.enforcement.migration_yugabyte_safety.walkers import (
    find_migration_violations,
)

__all__ = [
    "MigrationYugabyteConfig",
    "find_migration_violations",
    "run_migration_enforcement",
]
