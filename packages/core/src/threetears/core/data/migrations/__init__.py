"""
canonical migration runner public surface.

this package replaces the previous single-file
``threetears.core.data.migrations`` module. the :class:`MigrationRunner`
now composes per-package migration registrations across the platform
rather than each package owning a standalone runner. see the task shard
``migrations-task-01`` for the motivation.

public API:

- :class:`MigrationRunner` — composes registered packages and applies
  pending migrations against one database session bound to a schema,
  holding the database-wide DDL lock for the whole run.
- :class:`MigrationSession` / :class:`ConnectionSession` — the
  one-connection store a run needs, and the wrapper that makes one from
  a single connection.
- :func:`database_ddl_lock` — the lock every DDL job in a database
  takes, one job per database at a time (see
  :mod:`~threetears.core.data.migrations.ddl_lock`).
- :class:`PackageMigrations` — per-package registration of versioned
  async migration callables, with optional ``depends_on`` edges.
- :class:`MigrationScope` — PLATFORM vs AGENT enum.
- :func:`render_migration_template` — renders the blessed template for
  authoring a new migration module.
- error types: :class:`MigrationError`, :class:`DuplicateVersionError`,
  :class:`MissingDependencyError`, :class:`MigrationFailedError`,
  :class:`SessionRequiredError`, :class:`DdlLockError`,
  :class:`DdlLockTimeoutError`, :class:`DdlLockReleaseError`.
"""

from __future__ import annotations

from threetears.core.data.migrations.ddl_lock import (
    DDL_LOCK_NAMESPACE,
    DdlLockLease,
    DdlLockPolicy,
    database_ddl_lock,
    ddl_lock_key,
)
from threetears.core.data.migrations.drift import (
    DriftReport,
    diff_expected_live,
    parse_ddl_to_expected,
    snapshot_live_schema,
)
from threetears.core.data.migrations.errors import (
    DdlLockError,
    DdlLockReleaseError,
    DdlLockTimeoutError,
    DuplicateVersionError,
    MigrationError,
    MigrationFailedError,
    MissingDependencyError,
    SessionRequiredError,
)
from threetears.core.data.migrations.helpers import (
    InboundFk,
    MigrationStore,
    add_check_constraint,
    add_column_with_backfill,
    add_index,
    add_partition_column,
    replace_check_constraint,
    replace_primary_key,
)
from threetears.core.data.migrations.preview import (
    CapturedStatement,
    PreviewStore,
)
from threetears.core.data.migrations.registry import (
    MigrationFunc,
    PackageMigrations,
)
from threetears.core.data.migrations.runner import MigrationRunner
from threetears.core.data.migrations.scope import MigrationScope
from threetears.core.data.migrations.session import (
    ConnectionSession,
    MigrationSession,
    SqlConnection,
)
from threetears.core.data.migrations.template import render_migration_template

__all__ = [
    "DDL_LOCK_NAMESPACE",
    "CapturedStatement",
    "ConnectionSession",
    "DdlLockError",
    "DdlLockLease",
    "DdlLockPolicy",
    "DdlLockReleaseError",
    "DdlLockTimeoutError",
    "DriftReport",
    "DuplicateVersionError",
    "InboundFk",
    "MigrationError",
    "MigrationFailedError",
    "MigrationFunc",
    "MigrationRunner",
    "MigrationScope",
    "MigrationSession",
    "MigrationStore",
    "MissingDependencyError",
    "PackageMigrations",
    "PreviewStore",
    "SessionRequiredError",
    "SqlConnection",
    "add_check_constraint",
    "add_column_with_backfill",
    "add_index",
    "add_partition_column",
    "database_ddl_lock",
    "ddl_lock_key",
    "diff_expected_live",
    "parse_ddl_to_expected",
    "render_migration_template",
    "replace_check_constraint",
    "replace_primary_key",
    "snapshot_live_schema",
]
