"""
canonical migration runner public surface.

this package replaces the previous single-file
``threetears.core.data.migrations`` module. the :class:`MigrationRunner`
now composes per-package migration registrations across the platform
rather than each package owning a standalone runner. see the task shard
``migrations-task-01`` for the motivation.

public API:

- :class:`MigrationRunner` — composes registered packages and applies
  pending migrations against a DataStore bound to a schema.
- :class:`PackageMigrations` — per-package registration of versioned
  async migration callables, with optional ``depends_on`` edges.
- :class:`MigrationScope` — PLATFORM vs AGENT enum.
- :func:`render_migration_template` — renders the blessed template for
  authoring a new migration module.
- error types: :class:`MigrationError`, :class:`DuplicateVersionError`,
  :class:`MissingDependencyError`, :class:`MigrationFailedError`.
"""

from __future__ import annotations

from threetears.core.data.migrations.drift import (
    DriftReport,
    diff_expected_live,
    parse_ddl_to_expected,
    snapshot_live_schema,
)
from threetears.core.data.migrations.errors import (
    DuplicateVersionError,
    MigrationError,
    MigrationFailedError,
    MissingDependencyError,
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
from threetears.core.data.migrations.template import render_migration_template

__all__ = [
    "CapturedStatement",
    "DriftReport",
    "DuplicateVersionError",
    "MigrationError",
    "MigrationFailedError",
    "MigrationFunc",
    "MigrationRunner",
    "MigrationScope",
    "MissingDependencyError",
    "PackageMigrations",
    "PreviewStore",
    "diff_expected_live",
    "parse_ddl_to_expected",
    "render_migration_template",
    "snapshot_live_schema",
]
