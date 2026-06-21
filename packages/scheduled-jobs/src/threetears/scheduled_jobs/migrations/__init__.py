"""scheduled-jobs package migrations.

Single entry point :func:`register` wires the package's versioned
migration callables into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`.

This package owns two tables for the default store:

- ``scheduled_jobs`` -- one row per active scheduled job. Carries the
  opaque ``kind`` (TEXT) + ``payload`` (JSONB) plus the scheduling
  columns. Standalone ``UNIQUE (job_id)`` so ``job_fires`` can reference
  the bare id.
- ``job_fires`` -- one row per job fire (history). FK ``job_id REFERENCES
  scheduled_jobs(job_id) ON DELETE CASCADE``.

Scope is :data:`MigrationScope.PLATFORM` with NO ``depends_on``: unlike
agent-wake (which is AGENT-scope and depends on ``conversations`` +
``agent_skills``), the generic store stands alone -- ``partition_key``
is a denormalised UUID with no FK, so there is no upstream table to
order after. A consumer that wants the tables in a per-tenant /
per-agent schema can re-register under a different scope from its own
migration wiring; the default registration targets the shared platform
schema.

Version history:

- v001 creates ``scheduled_jobs`` + ``job_fires`` + indexes.
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

from threetears.scheduled_jobs.migrations.v001_create_scheduled_jobs import (
    create_scheduled_jobs,
)

PACKAGE_NAME = "scheduled_jobs"


def register(runner: MigrationRunner) -> PackageMigrations:
    """Register scheduled-jobs migrations with ``runner``.

    Produces a platform-scoped :class:`PackageMigrations` with no
    ``depends_on`` edges (the generic store stands alone). Attaches every
    migration in version order and calls ``runner.register``.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.PLATFORM,
    )
    pkg.version(1)(create_scheduled_jobs)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "create_scheduled_jobs",
    "register",
]
