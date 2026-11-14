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
- v003 builds ``idx_scheduled_jobs_next_fire_asc`` (the due scan, ``next_fire_at ASC``: range-sharded on
  YugabyteDB) beside v001's ``idx_scheduled_jobs_next_fire`` and drops it.
- v004 builds ``idx_scheduled_jobs_next_fire_key`` (``next_fire_at ASC, job_id ASC``: the due scan's page ordered by
  its time completed by the key) beside v003's ``idx_scheduled_jobs_next_fire_asc`` and drops it.
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
from threetears.scheduled_jobs.migrations.v002_dispatching_fires_index import (
    add_dispatching_fires_index,
)
from threetears.scheduled_jobs.migrations.v003_next_fire_range_index import (
    range_shard_next_fire_index,
)
from threetears.scheduled_jobs.migrations.v004_next_fire_key_index import (
    add_next_fire_key_index,
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
    pkg.version(2)(add_dispatching_fires_index)
    pkg.version(3)(range_shard_next_fire_index)
    pkg.version(4)(add_next_fire_key_index)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_dispatching_fires_index",
    "create_scheduled_jobs",
    "range_shard_next_fire_index",
    "register",
]
