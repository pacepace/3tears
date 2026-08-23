"""
agent-identity package migrations.

Single entry point :func:`register` wires the package's versioned
migration callables into a shared :class:`~threetears.core.data.
migrations.runner.MigrationRunner`. agent-identity owns exactly one
table in an agent schema:

- ``identity_versions`` -- the versioned identity-block store (a linear
  parent-pointer version chain per identity block; propose→consent→apply
  lifecycle; one active version per scope+block).

version history:

- v001 creates the identity_versions table, the ``identity_block_key`` +
  ``identity_version_status`` PG enums, and the three indexes (the partial
  UNIQUE active-per-block invariant, the block-history btree, the partial
  pending-queue btree). Self-evolution, 3tears v0.15.0.
- v002 adds ``wants`` / ``needs`` to the ``identity_block_key`` enum,
  making both proposable identity blocks. 3tears v0.27.0.

No soft-ref provenance columns (no ``conversation_id`` / ``source_*``), so
the package declares no cross-package apply-order dependency.
"""

from __future__ import annotations

from threetears.agent.identity.migrations.v001_create_identity_versions import (
    create_identity_versions_table,
)
from threetears.agent.identity.migrations.v002_add_wants_needs_block_keys import (
    add_wants_needs_block_keys,
)
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "agent_identity"


def register(runner: MigrationRunner) -> PackageMigrations:
    """
    register agent-identity migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(
        name=PACKAGE_NAME,
        scope=MigrationScope.AGENT,
    )
    pkg.version(1)(create_identity_versions_table)
    pkg.version(2)(add_wants_needs_block_keys)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_wants_needs_block_keys",
    "create_identity_versions_table",
    "register",
]
