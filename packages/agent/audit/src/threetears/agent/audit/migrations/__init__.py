"""
agent-audit package migrations.

:func:`register` wires the package's versioned migrations into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`. agent-audit
owns one platform-scoped table, ``audit_events``.

version history:

- v001 creates ``audit_events`` and its four indexes (3tears v0.59.0).
- v002 builds ``idx_audit_events_time_asc`` (``timestamp ASC``: range-sharded on YugabyteDB) and drops
  ``idx_audit_events_time`` (3tears v0.95.0).
- v003 (CONTRACT, held) -- drop ``idx_audit_events_customer_time`` (no read uses it): attached only with
  ``held_contract=True`` (the test schemas now; every deployment from the release after this one).
"""

from __future__ import annotations

from threetears.agent.audit.migrations.v001_create_audit_events import create_audit_events
from threetears.agent.audit.migrations.v002_audit_time_range_index import range_shard_audit_time_index
from threetears.agent.audit.migrations.v003_contract_customer_time_index_dropped import drop_unread_audit_index
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations

PACKAGE_NAME = "agent_audit"


def register(runner: MigrationRunner, *, held_contract: bool = False) -> PackageMigrations:
    """
    register agent-audit migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param held_contract: attach v003, the held contract step (drops the unread index)
    :ptype held_contract: bool
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=MigrationScope.PLATFORM)
    pkg.version(1)(create_audit_events)
    pkg.version(2)(range_shard_audit_time_index)
    if held_contract:
        pkg.version(3)(drop_unread_audit_index)
    runner.register(pkg)
    return pkg


__all__ = ["PACKAGE_NAME", "create_audit_events", "register"]
