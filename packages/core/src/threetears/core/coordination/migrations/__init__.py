"""3tears core coordination migrations.

:func:`register` wires this package's versioned migrations into a shared
:class:`~threetears.core.data.migrations.runner.MigrationRunner`. The package owns the four tables
the durable coordination primitives keep their state in: ``coordination_counters``,
``coordination_claims``, ``coordination_revocations`` and ``coordination_redemptions``.

PLATFORM scope: the tables land in whatever schema the caller binds through ``search_path`` before
calling :meth:`MigrationRunner.apply_for_platform_schema`. A consumer whose primitives run in an
agent schema re-registers this package at AGENT scope, as scriob already does for
``scheduled_jobs`` and ``epoch``. A consumer that cannot run DDL at all -- an agent or tool pod,
whose broker refuses it -- declares the same schemas in its data section instead, from
:data:`~threetears.core.coordination.tables.COORDINATION_TABLE_SCHEMAS`.

version history:

- v001 -- create the four coordination tables, rendered from their collections' declared schemas.
- v002 -- add the compare-and-swap order columns (``l2_epoch``, ``l2_revision``) to the counters,
  claims and redemptions tables, and backfill existing rows to the order floor.
- v003 -- create ``collection_write_marks`` (and, until 0.95, the trigger function that filled it).
- v004 -- index ``collection_write_marks`` by ``(table_name, id)``, so the broker's read of a commit's
  marks reads that table's, not the whole queue.
- v005 -- each coordination table's expiry index range-sharded (``expires_at ASC``), built beside the hashed one
  before it is dropped, so a YugabyteDB sweep seeks it rather than reading every tablet.
- v006 -- every write-mark trigger and the mark function taken down: a lost advance is announced by its writer's
  backstop from the change notes (:mod:`threetears.core.collections.lost_writes`), with no L3 write a commit. The
  queue's table stays, empty, until the contract release (:data:`~threetears.core.ddl.write_marks
  .WRITE_MARKS_CONTRACT_RELEASE`).
- v007 -- ``coordination_revocations`` carries its epoch (``ROW_STAMPED``): the epoch backfilled a page of keys a
  statement, the gone flag, the stored restore bucket, its ``(epoch_bucket, epoch)`` index and declaration, so a
  restore of a lost change log reads it a bucket at a time and the change-note ring can be retired.
"""

from __future__ import annotations

from threetears.core.coordination.migrations.v001_create_coordination_tables import (
    create_coordination_tables,
)
from threetears.core.coordination.migrations.v002_add_l2_order_columns import add_l2_order_columns
from threetears.core.coordination.migrations.v003_create_collection_write_marks import (
    create_collection_write_marks,
)
from threetears.core.coordination.migrations.v004_index_write_marks_by_table import index_write_marks_by_table
from threetears.core.coordination.migrations.v005_range_expiry_indexes import range_expiry_indexes
from threetears.core.coordination.migrations.v006_drop_write_mark_triggers import drop_write_mark_triggers
from threetears.core.coordination.migrations.v007_revocations_carry_their_epoch import (
    revocations_carry_their_epoch,
)
from threetears.core.coordination.tables import CoordinationRevocationsCollection
from threetears.core.data.migrations import (
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

PACKAGE_NAME = "coordination"


def register(runner: MigrationRunner, *, scope: MigrationScope = MigrationScope.PLATFORM) -> PackageMigrations:
    """register the coordination migrations with the given runner.

    :param runner: canonical migration runner to register with
    :ptype runner: MigrationRunner
    :param scope: which schema the tables belong in; PLATFORM by default, AGENT for a consumer
        whose primitives run in an agent schema
    :ptype scope: MigrationScope
    :return: populated package registration
    :rtype: PackageMigrations
    """
    pkg = PackageMigrations(name=PACKAGE_NAME, scope=scope)
    pkg.version(1)(create_coordination_tables)
    pkg.version(2)(add_l2_order_columns)
    pkg.version(3)(create_collection_write_marks)
    pkg.version(4)(index_write_marks_by_table)
    pkg.version(5)(range_expiry_indexes)
    pkg.version(6)(drop_write_mark_triggers)
    pkg.version(7)(revocations_carry_their_epoch)
    # the revocation list caches absences: it carries its change-note triggers from the run that creates it
    pkg.mark_write_generation(CoordinationRevocationsCollection.schema)
    runner.register(pkg)
    return pkg


__all__ = [
    "PACKAGE_NAME",
    "add_l2_order_columns",
    "create_collection_write_marks",
    "create_coordination_tables",
    "register",
]
