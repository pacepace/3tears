"""a stamping migration's table names what its declaration renders, and may live in a second schema.

``StampingTable`` carries the table's declared invalidation columns and moveable tenant columns, so the restore comment
a migration writes is the one ``TableSchema.create_statements`` renders (a restored record carries each column as a
write's record does). A table of a second schema of the database (the hub's audit table, in ``<platform>_audit``) is
named qualified in its statements; its index keeps the bare name and is commented on under that schema.
"""

from __future__ import annotations

import json

from threetears.core.data.migrations.stamp_tables import StampingTable, bucket_statements, stamp_existing_table
from threetears.core.testing.stamped_tables import MigrationStore


def _declared(comment: str) -> dict[str, object]:
    text = comment.split(" IS '", 1)[1][:-1].replace("''", "'")
    declared: dict[str, object] = json.loads(text)
    return declared


def test_the_declaration_names_the_tables_invalidation_and_moveable_columns() -> None:
    table = StampingTable("datasets", (("id", "uuid"),), columns=("customer_id", "name"), moveable=("customer_id",))
    *_, comment = bucket_statements(table, yugabyte=False)
    declared = _declared(comment)
    assert declared["moveable"] == ["customer_id"], declared
    assert "name" in str(declared), declared


def test_a_table_of_a_second_schema_is_named_qualified_its_index_bare() -> None:
    table = StampingTable("audit_events", (("row_scope", "text"), ("id", "uuid")), schema="platform_audit")
    added, created, comment = bucket_statements(table, yugabyte=False)
    assert added.startswith('ALTER TABLE "platform_audit"."audit_events" ADD COLUMN'), added
    assert f'CREATE INDEX IF NOT EXISTS "{table.index}" ON "platform_audit"."audit_events"' in created, created
    assert comment.startswith(f'COMMENT ON INDEX "platform_audit"."{table.index}" IS'), comment


async def test_a_table_of_a_second_schema_takes_every_step_qualified() -> None:
    table = StampingTable("audit_events", (("id", "uuid"),), schema="platform_audit")
    store = MigrationStore({})
    await stamp_existing_table(store, table, yugabyte=False)
    steps = [sql for sql, _ in store.statements if sql.startswith(("ALTER", "CREATE", "COMMENT"))]
    assert steps[:2] == [
        'ALTER TABLE "platform_audit"."audit_events" ADD COLUMN IF NOT EXISTS epoch bigint',
        'ALTER TABLE "platform_audit"."audit_events" ADD COLUMN IF NOT EXISTS gone boolean NOT NULL DEFAULT false',
    ], steps
    assert steps[2] == 'ALTER TABLE "platform_audit"."audit_events" ALTER COLUMN epoch SET NOT NULL', steps
    reads = [sql for sql, _ in store.statements if sql.startswith("SELECT") and "audit_events" in sql]
    assert reads and all("platform_audit.audit_events" in sql for sql in reads), reads
