"""``namespaces`` is keyed by ``namespace_id`` alone and ``row_scope`` is a moveable column: declared so, and taken
there by agent_acl v007, idempotently, in the outage-window cutover.

Why (Pace, 2026-11-25): a rescope changed half the old key ``(row_scope, namespace_id)`` in place, so with the
change-note ring retired a restore never heard the old key go away; gone-plus-new is not possible, as ``namespace_id``
and ``name`` are unique and foreign-key targets.
"""

from __future__ import annotations

import json
from typing import Any

from threetears.agent.acl.collections import NamespaceCollection
from threetears.agent.acl.migrations import register
from threetears.agent.acl.migrations.v006_access_tables_carry_their_epoch import STAMPED_TABLES
from threetears.agent.acl.migrations.v007_namespaces_keyed_by_id import (
    NAMESPACES,
    namespaces_keyed_by_id,
)
from threetears.core.collections.tenant_moves import TENANT_MOVES_SCHEMA
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.migrations.stamp_tables import bucket_statements, stamping_table_of

_SCHEMA = NamespaceCollection.schema

#: what YugabyteDB's ``version()`` answers, as the migration probes it
_YUGABYTE = "PostgreSQL 15.12-YB-2025.1.0.0-b0 on x86_64"

#: v007's scope index, as its statement reads with its comments' line ends spaced (:func:`_writes`)
_SCOPE_ID_INDEX = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.page_ids "
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespace_ids_page_by_key "
    "-- reason: every namespace id a page at a time, a scope at a time: the key alone is hashed on YugabyteDB "
    "CREATE INDEX IF NOT EXISTS idx_namespaces_scope_id ON namespaces (row_scope, namespace_id)"
)


class _Catalog:
    """the migration's store over a ``namespaces`` the catalog describes: its primary key's columns and its bucket
    index's declaration. Every statement is kept, whitespace folded.

    :ivar primary: the primary key's columns, in key order
    :ivar declared: the bucket index's comment, or None
    :ivar statements: every statement sent, in order
    """

    def __init__(self, primary: tuple[str, ...], declared: str | None) -> None:
        self.primary = primary
        self.declared = declared
        self.statements: list[str] = []

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        folded = " ".join(sql.split())
        self.statements.append(folded)
        if folded.startswith("SELECT version()"):
            return [{"version": _YUGABYTE}]
        if "pg_constraint" in folded:
            return [{"name": "namespaces_pkey", "column_name": column} for column in self.primary]
        if "obj_description" in folded:
            return [{"declared": self.declared}]
        return []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append(" ".join(sql.split()))
        return "OK"


def _comment(statement: str) -> str:
    return statement.split(" IS '", 1)[1][:-1].replace("''", "'")


def _v006_comment() -> str:
    [frozen] = [table for table in STAMPED_TABLES if table.name == "namespaces"]
    return _comment(bucket_statements(frozen, yugabyte=True)[2])


def _writes(store: _Catalog) -> list[str]:
    return [sql for sql in store.statements if not sql.startswith("SELECT")]


def _moves_table() -> list[str]:
    """the moves table's DDL as v007 sends it: the declaration's, each CREATE made IF NOT EXISTS, whitespace folded."""
    sent: list[str] = []
    for statement in TENANT_MOVES_SCHEMA.create_statements(yugabyte=True):
        folded = " ".join(statement.split())
        for create in ("CREATE TABLE ", "CREATE UNIQUE INDEX ", "CREATE INDEX "):
            if folded.startswith(create):
                folded = f"{create}IF NOT EXISTS {folded[len(create) :]}"
                break
        sent.append(folded)
    return sent


def test_namespaces_is_keyed_by_its_id_with_row_scope_moveable() -> None:
    """Flips if the key is anything but ``namespace_id``, ``row_scope`` stays a key or partition column, or stops being
    declared moveable and named on every change (its restore declaration says each)."""
    assert tuple(_SCHEMA.pk_columns) == ("namespace_id",)
    assert NamespaceCollection.primary_key_column == "namespace_id"
    assert _SCHEMA.partition_column is None
    assert _SCHEMA.moveable == ("row_scope",)
    assert "row_scope" in NamespaceCollection.invalidation_columns
    [comment] = [each for each in _SCHEMA.create_statements(yugabyte=True) if each.startswith("COMMENT ON INDEX")]
    declared = json.loads(_comment(comment))
    assert (declared["keys"], declared["moveable"], declared["columns"]) == (
        ["namespace_id"],
        ["row_scope"],
        ["row_scope"],
    )


def test_v007_names_the_table_as_its_declaration_renders_it() -> None:
    """Flips if v007's frozen table, bucket, index or declaration drift from what the schema renders."""
    assert register(MigrationRunner()).versions[7] is namespaces_keyed_by_id
    assert stamping_table_of(_SCHEMA) == NAMESPACES
    rendered = _SCHEMA.create_statements(yugabyte=True)
    [create] = [each for each in rendered if each.lstrip().startswith("CREATE TABLE")]
    added, index, comment = bucket_statements(NAMESPACES, yugabyte=True)
    expression = added.split("GENERATED ALWAYS AS (", 1)[1].rsplit(") STORED", 1)[0]
    assert f"GENERATED ALWAYS AS ({expression}) STORED" in create, (expression, create)
    [declared_index] = [each for each in rendered if each.startswith("CREATE INDEX") and NAMESPACES.index in each]
    assert index.split("\n")[-1].replace('"', "").replace(" IF NOT EXISTS", "") == declared_index.replace('"', "")
    assert comment in rendered
    # the moved columns as the declaration renders them: the scope's text and an epoch
    assert "row_scope_moved_from TEXT" in create and "row_scope_moved_epoch BIGINT" in create, create


async def test_v007_rekeys_a_v006_table_in_place() -> None:
    """Flips if v007 stops adding the moved columns, the scope index, the key on namespace_id alone, or the bucket
    rebuilt for the new key with its declaration last; or touches a row (no DML, no TRUNCATE, no DELETE)."""
    store = _Catalog(("row_scope", "namespace_id"), _v006_comment())
    await namespaces_keyed_by_id(store)
    assert _writes(store) == [
        *_moves_table(),
        'ALTER TABLE "namespaces" ADD COLUMN IF NOT EXISTS row_scope_moved_from text',
        'ALTER TABLE "namespaces" ADD COLUMN IF NOT EXISTS row_scope_moved_epoch bigint',
        _SCOPE_ID_INDEX,
        'ALTER TABLE "namespaces" DROP CONSTRAINT IF EXISTS "namespaces_pkey"',
        'ALTER TABLE "namespaces" ADD CONSTRAINT namespaces_pkey PRIMARY KEY (namespace_id)',
        f'DROP INDEX IF EXISTS "{NAMESPACES.index}"',
        'ALTER TABLE "namespaces" DROP COLUMN IF EXISTS epoch_bucket',
        *(" ".join(each.split()) for each in bucket_statements(NAMESPACES, yugabyte=True)),
    ], _writes(store)
    assert not [sql for sql in store.statements if sql.upper().startswith(("UPDATE", "INSERT", "TRUNCATE", "DELETE"))]


async def test_v007_creates_the_moves_table_its_moveable_column_needs() -> None:
    """Flips if v007 stops creating ``threetears_tenant_moves`` before it rekeys, as the declaration renders it, or
    sends any of it without ``IF NOT EXISTS``: a schema running only agent_acl would be refused at the run's end
    (refuse_moveable_without_moves), and one whose deployment made the table already would fail on it."""
    store = _Catalog(("row_scope", "namespace_id"), _v006_comment())
    await namespaces_keyed_by_id(store)
    moves = _moves_table()
    assert moves and moves[0].startswith("CREATE TABLE IF NOT EXISTS threetears_tenant_moves ("), moves
    assert all("IF NOT EXISTS" in each for each in moves if each.startswith("CREATE")), moves
    assert _writes(store)[: len(moves)] == moves, _writes(store)


async def test_v007_run_again_drops_nothing() -> None:
    """Flips if a second run (the table already keyed by namespace_id, its bucket declared for it) drops the key, the
    bucket or its index again."""
    [comment] = [each for each in _SCHEMA.create_statements(yugabyte=True) if each.startswith("COMMENT ON INDEX")]
    store = _Catalog(("namespace_id",), _comment(comment))
    await namespaces_keyed_by_id(store)
    # the moves table's DDL leads every run, each statement IF NOT EXISTS (a no-op the second time); past it, nothing
    # is dropped or rekeyed
    moves = _moves_table()
    assert _writes(store)[: len(moves)] == moves, _writes(store)
    rekeying = _writes(store)[len(moves) :]
    assert not [sql for sql in rekeying if "DROP" in sql or "PRIMARY KEY" in sql], rekeying
    assert _writes(store)[-1] == " ".join(bucket_statements(NAMESPACES, yugabyte=True)[2].split())


async def test_v007_stopped_between_its_steps_takes_them_again() -> None:
    """Flips if a run stopped after the old key was dropped (no key) or the bucket's index was dropped (no
    declaration) does not finish the step."""
    store = _Catalog((), None)
    await namespaces_keyed_by_id(store)
    writes = _writes(store)
    assert 'ALTER TABLE "namespaces" ADD CONSTRAINT namespaces_pkey PRIMARY KEY (namespace_id)' in writes
    assert 'ALTER TABLE "namespaces" DROP COLUMN IF EXISTS epoch_bucket' in writes
    assert writes[-3:] == [" ".join(each.split()) for each in bucket_statements(NAMESPACES, yugabyte=True)]
