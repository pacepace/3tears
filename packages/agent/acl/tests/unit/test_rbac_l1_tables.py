"""The rbac L1 mirror is GENERATED from the canonical schemas, not retyped.

Three copies of this metadata existed -- registry, hub, agent pod -- and two had fallen five
columns behind ``NamespaceCollection.schema`` (``tool_eligible`` / ``skill_eligible`` /
``face_api`` / ``face_mcp`` / ``face_platform_tool``). A hand-maintained SINGLE copy would
have moved that drift one column later rather than closing it, so the one copy is emitted by
:meth:`TableSchema.to_sqlalchemy_table`.

These tests assert the generation, not a column list: a list retyped here would be the fourth
copy. The one literal check is the five drifted columns, kept because naming them is what
makes the regression legible if it ever recurs.
"""

from __future__ import annotations

import sqlalchemy as sa
from threetears.core.collections.schema_backed import SchemaBackedCollection

import pytest

from threetears.agent.acl.tables import (
    RBAC_L1_COLLECTIONS,
    RBAC_L1_TABLE_NAMES,
    register_rbac_l1_tables,
)

#: the five columns the registry's and the hub's hand-written copies were missing. named
#: literally because the point of the test is the specific regression, not the general rule
#: the parametrized tests below already cover.
_DRIFTED_NAMESPACE_COLUMNS = (
    "tool_eligible",
    "skill_eligible",
    "face_api",
    "face_mcp",
    "face_platform_tool",
)


def test_the_five_rbac_tables_register() -> None:
    """every rbac table the evaluator reads lands on the caller's metadata."""
    md = sa.MetaData()
    tables = register_rbac_l1_tables(md)

    assert set(tables) == {"namespaces", "groups", "group_members", "roles", "role_assignments"}
    assert set(md.tables) == set(tables)
    assert RBAC_L1_TABLE_NAMES == frozenset(tables)


@pytest.mark.parametrize("collection_cls", RBAC_L1_COLLECTIONS, ids=lambda cls: cls.__name__)
def test_every_canonical_column_is_present(collection_cls: type[SchemaBackedCollection]) -> None:
    """the mirror carries every column the canonical TableSchema declares.

    a missing column is not a cache miss: ``BaseCollection.write_to_cache_sync`` raises
    ``sqlite3.OperationalError: table <name> has no column named <field>`` on every write.
    """
    tables = register_rbac_l1_tables(sa.MetaData())
    table = tables[collection_cls.schema.name]

    declared = {col.name for col in collection_cls.schema.columns}
    mirrored = {col.name for col in table.columns}
    assert declared == mirrored, f"{collection_cls.__name__} mirror differs from its schema"


@pytest.mark.parametrize("collection_cls", RBAC_L1_COLLECTIONS, ids=lambda cls: cls.__name__)
def test_primary_key_matches_the_canonical_schema(
    collection_cls: type[SchemaBackedCollection],
) -> None:
    """SQLite's UPSERT needs the ``ON CONFLICT`` columns to be a real PK.

    a single-column mirror against a composite-PK canonical schema trips ``ON CONFLICT clause
    does not match any PRIMARY KEY or UNIQUE constraint`` on every create.
    """
    tables = register_rbac_l1_tables(sa.MetaData())
    table = tables[collection_cls.schema.name]

    declared = collection_cls.schema.primary_key
    expected = frozenset({declared}) if isinstance(declared, str) else frozenset(declared)
    assert frozenset(col.name for col in table.primary_key.columns) == expected


def test_the_namespaces_mirror_carries_the_columns_that_drifted() -> None:
    """the eligibility + face flags round-trip; their absence WAS the live bug."""
    tables = register_rbac_l1_tables(sa.MetaData())
    mirrored = {col.name for col in tables["namespaces"].columns}

    assert set(_DRIFTED_NAMESPACE_COLUMNS) <= mirrored


def test_the_drifted_columns_are_still_declared_canonically() -> None:
    """scope sanity: the check above is vacuous if the schema stopped declaring them."""
    from threetears.agent.acl.collections import NamespaceCollection

    declared = {col.name for col in NamespaceCollection.schema.columns}
    assert set(_DRIFTED_NAMESPACE_COLUMNS) <= declared


def test_registration_is_idempotent_on_one_metadata() -> None:
    """a second call returns the tables already registered rather than raising.

    the hub and the agent pod both register the rbac mirror onto a module-level ``MetaData``
    that other tables share; a re-import or a second wiring pass must not explode.
    """
    md = sa.MetaData()
    first = register_rbac_l1_tables(md)
    second = register_rbac_l1_tables(md)

    assert all(first[name] is second[name] for name in first)


def test_a_namespace_row_with_the_face_flags_round_trips_through_l1() -> None:
    """the generated mirror accepts a write carrying the drifted columns.

    the column-set assertions above compare declarations; this one drives the real
    ``SQLiteBackend`` upsert path the missing columns used to break.
    """
    from uuid import uuid7

    from threetears.core.cache.sqlite import SQLiteBackend

    md = sa.MetaData()
    register_rbac_l1_tables(md)
    backend = SQLiteBackend(db_name=f"rbac_l1_tables_{uuid7().hex}")
    backend.initialize(md)

    namespace_id = uuid7()
    row = {
        "row_scope": "platform",
        "namespace_id": namespace_id,
        "name": "tools.probe",
        "namespace_type": "tool",
        "tool_eligible": True,
        "skill_eligible": False,
        "face_api": True,
        "face_mcp": False,
        "face_platform_tool": True,
    }
    backend.upsert("namespaces", row, primary_key=("row_scope", "namespace_id"))

    stored = backend.select_by_id(
        "namespaces",
        ("platform", namespace_id),
        primary_key=("row_scope", "namespace_id"),
    )
    assert stored is not None
    assert stored["tool_eligible"] is True
    assert stored["face_api"] is True
    assert stored["face_mcp"] is False
    backend.reset()
