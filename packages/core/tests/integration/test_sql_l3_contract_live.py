"""``SqlL3Backend`` over a real Postgres keeps the L3 contract (``tests/l3_contract.py``).

The same class the SQLite store runs in the unit suite (``tests/unit/backends/test_sqlite_l3_contract.py``): what a
schema-backed collection may rely on of its L3 is held on both, so a deployment moves between them by configuration.
The tables are created from the contract's schemas through SQLAlchemy's Postgres DDL, in a schema of the test's own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from packages.core.tests.l3_contract import CONTRACT_SCHEMAS, CollectionOverStoreContract, SqlStoreContract
from threetears.core.backends import SqlL3Backend, l3_store

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(db_container: str) -> AsyncIterator[SqlL3Backend]:
    """a store over a pool whose connections search a fresh schema holding the contract's empty tables."""
    schema_name = f"l3c_{uuid.uuid4().hex[:12]}"
    metadata = MetaData()
    ddl = [
        str(CreateTable(schema.to_sqlalchemy_table(metadata)).compile(dialect=cast(Any, postgresql).dialect()))
        for schema in CONTRACT_SCHEMAS
    ]

    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema_name}")
        await admin.execute(f"SET search_path TO {schema_name}")
        for statement in ddl:
            await admin.execute(statement)
    finally:
        await admin.close()
    try:
        # the configuration's one setting, as a deployment names its L3; a server setting rather than a SET, since a
        # pool resets session state each time it takes a connection back
        async with l3_store(db_container, server_settings={"search_path": schema_name}) as backend:
            assert isinstance(backend, SqlL3Backend)
            for schema in CONTRACT_SCHEMAS:
                backend.register_schema(schema.name, schema)
            yield backend
    finally:
        admin = await asyncpg.connect(db_container)
        try:
            await admin.execute(f"DROP SCHEMA {schema_name} CASCADE")
        finally:
            await admin.close()


class TestTheSqlStoreKeepsTheL3Contract(SqlStoreContract):
    duplicate_error = asyncpg.UniqueViolationError


class TestCollectionsOverTheSqlStore(CollectionOverStoreContract):
    """a collection over this store, with no L2 and with L2."""
