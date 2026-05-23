"""Integration-test fixtures for the channels package.

Mirrors :mod:`packages.agent.wake.tests.integration.conftest`: the
session-scoped ``db_container`` fixture comes from the canonical
harness in :mod:`threetears.core.testing.fixtures` (wired via the
workspace-root ``conftest.py`` ``pytest_plugins`` line); the per-test
``pg_schema`` fixture creates a fresh schema so each test starts
clean.

The channels package's webhook receiver integration tests need the
wake + skills + conversations schemas applied (the receiver delegates
to :func:`webhook_receive` which reads from ``webhook_subscriptions``
and writes to ``wake_fires``); the per-test ``_apply_schema`` helper
in ``test_webhook_e2e.py`` registers all three migration packs
against the schema this fixture creates.

No ``__init__.py`` under ``tests/integration/`` so this conftest's
module name is the full rootdir-relative path (matches the wake
package's integration conftest); a shared module name like
``tests.integration.conftest`` would collide with other packages'
integration conftests under pytest's ``--import-mode=importlib``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """Alias for :func:`threetears.core.testing.fixtures.db_container`.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """Create a fresh schema per test and yield ``(pg_url, schema_name)``.

    The schema is dropped on teardown so each test gets a clean slate.

    :param pg_url: testcontainer URL
    :ptype pg_url: str
    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"ch_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()
