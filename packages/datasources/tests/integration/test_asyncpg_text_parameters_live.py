"""a page's last key, carried as text, binds against a Postgres timestamp key.

The datasource rail carries every value JSON cannot as its text, and a paged read hands its last row's key back as
the next page's parameters in that form: ``2026-11-04 05:33:12`` for a timestamp. asyncpg binds by the type Postgres
declares for each placeholder and refused text for a timestamp (``DataError``), so a paged read of a relation whose
key holds a timestamp failed on its second page. Against a real Postgres, every driver path binds it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import asyncpg
import pytest

from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.query_client import keyset_condition

pytestmark = pytest.mark.integration

_SCHEMA = "ds_it_text_params"


def _connect_kwargs(db_url: str) -> dict[str, Any]:
    parts = urlsplit(db_url)
    return {
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": (parts.path or "/postgres").lstrip("/"),
        "user": parts.username or "postgres",
        "password": parts.password or "",
    }


@pytest.fixture
async def driver(db_container: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncpgDriver]:
    """a driver over a table keyed by a timestamp and an id, four rows."""
    kwargs = _connect_kwargs(db_container)
    conn = await asyncpg.connect(**kwargs)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE')
        await conn.execute(f'CREATE SCHEMA "{_SCHEMA}"')
        await conn.execute(
            f'CREATE TABLE "{_SCHEMA}".runs (observed_at timestamp NOT NULL, id integer NOT NULL, '
            "noted timestamptz, PRIMARY KEY (observed_at, id))"
        )
        await conn.execute(
            f'INSERT INTO "{_SCHEMA}".runs VALUES '
            "('2026-11-04 05:33:12', 1, '2026-11-04 05:33:12+00'), ('2026-11-04 05:33:12', 2, NULL), "
            "('2026-11-04 06:00:00', 1, NULL), ('2026-11-05 00:00:00', 1, NULL)"
        )
    finally:
        await conn.close()
    monkeypatch.setenv("TEXT_PARAMS_TEST_PW", kwargs["password"])
    config = PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host=kwargs["host"],
        port=kwargs["port"],
        database=kwargs["database"],
        username=kwargs["user"],
        password_ref="env://TEXT_PARAMS_TEST_PW",
        pool_min_size=1,
        pool_max_size=2,
        command_timeout_seconds=10,
        allowed_schemas=[_SCHEMA],
    )
    built = AsyncpgDriver(config)
    try:
        yield built
    finally:
        await built.close()


def _next_page() -> tuple[str, list[Any]]:
    """the rail's next-page predicate after the row ``('2026-11-04 05:33:12', 2)``, its key as the rail carries it."""
    condition, params = keyset_condition(["observed_at", "id"], ["2026-11-04 05:33:12", 2], earlier=">", last=">")
    return f"SELECT id, observed_at FROM {_SCHEMA}.runs WHERE {condition} ORDER BY observed_at, id", params


class TestATextKeyBindsAsItsDeclaredType:
    async def test_the_next_page_of_a_timestamp_key_is_read(self, driver: AsyncpgDriver) -> None:
        sql, params = _next_page()
        rows = await driver.fetch(sql, *params)
        assert [(row["observed_at"].isoformat(sep=" "), row["id"]) for row in rows] == [
            ("2026-11-04 06:00:00", 1),
            ("2026-11-05 00:00:00", 1),
        ]

    async def test_a_bounded_read_and_a_stream_bind_it_too(self, driver: AsyncpgDriver) -> None:
        sql, params = _next_page()
        assert len(await driver.fetch_at_most(sql, *params, max_rows=1)) == 1
        assert [row["id"] async for row in driver.fetch_iter(sql, *params)] == [1, 1]

    async def test_a_timestamptz_and_a_statement_bind_it(self, driver: AsyncpgDriver) -> None:
        rows = await driver.fetch(f"SELECT id FROM {_SCHEMA}.runs WHERE noted = $1", "2026-11-04T05:33:12+00:00")
        assert [row["id"] for row in rows] == [1]
        await driver.execute(f"DELETE FROM {_SCHEMA}.runs WHERE observed_at < $1", "2026-11-04 06:00:00")
        assert len(await driver.fetch(f"SELECT id FROM {_SCHEMA}.runs")) == 2

    async def test_text_that_is_no_timestamp_is_still_refused(self, driver: AsyncpgDriver) -> None:
        with pytest.raises(asyncpg.exceptions.DataError):
            await driver.fetch(f"SELECT id FROM {_SCHEMA}.runs WHERE observed_at > $1", "not a time")
