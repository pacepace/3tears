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
            "noted timestamptz, label text, PRIMARY KEY (observed_at, id))"
        )
        await conn.execute(f'CREATE SEQUENCE "{_SCHEMA}".calls')
        await conn.execute(
            f'INSERT INTO "{_SCHEMA}".runs VALUES '
            "('2026-11-04 05:33:12', 1, '2026-11-04 05:33:12+00', 'a'), ('2026-11-04 05:33:12', 2, NULL, 'b'), "
            "('2026-11-04 06:00:00', 1, NULL, 'c'), ('2026-11-05 00:00:00', 1, NULL, 'd')"
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

    async def test_text_converts_as_postgres_casts_it_at_the_time_zone_edges(self, driver: AsyncpgDriver) -> None:
        # a naive time at a timestamptz is read in the session's zone (UTC here), as Postgres's own cast reads it
        rows = await driver.fetch(f"SELECT id FROM {_SCHEMA}.runs WHERE noted = $1", "2026-11-04 05:33:12")
        assert [row["id"] for row in rows] == [1]
        # an offset at a plain timestamp is dropped, as Postgres's cast drops it, rather than refused
        rows = await driver.fetch(
            f"SELECT id FROM {_SCHEMA}.runs WHERE observed_at = $1 ORDER BY id", "2026-11-04 06:00:00+02"
        )
        assert [row["id"] for row in rows] == [1]

    async def test_the_retry_is_said_once_per_statement(
        self, driver: AsyncpgDriver, caplog: pytest.LogCaptureFixture
    ) -> None:
        sql = f"SELECT id FROM {_SCHEMA}.runs WHERE observed_at > $1 AND id >= 0"
        with caplog.at_level("INFO", logger="threetears.datasources.drivers.asyncpg_driver"):
            await driver.fetch(sql, "2026-11-04 05:33:12")
            await driver.fetch(sql, "2026-11-04 06:00:00")
        said = [r for r in caplog.records if "bound as their declared types" in r.getMessage()]
        assert len(said) == 1, said


class TestAServerErrorIsNeverRetried:
    """a server's own data error is raised as it is, and the statement runs once."""

    _DIVIDES = "SELECT nextval('{schema}.calls') / (CASE WHEN $1::uuid IS NOT NULL THEN 0 ELSE 1 END) AS q"

    async def _calls(self, db_container: str) -> int:
        conn = await asyncpg.connect(**_connect_kwargs(db_container))
        try:
            value: int = await conn.fetchval(
                f'SELECT CASE WHEN is_called THEN last_value ELSE 0 END FROM "{_SCHEMA}".calls'
            )
        finally:
            await conn.close()
        return value

    async def test_outside_a_transaction(self, driver: AsyncpgDriver, db_container: str) -> None:
        with pytest.raises(asyncpg.exceptions.DivisionByZeroError):
            await driver.fetch(self._DIVIDES.format(schema=_SCHEMA), "0198a3f2-1c4e-7b9a-8d2f-3e4a5b6c7d8e")
        assert await self._calls(db_container) == 1, "the statement ran twice"

    async def test_inside_a_statement_timeouts_transaction(self, driver: AsyncpgDriver, db_container: str) -> None:
        with pytest.raises(asyncpg.exceptions.DivisionByZeroError):
            await driver.fetch(
                self._DIVIDES.format(schema=_SCHEMA), "0198a3f2-1c4e-7b9a-8d2f-3e4a5b6c7d8e", timeout_seconds=5
            )
        assert await self._calls(db_container) == 1, "the statement ran twice"

    async def test_inside_a_cursors_transaction(self, driver: AsyncpgDriver, db_container: str) -> None:
        with pytest.raises(asyncpg.exceptions.DivisionByZeroError):
            await driver.fetch_at_most(
                self._DIVIDES.format(schema=_SCHEMA), "0198a3f2-1c4e-7b9a-8d2f-3e4a5b6c7d8e", max_rows=1
            )
        assert await self._calls(db_container) == 1, "the statement ran twice"


class TestAStreamPaysNothingForTextItBinds:
    async def test_a_text_parameter_at_a_text_placeholder_prepares_nothing_more(
        self, driver: AsyncpgDriver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prepared: list[str] = []
        real_prepare = asyncpg.connection.Connection.prepare

        async def counting(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
            prepared.append(query)
            return await real_prepare(self, query, *args, **kwargs)

        monkeypatch.setattr(asyncpg.connection.Connection, "prepare", counting)
        rows = [row["id"] async for row in driver.fetch_iter(f"SELECT id FROM {_SCHEMA}.runs WHERE label = $1", "c")]
        assert rows == [1]
        assert prepared == [], f"a stream prepared on a bind nothing refused: {prepared}"
