"""live integration tests for :class:`AsyncpgDriver` against a testcontainer.

verifies the contract end-to-end against a real postgres:

- pool creation + ``test_connection`` round-trip
- ``fetch`` / ``execute`` with ``$N`` placeholders
- ``fetch_iter`` server-side streaming (memory-bounded)
- ``list_tables`` / ``list_columns`` / ``table_hashes`` discover seed schema
- Tier-2 hash byte-equivalence between python helper + warehouse MD5
- cancellation propagation via :meth:`Connection.cancel` (NOT terminate)
- AGENT_INTERNAL borrowed-pool: driver does NOT close the borrowed pool
- microbenchmark guard rail (DS-10-13; gated, manual run only)

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import hashlib
import tracemalloc
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    PostgresConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.base import Driver
from threetears.datasources.entities import DataSourceType

from ..unit._helpers.cancellation_contract import (
    DriverCancellationContractTest,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# DSN helpers
# ---------------------------------------------------------------------------


def _parse_db_url(db_url: str) -> dict[str, Any]:
    """parse the testcontainer URL into asyncpg connect kwargs.

    :param db_url: ``postgresql://user:pw@host:port/db`` style URL
    :ptype db_url: str
    :return: dict with host/port/database/user/password keys
    :rtype: dict[str, Any]
    """
    from urllib.parse import urlsplit

    parts = urlsplit(db_url)
    return {
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": (parts.path or "/postgres").lstrip("/"),
        "username": parts.username or "postgres",
        "password": parts.password or "",
    }


def _make_config_for_container(
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allowed_schemas: list[str] | None = None,
) -> PostgresConnectionConfig:
    """build a :class:`PostgresConnectionConfig` for the seeded container.

    sets the password env var the config expects via monkeypatch so
    the test doesn't write to the real process env.

    :param allowed_schemas: optional list to thread into the config's
        ``allowed_schemas``; defaults to ``[]`` so the backend's
        default ``search_path`` applies
    :ptype allowed_schemas: list[str] | None
    """
    parsed = _parse_db_url(db_url)
    monkeypatch.setenv("ASYNCPG_DRIVER_TEST_PW", parsed["password"])
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        username=parsed["username"],
        password_ref="env://ASYNCPG_DRIVER_TEST_PW",
        pool_min_size=1,
        pool_max_size=2,
        command_timeout_seconds=10,
        allowed_schemas=allowed_schemas or [],
    )


# ---------------------------------------------------------------------------
# Seed fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_schema(db_container: str) -> AsyncIterator[tuple[str, str]]:
    """provision a fresh test schema with a small known table.

    yields ``(db_url, schema_name)`` so individual tests can build
    their own driver against the same container + know the seeded
    schema name. teardown drops the schema.
    """
    parsed = _parse_db_url(db_container)
    schema = "ds_it_asyncpg"
    conn = await asyncpg.connect(
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        user=parsed["username"],
        password=parsed["password"],
    )
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}"."widgets" (id integer NOT NULL, name text NOT NULL, weight double precision)'
        )
        await conn.execute(
            f'INSERT INTO "{schema}"."widgets" (id, name, weight) '
            "VALUES (1, 'alpha', 1.5), (2, 'beta', NULL), (3, 'gamma', 3.14)"
        )
        yield db_container, schema
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


# ---------------------------------------------------------------------------
# Python-side helper for Tier-2 hash byte-equivalence
# ---------------------------------------------------------------------------


def _python_column_hash(cols: list[dict[str, Any]]) -> str:
    """python-side MD5 over the column shape; cross-language invariant.

    payload formula: ``column_name + ':' + data_type + ':' + (is_nullable or '')``
    per column, joined by ``','`` in ascending ``ordinal_position``.
    matches the SQL ``MD5(STRING_AGG(...))`` in
    :data:`_POSTGRES_TABLE_HASHES_SQL` byte-for-byte.

    TODO(datasource-task-13): shard 13 lifts this helper into
    ``threetears.datasources.introspection`` as the canonical
    python-side hash (per DS-13-14). until then it lives in the test
    module so the cross-check stays local to the assertion.

    :param cols: column rows (must have ``column_name``, ``data_type``,
        ``is_nullable``, ``ordinal_position`` keys)
    :ptype cols: list[dict[str, Any]]
    :return: hex MD5 digest
    :rtype: str
    """
    payload = ",".join(
        f"{c['column_name']}:{c['data_type']}:{(c['is_nullable'] or '')}"
        for c in sorted(cols, key=lambda c: c["ordinal_position"])
    )
    return hashlib.md5(payload.encode()).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """driver works end-to-end against the testcontainer."""

    @pytest.mark.asyncio
    async def test_test_connection_round_trips(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """:meth:`test_connection` succeeds against the real container."""
        db_url, _schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            await driver.test_connection()
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_fetch_and_execute_with_placeholders(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``$1`` placeholders work end-to-end against the container."""
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            rows = await driver.fetch(
                f'SELECT name, weight FROM "{schema}"."widgets" WHERE id = $1',
                1,
            )
            assert rows == [{"name": "alpha", "weight": 1.5}]
            await driver.execute(
                f'INSERT INTO "{schema}"."widgets" (id, name) VALUES ($1, $2)',
                99,
                "echo",
            )
            rows2 = await driver.fetch(f'SELECT name FROM "{schema}"."widgets" WHERE id = $1', 99)
            assert rows2 == [{"name": "echo"}]
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_list_tables_discovers_seed(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """:meth:`list_tables` returns the seeded table."""
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            tables = await driver.list_tables([schema])
            assert {"table_schema": schema, "table_name": "widgets"} in tables
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_list_columns_preserves_raw_is_nullable(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``is_nullable`` is the raw ``'YES'``/``'NO'`` string."""
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            cols = await driver.list_columns([schema])
            id_col = next(c for c in cols if c["table_name"] == "widgets" and c["column_name"] == "id")
            weight_col = next(c for c in cols if c["table_name"] == "widgets" and c["column_name"] == "weight")
            assert id_col["is_nullable"] == "NO"
            assert weight_col["is_nullable"] == "YES"
            # NOT a bool
            assert isinstance(id_col["is_nullable"], str)
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# search_path: connection-scope ``SET search_path`` from allowed_schemas
# ---------------------------------------------------------------------------


class TestSearchPathOnOpen:
    """live proof that ``allowed_schemas`` -> per-conn ``SET search_path``.

    the unit suite verifies the SQL we ship to asyncpg; this test
    proves Postgres actually accepts it AND that an unqualified
    table reference resolves through the seeded schema after the
    pool's ``init`` callback fires.
    """

    @pytest.mark.asyncio
    async def test_unqualified_table_resolves_via_search_path(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """with ``allowed_schemas=[seeded]``, ``SELECT * FROM widgets`` works.

        without the search_path set the same statement would fail with
        ``UndefinedTableError`` because the table lives in a non-default
        schema. seeing the row come back is the live signal that the
        connection-scope ``SET`` took effect.
        """
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch, allowed_schemas=[schema])
        driver = AsyncpgDriver(config)
        try:
            rows = await driver.fetch("SELECT name FROM widgets WHERE id = $1", 1)
            assert rows == [{"name": "alpha"}]
            # cross-check the session-scope GUC the server reports
            current = await driver.fetch("SHOW search_path")
            # asyncpg returns the raw quoted form; assert via substring
            assert schema in current[0]["search_path"]
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_empty_allowed_schemas_leaves_default_search_path(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """no ``allowed_schemas`` -> backend default ``search_path`` is intact.

        proves the absence of the init callback doesn't bleed into a
        connection's state in any other way.
        """
        db_url, schema = seeded_schema
        # explicit empty (the default) -- prove the table is unreachable
        # without qualification when no search_path is wired.
        config = _make_config_for_container(db_url, monkeypatch, allowed_schemas=[])
        driver = AsyncpgDriver(config)
        try:
            # default Postgres search_path is ``"$user", public`` -- the
            # seeded schema is NOT in it; unqualified select must fail.
            with pytest.raises(Exception, match="widgets"):
                await driver.fetch("SELECT * FROM widgets")
            # but qualified access still works
            rows = await driver.fetch(f'SELECT name FROM "{schema}"."widgets" WHERE id = $1', 1)
            assert rows == [{"name": "alpha"}]
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# fetch_iter streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """:meth:`fetch_iter` streams via server-side cursor (DS-10-09)."""

    @pytest.mark.asyncio
    async def test_fetch_iter_streams_large_result(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """seed 10k rows + iterate; assert memory growth stays bounded.

        compares peak tracemalloc against the materialize-everything
        path (``fetch``). the streaming path SHOULD use significantly
        less memory at peak. exact bytes vary by interpreter; we
        require fetch_iter peak to be at most 60% of fetch peak.
        """
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            # seed 10k rows
            await driver.execute(f'CREATE TABLE "{schema}"."big" (id integer, payload text)')
            # batch insert for speed
            payload = "x" * 200
            values = ", ".join(f"({i}, '{payload}')" for i in range(10000))
            await driver.execute(f'INSERT INTO "{schema}"."big" (id, payload) VALUES {values}')

            # measure peak memory for materialize-everything
            tracemalloc.start()
            rows = await driver.fetch(f'SELECT id, payload FROM "{schema}"."big" ORDER BY id')
            fetch_current, fetch_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert len(rows) == 10000
            del rows

            # measure peak memory for streaming
            count = 0
            tracemalloc.start()
            async for _row in driver.fetch_iter(f'SELECT id, payload FROM "{schema}"."big" ORDER BY id'):
                count += 1
            stream_current, stream_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert count == 10000

            # streaming should be substantially smaller than full materialization
            assert stream_peak < fetch_peak * 0.6, (
                f"fetch_iter peak {stream_peak} is not substantially below fetch peak {fetch_peak}"
            )
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# Tier-2 hash byte-equivalence
# ---------------------------------------------------------------------------


class TestTier2HashEquivalence:
    """python-side hash MUST byte-equal the warehouse-side MD5."""

    @pytest.mark.asyncio
    async def test_python_and_sql_hashes_agree(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """call :meth:`table_hashes` + recompute in python; assert equality.

        this is the cross-language invariant that makes the Tier-2
        change-probe work. if it ever fails: investigate WHICH side
        diverged before touching the SQL (the SQL is the contract;
        python helper might be wrong).
        """
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            sql_hashes = await driver.table_hashes([schema])
            cols = await driver.list_columns([schema])
            widgets_cols = [c for c in cols if c["table_name"] == "widgets"]
            python_hash = _python_column_hash(widgets_cols)  # type: ignore[arg-type]
            assert sql_hashes[(schema, "widgets")] == python_hash
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# Cancellation contract (DS-10-08)
# ---------------------------------------------------------------------------


class TestAsyncpgDriverCancellation(DriverCancellationContractTest):
    """inherit the canonical cancellation contract; supply slow driver + SQL.

    the mixin runs the standard cancel-propagation assertions
    (``fetch`` + ``execute``). this concrete class adds the asyncpg-
    specific "connection is returned cleanly to the pool" assertion
    on top.
    """

    # pytest needs to discover the mixin's tests; the fixture-request
    # pattern below lets the mixin's @pytest.mark.asyncio methods see
    # the seeded_schema fixture without an extra setup dance.
    @pytest.fixture(autouse=True)
    def _wire_fixtures(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """capture fixtures so the mixin methods can build a driver."""
        self._seeded_schema = seeded_schema
        self._monkeypatch = monkeypatch

    async def make_slow_driver(self) -> Driver:
        """build an :class:`AsyncpgDriver` against the testcontainer."""
        db_url, _schema = self._seeded_schema
        config = _make_config_for_container(db_url, self._monkeypatch)
        # bump command_timeout so the pg_sleep doesn't trip it before
        # the cancellation fires.
        config_dict = config.model_dump()
        config_dict["command_timeout_seconds"] = 30
        config = PostgresConnectionConfig(**config_dict)
        return AsyncpgDriver(config)

    def slow_sql(self) -> str:
        """return a postgres-native slow query."""
        return "SELECT pg_sleep(5)"

    @pytest.mark.asyncio
    async def test_cancellation_returns_connection_cleanly_to_pool(
        self,
    ) -> None:
        """after cancellation, the connection stays in the pool (not evicted).

        verifies the cancel-vs-terminate discipline: ``cancel()``
        keeps the connection alive; ``terminate()`` evicts it. after
        a cancelled fetch, issuing a follow-up query MUST work
        without the pool re-opening a fresh connection.
        """
        driver = await self.make_slow_driver()
        try:
            task = asyncio.create_task(driver.fetch(self.slow_sql()))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # follow-up query MUST succeed on the same pool
            rows = await driver.fetch("SELECT 42 AS x")
            assert rows == [{"x": 42}]
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# AGENT_INTERNAL borrowed-pool path
# ---------------------------------------------------------------------------


class TestBorrowedPoolLive:
    """borrowed-pool driver works against a real pool, doesn't close it."""

    @pytest.mark.asyncio
    async def test_borrowed_pool_query_and_close_does_not_close_pool(
        self,
        seeded_schema: tuple[str, str],
    ) -> None:
        """construct a pool, hand to driver, query, close driver, pool stays open."""
        db_url, schema = seeded_schema
        parsed = _parse_db_url(db_url)
        pool = await asyncpg.create_pool(
            host=parsed["host"],
            port=parsed["port"],
            database=parsed["database"],
            user=parsed["username"],
            password=parsed["password"],
            min_size=1,
            max_size=2,
        )
        assert pool is not None
        try:
            config = AgentInternalConnectionConfig(
                datasource_type=DataSourceType.AGENT_INTERNAL,
                schema_name=schema,
            )
            driver = AsyncpgDriver(config, external_pool=pool)
            rows = await driver.fetch(f'SELECT id FROM "{schema}"."widgets" ORDER BY id')
            assert [r["id"] for r in rows] == [1, 2, 3]
            # close the driver -- the pool must remain open
            await driver.close()
            assert not pool.is_closing()
            # the pool is still usable by an external caller
            async with pool.acquire() as conn:
                val = await conn.fetchval("SELECT 7")
                assert val == 7
        finally:
            await pool.close()


# ---------------------------------------------------------------------------
# Microbenchmark (DS-10-13, P1)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="DS-10-13: P1 microbenchmark; manual run only. "
    "bar is <1ms median added latency vs raw pool.fetch over 100 iterations."
)
@pytest.mark.asyncio
async def test_borrowed_pool_microbenchmark_under_one_ms(
    seeded_schema: tuple[str, str],
) -> None:
    """DS-10-13 perf guard: driver wrapper adds <1ms median latency.

    skipped by default; flip the skip marker locally to run. the
    bar lives in the marker reason so a future reviewer can see the
    expected target without grepping for the issue.
    """
    import statistics
    import time

    db_url, schema = seeded_schema
    parsed = _parse_db_url(db_url)
    pool = await asyncpg.create_pool(
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        user=parsed["username"],
        password=parsed["password"],
        min_size=1,
        max_size=2,
    )
    assert pool is not None
    try:
        config = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name=schema,
        )
        driver = AsyncpgDriver(config, external_pool=pool)
        try:
            # warm the connection
            await driver.fetch("SELECT 1")

            raw_durations: list[float] = []
            wrapped_durations: list[float] = []

            for _ in range(100):
                start = time.monotonic()
                await pool.fetch("SELECT 1")
                raw_durations.append(time.monotonic() - start)
                start = time.monotonic()
                await driver.fetch("SELECT 1")
                wrapped_durations.append(time.monotonic() - start)

            raw_median = statistics.median(raw_durations)
            wrapped_median = statistics.median(wrapped_durations)
            added_latency = wrapped_median - raw_median
            assert added_latency < 0.001, f"wrapper added {added_latency * 1000:.3f}ms median; bar is <1ms"
        finally:
            await driver.close()
    finally:
        await pool.close()
