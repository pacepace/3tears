"""live integration tests for :class:`AsyncpgDriver` against a testcontainer.

verifies the contract end-to-end against a real postgres:

- pool creation + ``test_connection`` round-trip
- ``fetch`` / ``execute`` with ``$N`` placeholders
- ``fetch_iter`` server-side streaming (memory-bounded)
- ``list_tables`` / ``list_columns`` / ``table_hashes`` discover seed schema
- Tier-2 hash byte-equivalence between python helper + warehouse MD5
- cancellation propagation (the driver issues no cancel of its own -- dsd-task-02)
- AGENT_INTERNAL borrowed-pool: driver does NOT close the borrowed pool
- microbenchmark guard rail (DS-10-13; gated, manual run only)

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import tracemalloc
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.datasources.introspection import compute_column_hash

from threetears.datasources.config import (
    BorrowedPoolConnectionConfig,
    PostgresConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.base import Driver
from threetears.datasources.entities import DataSourceType

from ..unit.helpers.cancellation_contract import (
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
    """python-side hash, delegating to the CANONICAL library helper.

    A test that re-implements the thing it verifies proves the two
    implementations agree, which is not the claim. Delegating means the
    assertion compares the WAREHOUSE against the LIBRARY, which is.

    :param cols: column rows carrying ``column_name``, ``data_type``,
        ``is_nullable``, ``ordinal_position``
    :ptype cols: list[dict[str, Any]]
    :return: hex MD5 digest
    :rtype: str
    """
    return compute_column_hash(cols)


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
# least-privilege users: the catalog is what the user can SELECT
# ---------------------------------------------------------------------------


_READER_ROLE = "ds_it_least_privilege_reader"
_READER_PASSWORD = "ds-it-reader"  # noqa: S105 - a throwaway role inside the test container


@pytest.fixture
async def least_privilege_reader(
    seeded_schema: tuple[str, str],
) -> AsyncIterator[tuple[str, str]]:
    """a login role that can SELECT ``widgets`` and a mixed-case ``"Gadgets"``, and only INSERT ``secrets``.

    the INSERT grant is the point: ``information_schema`` lists a table its user holds ANY
    privilege on, so ``secrets`` is visible to this role and still unreadable -- the shape the
    reports warehouse users had, where the catalog listed tables every read of which raised 42501.

    yields ``(db_url_as_the_reader, schema)``.
    """
    db_url, schema = seeded_schema
    parsed = _parse_db_url(db_url)
    conn = await asyncpg.connect(
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        user=parsed["username"],
        password=parsed["password"],
    )
    try:
        await conn.execute(f'CREATE TABLE "{schema}"."secrets" (id integer, token text)')
        await conn.execute(f'CREATE TABLE "{schema}"."Gadgets" (id integer)')
        await conn.execute(f"DROP ROLE IF EXISTS {_READER_ROLE}")
        await conn.execute(f"CREATE ROLE {_READER_ROLE} LOGIN PASSWORD '{_READER_PASSWORD}'")
        await conn.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO {_READER_ROLE}')
        await conn.execute(f'GRANT SELECT ON "{schema}"."widgets", "{schema}"."Gadgets" TO {_READER_ROLE}')
        await conn.execute(f'GRANT INSERT ON "{schema}"."secrets" TO {_READER_ROLE}')
        reader_url = (
            f"postgresql://{_READER_ROLE}:{_READER_PASSWORD}@{parsed['host']}:{parsed['port']}/{parsed['database']}"
        )
        yield reader_url, schema
    finally:
        # the role owns no objects but holds grants on the schema; drop those first
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.execute(f"DROP ROLE IF EXISTS {_READER_ROLE}")
        await conn.close()


class TestIntrospectionCatalogsOnlySelectableTables:
    """a table the datasource user cannot SELECT never reaches the catalog, the hash probe or the columns."""

    @pytest.mark.asyncio
    async def test_every_catalog_read_sees_only_the_selectable_tables(
        self,
        least_privilege_reader: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reader_url, schema = least_privilege_reader
        config = _make_config_for_container(reader_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            tables = {row["table_name"] for row in await driver.list_tables([schema])}
            columns = {row["table_name"] for row in await driver.list_columns([schema])}
            hashed = {table for _schema, table in await driver.table_hashes([schema])}
        finally:
            await driver.close()
        assert tables == {"widgets", "Gadgets"}
        assert columns == {"widgets", "Gadgets"}
        assert hashed == {"widgets", "Gadgets"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("read", ["list_tables", "list_columns", "table_hashes"])
    async def test_a_table_dropped_while_the_catalog_is_read_is_skipped_not_fatal(
        self, seeded_schema: tuple[str, str], read: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a relation dropped mid-read (a dbt promote's DROP, a rename) must not fail the catalog read.

        deterministic: the driver reads on a connection holding a REPEATABLE READ snapshot taken
        before a second session drops the table, so ``information_schema`` still returns its rows. a
        privilege check by NAME resolves against the current catalog and raises ``relation does not
        exist`` -- for every table in scope; by OID it answers NULL and the relation is left out.
        """
        db_url, schema = seeded_schema
        parsed = _parse_db_url(db_url)
        connect = {
            "host": parsed["host"],
            "port": parsed["port"],
            "database": parsed["database"],
            "user": parsed["username"],
            "password": parsed["password"],
        }
        reader = await asyncpg.connect(**connect)
        dropper = await asyncpg.connect(**connect)
        driver = AsyncpgDriver(_make_config_for_container(db_url, monkeypatch), external_pool=_OneConnection(reader))
        try:
            await dropper.execute(f'CREATE TABLE "{schema}"."doomed" (id integer)')
            async with reader.transaction(isolation="repeatable_read"):
                await reader.fetchval("SELECT 1")  # the snapshot is taken here
                await dropper.execute(f'DROP TABLE "{schema}"."doomed"')
                seen = await reader.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = $1", schema
                )
                assert "doomed" in {r["table_name"] for r in seen}, (
                    "the snapshot no longer shows the dropped table; the test proves nothing"
                )
                answered = await getattr(driver, read)([schema])
            names = {key[1] for key in answered} if read == "table_hashes" else {r["table_name"] for r in answered}
            assert names == {"widgets"}
        finally:
            await reader.close()
            await dropper.close()


class _OneConnection:
    """a pool lending one connection: the driver reads inside the transaction the test holds open on it."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    @contextlib.asynccontextmanager
    async def acquire(self, **_: Any) -> AsyncIterator[asyncpg.Connection]:
        yield self._conn

    async def close(self) -> None:
        """the test closes the connection itself."""


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

    @pytest.mark.asyncio
    async def test_fetch_at_most_reads_the_first_rows_in_order_and_holds_no_more(
        self,
        seeded_schema: tuple[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a statement with no LIMIT of its own over 10k rows: 3 rows read, in order, and memory far below a full read."""
        db_url, schema = seeded_schema
        config = _make_config_for_container(db_url, monkeypatch)
        driver = AsyncpgDriver(config)
        try:
            await driver.execute(f'CREATE TABLE "{schema}"."bounded" (id integer, payload text)')
            payload = "x" * 200
            values = ", ".join(f"({i}, '{payload}')" for i in range(10000))
            await driver.execute(f'INSERT INTO "{schema}"."bounded" (id, payload) VALUES {values}')
            statement = f'SELECT id, payload FROM "{schema}"."bounded" WHERE id >= $1 ORDER BY id DESC'

            tracemalloc.start()
            everything = await driver.fetch(statement, 0, timeout_seconds=30)
            _, fetch_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del everything

            tracemalloc.start()
            rows = await driver.fetch_at_most(statement, 0, max_rows=3, timeout_seconds=30)
            _, bounded_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert [row["id"] for row in rows] == [9999, 9998, 9997]
            assert bounded_peak < fetch_peak * 0.1, (
                f"bounded read peak {bounded_peak} against a full read's {fetch_peak}"
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
        """after cancellation, the connection stays usable (not evicted).

        the driver calls neither ``cancel`` (which asyncpg does not
        have) nor ``terminate`` (which would evict the connection):
        asyncpg's own protocol requests the backend cancel and
        ``Pool.release`` resets the session. after a cancelled fetch,
        issuing a follow-up query MUST therefore work.
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
            config = BorrowedPoolConnectionConfig(
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
        config = BorrowedPoolConnectionConfig(
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
