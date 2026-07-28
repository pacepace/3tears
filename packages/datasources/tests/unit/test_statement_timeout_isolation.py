"""per-statement timeout isolation + read-path transaction close (dsd-task-01).

covers DSD-01-03 / DSD-01-04 / DSD-01-05.

**the leak test is the point of this module.** in production the leak
is nondeterministic -- a bounded aggregate that lowers
``statement_timeout`` with a session-level ``SET`` hands the low value
to whatever borrows that cached connection next, and a build inheriting
it dies early depending on which connection it drew. here it is
deterministic: borrow, apply a short override, release, re-borrow, and
assert the second borrower does not observe the first's timeout.

also covers the two structural guards that keep the leak from coming
back one careless line at a time:

- an AST scan for a bare ``SET statement_timeout`` outside the two
  legitimate sites (connection open, release-path reset)
- release-path symmetry: the release path both closes the transaction
  and restores the ceiling, in a ``finally``
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from threetears.datasources.config import (
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

from ._helpers.driver_shims import (
    build_mock_redshift_connection,
    build_transaction_capable_pool,
)

_DRIVERS_DIR = Path(__file__).resolve().parents[2] / "src" / "threetears" / "datasources" / "drivers"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def redshift_config() -> RedshiftConnectionConfig:
    """redshift config whose ceiling (300s) is distinguishable from overrides.

    :return: config with a one-connection cache so re-borrow is guaranteed
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_ref=None,
        executor_max_workers=1,
        connection_cache_size=1,
        query_timeout_seconds=300,
    )


@pytest.fixture
def postgres_config() -> PostgresConnectionConfig:
    """postgres config for the mocked asyncpg driver.

    :return: config for the mocked asyncpg driver
    :rtype: PostgresConnectionConfig
    """
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
    )


def _timeout_statements(sql_log: list[str]) -> list[str]:
    """return every statement that touches ``statement_timeout``.

    :param sql_log: full statement log from the mocked backend
    :ptype sql_log: list[str]
    :return: statements mentioning statement_timeout, in issue order
    :rtype: list[str]
    """
    return [sql for sql in sql_log if "statement_timeout" in sql]


def _session_scoped_timeout_statements(sql_log: list[str]) -> list[str]:
    """return only the SESSION-scoped ``SET statement_timeout`` statements.

    ``SET LOCAL`` unwinds with the transaction and cannot leak; a bare
    ``SET`` persists on the cached connection and is exactly the leak.
    ``RESET`` is the release path putting the session back, which is
    the opposite of a leak.

    :param sql_log: full statement log from the mocked backend
    :ptype sql_log: list[str]
    :return: session-scoped SET statements, in issue order
    :rtype: list[str]
    """
    return [
        sql for sql in _timeout_statements(sql_log) if not sql.startswith("SET LOCAL ") and not sql.startswith("RESET ")
    ]


# ---------------------------------------------------------------------------
# DSD-01-03: the override reaches the session
# ---------------------------------------------------------------------------


class TestRedshiftPerStatementOverride:
    """``fetch`` / ``execute`` accept a per-statement timeout override."""

    @pytest.mark.asyncio
    async def test_fetch_without_override_issues_no_extra_timeout_statement(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """no override -> only the open-time ceiling is ever set."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            timeouts = _timeout_statements(conn.sql_log)
            assert timeouts == ["SET statement_timeout TO 300000"]
            await driver.close()

    @pytest.mark.asyncio
    async def test_fetch_override_uses_set_local(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """DSD-01-04: the override is transaction-local, never a bare SET."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1", timeout_seconds=120)
            assert "SET LOCAL statement_timeout TO 120000" in conn.sql_log
            await driver.close()

    @pytest.mark.asyncio
    async def test_override_precedes_the_caller_statement(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the SET LOCAL must land before the statement it is meant to bound."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT count(*) FROM big", timeout_seconds=120)
            set_local_index = conn.sql_log.index("SET LOCAL statement_timeout TO 120000")
            statement_index = conn.sql_log.index("SELECT count(*) FROM big")
            assert set_local_index < statement_index
            await driver.close()

    @pytest.mark.asyncio
    async def test_execute_override_uses_set_local(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the reaper's ``DROP`` gets its own bound the same way."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.execute("DROP TABLE t", timeout_seconds=300)
            assert "SET LOCAL statement_timeout TO 300000" in conn.sql_log
            await driver.close()

    @pytest.mark.asyncio
    async def test_transaction_statement_override_uses_set_local(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the override is available inside a transaction too."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            async with driver.transaction() as transaction:
                await transaction.execute("CREATE TABLE t AS SELECT 1", timeout_seconds=14400)
            assert "SET LOCAL statement_timeout TO 14400000" in conn.sql_log
            await driver.close()

    @pytest.mark.asyncio
    async def test_non_positive_override_rejected(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """a non-positive override is a caller bug, not a silent no-timeout."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            with pytest.raises(ValueError, match="positive"):
                await driver.fetch("SELECT 1", timeout_seconds=0)
            await driver.close()


class TestAsyncpgPerStatementOverride:
    """the asyncpg driver presents the same override surface."""

    @pytest.mark.asyncio
    async def test_fetch_without_override_issues_no_timeout_statement(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """no override -> no statement_timeout traffic at all."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.fetch("SELECT 1")
        assert _timeout_statements(pool.sql_log) == []
        await driver.close()

    @pytest.mark.asyncio
    async def test_fetch_override_uses_set_local(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """DSD-01-04 on the stand-in: transaction-local, never a bare SET."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.fetch("SELECT 1", timeout_seconds=120)
        assert "SET LOCAL statement_timeout TO 120000" in pool.sql_log
        assert _session_scoped_timeout_statements(pool.sql_log) == []
        await driver.close()

    @pytest.mark.asyncio
    async def test_override_runs_inside_a_transaction(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """``SET LOCAL`` outside a transaction block is a no-op with a warning.

        so the override MUST open one; the mocked transaction handle
        proves the driver did.
        """
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.fetch("SELECT 1", timeout_seconds=120)
        assert pool.transaction_handle.start.await_count == 1
        assert pool.transaction_handle.commit.await_count == 1
        await driver.close()

    @pytest.mark.asyncio
    async def test_execute_override_uses_set_local(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """the same override surface on the write path."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.execute("DROP TABLE t", timeout_seconds=300)
        assert "SET LOCAL statement_timeout TO 300000" in pool.sql_log
        await driver.close()

    @pytest.mark.asyncio
    async def test_non_positive_override_rejected(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """a non-positive override is a caller bug on both drivers."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        with pytest.raises(ValueError, match="positive"):
            await driver.fetch("SELECT 1", timeout_seconds=-1)
        await driver.close()


# ---------------------------------------------------------------------------
# DSD-01-04: THE LEAK TEST
# ---------------------------------------------------------------------------


class TestTimeoutDoesNotLeakAcrossCacheHit:
    """borrow, set short, release, re-borrow: the timeout must not survive.

    the cache-hit case is the one the original bug hinged on -- the
    timeout is applied only at connection open and only the search path
    is re-applied on a hit, so a session-level ``SET`` from the previous
    borrower is still in force when a build draws that connection.
    """

    @pytest.mark.asyncio
    async def test_short_override_does_not_reach_next_borrower(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """a 120s bounded aggregate must not hand 120s to the next borrower."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ) as connect_mock:
            driver = RedshiftDriver(redshift_config)
            # borrow 1: the bounded aggregate lowers the timeout
            await driver.fetch("SELECT count(*) FROM t", timeout_seconds=120)
            first_borrow_end = len(conn.sql_log)
            # borrow 2: the build. same connection (cache hit), long bound
            await driver.execute("CREATE TABLE big AS SELECT 1", timeout_seconds=14400)
            connect_mock.assert_called_once()  # proves the cache hit
            second_borrow = conn.sql_log[first_borrow_end:]
            # the ONLY timeout the second borrower set is its own
            assert [sql for sql in second_borrow if "statement_timeout" in sql and sql.startswith("SET LOCAL")] == [
                "SET LOCAL statement_timeout TO 14400000"
            ]
            # and no session-scoped SET carried 120000 anywhere
            assert not any("120000" in sql for sql in _session_scoped_timeout_statements(conn.sql_log))
            await driver.close()

    @pytest.mark.asyncio
    async def test_release_restores_the_ceiling_after_an_override(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """DSD-01-04 belt-and-braces: the release path re-asserts the ceiling.

        ``SET LOCAL`` already makes the leak structurally impossible.
        this reset is the second line of defence for the case where the
        engine degrades ``SET LOCAL`` to session scope -- which cannot
        be verified from a unit test against a real cluster, so the
        driver does not rely on it alone.
        """
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1", timeout_seconds=120)
            # open-time ceiling + release-time restore = two session SETs
            assert _session_scoped_timeout_statements(conn.sql_log) == [
                "SET statement_timeout TO 300000",
                "SET statement_timeout TO 300000",
            ]
            await driver.close()

    @pytest.mark.asyncio
    async def test_release_restores_the_ceiling_on_the_error_path(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """a reset skipped on the error path IS the leak; it must be in a finally."""
        conn = build_mock_redshift_connection()

        class _ProgrammingError(Exception):
            pass

        conn.fail_caller_statement_at(1, _ProgrammingError("relation does not exist"))
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            with pytest.raises(_ProgrammingError):
                await driver.fetch("SELECT * FROM missing", timeout_seconds=120)
            assert _session_scoped_timeout_statements(conn.sql_log) == [
                "SET statement_timeout TO 300000",
                "SET statement_timeout TO 300000",
            ]
            await driver.close()

    @pytest.mark.asyncio
    async def test_no_reset_round_trip_when_no_override_was_applied(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the hot path pays only the transaction close, not a timeout reset."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            assert _session_scoped_timeout_statements(conn.sql_log) == ["SET statement_timeout TO 300000"]
            await driver.close()

    @pytest.mark.asyncio
    async def test_asyncpg_release_resets_after_an_override(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """release-path symmetry: the stand-in resets too, or it diverges silently."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.fetch("SELECT 1", timeout_seconds=120)
        assert "RESET statement_timeout" in pool.sql_log
        await driver.close()

    @pytest.mark.asyncio
    async def test_asyncpg_release_resets_on_the_error_path(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """the reset must be in a ``finally`` on the stand-in as well."""

        class _PostgresError(Exception):
            pass

        pool = build_transaction_capable_pool()
        pool.fail_caller_statement_at(1, _PostgresError("boom"))
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        with pytest.raises(_PostgresError):
            await driver.fetch("SELECT * FROM missing", timeout_seconds=120)
        assert "RESET statement_timeout" in pool.sql_log
        await driver.close()


# ---------------------------------------------------------------------------
# DSD-01-05: the read path closes its transaction on release
# ---------------------------------------------------------------------------


class TestReadPathClosesTransaction:
    """a completed SELECT must leave no open snapshot on the cached connection."""

    @pytest.mark.asyncio
    async def test_successful_fetch_rolls_back_before_release(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """redshift_connector auto-opens a transaction block per statement.

        ``cursor.execute`` issues ``begin transaction`` whenever the
        session is idle and autocommit is off, so a completed SELECT
        holds a snapshot and its locks until something ends the block.
        that is what makes a ``DROP`` block behind a *finished* query.
        """
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            conn.rollback.assert_called()
            await driver.close()

    @pytest.mark.asyncio
    async def test_transaction_closed_before_connection_enters_cache(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """ordering matters: a connection must never enter the cache mid-snapshot."""
        conn = build_mock_redshift_connection()
        cache_state_at_rollback: list[int] = []

        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            conn.rollback.side_effect = lambda: cache_state_at_rollback.append(len(driver._cache))  # noqa: SLF001
            await driver.fetch("SELECT 1")
            assert cache_state_at_rollback, "release path never closed the transaction"
            assert all(size == 0 for size in cache_state_at_rollback), (
                "the connection was already back in the cache when the transaction closed"
            )
            await driver.close()

    @pytest.mark.asyncio
    async def test_fetch_iter_closes_transaction_on_release(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the streaming path shares the release contract."""
        conn = build_mock_redshift_connection(description=[("a", None)])
        conn.recorded_cursor.fetchmany = lambda *a, **k: []
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            async for _row in driver.fetch_iter("SELECT a FROM t"):
                pass  # pragma: no cover -- empty stream
            conn.rollback.assert_called()
            await driver.close()

    @pytest.mark.asyncio
    async def test_open_time_session_settings_are_committed(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """the open-time SETs must survive the release-path ROLLBACK.

        an uncommitted session-level ``SET`` issued inside the block
        ``cursor.execute`` auto-opened is DISCARDED by a later
        ``ROLLBACK``. committing at open is what keeps the ceiling and
        the search path in force for the connection's whole life.
        """
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            timeout_index = conn.sql_log.index("SET statement_timeout TO 300000")
            assert timeout_index >= 0
            assert conn.commit.call_count >= 1, "open-time session settings were never committed"
            await driver.close()


# ---------------------------------------------------------------------------
# Structural guards -- the leak is one careless line away from returning
# ---------------------------------------------------------------------------


def _iter_driver_modules() -> list[Path]:
    """return every driver module to AST-scan.

    :return: sorted list of driver module paths
    :rtype: list[Path]
    """
    return sorted(_DRIVERS_DIR.rglob("*.py"))


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """collect the ids of every string constant used as a docstring or bare expression.

    prose talks about ``SET statement_timeout`` constantly -- correctly
    so, since the mechanism has to be explained. the scan is about
    EXECUTABLE literals, so anything sitting in an ``ast.Expr``
    statement is not code.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: object ids of prose string constants
    :rtype: set[int]
    """
    prose: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            prose.add(id(node.value))
    return prose


def _bare_set_statement_timeout_literals(path: Path) -> list[tuple[int, str]]:
    """find executable string literals that SET ``statement_timeout`` without ``LOCAL``.

    the module-level template constants the drivers build their SQL
    from are exempt by name -- they are the two legitimate
    session-scoped sites (connection open, release-path reset) and both
    are asserted behaviourally above. any OTHER bare ``SET
    statement_timeout`` literal is the leak mechanism returning.

    :param path: driver module to scan
    :ptype path: Path
    :return: list of ``(lineno, literal)`` offenders
    :rtype: list[tuple[int, str]]
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    prose = _docstring_constant_ids(tree)
    exempt_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name.endswith("_SQL_TEMPLATE") or name.endswith("_SQL") for name in targets):
                exempt_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in prose:
            continue
        text = node.value
        if "statement_timeout" not in text:
            continue
        if "SET LOCAL" in text or "RESET " in text:
            continue
        if "SET " not in text:
            continue
        if node.lineno in exempt_lines:
            continue
        hits.append((node.lineno, text))
    return hits


@pytest.mark.parametrize("driver_module", _iter_driver_modules(), ids=lambda p: p.name)
def test_no_ad_hoc_bare_set_statement_timeout(driver_module: Path) -> None:
    """a bare ``SET statement_timeout`` outside the two template constants is the leak.

    the session-scoped form persists on the cached connection and hands
    the previous borrower's bound to the next one. every per-statement
    override MUST be ``SET LOCAL``.

    :param driver_module: driver module under scan
    :ptype driver_module: Path
    :return: nothing
    :rtype: None
    """
    hits = _bare_set_statement_timeout_literals(driver_module)
    assert hits == [], (
        f"bare session-scoped SET statement_timeout in {driver_module.name}: {hits}; "
        "per-statement overrides must use SET LOCAL"
    )


def _release_path_body(source: str, function_name: str) -> ast.AsyncFunctionDef:
    """return the AST node for ``function_name`` in ``source``.

    :param source: module source text
    :ptype source: str
    :param function_name: async function to locate
    :ptype function_name: str
    :return: the function node
    :rtype: ast.AsyncFunctionDef
    :raises AssertionError: when the function is absent
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found; the release path was renamed or removed")


def test_redshift_release_path_runs_in_a_finally() -> None:
    """release-path symmetry: the checkout finish is reached from a ``finally``.

    a reset skipped on the error path is exactly the leak, and the
    transaction close and the timeout restore share one code path so
    they cannot drift apart.
    """
    source = (_DRIVERS_DIR / "redshift_driver.py").read_text()
    node = _release_path_body(source, "_run_with_connection")
    finally_calls: list[str] = []
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Try):
            for handler in stmt.finalbody:
                for inner in ast.walk(handler):
                    if isinstance(inner, ast.Attribute):
                        finally_calls.append(inner.attr)
    assert "_finish_checkout" in finally_calls, (
        "_run_with_connection must reach _finish_checkout from a finally; "
        f"finally body referenced {sorted(set(finally_calls))}"
    )


def test_asyncpg_release_reset_runs_in_a_finally() -> None:
    """the stand-in's timeout reset is reached from a ``finally`` too."""
    source = (_DRIVERS_DIR / "asyncpg_driver.py").read_text()
    node = _release_path_body(source, "_acquire_and_run")
    finally_names: list[str] = []
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Try):
            for handler in stmt.finalbody:
                for inner in ast.walk(handler):
                    if isinstance(inner, ast.Attribute):
                        finally_names.append(inner.attr)
                    elif isinstance(inner, ast.Name):
                        finally_names.append(inner.id)
    assert "_reset_statement_timeout" in finally_names, (
        "_acquire_and_run must reset the statement timeout from a finally; "
        f"finally body referenced {sorted(set(finally_names))}"
    )


def test_shared_timeout_sql_helpers_are_used_by_both_drivers() -> None:
    """both drivers build the SET LOCAL from the ONE shared helper.

    two hand-rolled format strings drift; the helper is the single
    place the millisecond conversion and the LOCAL scoping live.
    """
    for module in ("redshift_driver.py", "asyncpg_driver.py"):
        source = (_DRIVERS_DIR / module).read_text()
        assert "build_set_local_statement_timeout_sql" in source, f"{module} does not use the shared SET LOCAL helper"
