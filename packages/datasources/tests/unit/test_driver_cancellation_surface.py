"""dsd-task-02: the drivers' cancellation claims, bound to the REAL backend surface.

this module exists because the previous asyncpg cancellation tests
mocked ``conn.cancel`` -- supplying a method
:class:`asyncpg.Connection` does not have. the driver called it at
three sites, every call raised :class:`AttributeError` inside
:meth:`Driver._with_cancellation`, the helper logged a warning and
swallowed it, and cancellation returned to the caller while the
backend statement ran to completion. a test whose own scaffolding
supplies the missing method can never see that.

so every assertion here binds against the installed library rather
than a mock of it:

- the attribute pins read :class:`asyncpg.Connection` and
  :class:`redshift_connector.Connection` directly, so the day either
  library gains or loses a cancellation verb this module fails and the
  driver's recorded decision gets revisited on purpose
- the AST pin walks the asyncpg driver module and fails on any
  cancellation-verb attribute access, so the removed call sites cannot
  quietly come back
- the behavioural tests drive a real :class:`AsyncpgDriver` over a
  connection mock built with ``spec=asyncpg.Connection``. the spec is
  the whole point: attribute access outside the real class surface
  raises :class:`AttributeError`, exactly as it does in production
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
import redshift_connector

from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers import asyncpg_driver as asyncpg_driver_module
from threetears.datasources.drivers import base as driver_base_module
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.entities import DataSourceType

from ._helpers.driver_shims import PoolAcquireHandle

#: attribute names that would mean "this driver cancels an in-flight
#: statement itself". the public one does not exist on
#: :class:`asyncpg.Connection`; the private ones are asyncpg's own
#: protocol machinery, which asyncpg already drives on our behalf.
_CANCELLATION_VERBS = frozenset({"cancel", "_cancel", "_cancel_current_command"})


class _RecordingLogger:
    """stand-in for a module ``log`` that records warning calls.

    the swallowed :class:`AttributeError` surfaced as exactly one
    ``log.warning`` from :meth:`Driver._with_cancellation`, so "no
    warning was emitted" is the observable that distinguishes a driver
    which performs no backend cancellation from one which tries and
    silently fails.

    :param warnings: accumulated warning messages, in call order
    :ptype warnings: list[str]
    """

    def __init__(self) -> None:
        """start with an empty warning log.

        :return: nothing
        :rtype: None
        """
        self.warnings: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """record a warning message.

        :param message: log message
        :ptype message: str
        :param args: positional interpolation arguments
        :ptype args: Any
        :param kwargs: keyword arguments (``extra`` and friends)
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        """
        self.warnings.append(message)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """swallow debug output.

        :param message: log message
        :ptype message: str
        :param args: positional interpolation arguments
        :ptype args: Any
        :param kwargs: keyword arguments
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        """

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """swallow info output.

        :param message: log message
        :ptype message: str
        :param args: positional interpolation arguments
        :ptype args: Any
        :param kwargs: keyword arguments
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        """


def _build_blocking_pool(block: asyncio.Event) -> MagicMock:
    """build an ``asyncpg.Pool`` mock whose connection blocks until ``block`` is set.

    the connection is ``MagicMock(spec=asyncpg.Connection)``: reading an
    attribute the real class does not carry raises
    :class:`AttributeError` here exactly as it would in production. that
    is the property the old mocks destroyed by assigning
    ``conn.cancel``.

    :param block: event the mocked statements await; never set, so the
        statement stays in flight until the awaiting task is cancelled
    :ptype block: asyncio.Event
    :return: pool mock with acquire / release / close wired, carrying
        the spec'd connection on ``pool.recorded_conn``
    :rtype: MagicMock
    """
    conn = MagicMock(spec=asyncpg.Connection, name="SpecdConn")

    async def _block(sql: str, *params: Any, **kwargs: Any) -> Any:
        """await the never-set event so the statement stays in flight."""
        await block.wait()
        return []

    conn.fetch = AsyncMock(side_effect=_block)
    conn.execute = AsyncMock(side_effect=_block)
    conn.fetchval = AsyncMock(side_effect=_block)

    transaction = MagicMock(name="SpecdTransaction")
    transaction.start = AsyncMock(return_value=None)
    transaction.commit = AsyncMock(return_value=None)
    transaction.rollback = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=transaction)

    pool = MagicMock(name="BlockingPool")
    # asyncpg's acquire is both awaitable and an async context manager;
    # the driver uses the awaitable form for a pinned transaction and
    # the context-manager form for single statements.
    pool.acquire = MagicMock(side_effect=lambda *a, **k: PoolAcquireHandle(pool, conn))
    pool.release = AsyncMock(return_value=None)
    pool.close = AsyncMock(return_value=None)
    pool.recorded_conn = conn
    pool.transaction_handle = transaction
    return pool


@pytest.fixture
def postgres_config() -> PostgresConnectionConfig:
    """minimal postgres config; no ``password_ref`` so nothing resolves a secret.

    :return: config the driver accepts without touching a backend
    :rtype: PostgresConnectionConfig
    """
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
    )


@pytest.fixture
def recording_log(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    """swap :mod:`threetears.datasources.drivers.base`'s logger for a recorder.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the recorder installed on the base module
    :rtype: _RecordingLogger
    """
    recorder = _RecordingLogger()
    monkeypatch.setattr(driver_base_module, "log", recorder)
    return recorder


class TestBackendCancellationSurface:
    """what the installed libraries actually expose, read from the classes.

    these are the pins. they do not touch driver code at all -- their
    only job is to fail the moment a library's cancellation surface
    changes, so the decision recorded in the asyncpg driver's module
    docstring is re-made deliberately instead of drifting.
    """

    def test_asyncpg_connection_has_no_public_cancel(self) -> None:
        """``asyncpg.Connection`` exposes no public ``cancel``.

        the whole defect in one assertion. if this ever fails, asyncpg
        gained a public cancellation verb and the driver's "we issue no
        cancel of our own" decision is worth revisiting.
        """
        assert not hasattr(asyncpg.Connection, "cancel")

    def test_asyncpg_cancellation_machinery_is_private(self) -> None:
        """asyncpg's only cancellation verbs are private protocol machinery.

        naming them explicitly documents what the alternative option
        would have had to reach for, and fails if the names move.
        """
        assert hasattr(asyncpg.Connection, "_cancel")
        assert hasattr(asyncpg.Connection, "_cancel_current_command")

    def test_asyncpg_cancels_on_its_own_when_the_waiter_is_cancelled(self) -> None:
        """asyncpg's protocol re-requests cancellation itself; the driver need not.

        ``Protocol._new_waiter`` registers ``_on_waiter_completed`` as a
        done-callback on every query waiter, and that callback calls
        ``_request_cancel()`` when the waiter was cancelled. so a
        cancelled task already produces a pgwire CancelRequest without
        the driver doing anything -- which is why the driver adding its
        own would be a duplicate racing asyncpg's state machine.
        """
        protocol_source = (Path(asyncpg.__file__).parent / "protocol" / "protocol.pyx").read_text()
        assert "self.waiter.add_done_callback(self._on_waiter_completed)" in protocol_source
        assert "con._cancel_current_command(self.cancel_sent_waiter)" in protocol_source

    def test_redshift_connection_has_no_cancel(self) -> None:
        """``redshift_connector`` exposes no cancellation verb on connection or cursor.

        the redshift driver already says so in its module docstring and
        cancels by terminating the server-side backend instead. this pin
        makes that documented claim testable rather than a comment.
        """
        assert not hasattr(redshift_connector.Connection, "cancel")
        assert not hasattr(redshift_connector.Cursor, "cancel")


class TestAsyncpgDriverMakesNoCancellationClaim:
    """the asyncpg driver module must not reference a cancellation verb.

    AST-based, so it catches the reintroduction of ``conn.cancel()`` at
    a fourth call site as readily as at the original three.
    """

    def test_module_contains_no_cancellation_verb_access(self) -> None:
        """no attribute access named ``cancel`` / ``_cancel`` / ``_cancel_current_command``."""
        source = Path(asyncpg_driver_module.__file__).read_text()
        tree = ast.parse(source)
        offenders = [
            f"{node.attr} at line {node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in _CANCELLATION_VERBS
        ]
        assert offenders == []


class TestAsyncpgDriverCancellationAgainstRealSurface:
    """cancellation behaviour driven over a ``spec=asyncpg.Connection`` mock.

    every test here would have caught the original defect: a driver that
    calls ``conn.cancel()`` against a spec'd connection raises
    :class:`AttributeError` inside :meth:`Driver._with_cancellation`,
    which logs the warning these tests assert is absent.
    """

    async def test_fetch_cancellation_touches_no_missing_attribute(
        self,
        postgres_config: PostgresConnectionConfig,
        recording_log: _RecordingLogger,
    ) -> None:
        """cancelling ``fetch`` propagates and emits no cancel-callback warning."""
        block = asyncio.Event()
        pool = _build_blocking_pool(block)
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        task = asyncio.create_task(driver.fetch("SELECT pg_sleep(30)"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recording_log.warnings == []

    async def test_execute_cancellation_touches_no_missing_attribute(
        self,
        postgres_config: PostgresConnectionConfig,
        recording_log: _RecordingLogger,
    ) -> None:
        """cancelling ``execute`` propagates and emits no cancel-callback warning."""
        block = asyncio.Event()
        pool = _build_blocking_pool(block)
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        task = asyncio.create_task(driver.execute("CREATE TABLE t AS SELECT pg_sleep(30)"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recording_log.warnings == []

    async def test_transaction_fetch_cancellation_touches_no_missing_attribute(
        self,
        postgres_config: PostgresConnectionConfig,
        recording_log: _RecordingLogger,
    ) -> None:
        """cancelling a transaction ``fetch`` propagates with no cancel-callback warning.

        the second of the three original call sites: it reached for
        ``checkout.conn.cancel()`` without even the counter bump the
        single-statement path had.
        """
        block = asyncio.Event()
        pool = _build_blocking_pool(block)
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        transaction = await driver.begin()
        task = asyncio.create_task(transaction.fetch("SELECT pg_sleep(30)"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recording_log.warnings == []

    async def test_transaction_execute_cancellation_touches_no_missing_attribute(
        self,
        postgres_config: PostgresConnectionConfig,
        recording_log: _RecordingLogger,
    ) -> None:
        """cancelling a transaction ``execute`` propagates with no cancel-callback warning.

        the third original call site.
        """
        block = asyncio.Event()
        pool = _build_blocking_pool(block)
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        transaction = await driver.begin()
        task = asyncio.create_task(transaction.execute("DROP TABLE t"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recording_log.warnings == []

    async def test_spec_would_have_caught_the_original_defect(
        self,
        postgres_config: PostgresConnectionConfig,
        recording_log: _RecordingLogger,
    ) -> None:
        """a cancel callback reaching for ``conn.cancel`` warns on a spec'd connection.

        the meta-test. it drives :meth:`Driver._with_cancellation` with
        exactly the callback the driver used to pass, over the same
        spec'd connection, and asserts the warning appears. without this
        the "no warning" assertions above could pass for the wrong
        reason -- e.g. because the recorder was never wired up.
        """
        block = asyncio.Event()
        pool = _build_blocking_pool(block)
        conn = pool.recorded_conn
        driver = AsyncpgDriver(postgres_config, external_pool=pool)

        async def _run() -> Any:
            """route a blocking statement through the shared helper with the old callback."""
            return await driver._with_cancellation(  # noqa: SLF001 -- exercising the shared helper directly
                lambda: conn.fetch("SELECT pg_sleep(30)"),
                cancel_callback=lambda: conn.cancel(),
            )

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recording_log.warnings == ["cancel callback failed; backend work may still be running"]
