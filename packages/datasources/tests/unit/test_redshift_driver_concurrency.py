"""concurrency guarantees for :class:`RedshiftDriver` under a burst of queries.

these pin the three bounds the per-datasource ``executor_max_workers`` /
``connection_cache_size`` sizing delivers, exercised by firing many concurrent
``fetch()`` calls against an instrumented fake ``redshift_connector``:

1. concurrent in-thread query EXECUTION never exceeds ``executor_max_workers``
   -- the bridge's bounded :class:`ThreadPoolExecutor` is the cap;
2. the warm-connection cache never retains more than ``connection_cache_size``
   connections -- the deque ``maxlen`` evicts + closes the overflow;
3. simultaneously-OPEN connections never exceed ``connection_cache_size`` -- the
   driver holds an acquisition :class:`asyncio.Semaphore` sized to the cache, so
   the (cache_size + 1)th concurrent caller WAITS before opening a connection
   instead of opening one past the warehouse user's CONNECTION LIMIT.

guarantee (3) is the hard cap that bounds connections to the Redshift user's
CONNECTION LIMIT: a burst of N concurrent fetch() can no longer open N
connections. it tracks ``connection_cache_size`` specifically, NOT
``executor_max_workers`` -- proven by ``test_open_cap_tracks_cache_size_not_workers``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

#: more concurrent calls than any pool below, so every bound is stressed.
_CONCURRENT_CALLS = 12
#: per-open / per-query holds long enough that concurrent work overlaps in
#: real wall-clock without making the suite slow.
_OPEN_HOLD_S = 0.03
_QUERY_HOLD_S = 0.05


class _ConcurrencyMeter:
    """thread-safe peak tracker for open connections + concurrent executes.

    the driver runs every backend call in an executor worker thread, so the
    counters are mutated off the event loop -- a lock is mandatory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.open_now = 0
        self.open_peak = 0
        self.exec_now = 0
        self.exec_peak = 0

    def on_open(self) -> None:
        """record a fresh connection opening."""
        with self._lock:
            self.open_now += 1
            self.open_peak = max(self.open_peak, self.open_now)

    def on_close(self) -> None:
        """record a connection closing."""
        with self._lock:
            self.open_now -= 1

    def on_exec_enter(self) -> None:
        """record a SELECT entering execution."""
        with self._lock:
            self.exec_now += 1
            self.exec_peak = max(self.exec_peak, self.exec_now)

    def on_exec_exit(self) -> None:
        """record a SELECT leaving execution."""
        with self._lock:
            self.exec_now -= 1


def _make_fake_connect(meter: _ConcurrencyMeter) -> Any:
    """build a ``connect()`` stand-in that tracks open/close + execute overlap.

    :param meter: peak tracker the fake increments/decrements
    :ptype meter: _ConcurrencyMeter
    :return: a callable matching ``redshift_connector.connect``'s kwargs shape
    :rtype: Any
    """

    def _connect(**_kwargs: Any) -> MagicMock:
        meter.on_open()
        time.sleep(_OPEN_HOLD_S)  # stand in for the TLS+auth handshake cost
        conn = MagicMock(name="FakeConn")

        def _cursor() -> MagicMock:
            cur = MagicMock(name="FakeCursor")
            cur.description = [("col", None)]

            def _execute(sql: str, *_a: Any) -> None:
                # only the real SELECT holds the connection busy; the per-open
                # SET statement_timeout / search_path calls are cheap setup.
                if sql.strip().upper().startswith("SELECT"):
                    meter.on_exec_enter()
                    time.sleep(_QUERY_HOLD_S)
                    meter.on_exec_exit()

            cur.execute = MagicMock(side_effect=_execute)
            cur.fetchall = MagicMock(return_value=[(1,)])
            cur.close = MagicMock(return_value=None)
            return cur

        conn.cursor = MagicMock(side_effect=_cursor)
        conn.commit = MagicMock(return_value=None)
        conn.rollback = MagicMock(return_value=None)
        conn.close = MagicMock(side_effect=meter.on_close)
        return conn

    return _connect


def _config(*, max_workers: int, cache_size: int) -> RedshiftConnectionConfig:
    """build a redshift config with an explicit pool sizing.

    :param max_workers: bridge executor ceiling
    :ptype max_workers: int
    :param cache_size: warm-connection cache size (and the open-connection cap)
    :ptype cache_size: int
    :return: config with the requested pool sizing
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_ref=None,
        executor_max_workers=max_workers,
        connection_cache_size=cache_size,
        query_timeout_seconds=60,
    )


async def _run_burst(
    meter: _ConcurrencyMeter,
    *,
    max_workers: int,
    cache_size: int,
    calls: int = _CONCURRENT_CALLS,
) -> RedshiftDriver:
    """fire ``calls`` concurrent fetch() through one driver; return it (open).

    :param meter: peak tracker wired into the fake connect
    :ptype meter: _ConcurrencyMeter
    :param max_workers: bridge executor ceiling
    :ptype max_workers: int
    :param cache_size: warm-connection cache size
    :ptype cache_size: int
    :param calls: number of concurrent fetch() calls to fire
    :ptype calls: int
    :return: the driver after the burst, before close
    :rtype: RedshiftDriver
    """
    with patch(
        "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
        side_effect=_make_fake_connect(meter),
    ):
        driver = RedshiftDriver(
            _config(max_workers=max_workers, cache_size=cache_size),
            datasource_name="concurrency-test",
        )
        results = await asyncio.gather(*[driver.fetch("SELECT 1") for _ in range(calls)])
        # every concurrent call returns its row -- the burst completes, none error,
        # and the semaphore never deadlocks under load.
        assert len(results) == calls
        assert all(rows == [{"col": 1}] for rows in results)
    return driver


@pytest.mark.asyncio
async def test_concurrent_execution_capped_at_max_workers() -> None:
    """concurrent query execution never exceeds the bridge's worker ceiling."""
    meter = _ConcurrencyMeter()
    driver = await _run_burst(meter, max_workers=5, cache_size=5)
    await driver.close()
    # the burst is larger than the pool, so execution saturates the cap...
    assert meter.exec_peak == 5
    # ...and never exceeds it.
    assert meter.exec_peak <= 5


@pytest.mark.asyncio
async def test_connection_cache_bounded_by_cache_size() -> None:
    """the warm-connection cache never retains more than ``connection_cache_size``."""
    meter = _ConcurrencyMeter()
    driver = await _run_burst(meter, max_workers=5, cache_size=5)
    assert driver._cache.maxlen == 5  # noqa: SLF001 - test inspects pool state
    assert len(driver._cache) <= 5  # noqa: SLF001
    await driver.close()


@pytest.mark.asyncio
async def test_open_connections_capped_at_cache_size() -> None:
    """the acquisition semaphore caps simultaneously-open connections at the cache size.

    a burst of 12 concurrent fetch() against a 5/5 pool opens at most 5
    connections -- the 6th caller waits on the semaphore rather than opening a
    connection past the warehouse user's CONNECTION LIMIT.
    """
    meter = _ConcurrencyMeter()
    driver = await _run_burst(meter, max_workers=5, cache_size=5)
    await driver.close()
    assert meter.open_peak <= 5


@pytest.mark.asyncio
async def test_open_cap_tracks_cache_size_not_workers() -> None:
    """the open-connection cap is ``connection_cache_size``, independent of workers.

    with more workers (5) than cache (3), the semaphore -- sized to the cache --
    is the binding cap: at most 3 connections open at once even though 5 executor
    threads are available.
    """
    meter = _ConcurrencyMeter()
    driver = await _run_burst(meter, max_workers=5, cache_size=3)
    await driver.close()
    assert meter.open_peak <= 3
