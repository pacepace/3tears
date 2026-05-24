"""reusable cancellation-contract test mixin for concrete-driver test classes.

DS-09-13: the cancellation contract is the most-likely-to-be-violated
invariant when a new driver lands. without a scaffold, the next driver
author "tests it manually" with no proof. this mixin gives them a
ready-made set of assertions to inherit.

usage in shards 10 / 11::

    class TestAsyncpgDriverCancellation(DriverCancellationContractTest):
        async def make_slow_driver(self) -> Driver:
            # real AsyncpgDriver against a local fixture; SQL that
            # blocks long enough to be reliably cancellable.
            return AsyncpgDriver(...)

        def slow_sql(self) -> str:
            return "SELECT pg_sleep(5)"

the mixin runs the standard cancellation assertions against whatever
driver + SQL the concrete test class supplies. concrete classes only
provide the two factory methods; the assertions themselves stay in
one place across all driver implementations.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod

import pytest

from threetears.datasources.drivers.base import Driver

__all__ = ["DriverCancellationContractTest"]


class DriverCancellationContractTest:
    """mixin asserting :class:`Driver` cancellation contract for any concrete driver.

    concrete-driver test classes inherit this mixin and supply
    :meth:`make_slow_driver` + :meth:`slow_sql`. the mixin runs the
    canonical cancellation assertions; the concrete class adds its
    own driver-specific tests on top.
    """

    @abstractmethod
    async def make_slow_driver(self) -> Driver:
        """return a fully-constructed concrete driver ready to use.

        the test will ``await driver.close()`` after each assertion;
        construct a fresh driver for each test.

        :return: driver instance with backend connectivity wired up
        :rtype: Driver
        """

    @abstractmethod
    def slow_sql(self) -> str:
        """return a SQL statement that blocks long enough to be reliably cancelled.

        typical shape: ``SELECT pg_sleep(5)`` for postgres-compatible
        backends, equivalents for redshift / snowflake / bigquery.

        :return: blocking SQL text
        :rtype: str
        """

    @pytest.mark.asyncio
    async def test_fetch_propagates_cancellation(self) -> None:
        """awaiting :meth:`fetch` cancellation re-raises CancelledError.

        the contract: cancelling the task running ``fetch`` MUST
        result in :class:`asyncio.CancelledError` propagating to the
        awaiting code, not a silent return or wrapped exception.
        """
        driver = await self.make_slow_driver()
        try:
            task = asyncio.create_task(driver.fetch(self.slow_sql()))
            # let the task actually start the backend call
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_execute_propagates_cancellation(self) -> None:
        """:meth:`execute` honours the same propagation contract as fetch."""
        driver = await self.make_slow_driver()
        try:
            task = asyncio.create_task(driver.execute(self.slow_sql()))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await driver.close()
