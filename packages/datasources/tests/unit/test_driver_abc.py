"""tests for the :class:`Driver` ABC contract.

scope:

- :class:`Driver` cannot be instantiated directly
- ``__abstractmethods__`` exactly matches the documented set (fetch_iter
  is NOT abstract -- it has a default impl)
- default :meth:`Driver.fetch_iter` yields whatever :meth:`fetch` returns
- :meth:`Driver._with_cancellation` propagates cancellation correctly
  AND invokes the cancel callback before re-raising
- :class:`TableRow` / :class:`ColumnRow` shapes are usable as the
  documented row dicts
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow

from ._helpers.fake_driver import FakeDriver


class TestDriverAbstractness:
    """Driver() raises; subclasses must implement the documented set."""

    def test_cannot_instantiate_raw_driver(self) -> None:
        with pytest.raises(TypeError):
            Driver()  # type: ignore[abstract]

    def test_abstract_methods_match_documented_set(self) -> None:
        """``__abstractmethods__`` MUST equal the documented set exactly.

        fetch_iter is NOT in the set -- it has a default implementation
        that yields from fetch().
        """
        expected = {
            "fetch",
            "execute",
            "list_tables",
            "list_columns",
            "table_hashes",
            "test_connection",
            "close",
        }
        assert set(Driver.__abstractmethods__) == expected

    def test_fetch_iter_is_not_abstract(self) -> None:
        assert "fetch_iter" not in Driver.__abstractmethods__


class TestFetchIterDefault:
    """default :meth:`Driver.fetch_iter` yields from :meth:`fetch`."""

    @pytest.mark.asyncio
    async def test_yields_each_fetch_row(self) -> None:
        rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        driver = FakeDriver(fetch_rows=rows)
        out: list[dict[str, object]] = []
        async for row in driver.fetch_iter("SELECT * FROM whatever"):
            out.append(row)
        assert out == rows

    @pytest.mark.asyncio
    async def test_yields_nothing_for_empty_fetch(self) -> None:
        driver = FakeDriver(fetch_rows=[])
        out: list[dict[str, object]] = []
        async for row in driver.fetch_iter("SELECT 1"):
            out.append(row)
        assert out == []


class TestWithCancellation:
    """:meth:`Driver._with_cancellation` propagates and fires the cancel callback.

    tests route through :meth:`FakeDriver.run_with_cancellation` (the
    public test seam) so this file doesn't need an SLF001 exemption.
    """

    @pytest.mark.asyncio
    async def test_success_path_returns_value(self) -> None:
        driver = FakeDriver()
        cancel_calls: list[str] = []

        async def op() -> int:
            return 42

        result = await driver.run_with_cancellation(
            op, cancel_callback=lambda: cancel_calls.append("called")
        )
        assert result == 42
        assert cancel_calls == []  # success path does not fire the callback

    @pytest.mark.asyncio
    async def test_cancellation_fires_sync_callback_then_reraises(self) -> None:
        driver = FakeDriver()
        cancel_calls: list[str] = []

        async def op() -> int:
            await asyncio.sleep(10)
            return 0  # unreachable

        async def run() -> int:
            return await driver.run_with_cancellation(
                op,
                cancel_callback=lambda: cancel_calls.append("called"),
            )

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancel_calls == ["called"]

    @pytest.mark.asyncio
    async def test_cancellation_awaits_async_callback(self) -> None:
        driver = FakeDriver()
        cancel_calls: list[str] = []

        async def op() -> int:
            await asyncio.sleep(10)
            return 0

        async def async_cancel() -> None:
            cancel_calls.append("async-called")

        async def run() -> int:
            return await driver.run_with_cancellation(
                op,
                cancel_callback=async_cancel,
            )

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancel_calls == ["async-called"]

    @pytest.mark.asyncio
    async def test_callback_exception_is_suppressed(self) -> None:
        """callback raising MUST NOT mask the original CancelledError."""
        driver = FakeDriver()

        async def op() -> int:
            await asyncio.sleep(10)
            return 0

        def broken_callback() -> None:
            raise ValueError("backend cancel hook failed")

        async def run() -> int:
            return await driver.run_with_cancellation(
                op, cancel_callback=broken_callback
            )

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestRowShapes:
    """TableRow + ColumnRow are TypedDicts with the documented keys."""

    def test_table_row_construction(self) -> None:
        row: TableRow = {"table_schema": "public", "table_name": "users"}
        assert row["table_schema"] == "public"
        assert row["table_name"] == "users"

    def test_column_row_keys_match_spec(self) -> None:
        row: ColumnRow = {
            "table_schema": "public",
            "table_name": "users",
            "column_name": "id",
            "data_type": "integer",
            "is_nullable": "NO",  # RAW string, NOT a bool -- spec
            "ordinal_position": 1,
        }
        assert row["is_nullable"] == "NO"
        assert isinstance(row["is_nullable"], str)
        assert row["ordinal_position"] == 1

    def test_column_row_is_nullable_accepts_empty_string(self) -> None:
        """warehouse columns where is_nullable is unknown return ``''``."""
        row: ColumnRow = {
            "table_schema": "s",
            "table_name": "t",
            "column_name": "c",
            "data_type": "text",
            "is_nullable": "",
            "ordinal_position": 1,
        }
        assert row["is_nullable"] == ""


class TestObservedDecoratorCompiles:
    """smoke-test that the observability decorator imports + is invokable.

    the decorator is exercised in earnest in the concrete-driver tests
    (shards 10 / 11). here we just verify the import path is good and
    the decorator wraps without errors. the OTel-passthrough branch
    is verified indirectly: if OTel isn't installed in the test
    environment, the decorator runs the passthrough by default and
    these tests still pass.
    """

    def test_observed_decorator_importable(self) -> None:
        from threetears.datasources.drivers.base import _observed

        assert callable(_observed)

    def test_observed_decorator_rejects_sync_function(self) -> None:
        from threetears.datasources.drivers.base import _observed

        def sync_method(self: object) -> int:
            return 1

        with pytest.raises(TypeError, match="async"):
            _observed("fake")(sync_method)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_observed_decorator_wraps_async_method(self) -> None:
        """decorator returns a coroutine function that produces the wrapped result."""
        from threetears.datasources.drivers.base import _observed

        @_observed("fake")
        async def m(self: object) -> int:
            return 7

        class Holder:
            pass

        result = await m(Holder())
        assert result == 7
