"""minimal in-memory :class:`Driver` subclass used across the unit-test suite.

:class:`FakeDriver` is the canonical "real driver shape, no backend lib"
fixture. tests for the ABC's contract (abstract-methods exactly match
the documented set; default ``fetch_iter`` yields what ``fetch``
returns; cancellation helper routes through ``_with_cancellation``)
construct a ``FakeDriver`` rather than mocking the abstract surface
piecemeal.

knobs:

- ``fetch_rows`` -- rows returned by :meth:`fetch` (and therefore the
  default :meth:`fetch_iter`).
- ``slow_seconds`` -- inject artificial latency on every call so
  cancellation tests can race against the operation.
- ``cancel_hook`` -- callable invoked from :meth:`_simulate_query` on
  :class:`asyncio.CancelledError`; lets tests assert the backend-
  cancel hook fired.
- ``raise_on_test_connection`` -- toggle whether
  :meth:`test_connection` raises (for testing failure-path callers).

deliberately NOT in the public API: only the test suite imports it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow

__all__ = ["FakeDriver"]


class FakeDriver(Driver):
    """real :class:`Driver` subclass with no backend behind it.

    every method is implemented but trivially: returns empty lists /
    dicts unless the caller seeds the constructor knobs.
    """

    def __init__(
        self,
        *,
        fetch_rows: list[dict[str, Any]] | None = None,
        table_rows: list[TableRow] | None = None,
        column_rows: list[ColumnRow] | None = None,
        table_hash_map: dict[tuple[str, str], str] | None = None,
        slow_seconds: float = 0.0,
        cancel_hook: Callable[[], Any] | None = None,
        raise_on_test_connection: bool = False,
    ) -> None:
        self._fetch_rows = fetch_rows or []
        self._table_rows = table_rows or []
        self._column_rows = column_rows or []
        self._table_hash_map = table_hash_map or {}
        self._slow_seconds = slow_seconds
        self._cancel_hook = cancel_hook
        self._raise_on_test_connection = raise_on_test_connection
        self._closed = False
        self.cancel_called = 0

    async def _simulate_query(self) -> None:
        """sleep ``slow_seconds`` so cancellation tests can race the operation.

        on :class:`asyncio.CancelledError`, invokes ``cancel_hook`` if
        configured, then re-raises. this is the surface
        :class:`Driver._with_cancellation` tests assert behaviour
        against.

        :return: nothing
        :rtype: None
        """
        if self._slow_seconds > 0:
            try:
                await asyncio.sleep(self._slow_seconds)
            except asyncio.CancelledError:
                self.cancel_called += 1
                if self._cancel_hook is not None:
                    self._cancel_hook()
                raise

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        await self._simulate_query()
        return list(self._fetch_rows)

    async def execute(self, sql: str, *params: Any) -> None:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        await self._simulate_query()

    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        return list(self._table_rows)

    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        return list(self._column_rows)

    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        return dict(self._table_hash_map)

    async def test_connection(self) -> None:
        if self._closed:
            raise RuntimeError("FakeDriver is closed")
        if self._raise_on_test_connection:
            raise RuntimeError("fake driver test_connection failure (seeded)")

    async def close(self) -> None:
        self._closed = True

    async def run_with_cancellation(
        self,
        coro_fn: Callable[[], Any],
        *,
        cancel_callback: Callable[[], Any],
    ) -> Any:
        """public test seam around :meth:`Driver._with_cancellation`.

        the helper is intentionally private on the ABC (concrete
        drivers route every backend call through it; callers never
        invoke it directly). tests still need to assert the
        propagation contract, so :class:`FakeDriver` exposes a thin
        public wrapper -- avoiding underscore access on the ABC from
        test files.

        :param coro_fn: see :meth:`Driver._with_cancellation`
        :ptype coro_fn: Callable[[], Awaitable[Any]]
        :param cancel_callback: see :meth:`Driver._with_cancellation`
        :ptype cancel_callback: Callable[[], Any]
        :return: the wrapped coroutine's result
        :rtype: Any
        """
        return await self._with_cancellation(coro_fn, cancel_callback=cancel_callback)
