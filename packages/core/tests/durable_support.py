"""a durable L3 a test hands the registry when the test is not about L3, and a raw pool whose statements go elsewhere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from threetears.core.backends.protocol import DurableStore, RawL3Pool


class NoL3(DurableStore):
    """a durable L3 with no ``transaction()``: it satisfies ``DurableStore``, so the registry does not wrap it as SQL.

    For a test not about L3. A bare ``object()`` there was wrapped in ``SqlL3Backend``, whose ``transaction()``
    calls ``.acquire()``; the flush took its atomic-batch path, failed, and fell back to per-entity writes through
    a broad ``except``. This goes to the per-entity writes directly, which is what such a test means. Its
    collections write and read their rows themselves, so a call that reaches this store is a test that went wrong.
    """

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        raise AssertionError(f"NoL3 read {table}: the test's collection reads its own rows")

    async def upsert(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: Sequence[str] | None = None,
        on_conflict: str = "update",
        cas: datetime | None = None,
        conn: Any = None,
    ) -> int:
        raise AssertionError(f"NoL3 wrote {table}: the test's collection writes its own rows")

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        raise AssertionError(f"NoL3 deleted from {table}: the test's collection deletes its own rows")

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        raise AssertionError(f"NoL3 scanned {table}: the test's collection reads its own rows")


class ConnOnlyPool(RawL3Pool):
    """a raw transport a collection wraps as its SQL store, for a test that hands every statement a connection.

    The collection reads what it needs of the transport (``rows_per_statement``, when a subclass states it) and
    sends its statements to the ``conn=`` given, so a statement reaching this pool is a test that went wrong.
    """

    def acquire(self) -> Any:
        raise AssertionError("ConnOnlyPool acquired: the test's statements go to the connection it gives")

    async def fetch(self, query: str, /, *params: Any) -> Any:
        raise AssertionError(f"ConnOnlyPool fetched {query!r}: the test's statements go to the connection it gives")

    async def fetchrow(self, query: str, /, *params: Any) -> Any:
        raise AssertionError(f"ConnOnlyPool fetched {query!r}: the test's statements go to the connection it gives")

    async def execute(self, query: str, /, *params: Any) -> Any:
        raise AssertionError(f"ConnOnlyPool ran {query!r}: the test's statements go to the connection it gives")
