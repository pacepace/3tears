"""an L3 backend that runs no statement and records each one asked of it: a plan test explains the statement the code
under test really writes.

A collection or helper run over :class:`CapturedL3` asks its questions as it would of a pool; each lands in
:attr:`CapturedL3.asked` (the SQL and its arguments, in order) and gets an empty answer, so the test then explains
that exact statement on a seeded database (:func:`threetears.core.testing.plans.explain_statement`). A statement the
test wants answered for real (a helper's one question of the server's version, which spells the page for that engine)
goes to ``real`` when ``passes`` says so. It implements :class:`~threetears.core.backends.protocol.L3Backend`, so it
is handed wherever a pool is (``CollectionRegistry.configure(l3_pool=...)``) without a cast; the structured surface
(``fetch_one``, ``upsert``, ...) is refused, as no plan test reads through it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

__all__ = ["CapturedL3"]


class CapturedL3:
    """an L3 pool that runs nothing and records each statement asked of it.

    :param status: what ``execute`` answers (a write's status)
    :ptype status: str
    :param value: what ``fetchval`` answers
    :ptype value: Any
    :param real: a connection the statements ``passes`` admits are run on, after they are recorded; none when None
    :ptype real: Any
    :param passes: which statements go to ``real``; every one when None (and ``real`` is given)
    :ptype passes: Callable[[str], bool] | None
    """

    def __init__(
        self,
        *,
        status: str = "UPDATE 0",
        value: Any = 0,
        real: Any = None,
        passes: Callable[[str], bool] | None = None,
    ) -> None:
        #: each statement asked, with its arguments, in order
        self.asked: list[tuple[str, tuple[Any, ...]]] = []
        self._status = status
        self._value = value
        self._real = real
        self._passes = passes

    def _runs(self, query: str, params: tuple[Any, ...]) -> bool:
        """record a statement; whether it goes to the real connection."""
        self.asked.append((query, params))
        return self._real is not None and (self._passes is None or self._passes(query))

    async def fetch(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> list[dict[str, Any]]:
        """record ``query``; no rows (or the real connection's).

        :param query: the statement
        :ptype query: str
        :param params: its arguments
        :ptype params: Any
        :param namespace: ignored
        :ptype namespace: str | None
        :param customer_scope: ignored
        :ptype customer_scope: UUID | None
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        if self._runs(query, params):
            return list(await self._real.fetch(query, *params))
        return []

    async def fetchrow(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> dict[str, Any] | None:
        """record ``query``; no row (or the real connection's).

        :param query: the statement
        :ptype query: str
        :param params: its arguments
        :ptype params: Any
        :param namespace: ignored
        :ptype namespace: str | None
        :param customer_scope: ignored
        :ptype customer_scope: UUID | None
        :return: the row, or None
        :rtype: dict[str, Any] | None
        """
        if self._runs(query, params):
            return await self._real.fetchrow(query, *params)  # type: ignore[no-any-return]
        return None

    async def fetchval(
        self, query: str, *params: Any, namespace: str | None = None, customer_scope: UUID | None = None
    ) -> Any:
        """record ``query``; the configured value (or the real connection's).

        :param query: the statement
        :ptype query: str
        :param params: its arguments
        :ptype params: Any
        :param namespace: ignored
        :ptype namespace: str | None
        :param customer_scope: ignored
        :ptype customer_scope: UUID | None
        :return: the value
        :rtype: Any
        """
        if self._runs(query, params):
            return await self._real.fetchval(query, *params)
        return self._value

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        """record ``query``; the configured status (or the real connection's).

        :param query: the statement
        :ptype query: str
        :param params: its arguments
        :ptype params: Any
        :param namespace: ignored
        :ptype namespace: str | None
        :return: the status
        :rtype: str
        """
        if self._runs(query, params):
            return str(await self._real.execute(query, *params))
        return self._status

    async def execute_batch(
        self, queries: list[dict[str, Any]], *, namespace: str | None = None, transaction: bool = True
    ) -> list[Any]:
        """refused: no plan test sends a batch.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records single statements")

    def acquire(self) -> Any:
        """refused: no plan test holds a connection of a captured pool.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool has no connection to lend")

    def transaction(self, namespace: str | None = None) -> Any:
        """refused: no plan test runs a transaction on a captured pool.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool runs no transaction")

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        """refused: the structured surface is not captured.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records SQL statements only")

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
        """refused: the structured surface is not captured.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records SQL statements only")

    async def delete(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> None:
        """refused: the structured surface is not captured.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records SQL statements only")

    async def scan(self, table: str, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """refused: the structured surface is not captured.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records SQL statements only")

    async def upsert_many(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        max_rows: int,
        max_bytes: int,
        conn: Any = None,
    ) -> int:
        """refused: the structured surface is not captured.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a captured pool records SQL statements only")
