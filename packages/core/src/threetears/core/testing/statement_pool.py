"""a raw L3 pool that answers only the statements a test says it answers.

A unit test hands a collection or a registry an L3 that records or answers the SQL it is sent, with
no database behind it. :class:`StatementPool` IS a
:class:`~threetears.core.backends.protocol.RawL3Pool`, so ``CollectionRegistry.configure(l3_pool=...)``
and a collection's ``l3_pool`` take it where they take an asyncpg pool, and the test type-checks
without a cast. A subclass overrides the statements its test answers; any other statement fails
loudly, naming the SQL, rather than answering as if a table were empty.

.. code-block:: python

    class _Recording(StatementPool):
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
            self.statements.append(sql)
            return []
"""

from __future__ import annotations

from typing import Any

from threetears.core.backends.protocol import RawL3Pool

__all__ = ["StatementPool"]


class StatementPool(RawL3Pool):
    """a :class:`RawL3Pool` with no table behind it: a statement no subclass answers raises."""

    def acquire(self) -> Any:
        """no connection to lend: every statement a test sends goes to the pool itself.

        :raises NotImplementedError: always
        """
        raise NotImplementedError("a statement pool lends no connections")

    async def fetch(self, query: str, /, *params: Any) -> Any:
        """a SELECT this test does not answer.

        :raises NotImplementedError: always, naming the statement
        """
        raise NotImplementedError(f"unanswered fetch: {query}")

    async def fetchrow(self, query: str, /, *params: Any) -> Any:
        """a single-row SELECT this test does not answer.

        :raises NotImplementedError: always, naming the statement
        """
        raise NotImplementedError(f"unanswered fetchrow: {query}")

    async def execute(self, query: str, /, *params: Any) -> Any:
        """a statement this test does not answer.

        :raises NotImplementedError: always, naming the statement
        """
        raise NotImplementedError(f"unanswered execute: {query}")
