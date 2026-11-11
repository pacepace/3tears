"""the connection a keyed statement is sent on: the two methods every keyed runner calls, asyncpg-shaped.

An asyncpg connection or pool, the L3 rail's proxy, and ``threetears.core.keyed``'s in-process readers all have them;
so may a store's own pool, which runs a keyed builder's statement with no 3tears core installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

__all__ = ["KeyedConnection"]


class KeyedConnection(Protocol):
    """what a keyed statement is run on: ``fetch`` for the rows it answers (a read, or a write's ``RETURNING``),
    ``execute`` for a write's status (``DELETE 3``). Each takes the statement with ``$n`` placeholders and its binds,
    as a builder in :mod:`threetears.interfaces.keyed` writes them."""

    async def fetch(self, query: str, /, *params: Any) -> Sequence[Any]:
        """the rows the statement answers, each read by column name.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its binds, ``$1`` first
        :ptype params: Any
        :return: the rows
        :rtype: Sequence[Any]
        """
        ...

    async def execute(self, query: str, /, *params: Any) -> Any:
        """run the statement, answering its status (``UPDATE 1``).

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its binds, ``$1`` first
        :ptype params: Any
        :return: the status
        :rtype: Any
        """
        ...
