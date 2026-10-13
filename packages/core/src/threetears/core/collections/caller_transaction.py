"""a caller-owned L3 transaction that collection writes join, with every cache settled after it ends.

``save_entity(entity, conn=conn)`` writes L3 inside a transaction the CALLER commits or rolls back,
so when the save returns its row is not yet a fact: L3 may still refuse the commit, or the caller
may roll it back. Anything a cache tier takes from it before then can outlive it:

- written to L2 or L1, a rolled-back row stays cached though L3 never held it;
- evicted instead, a reader in the window reads L3's committed -- older -- row and caches it, and
  once the caller commits nothing is left to evict that.

So the caches cannot be settled from inside the save. They are settled when the transaction has
ended, whichever way it ended, and only the code that ends it knows when that is. A
:class:`CallerTransaction` is that code: it opens the transaction on the caller's connection,
collection writes that join it (``conn=`` the same connection) record the keys they wrote, and when
the transaction has committed or rolled back each key is evicted from L1 and L2 and the eviction
is broadcast. The next read of the key takes whichever row L3 ended with.

On a collection switched on to carry a write generation
(:attr:`~threetears.core.collections.base.BaseCollection.write_generation`), settling is also where
the generation advances: once per table for the whole transaction, however many of its rows the
transaction wrote and however many collection instances of the table wrote them, with every row's
broadcast naming that one advance and the number of rows it covers. It advances whichever
way the transaction ended, for the reason the eviction runs either way.

Evicting rather than writing the committed row is deliberate: it is correct whether the
transaction committed, rolled back, or rolled back a savepoint that a save joined, with no need to
know which. A save refuses a ``conn`` no :class:`CallerTransaction` opened, rather than guess.

Usage::

    async with pool.acquire() as conn:
        async with CallerTransaction(conn):
            await conn.execute(...)
            await collection.save_entity(entity, conn=conn)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

from threetears.core.exceptions import GenerationUnavailableError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.collections.base import BaseCollection

__all__ = ["CallerTransaction", "SharedAdvance", "shared_advance_for"]

log = get_logger(__name__)

#: every CallerTransaction open in this context, outermost first. A context variable, so a task
#: started inside the transaction still finds it, and one started outside never does.
_OPEN: ContextVar[tuple[CallerTransaction, ...]] = ContextVar("threetears_caller_transactions", default=())


@dataclass(slots=True)
class SharedAdvance:
    """one table's write-generation advance for a settling transaction that more than one collection
    instance of the table wrote.

    The first instance to advance makes it, for every row of the table the transaction wrote; each
    later one stamps its broadcasts with that same advance. A follower then counts the advance
    complete only once it has heard every row's broadcast, whichever instance sent it.

    :ivar rows: how many rows of the table the transaction settles, across every instance
    :ivar bump: the advance, once the first instance has made it; ``None`` before
    """

    rows: int
    bump: object | None = None


#: the shared advances of the transaction settling in this context, by table.
_SHARED: ContextVar[Mapping[str, SharedAdvance]] = ContextVar("threetears_shared_advances", default={})


def shared_advance_for(table_name: str) -> SharedAdvance | None:
    """the advance a settling transaction shares among the instances of ``table_name``, if any.

    :param table_name: the table a collection is advancing
    :ptype table_name: str
    :return: the shared advance, or ``None`` when the collection advances on its own
    :rtype: SharedAdvance | None
    """
    return _SHARED.get().get(table_name)


class CallerTransaction:
    """open a transaction on a caller's L3 connection, and settle the caches its writes touched once it ends.

    Nests: an inner one on the same connection opens a savepoint (whatever the connection's own
    ``transaction()`` does when nested) and leaves settling to the outermost, which is the only one
    whose end makes the writes final.

    :ivar conn: the connection the transaction is open on
    """

    __slots__ = ("_enrolled", "_options", "_outermost", "_token", "_transaction", "conn")

    def __init__(self, conn: Any, **options: Any) -> None:
        """name the connection to open the transaction on.

        :param conn: the backend-specific connection (an asyncpg connection, or the proxy
            connection :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend` yields)
        :ptype conn: Any
        :param options: passed to the connection's ``transaction()`` -- isolation, and the like
        :ptype options: Any
        """
        self.conn = conn
        self._options = options
        self._enrolled: list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]] = []
        self._outermost: CallerTransaction | None = None
        self._token: Token[tuple[CallerTransaction, ...]] | None = None
        self._transaction: Any = None

    @staticmethod
    def enclosing(conn: Any) -> CallerTransaction | None:
        """the outermost :class:`CallerTransaction` open on ``conn`` in this context, or ``None``.

        :param conn: the connection a write was handed
        :ptype conn: Any
        :return: the transaction whose end settles the write's caches, or ``None``
        :rtype: CallerTransaction | None
        """
        found: CallerTransaction | None = None
        for scope in _OPEN.get():
            if scope.conn is conn:
                found = scope
                break
        return found

    @staticmethod
    def join(conn: Any, *, writer: str) -> CallerTransaction:
        """the :class:`CallerTransaction` a write handed ``conn`` joins; refuse a connection none opened.

        The one refusal every collection write that takes ``conn=`` makes: a row written inside a
        transaction the caller ends later is not final when the write returns, and only the
        transaction's end is where every cache of it can be settled. A connection whose transaction
        this class did not open has no such end to settle at.

        :param conn: the connection the write was handed
        :ptype conn: Any
        :param writer: the refusing call, as the caller wrote it (``"Schedules.resume"``)
        :ptype writer: str
        :return: the outermost :class:`CallerTransaction` open on ``conn``
        :rtype: CallerTransaction
        :raises ValueError: when no :class:`CallerTransaction` is open on ``conn`` in this context
        """
        found = CallerTransaction.enclosing(conn)
        if found is None:
            raise ValueError(
                f"{writer}(conn=...) needs the connection's transaction opened by "
                f"threetears.core.collections.CallerTransaction(conn): the row is not final until the "
                f"caller's transaction ends, and only that is where every cache of it can be settled"
            )
        return found

    def enroll(self, collection: BaseCollection[Any], entity_id: Any, *, row: dict[str, Any] | None = None) -> None:
        """record that ``collection`` wrote ``entity_id`` in this transaction, to be evicted when it ends.

        A key written twice is evicted twice, which costs one more L2 delete and nothing else.

        :param collection: the collection that wrote the row
        :ptype collection: BaseCollection
        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param row: the row as the write sent it, when the write saw one; the eviction's broadcast
            takes the collection's declared invalidation columns from it
        :ptype row: dict[str, Any] | None
        :return: nothing
        :rtype: None
        """
        self._enrolled.append((collection, entity_id, row))

    async def __aenter__(self) -> CallerTransaction:
        """open the transaction and make it findable from the writes that join it.

        :return: this transaction
        :rtype: CallerTransaction
        """
        self._outermost = CallerTransaction.enclosing(self.conn)
        transaction = self.conn.transaction(**self._options)
        await transaction.__aenter__()
        self._transaction = transaction
        self._token = _OPEN.set((*_OPEN.get(), self))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """end the transaction, then -- outermost only -- evict every key its writes touched.

        The eviction runs however the transaction ended, a failed commit included, and runs to
        completion even when the caller is cancelled meanwhile: an eviction left half done leaves
        exactly the stale row it exists to remove.

        :param exc_type: the exception type the body raised, or ``None``
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception the body raised, or ``None``
        :ptype exc: BaseException | None
        :param traceback: its traceback, or ``None``
        :ptype traceback: TracebackType | None
        :return: whatever the connection's transaction returned, so it may suppress as it would alone
        :rtype: bool
        :raises GenerationUnavailableError: when the transaction ended cleanly and a collection
            switched on to carry a write generation could not advance it. Every row was still
            evicted and broadcast. A body or a commit that raised keeps its own error
        """
        generation_failure: GenerationUnavailableError | None = None
        try:
            suppressed = await self._transaction.__aexit__(exc_type, exc, traceback)
        finally:
            if self._token is not None:
                _OPEN.reset(self._token)
                self._token = None
            if self._outermost is None:
                generation_failure = await asyncio.shield(self._settle(committed=exc_type is None))
        if generation_failure is not None and exc_type is None:
            raise generation_failure
        return bool(suppressed)

    async def _settle(self, *, committed: bool) -> GenerationUnavailableError | None:
        """evict every key a write joined to this transaction touched, from L1 and L2, and broadcast it.

        One call per collection, and one advance per table: when two collection instances of one
        table wrote rows, the first to advance does so for the rows of both (:class:`SharedAdvance`),
        so every broadcast names the same advance and the same count. One collection's failed advance
        does not stop the next collection's rows being settled.

        :param committed: whether the body finished without raising, for the log; the commit itself
            may still have failed, and the eviction is the same either way
        :ptype committed: bool
        :return: the first failure to advance a write generation, for the caller to raise; ``None``
            when every advance that was owed was made
        :rtype: GenerationUnavailableError | None
        """
        enrolled = list(self._enrolled)
        self._enrolled.clear()
        # one call per collection, with every key it wrote, in the order written: a load of many
        # rows settles them together rather than one round trip at a time
        by_collection: dict[int, tuple[BaseCollection[Any], list[Any], list[dict[str, Any] | None]]] = {}
        for collection, entity_id, row in enrolled:
            _, keys, rows = by_collection.setdefault(id(collection), (collection, [], []))
            keys.append(entity_id)
            rows.append(row)
        per_table: dict[str, list[int]] = {}
        for collection, keys, _ in by_collection.values():
            per_table.setdefault(collection.table_name, []).append(len(keys))
        shared = {table: SharedAdvance(rows=sum(counts)) for table, counts in per_table.items() if len(counts) > 1}
        sharing = _SHARED.set(shared)
        try:
            failure = await self._settle_each(by_collection)
        finally:
            _SHARED.reset(sharing)
        if enrolled:
            log.debug(
                "caller transaction ended; the rows its writes touched were evicted from every cache",
                extra={"extra_data": {"rows": len(enrolled), "body_completed": committed}},
            )
        return failure

    @staticmethod
    async def _settle_each(
        by_collection: dict[int, tuple[BaseCollection[Any], list[Any], list[dict[str, Any] | None]]],
    ) -> GenerationUnavailableError | None:
        """settle each collection's keys, in turn.

        :param by_collection: each collection with the keys it wrote and the rows it saw
        :ptype by_collection: dict[int, tuple[BaseCollection, list[Any], list[dict[str, Any] | None]]]
        :return: the first failure to advance a write generation; ``None`` when there was none
        :rtype: GenerationUnavailableError | None
        """
        failure: GenerationUnavailableError | None = None
        for collection, keys, rows in by_collection.values():
            try:
                if collection.invalidation_columns:
                    await collection.invalidate_cache_many(keys, rows=rows)
                else:
                    # the call every collection has always been given, so one that overrides it
                    # without the keyword is still settled
                    await collection.invalidate_cache_many(keys)
            except GenerationUnavailableError as exc:
                # logged where the advance failed; this collection's rows were all evicted
                if failure is None:
                    failure = exc
        return failure
