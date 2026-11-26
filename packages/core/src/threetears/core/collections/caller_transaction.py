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

On a table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`),
the end is also where the commit's delta file is saved, BEFORE the commit (:mod:`~threetears.core.collections.delta_files`):
each write of the table records what it wrote here beside its enrollment (:meth:`record_rows`, :meth:`record_arrow`,
:meth:`record_deletes`, and a key-prefix delete its prefixes, :meth:`record_prefixes`), and a commit is refused, the
transaction rolled back, unless every key and every key prefix the transaction enrolled for the table is in the file
and the file is saved. That is the enforcement every writer of the table goes through: a write that enrolls a key or a
prefix it did not record, or a pod with no delta files directory configured, all refuse the commit rather than make a
version no file holds.
The names saved are :attr:`delta_files`, for the commit's epoch record; once the commit has landed,
:attr:`committed_generations` names the generation token it wrote for each table, which the writer folds the files by
(:meth:`~threetears.core.collections.delta_fold.DeltaFolds.committed`).

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
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from itertools import chain, repeat
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Final, Protocol

from threetears.core.backends.broker_generation import (
    StampedScopes,
    collecting_advanced_generations,
    holding_touched_scopes,
    committed_generations_record,
    naming_commit_generations,
    naming_commit_s_min,
    naming_commit_scopes,
    naming_delta_files,
    releasing_touched_scopes,
    note_statement_stamp,
    reply_generations_since,
)
from threetears.core.collections.delta_files import (
    OWN_DELTA_SAVES,
    DeltaFileRefusedError,
    TableDelta,
    delta_files_root,
    versioned_write_id,
)
from threetears.core.collections.delta_save import save_in_worker
from threetears.core.collections.epoch_change import EVERY_STATE_NAMED, ChangedRow, recordable_scopes
from threetears.core.ddl.row_epoch import (
    MAX_TRANSACTION_AGE_MS,
    TransactionTooOldError,
    UnstampedWriteError,
    epoch_wall_ms,
    next_epoch,
    touch_last_epoch_sql,
    wall_clock_ms,
)
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.frozen_state import with_chunk
from threetears.core.loop_turns import LoopTurns, emptied, stepped
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.collections.base import BaseCollection

__all__ = ["CallerTransaction", "SharedAdvance", "save_delta_files", "shared_advance_for"]

log = get_logger(__name__)

#: the most scopes one touch statement names (:meth:`CallerTransaction._touch_scopes`), one array of them, as
#: ``ScopeEpochs`` batches its own
_TOUCH_BATCH: Final = 500

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

    :ivar changed: every row of the table the transaction settles, across every instance: the
        advance's change record and its row count
    :ivar bump: the advance, once the first instance has made it; ``None`` before
    """

    changed: list[ChangedRow]
    bump: object | None = None

    @property
    def rows(self) -> int:
        """how many rows of the table the transaction settles, across every instance.

        :return: the count
        :rtype: int
        """
        return len(self.changed)


class _PrefixSettling(Protocol):
    """a collection that deleted key prefixes on a caller's transaction, settled by them when it ends
    (:meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.delete_prefixes`).
    """

    @property
    def table_name(self) -> str:
        """the table.

        :return: its name
        :rtype: str
        """
        ...

    @property
    def writes_delta_files(self) -> bool:
        """whether the table opts in to delta files, which a key-prefix delete cannot keep.

        :return: whether it does
        :rtype: bool
        """
        ...

    async def settle_prefixes(self, prefixes: Sequence[Mapping[str, str]]) -> None:
        """settle every cache of the rows of ``prefixes`` once the transaction that deleted them has ended.

        :param prefixes: each prefix, the key's leading columns and their values as text
        :ptype prefixes: Sequence[Mapping[str, str]]
        :return: nothing
        :rtype: None
        :raises GenerationUnavailableError: when the table's write generation had to advance and could not
        """
        ...


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


#: from how many enrolled rows a settling transaction groups them and builds their change records on a worker thread,
#: not the loop: fewer cost less there than the thread does (the count from which L1 evictions leave the loop too)
_SETTLE_OFF_LOOP_ROWS: Final = 64

_ByCollection = dict[int, tuple["BaseCollection[Any]", list[Any], list[dict[str, Any] | None]]]

#: one enrollment: the collection that wrote a row, its key, and the row as the write saw it
type _Enrollment = tuple["BaseCollection[Any]", Any, dict[str, Any] | None]

#: one restatement: how many enrollments were made before it, the collection, and each key onto its row as stored
type _Restatement = tuple[int, "BaseCollection[Any]", Mapping[Any, dict[str, Any]]]


def _enrollments(
    chunks: Sequence[tuple[_Enrollment, ...]],
    restated: Sequence[tuple[_Restatement, ...]],
    loop: asyncio.AbstractEventLoop | None = None,
) -> list[_Enrollment]:
    """a transaction's enrollments, in order, each restatement laid over the latest enrollment of each of its keys
    made before it (:meth:`CallerTransaction.restate_many`); pure, so a thread may make them.

    :param chunks: the enrollments, oldest chunk first
    :ptype chunks: Sequence[tuple[_Enrollment, ...]]
    :param restated: the restatements, oldest chunk first
    :ptype restated: Sequence[tuple[_Restatement, ...]]
    :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`); None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the enrollments, a list of this call's own
    :rtype: list[_Enrollment]
    """
    turns = LoopTurns(loop) if loop is not None else None
    enrolled = list(chain.from_iterable(chunks))
    for upto, collection, rows in chain.from_iterable(restated):
        left = dict(rows)
        indices = range(upto - 1, -1, -1)
        for index in indices if turns is None else stepped(indices, turns):
            if not left:
                break
            enrolled_by, key, _ = enrolled[index]
            if enrolled_by is collection and key in left:
                enrolled[index] = (collection, key, left.pop(key))
    return enrolled


def _grouped_enrollments(
    chunks: Sequence[tuple[_Enrollment, ...]],
    restated: Sequence[tuple[_Restatement, ...]],
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[list[_Enrollment], _ByCollection, dict[str, SharedAdvance]]:
    """:func:`_enrollments`, then :func:`_grouped` of them, in one call; pure, so a thread may make them.

    :param chunks: the enrollments, oldest chunk first
    :ptype chunks: Sequence[tuple[_Enrollment, ...]]
    :param restated: the restatements, oldest chunk first
    :ptype restated: Sequence[tuple[_Restatement, ...]]
    :param loop: the caller's event loop, handed the thread's turns; None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the enrollments, and their grouping as :func:`_grouped` makes it
    :rtype: tuple[list[_Enrollment], _ByCollection, dict[str, SharedAdvance]]
    """
    enrolled = _enrollments(chunks, restated, loop)
    by_collection, shared = _grouped(enrolled, loop)
    return enrolled, by_collection, shared


def _mixed_writes(table: str) -> str:
    """the refusal of a keyed write and a key-prefix delete of one table in one transaction.

    :param table: the table
    :ptype table: str
    :return: the message
    :rtype: str
    """
    return (
        f"{table}: one transaction both deleted key prefixes of the table and wrote rows of it by key; its commit is "
        f"recorded as one epoch, which names rows or prefixes, never both. Write the rows in a transaction of their own"
    )


def _grouped(
    enrolled: list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]],
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[_ByCollection, dict[str, SharedAdvance]]:
    """a settling transaction's enrollments, by collection, and the advances its tables share; pure, so a thread may
    make them.

    One call per collection, with every key it wrote, in the order written: a load of many rows settles them together
    rather than one round trip at a time. Only a table more than one instance wrote, and one of them advances, shares an
    advance; no change record is built for any other here: an instance that advances alone builds its own, and one
    that advances nothing needs none. The 33,786 records of a shared advance no instance made cost the commit a
    garbage collection pass with the GIL held, 12 ms on top (2026-11-16).

    :param enrolled: every key the transaction's writes enrolled, in order
    :ptype enrolled: list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]
    :param loop: the caller's event loop, handed the thread's turns (:class:`LoopTurns`): without them a 33,786-key
        commit's grouping held the interpreter lock 1.6-3.3 ms at a stretch (2026-11-16); None on the loop itself
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: each collection with its keys and the rows its writes saw; each table more than one instance wrote,
        with the advance they share
    :rtype: tuple[dict[int, tuple[BaseCollection[Any], list[Any], list[dict[str, Any] | None]]],
        dict[str, SharedAdvance]]
    """
    turns = LoopTurns(loop) if loop is not None else None
    by_collection: _ByCollection = {}
    for collection, entity_id, row in enrolled if turns is None else stepped(enrolled, turns):
        _, keys, rows = by_collection.setdefault(id(collection), (collection, [], []))
        keys.append(entity_id)
        rows.append(row)
    instances: dict[str, list[tuple[BaseCollection[Any], list[Any], list[dict[str, Any] | None]]]] = {}
    for each in by_collection.values():
        instances.setdefault(each[0].table_name, []).append(each)
    shared = {
        table: SharedAdvance(
            changed=[
                collection.changed_row(key, row)
                for collection, keys, rows in written
                for key, row in (
                    zip(keys, rows, strict=True) if turns is None else stepped(zip(keys, rows, strict=True), turns)
                )
            ]
        )
        for table, written in instances.items()
        if len(written) > 1 and any(collection.advances_write_generation for collection, _, _ in written)
    }
    return by_collection, shared


def _released(
    enrolled: list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]],
    by_collection: _ByCollection,
    shared: Mapping[str, SharedAdvance],
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """empty a settled transaction's enrollments, their grouping and its shared advances' change records a slice at a
    time, taking turns with the caller's loop (:func:`~threetears.core.loop_turns.emptied`); for a thread. A slice's
    rows are freed in one C call that keeps the GIL throughout: 1,024 rows of 179 columns a slice, with no turns
    between, held the loop 4-5 ms a slice (the WD2 bench, 2026-11-16).

    The grouping's lists were lent to each collection's ``invalidate_cache_many`` for the call, which keeps none of
    them; emptying them in place frees their rows here rather than wherever the last reference happens to drop.

    :param enrolled: the transaction's enrollments, settled
    :ptype enrolled: list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]
    :param by_collection: their grouping by collection (:func:`_grouped`)
    :ptype by_collection: dict[int, tuple[BaseCollection[Any], list[Any], list[dict[str, Any] | None]]]
    :param shared: the advances its tables' instances shared (:func:`_grouped`)
    :ptype shared: Mapping[str, SharedAdvance]
    :param loop: the caller's event loop, handed the thread's turns; None off any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: nothing
    :rtype: None
    """
    # an advance made handed its records to the generation source, which may keep them: those are only let go of.
    # One no instance made never left this module
    held: list[list[Any]] = [advance.changed for advance in shared.values() if advance.bump is None]
    for _, keys, rows in by_collection.values():
        held += (rows, keys)
    emptied([*held, enrolled], loop)


class CallerTransaction:
    """open a transaction on a caller's L3 connection, and settle the caches its writes touched once it ends.

    Nests: an inner one on the same connection opens a savepoint (whatever the connection's own
    ``transaction()`` does when nested) and leaves settling to the outermost, which is the only one
    whose end makes the writes final.

    :ivar conn: the connection the transaction is open on
    """

    __slots__ = (
        "_before_commit",
        "_committed_generations",
        "_deltas",
        "_enrolled",
        "_enrolled_count",
        "_options",
        "_outermost",
        "_reaches",
        "_restated",
        "_s_max",
        "_s_min",
        "_saved_delta_files",
        "_scopes",
        "_states",
        "_token",
        "_touch_token",
        "_touched",
        "_transaction",
        "_write_version",
        "conn",
    )

    def __init__(self, conn: Any, *, timeout_ms: int | None = None, **options: Any) -> None:
        """name the connection to open the transaction on.

        :param conn: the backend-specific connection (an asyncpg connection, or the proxy
            connection :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend` yields)
        :ptype conn: Any
        :param timeout_ms: the transaction's deadline, for a write sized to its work: handed to a connection that
            takes one (``takes_transaction_timeout``, the L3 broker's: its statements' timeout, and how long the hub
            lets each of its requests wait for a slot). A direct pool has no broker to queue at and takes none: its
            statements run under the database's own timeout. None for the connection's own
        :ptype timeout_ms: int | None
        :param options: passed to the connection's ``transaction()`` -- isolation, and the like
        :ptype options: Any
        """
        self.conn = conn
        if timeout_ms is not None and getattr(conn, "takes_transaction_timeout", False) is True:
            options = {**options, "timeout_ms": timeout_ms}
        self._options = options
        # every key the transaction's writes enrolled, in order: append-only chunks, each a tuple, merged
        # log-structured (with_chunk) and replaced whole, never changed in place; restatements are laid over them when
        # read (_enrollments), so a thread handed them never sees them change
        self._enrolled: tuple[tuple[_Enrollment, ...], ...] = ()
        self._enrolled_count = 0
        self._restated: tuple[tuple[_Restatement, ...], ...] = ()
        # the key prefixes deleted whole, a table's collection and prefixes each: settled by prefix, never a key at a
        # time. A tuple replaced whole: a transaction deletes the prefixes of a table or two, looked up by name
        self._reaches: tuple[tuple[_PrefixSettling, tuple[dict[str, str], ...]], ...] = ()
        self._outermost: CallerTransaction | None = None
        self._token: Token[tuple[CallerTransaction, ...]] | None = None
        self._transaction: Any = None
        # the transaction's own state, owned by the task that runs it: each map a read-only snapshot replaced whole,
        # never changed in place, so what a thread is handed is never changed under it
        # what each table that opts in to delta files was written, by table: saved as its file before the commit
        self._deltas: Mapping[str, TableDelta] = MappingProxyType({})
        self._saved_delta_files: Mapping[str, str] = MappingProxyType({})
        self._committed_generations: Mapping[str, str] = MappingProxyType({})
        # the scopes (states) the writer's seqlock touched in this transaction (ScopeEpochs.touch), each once: its
        # commit names them beside its delta files (decision 31)
        self._scopes: frozenset[str] = frozenset()
        # every state the delta files' rows are of, read off them when saved (TableDelta.named_states); None when any
        # one table's cannot be told: the commit's scopes are then not marked complete (decision 54 (c))
        self._states: frozenset[str] | None = None
        # the write's version its data writes took their turn at (ScopeEpochs.touch): its delta files are named by it
        # (decision 53); None for a transaction no fenced write joined, whose files keep the older, unversioned name
        self._write_version: int | None = None
        # the lowest and highest row epoch this transaction stamped (row_epoch): the lowest is its commit's s_min, and
        # its age at the COMMIT (refused past W); the highest is what its writer's scope epochs record as last_epoch
        self._s_min: int | None = None
        self._s_max: int | None = None
        # what runs on the transaction's connection just before its COMMIT, given the highest stamp, in order
        # (before_commit): a tuple replaced whole, owned by the task that runs the transaction
        self._before_commit: tuple[Callable[[Any, int], Awaitable[None]], ...] = ()
        # the scope epochs scopes this transaction's stamped writes touched, by scope epochs table: each one's last_epoch
        # raised to the highest stamp just before the COMMIT (_touch_scopes). Read off the rows by the store
        # (stamped_send) and named by ScopeEpochs.touch alike, so a scope both name is one index entry. A read-only
        # map of frozensets replaced whole, owned by the task that runs the transaction; a scope count, never a row's
        self._touched: Mapping[str, frozenset[str]] = MappingProxyType({})
        self._touch_token: Token[Any] | None = None

    @property
    def delta_files(self) -> Mapping[str, str]:
        """each delta file this transaction saved before its commit, by table: the names, under the delta files'
        directory, its epoch record names. Empty until the transaction has ended, and for a transaction that wrote no
        table that opts in.

        :return: table -> its file's name
        :rtype: Mapping[str, str]
        """
        return self._saved_delta_files

    @property
    def committed_generations(self) -> Mapping[str, str]:
        """each generation token this transaction's commit wrote, by table: what a part folded past its delta file is
        current to (:meth:`~threetears.core.collections.delta_fold.DeltaFolds.committed`). Through the L3 broker, every
        token its commit reply named (``l3.tx.commit``); on a direct pool, each token the advances its settling made
        wrote. Empty until the outermost transaction has ended, and for one that rolled back or whose commit failed.

        :return: table -> the token
        :rtype: Mapping[str, str]
        """
        return self._committed_generations

    def _delta_of(self, collection: BaseCollection[Any]) -> TableDelta:
        """the delta this transaction keeps for ``collection``'s table, made at its first write.

        :param collection: a collection whose table opts in to delta files
        :ptype collection: BaseCollection
        :return: the table's delta
        :rtype: TableDelta
        """
        delta = self._deltas.get(collection.table_name)
        if delta is None:
            delta = collection.table_delta()
            self._deltas = MappingProxyType({**self._deltas, collection.table_name: delta})
        return delta

    def record_rows(
        self,
        collection: BaseCollection[Any],
        rows: Sequence[Mapping[str, Any]],
        written: Sequence[str] | Sequence[Sequence[str]],
    ) -> None:
        """record rows ``collection`` wrote in this transaction, as written, and the columns their statements set in a
        row L3 already held, for its table's delta file; nothing for a table that does not opt in. One row a key the
        write enrolled.

        :param collection: the collection that wrote them
        :ptype collection: BaseCollection
        :param rows: the rows, each keyed by column
        :ptype rows: Sequence[Mapping[str, Any]]
        :param written: the columns set (:func:`~threetears.core.backends.schema_sql.upsert_set_columns`): one list
            for every row (a bulk statement's), or one a row
        :ptype written: Sequence[str] | Sequence[Sequence[str]]
        :return: nothing
        :rtype: None
        """
        if collection.writes_delta_files:
            self._delta_of(collection).rows(rows, written)

    def record_arrow(self, collection: BaseCollection[Any], rows: Any, written: Sequence[str]) -> None:
        """:meth:`record_rows` for rows written as an Arrow table, one statement's ``SET`` list for them all.

        :param collection: the collection that wrote them
        :ptype collection: BaseCollection
        :param rows: the rows, a column each
        :ptype rows: pyarrow.Table
        :param written: the columns the statement sets in a row already held
        :ptype written: Sequence[str]
        :return: nothing
        :rtype: None
        """
        if collection.writes_delta_files:
            self._delta_of(collection).arrow(rows, written)

    def record_deletes(
        self, collection: BaseCollection[Any], keys: Sequence[Sequence[Any]], *, scope: Any = None
    ) -> None:
        """record keys ``collection`` deleted in this transaction, each its values in key order, as its table's delete
        markers; nothing for a table that does not opt in. One key a key the delete enrolled.

        :param collection: the collection that deleted them
        :ptype collection: BaseCollection
        :param keys: the keys
        :ptype keys: Sequence[Sequence[Any]]
        :param scope: for a table whose key does not name a row's state, the keys' state, which each marker carries
            (:meth:`~threetears.core.collections.delta_files.TableDelta.deletes`, decision 27); None otherwise
        :ptype scope: Any
        :return: nothing
        :rtype: None
        :raises DeltaFileRefusedError: when the table's markers carry a state and none is given
        """
        if collection.writes_delta_files:
            self._delta_of(collection).deletes(keys, scope=scope)

    def record_prefixes(self, collection: BaseCollection[Any], prefixes: Sequence[Mapping[str, Any]]) -> None:
        """record key prefixes ``collection`` deleted whole in this transaction, as its table's delta file names them
        (decision 23: a copy and the fold clear their own rows under each); nothing for a table that does not opt in.
        Every prefix the delete enrolled (:meth:`enroll_prefixes`).

        :param collection: the collection that deleted them
        :ptype collection: BaseCollection
        :param prefixes: each prefix, the key's leading columns and their values
        :ptype prefixes: Sequence[Mapping[str, Any]]
        :return: nothing
        :rtype: None
        """
        if collection.writes_delta_files:
            self._delta_of(collection).clears(prefixes)

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
    def stamp_for(conn: Any, table: str) -> int:
        """the stamp one write statement of a stamped table (``ROW_STAMPED``) carries: from ``conn``'s transaction when
        it joins one (:meth:`in_place_epoch`, counted in the commit's ``s_min``), else from the process clock, noted as
        that one-statement write's own ``s_min`` (``note_statement_stamp``), so the epoch record of its own commit
        names it and the restore scrub checks its margin. The one draw of every stamped write, the ones sent outside
        the store's stamped requests too (a keyed update's ``stamped_values``, a declared table's ``write_row``).

        :param conn: the connection the write runs on, or None for the pool
        :ptype conn: Any
        :param table: the table written, as its advance names it
        :ptype table: str
        :return: the stamp
        :rtype: int
        """
        transaction = None if conn is None else CallerTransaction.enclosing(conn)
        if transaction is not None:
            return transaction.in_place_epoch()
        stamp = next_epoch()
        note_statement_stamp(table, stamp)
        return stamp

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
        :raises ValueError: when the transaction deleted key prefixes of the table (:meth:`enroll_prefixes`)
        """
        if self._reaches:
            self._refuse_keyed_beside_prefixes((collection.table_name,))
        self._enroll_chunk(((collection, entity_id, row),))

    @staticmethod
    def enrollments(
        collection: BaseCollection[Any],
        entity_ids: Sequence[Any],
        rows: Sequence[dict[str, Any] | None],
        turns: LoopTurns | None = None,
    ) -> list[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]:
        """the enrollments :meth:`enroll` makes for each of ``entity_ids`` with its row, in order, for
        :meth:`enroll_all`; pure, so a thread may make them.

        A write of many rows makes them on a worker thread and hands them to :meth:`enroll_all` on the loop: one
        enrollment a key, 33,786 of them made on the loop, set off the collector there too (CPython 3.14 walks its
        old generation in increments a young collection starts), the loop held with them.

        :param collection: the collection that wrote the rows
        :ptype collection: BaseCollection
        :param entity_ids: each row's pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_ids: Sequence[Any]
        :param rows: each row as the write sent it, or None, as :meth:`enroll`'s ``row``; as many as ``entity_ids``
        :ptype rows: Sequence[dict[str, Any] | None]
        :param turns: the making thread's turns with the caller's loop (:func:`~threetears.core.loop_turns.stepped`);
            None to make them in one call
        :ptype turns: LoopTurns | None
        :return: the enrollments
        :rtype: list[tuple[BaseCollection, Any, dict[str, Any] | None]]
        :raises ValueError: when ``rows`` and ``entity_ids`` differ in length
        """
        made = zip(repeat(collection, len(entity_ids)), entity_ids, rows, strict=True)
        return list(made if turns is None else stepped(made, turns))

    def enroll_all(self, enrollments: Sequence[tuple[BaseCollection[Any], Any, dict[str, Any] | None]]) -> None:
        """record every one of ``enrollments`` (:meth:`enrollments`), in order, as :meth:`enroll` records one.

        :param enrollments: the enrollments
        :ptype enrollments: Sequence[tuple[BaseCollection, Any, dict[str, Any] | None]]
        :return: nothing
        :rtype: None
        :raises ValueError: when the transaction deleted key prefixes of a table they name (:meth:`enroll_prefixes`)
        """
        if self._reaches:
            self._refuse_keyed_beside_prefixes({collection.table_name for collection, _, _ in enrollments})
        self._enroll_chunk(tuple(enrollments))

    def _enroll_chunk(self, chunk: tuple[_Enrollment, ...]) -> None:
        """append ``chunk`` to the enrollments, which are replaced whole.

        :param chunk: the enrollments, in order
        :ptype chunk: tuple[_Enrollment, ...]
        :return: nothing
        :rtype: None
        """
        if chunk:
            self._enrolled = with_chunk(self._enrolled, chunk)
            self._enrolled_count += len(chunk)

    def note_scopes(self, scopes: Sequence[str]) -> None:
        """record that this transaction's writes touched ``scopes`` (the states, ``state:CA``, as the writer's seqlock
        names them): a commit that saves a delta file names them (``COMMIT_SCOPES_FIELD``), so a record its broker
        writes keys unknown names the states to re-read should the file be lost (decision 31).

        :param scopes: the scopes
        :ptype scopes: Sequence[str]
        :return: nothing
        :rtype: None
        """
        self._scopes = self._scopes | frozenset(scopes)

    def note_write_version(self, version: int) -> None:
        """record the write version this transaction's data writes took their turn at
        (:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.touch`): its delta files are named by it
        (:func:`~threetears.core.collections.delta_files.versioned_write_id`, decision 53), so a full's manifest covers
        them exactly. One version a transaction: a second, different one is refused (a transaction fenced at two writes
        would name its files by neither).

        :param version: the write's version
        :ptype version: int
        :return: nothing
        :rtype: None
        :raises ValueError: when the transaction already noted another version
        """
        if self._write_version is not None and self._write_version != version:
            raise ValueError(
                f"a transaction's data writes took their turn at write {self._write_version}, not {version}: its delta "
                f"files are named by one write's version"
            )
        self._write_version = version

    def row_epoch(self, table: str) -> int:
        """a fresh epoch for one write statement of ``table`` on this transaction, for a table that declares a row epoch
        (:mod:`~threetears.core.ddl.row_epoch`): the process's hybrid logical clock's next stamp, so it sorts newer than
        every epoch drawn before it, in this transaction or any other of the write, and than the previous writer's
        (``ScopeEpochs.begin`` moved the clock past its ``last_epoch``). Only once the transaction took a write's turn
        (:meth:`note_write_version`): that turn is the fence that orders this writer after the previous one.

        :param table: the table written, for the refusal
        :ptype table: str
        :return: the epoch
        :rtype: int
        :raises UnstampedWriteError: when no write took its turn in this transaction: its rows would carry no version
        """
        if self._write_version is None:
            raise UnstampedWriteError(
                f"{table} declares a row epoch, so every write carries its write's epoch, and this transaction took no "
                f"write's turn: call ScopeEpochs.touch(version, ...) on it before writing the table"
            )
        return self.in_place_epoch()

    def in_place_epoch(self) -> int:
        """a fresh epoch for one write statement on this transaction, with or without a write's turn: the process
        clock's next stamp, counted in the transaction's lowest and highest stamps (its ``s_min``, its age refusal, its
        ``before_commit`` steps). What a stamped table's write (``ROW_STAMPED``: one row a key, updated in place) takes
        when its transaction took no turn (``SchemaBackedCollection._epoch_stamp``).

        WHY a stamped table needs no write's turn (:meth:`row_epoch` needs one): its row's epoch rises on every write
        (``GREATEST(epoch + 1, <new>)``, ``rising_epoch_assignment``), so it never goes back whichever process's clock
        stamped it, and a restore reads every row stamped since X. The turn orders a versioned table's versions against
        the previous writer's, which a stamped table does not keep. The hub's own tables have no seqlock writer to
        take a turn from.

        :return: the epoch
        :rtype: int
        """
        epoch = next_epoch()
        # the clock only rises in this process, so the first stamp is the lowest and the last the highest
        if self._s_min is None:
            self._s_min = epoch
        self._s_max = epoch
        return epoch

    def before_commit(self, step: Callable[[Any, int], Awaitable[None]]) -> None:
        """run ``step(conn, s_max)`` on this transaction's connection just before its COMMIT, once the body ended
        cleanly and only when the transaction stamped a row epoch: ``s_max`` is the highest it stamped. In the order
        given. A step that raises refuses the commit: the transaction is rolled back and settled, and the error is the
        caller's. How ``ScopeEpochs`` records the stamp a write left on the scopes it touched (``last_epoch``).

        :param step: the step
        :ptype step: Callable[[Any, int], Awaitable[None]]
        :return: nothing
        :rtype: None
        """
        self._before_commit = (*self._before_commit, step)

    def touch_scopes(self, epochs: str, scopes: Iterable[str]) -> None:
        """raise ``last_epoch`` of ``scopes`` of the scope epochs table ``epochs`` to this transaction's highest stamp
        just before its COMMIT, when it stamped one: how a stamped write's scopes reach the restore, which finds the
        scopes changed since X by ``last_epoch`` alone (``ddl.row_epoch``). Every scope named across the transaction is
        touched once, by one statement a batch of scopes, whoever named it (the store, from the rows it sent; or
        ``ScopeEpochs.touch``).

        :param epochs: the scope epochs table, as the transaction's connection names it
        :ptype epochs: str
        :param scopes: the scopes
        :ptype scopes: Iterable[str]
        :return: nothing
        :rtype: None
        """
        named = frozenset(scopes)
        if not named:
            return
        held = self._touched.get(epochs, frozenset())
        if named <= held:
            return
        self._touched = MappingProxyType({**self._touched, epochs: held | named})

    def _take_stamped_scopes(self, touched: StampedScopes) -> None:
        """take one stamped write's scopes and stamps, read off its rows by the store (``stamped_send``): its scopes to
        touch, and its stamps into the transaction's lowest and highest, which its age and its touch are measured by.

        :param touched: the write's scopes and stamps
        :ptype touched: StampedScopes
        :return: nothing
        :rtype: None
        """
        self.touch_scopes(touched.epochs, touched.scopes)
        self._s_min = touched.s_min if self._s_min is None else min(self._s_min, touched.s_min)
        self._s_max = touched.s_max if self._s_max is None else max(self._s_max, touched.s_max)

    async def _touch_scopes(self, s_max: int) -> None:
        """touch every scope this transaction's stamped writes named with ``s_max`` (:meth:`touch_scopes`), on its
        connection, just before its COMMIT: one statement a batch of scopes, the scopes in order.

        :param s_max: the highest stamp the transaction wrote
        :ptype s_max: int
        :return: nothing
        :rtype: None
        """
        for epochs, scopes in sorted(self._touched.items()):
            ordered = sorted(scopes)
            quoted = quote_identifier(epochs)
            for start in range(0, len(ordered), _TOUCH_BATCH):
                batch = ordered[start : start + _TOUCH_BATCH]
                await self.conn.execute(touch_last_epoch_sql(quoted), s_max, batch)

    async def _ready_to_commit(self) -> None:
        """refuse a commit older than W, touch the scopes its stamped writes named (:meth:`touch_scopes`), then run the
        steps owed before the COMMIT (:meth:`before_commit`).

        :return: nothing
        :rtype: None
        :raises TransactionTooOldError: when the transaction's first stamp is older than
            :data:`~threetears.core.ddl.row_epoch.MAX_TRANSACTION_AGE_MS`
        """
        if self._s_min is None or self._s_max is None:
            return
        age = wall_clock_ms() - epoch_wall_ms(self._s_min)
        if age > MAX_TRANSACTION_AGE_MS:
            # WHY: a restore reads back by M = W + A + S from the keeper's mark; a row committed older than W could sit
            # below that floor and be missed (design-append-only-l3.md, decision 3)
            raise TransactionTooOldError(
                f"this transaction's first row epoch is {age} ms old, past the {MAX_TRANSACTION_AGE_MS} ms a "
                "transaction may run: its commit is refused and it is rolled back. Write in smaller transactions"
            )
        steps, s_max = self._before_commit, self._s_max
        await self._touch_scopes(s_max)
        for step in steps:
            await step(self.conn, s_max)

    def enroll_prefixes(self, collection: _PrefixSettling, prefixes: Sequence[Mapping[str, str]]) -> None:
        """record that ``collection`` deleted every row of ``prefixes`` in this transaction, to be settled by those
        prefixes when it ends: the table's caches dropped and its generation advanced once, its epoch naming the
        prefixes (a follower reads each as a key range), never a key at a time.

        **One kind of write a table a transaction.** A table whose prefixes this transaction deletes takes no keyed
        write in it, before or after, and the reverse: its commit is recorded as ONE epoch, which names either rows or
        prefixes. Write the table's new rows in a transaction of their own.

        :param collection: the collection that deleted them
        :ptype collection: _PrefixSettling
        :param prefixes: each prefix, the key's leading columns and their values as text
        :ptype prefixes: Sequence[Mapping[str, str]]
        :return: nothing
        :rtype: None
        :raises ValueError: when the transaction wrote keyed rows of the table
        """
        table = collection.table_name
        if any(enrolled.table_name == table for chunk in self._enrolled for enrolled, _, _ in chunk):
            raise ValueError(_mixed_writes(table))
        reaches = self._reaches
        at = next((index for index, (each, _) in enumerate(reaches) if each.table_name == table), None)
        owner, held = (collection, ()) if at is None else reaches[at]
        kept = list(held)
        for prefix in prefixes:
            named = dict(prefix)
            if named not in kept:
                kept.append(named)
        entry = (owner, tuple(kept))
        self._reaches = (*reaches, entry) if at is None else (*reaches[:at], entry, *reaches[at + 1 :])

    def _refuse_keyed_beside_prefixes(self, tables: Iterable[str]) -> None:
        """refuse a keyed write of a table whose prefixes this transaction deleted (:meth:`enroll_prefixes`).

        :param tables: the tables the keyed write names
        :ptype tables: Iterable[str]
        :return: nothing
        :rtype: None
        :raises ValueError: when one of them had prefixes deleted
        """
        cleared = {collection.table_name for collection, _ in self._reaches}
        for table in tables:
            if table in cleared:
                raise ValueError(_mixed_writes(table))

    def restate(self, collection: BaseCollection[Any], entity_id: Any, *, row: dict[str, Any]) -> None:
        """replace the row ``collection``'s latest enrollment of ``entity_id`` carries, once the write knows it as stored.

        A write enrolls before it writes, with the row as sent. A write that read its row back
        restates it, so the eviction's broadcast takes its invalidation columns from the row as
        stored. A key not enrolled is left alone.

        :param collection: the collection that wrote the row
        :ptype collection: BaseCollection
        :param entity_id: pk value (single-pk) or tuple of pk values in declared order
        :ptype entity_id: Any
        :param row: the row as stored
        :ptype row: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._restated = with_chunk(
            self._restated, ((self._enrolled_count, collection, MappingProxyType({entity_id: row})),)
        )

    def restate_many(
        self, collection: BaseCollection[Any], rows: Mapping[Any, dict[str, Any]], turns: LoopTurns | None = None
    ) -> None:
        """:meth:`restate` each key of ``rows`` with its row, in one pass over the enrollments.

        Recorded here, on the loop, as one restatement replaced whole into the transaction's; it is laid over the
        enrollments when they are read (:func:`_enrollments`), off the loop for a large transaction: a bulk delete of
        5,000 rows restated a key at a time took about a second and a half, and 33,786 keys restated in one pass held
        the loop 2.6 ms (2026-11-16). It covers the enrollments made before the call: one a later write makes keeps its
        own row.

        :param collection: the collection that enrolled them
        :ptype collection: BaseCollection[Any]
        :param rows: each key, as enrolled, onto the row it now names
        :ptype rows: Mapping[Any, dict[str, Any]]
        :param turns: turns with the caller's loop while ``rows`` is copied (:func:`~threetears.core.loop_turns.stepped`);
            None to copy it in one call
        :ptype turns: LoopTurns | None
        :return: nothing
        :rtype: None
        """
        held = dict(rows.items() if turns is None else stepped(rows.items(), turns))
        self._restated = with_chunk(self._restated, ((self._enrolled_count, collection, MappingProxyType(held)),))

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
        if self._outermost is None:
            # the store hands every stamped write made on this connection its scopes here, before it is sent
            self._touch_token = holding_touched_scopes(self.conn, self._take_stamped_scopes)
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
        :raises DeltaFileRefusedError: when a table that opts in to delta files cannot have its file saved before
            the commit; the transaction was rolled back
        """
        if self._outermost is None and exc_type is None:
            # saved before the commit, so every committed version has its file; a save that fails ends the
            # transaction as a body that raised does, rolled back and settled, and its error is the caller's. A
            # commit older than W, or a step owed before it that fails, ends it the same way
            try:
                await self._ready_to_commit()
                self._saved_delta_files = MappingProxyType(await self._save_delta_files())
            except BaseException as error:  # prawduct:allow prawduct/broad-except -- the commit is refused: rolled back and settled, then re-raised
                await self._end(type(error), error, error.__traceback__)
                raise
            finally:
                self._deltas = MappingProxyType({})
        suppressed, generation_failure = await self._end(exc_type, exc, traceback)
        if generation_failure is not None and exc_type is None:
            raise generation_failure
        return suppressed

    async def _end(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> tuple[bool, GenerationUnavailableError | None]:
        """end the connection's transaction as the body ended, then -- outermost only -- settle every key its writes
        touched.

        :param exc_type: the exception type the body raised, or ``None``
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception the body raised, or ``None``
        :ptype exc: BaseException | None
        :param traceback: its traceback, or ``None``
        :ptype traceback: TracebackType | None
        :return: whether the connection's transaction suppressed the exception, and the first failure to advance a
            write generation
        :rtype: tuple[bool, GenerationUnavailableError | None]
        """
        generation_failure: GenerationUnavailableError | None = None
        # a commit names its delta files: its l3.tx.commit carries them to the broker that writes its epoch records,
        # and an advance this process records itself names them. Only a commit that landed: a record naming the file
        # of a commit that failed would hand every copy rows L3 never took
        files: Mapping[str, str] = self._saved_delta_files if exc_type is None else MappingProxyType({})
        # the states only a commit with a delta file names, bounded: one too wide names none (decision 31); marked
        # complete only when read off its files' rows (decision 54 (c))
        scopes = _commit_scopes(self._scopes, self._states) if files else []
        # the commit's lowest row epoch: its l3.tx.commit carries it, and every epoch record of it names it (s_min)
        s_min = self._s_min if exc_type is None else None
        landed = False
        # the broker's reply to the commit replaces this task's record of its generations; a direct pool's commit
        # leaves it, so a record that did not change since names nothing of this commit's
        before = committed_generations_record()
        try:
            with naming_delta_files(files), naming_commit_scopes(scopes), naming_commit_s_min(s_min):
                suppressed = await self._transaction.__aexit__(exc_type, exc, traceback)
            landed = True
        finally:
            if self._saved_delta_files:
                # a commit that did not land leaves its files orphans, the only files the fold's sweep removes
                OWN_DELTA_SAVES.settled(list(self._saved_delta_files.values()), landed=landed and exc_type is None)
            if self._touch_token is not None:
                releasing_touched_scopes(self._touch_token)
                self._touch_token = None
            if self._token is not None:
                _OPEN.reset(self._token)
                self._token = None
            if self._outermost is None:
                # the broker's tokens for this commit, read before the settle: a collection that advances nothing
                # itself (its table advanced by the broker's trigger) stamps the keys it tells a copy with its table's,
                # so the copy merges them from the commit's delta file and reads none from L3 (named_commit_generation)
                replied = reply_generations_since(before) if landed and exc_type is None else {}
                with (
                    naming_delta_files(files if landed else {}),
                    naming_commit_scopes(scopes if landed else []),
                    naming_commit_s_min(s_min if landed else None),
                    naming_commit_generations(replied),
                ):
                    generation_failure, advanced = await asyncio.shield(
                        self._settle_collecting(committed=exc_type is None)
                    )
                if landed and exc_type is None:
                    self._committed_generations = MappingProxyType({**replied, **advanced})
        return bool(suppressed), generation_failure

    async def _save_delta_files(self) -> dict[str, str]:
        """save each delta file this transaction owes, before its commit, and name them.

        :return: table -> its file's name under the delta files' directory; empty when no table written opts in
        :rtype: dict[str, str]
        :raises DeltaFileRefusedError: when a table that opts in cannot have its file: a key or a key prefix enrolled
            that no write recorded, or no delta files directory configured
        :raises OSError: when a file cannot be written
        """
        # frozen as handed over: a thread counts them while the loop runs on, and never sees a list the loop changes
        enrolled = tuple(chain.from_iterable(self._enrolled))
        reaches = self._reaches
        deltas = self._deltas
        if not deltas and len(enrolled) < _SETTLE_OFF_LOOP_ROWS:
            owed = _owed_deltas(enrolled, reaches, deltas)
        elif not deltas and not enrolled:
            owed = {}
        else:
            # a key a row of a commit of 33,786 rows: counted off the loop, as settling groups them, and the states its
            # rows are of read there too
            owed, self._states = await asyncio.to_thread(_owed_and_states, enrolled, reaches, deltas)
        return await save_delta_files(owed, version=self._write_version)

    async def _settle_collecting(
        self, *, committed: bool
    ) -> tuple[GenerationUnavailableError | None, Mapping[str, str]]:
        """:meth:`_settle`, collecting the token each advance it makes writes, by table: the collector is this task's
        own (the shielded settle's), and the caller gets a read-only copy back, never a dict the settle fills.

        :param committed: as :meth:`_settle`
        :ptype committed: bool
        :return: the first failure to advance, and table -> the token its advance wrote
        :rtype: tuple[GenerationUnavailableError | None, Mapping[str, str]]
        """
        advanced: dict[str, str] = {}
        with collecting_advanced_generations(advanced):
            failure = await self._settle(committed=committed)
        return failure, MappingProxyType(dict(advanced))

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
        chunks, restated, reaches, count = self._enrolled, self._restated, self._reaches, self._enrolled_count
        self._enrolled, self._restated, self._reaches, self._enrolled_count = (), (), (), 0
        if count < _SETTLE_OFF_LOOP_ROWS:
            enrolled, by_collection, shared = _grouped_enrollments(chunks, restated)
        else:
            # a commit of 33,786 rows held the loop 41-71 ms here grouping its keys and building a change record
            # a key (the WD2 bench, 2026-11-16, 179 columns, save_rows and save_arrow alike); on a thread the loop
            # is handed a turn every half millisecond (LoopTurns). Pure work over this transaction's own list, awaited
            # before anything is settled, so the order every cache hears its keys in is unchanged
            enrolled, by_collection, shared = await asyncio.to_thread(
                _grouped_enrollments, chunks, restated, asyncio.get_running_loop()
            )
        # the chunks no longer hold the rows alone: the enrollments do, freed off the loop below (_released)
        del chunks, restated
        sharing = _SHARED.set(shared)
        try:
            failure = await self._settle_each(by_collection)
        finally:
            _SHARED.reset(sharing)
        for collection, prefixes in reaches:
            try:
                await collection.settle_prefixes(prefixes)
            except GenerationUnavailableError as exc:
                # logged where the advance failed; the table's caches were dropped
                if failure is None:
                    failure = exc
        if count >= _SETTLE_OFF_LOOP_ROWS:
            # the enrollments often hold the only reference to the rows a write sent: freeing 33,786 rows of 179
            # columns at once held the loop 25 ms (save_rows, the WD2 bench); freed a slice at a time on a thread
            await asyncio.to_thread(_released, enrolled, by_collection, shared, asyncio.get_running_loop())
        if count:
            log.debug(
                "caller transaction ended; the rows its writes touched were evicted from every cache",
                extra={"extra_data": {"rows": count, "body_completed": committed}},
            )
        return failure

    @staticmethod
    async def _settle_each(by_collection: _ByCollection) -> GenerationUnavailableError | None:
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


def _commit_scopes(touched: frozenset[str], states: frozenset[str] | None) -> list[str]:
    """the scopes a commit with delta files names (``COMMIT_SCOPES_FIELD``): those its writes touched, and, when its
    files' rows said every state they are of (:meth:`~threetears.core.collections.delta_files.TableDelta.named_states`),
    those states and the mark that they are complete
    (:data:`~threetears.core.collections.epoch_change.EVERY_STATE_NAMED`). Bounded whole (``recordable_scopes``): too
    many names none, never a subset.

    Why the mark only from the rows: the touched scopes are the writer's word, which nothing ties to the rows it wrote;
    a record naming fewer states than its commit wrote would leave a copy that reloads just those states with the old
    rows of the rest (decision 54 (c)).

    :param touched: the scopes the transaction's writes touched (``ScopeEpochs.touch``)
    :ptype touched: frozenset[str]
    :param states: every state its delta files' rows are of; None when any table's cannot be told
    :ptype states: frozenset[str] | None
    :return: the scopes to name; empty for none
    :rtype: list[str]
    """
    if states is None:
        return recordable_scopes(touched) or []
    return recordable_scopes(touched | states | {EVERY_STATE_NAMED}) or []


def _owed_and_states(
    enrolled: Sequence[tuple[BaseCollection[Any], Any, dict[str, Any] | None]],
    reaches: Sequence[tuple[_PrefixSettling, Sequence[Mapping[str, str]]]],
    deltas: Mapping[str, TableDelta],
) -> tuple[dict[str, TableDelta], frozenset[str] | None]:
    """:func:`_owed_deltas`, and every state the owed deltas' rows are of (None when any one's cannot be told). Pure,
    for a thread.

    :param enrolled: the transaction's enrollments
    :ptype enrolled: Sequence[tuple[BaseCollection, Any, dict[str, Any] | None]]
    :param reaches: the key prefixes it deleted, by collection
    :ptype reaches: Sequence[tuple[_PrefixSettling, Sequence[Mapping[str, str]]]]
    :param deltas: what its writes recorded, by table
    :ptype deltas: Mapping[str, TableDelta]
    :return: the owed deltas, and their states
    :rtype: tuple[dict[str, TableDelta], frozenset[str] | None]
    :raises DeltaFileRefusedError: as :func:`_owed_deltas`
    """
    owed = _owed_deltas(enrolled, reaches, deltas)
    states: frozenset[str] = frozenset()
    for delta in owed.values():
        named = delta.named_states()
        if named is None:
            return owed, None
        states |= named
    return owed, states if owed else None


def _owed_deltas(
    enrolled: Sequence[tuple[BaseCollection[Any], Any, dict[str, Any] | None]],
    reaches: Sequence[tuple[_PrefixSettling, Sequence[Mapping[str, str]]]],
    deltas: Mapping[str, TableDelta],
) -> dict[str, TableDelta]:
    """the delta each table that opts in owes a file for, checked whole: every key a write enrolled for the table is
    a row or delete marker some write recorded, and nothing was recorded that no write enrolled. Pure, for a thread.

    :param enrolled: the transaction's enrollments
    :ptype enrolled: Sequence[tuple[BaseCollection, Any, dict[str, Any] | None]]
    :param reaches: the key prefixes it deleted, by collection
    :ptype reaches: Sequence[tuple[_PrefixSettling, Sequence[Mapping[str, str]]]]
    :param deltas: what its writes recorded, by table
    :ptype deltas: Mapping[str, TableDelta]
    :return: table -> its delta, for each table that opts in
    :rtype: dict[str, TableDelta]
    :raises DeltaFileRefusedError: when a table that opts in is not wholly recorded
    """
    # counted, as keys are: an enrolled prefix is its values as text (str), a recorded one as a key names them
    # (prefix_text), and the two spell a time differently; each holds a prefix once
    reached: dict[str, int] = {}
    for collection, prefixes in reaches:
        if collection.writes_delta_files:
            reached[collection.table_name] = reached.get(collection.table_name, 0) + len(prefixes)
    for table, count in reached.items():
        recorded = len(deltas[table].prefixes) if table in deltas else 0
        if recorded != count:
            raise DeltaFileRefusedError(
                f"{table}: the transaction deleted {count} key prefixes of a table that opts in to delta files and "
                f"recorded {recorded} for its file (record_prefixes); refused rather than committed without them"
            )
    counts = Counter(collection.table_name for collection, _, _ in enrolled if collection.writes_delta_files)
    for table, count in counts.items():
        recorded = deltas[table].entries if table in deltas else 0
        if recorded != count:
            raise DeltaFileRefusedError(
                f"{table}: the transaction wrote {count} keys of a table that opts in to delta files and recorded "
                f"{recorded} for its file; a write that records none is refused rather than committed without them"
            )
    unenrolled = sorted(set(deltas) - set(counts) - set(reached))
    if unenrolled:
        raise DeltaFileRefusedError(
            f"{unenrolled}: rows were recorded for a delta file that no write enrolled, a write that would reach L3 "
            f"without settling its keys; it is refused"
        )
    return {table: deltas[table] for table in (*counts, *reached)}


async def save_delta_files(deltas: Mapping[str, TableDelta], *, version: int | None) -> dict[str, str]:
    """save each table's delta file before the commit that wrote it, one write id for them all, in a worker process
    (:func:`~threetears.core.collections.delta_save.save_in_worker`).

    The one way a write saves its files: a caller's transaction (:class:`CallerTransaction`) and a load from files
    (:meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.load_files`) alike.

    :param deltas: table -> what the write wrote to it
    :ptype deltas: Mapping[str, TableDelta]
    :param version: the write's version (``ScopeEpochs.begin``), which names the files (decision 53: a full covers
        every file below a version, exactly); None names them in the older, unversioned form, which a recovery always
        reads (decision 53's expand step). Required, no default: a writer under a lease says its version at the call
    :ptype version: int | None
    :return: table -> its file's name under the delta files' directory, as the commit's epoch record names it
    :rtype: dict[str, str]
    :raises DeltaFileRefusedError: when no delta files directory is configured
    :raises OSError: when a file cannot be written
    """
    if not deltas:
        return {}
    root = delta_files_root()
    if root is None:
        raise DeltaFileRefusedError(
            f"{sorted(deltas)}: the table opts in to delta files and no delta files directory is configured "
            f"(THREETEARS_DELTA_FILES_DIR); the write is refused rather than committed without its file"
        )
    write_id = uuid.uuid7().hex if version is None else versioned_write_id(version, uuid.uuid7().hex)
    # in flight from here until the commit settles (:data:`~threetears.core.collections.delta_files.OWN_DELTA_SAVES`): what the orphan sweep goes by
    names = OWN_DELTA_SAVES.saving(list(deltas), write_id)
    try:
        # built and saved in a worker process; the commit waits for the files (decision 11)
        return await save_in_worker(root, dict(deltas), write_id)
    except BaseException:
        # the commit is refused: a file saved before the one that failed is an orphan
        OWN_DELTA_SAVES.settled(names, landed=False)
        raise
