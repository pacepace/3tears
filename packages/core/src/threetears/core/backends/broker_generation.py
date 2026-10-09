"""a pod's write generations, advanced by the L3 broker that committed its writes.

A pod cannot write the epoch bucket: a pod that could would be able to fake an advance the whole
fleet acts on. So the hub's L3 broker, which ends every transaction a pod's write runs in, advances
the generation of each table the commit wrote that is switched on, after the commit, and names the
token it wrote for each one in its reply. This module is the pod's half of that:

- :class:`NatsProxyL3Backend <threetears.core.backends.nats_proxy.NatsProxyL3Backend>` hands every
  reply that ends a write to :func:`record_reply_generations`, which keeps the tokens it names for
  the code that made the request;
- :class:`BrokerGenerationSource` is the :class:`~threetears.core.collections.generation.GenerationSource`
  a pod's registry is wired with. A collection calls ``advance`` after its commit, exactly as it does
  over the epoch bucket directly, and is handed the token the broker's reply carried for its table.

**"The commit just made" is the calling code's own.** Tokens are kept in a context variable, so a
collection is answered only from replies to requests made in its own asyncio task, never another
task's: two tasks writing one table concurrently are each handed their own commit's token. A task
started while tokens are held (``asyncio.shield`` around a transaction's settling is one) reads the
tokens of the task that started it; one that makes a write request of its own starts its own
record, and leaves the record it inherited as it was.

**One commit, one token per table, taken as often as that commit's settling asks.** Every advance
of a table for the same commit is handed the same token and raises nothing: a collection that
updates many rows in one statement and then evicts them one by one, or two collection instances of
one table settling one transaction, advance the table more than once for one bump on the wire. A
token stays until the commit's scope ends: a later reply naming the same table replaces it, and a
transaction that rolls back, a commit the broker refuses, and a commit whose reply never came back
drop every token the task holds, so nothing settled after them is handed an earlier commit's.

**What a reply says.** Every reply that ends a commit is read -- ``l3.query``, ``l3.batch`` (a batch
run statement by statement that failed partway included, since the statements before the failure
committed) and ``l3.tx.commit``. :data:`GENERATIONS_REPLY_FIELD` maps each switched-on table the commit wrote
to the token the broker's advance wrote. :data:`GENERATIONS_FAILED_REPLY_FIELD` lists each one the
broker could not advance, beside :data:`GENERATION_UNAVAILABLE_ERROR_CODE`, on a reply that is
still a success: the rows were committed, so the write must not be retried. A collection advancing
such a table raises :class:`~threetears.core.exceptions.GenerationUnavailableError`. Both fields are
optional: a pod built before them reads the reply as before, and a reply from a broker built before
them names no table, so a switched-on collection's advance raises rather than claim an advance
nobody made.

**How each side knows a table's writes advance.** The pod: the collection's own class, which calls
``advance`` when it declares ``write_generation = WRITE_GENERATION`` or caches absences. The broker:
the same class,
imported in the hub, through
:func:`~threetears.core.collections.base.tables_with_write_generation`. One class per table is what
keeps the two answers the same.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from threetears.core.exceptions import GenerationNotCommittedError, GenerationUnavailableError
from threetears.observe import get_logger

__all__ = [
    "GENERATIONS_FAILED_REPLY_FIELD",
    "GENERATIONS_REPLY_FIELD",
    "GENERATION_UNAVAILABLE_ERROR_CODE",
    "BrokerGenerationSource",
    "GenerationReader",
    "forget_reply_generations",
    "record_reply_generations",
]

_logger = get_logger(__name__)

#: the reply field naming, for each switched-on table a commit wrote, the generation token the
#: broker's advance after that commit wrote. The hub imports this name.
GENERATIONS_REPLY_FIELD: Final = "generations"

#: the reply field listing each switched-on table the commit wrote whose generation the broker could
#: not advance. The hub imports this name.
GENERATIONS_FAILED_REPLY_FIELD: Final = "generations_failed"

#: the broker's ``error_code`` on a reply whose statements COMMITTED and whose generation advance
#: failed for at least one table. The reply is still ``success``: the write landed and must not be
#: sent again. The hub imports this name.
GENERATION_UNAVAILABLE_ERROR_CODE: Final = "GENERATION_UNAVAILABLE"


@dataclass(slots=True)
class _Committed:
    """the generations the broker named for one task's commits, until a collection takes them.

    :ivar owner: the task whose requests filled this record; a task that writes and is not the
        owner starts a record of its own rather than adding to an inherited one
    :ivar tokens: table to the token the broker wrote for it
    :ivar failed: table to why the broker could not advance it
    :ivar not_committed: why the task's last commit landed nothing (rolled back, refused, or its
        outcome unknown), until a later reply names a generation; ``None`` otherwise
    """

    owner: weakref.ReferenceType[asyncio.Task[Any]] | None
    tokens: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    not_committed: str | None = None


_COMMITTED: ContextVar[_Committed | None] = ContextVar("threetears_broker_committed_generations", default=None)


def _current_task_ref() -> weakref.ReferenceType[asyncio.Task[Any]] | None:
    """a weak reference to the running task, or ``None`` outside one.

    :return: the reference
    :rtype: weakref.ReferenceType[asyncio.Task[Any]] | None
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return None if task is None else weakref.ref(task)


def _own_record() -> _Committed:
    """this task's record, started here when the one in context belongs to another task or none.

    :return: the record
    :rtype: _Committed
    """
    owner = _current_task_ref()
    running = None if owner is None else owner()
    held = _COMMITTED.get()
    holder = None if held is None or held.owner is None else held.owner()
    if held is None or holder is not running:
        held = _Committed(owner=owner)
        _COMMITTED.set(held)
    return held


def record_reply_generations(response: Mapping[str, Any]) -> None:
    """keep the generations a successful reply to a write names, for the collection that asked.

    Called by the L3 proxy on every reply that ends a commit: a successful ``l3.query`` or
    ``l3.tx.commit``, and every ``l3.batch`` reply, a statement-by-statement batch that failed
    partway included. A reply that names none changes nothing. A table both named and listed failed
    is failed.

    :param response: the parsed reply
    :ptype response: Mapping[str, Any]
    :return: nothing
    :rtype: None
    """
    tokens = response.get(GENERATIONS_REPLY_FIELD)
    failed = response.get(GENERATIONS_FAILED_REPLY_FIELD)
    named = {
        table: token
        for table, token in (tokens.items() if isinstance(tokens, Mapping) else ())
        if isinstance(table, str) and isinstance(token, str) and token
    }
    unadvanced = [table for table in (failed if isinstance(failed, list) else ()) if isinstance(table, str)]
    if not named and not unadvanced:
        return
    held = _own_record()
    held.not_committed = None
    for table, token in named.items():
        held.tokens[table] = token
        held.failed.pop(table, None)
    if unadvanced:
        reason = str(response.get("error_message") or "the broker could not advance the generation")
        _logger.error(
            "the L3 broker committed this write and could not advance the write generation of a table it "
            "wrote; a pod following the table cannot tell from it that this write happened",
            extra={"extra_data": {"tables": sorted(unadvanced), "error_code": response.get("error_code")}},
        )
        for table in unadvanced:
            held.tokens.pop(table, None)
            held.failed[table] = (
                f"the L3 broker committed the write to {table!r} and could not advance its write generation: {reason}"
            )


def forget_reply_generations(reason: str) -> None:
    """drop every generation this task holds, because its last commit landed nothing.

    Called when a transaction is rolled back, its commit is refused, or the commit's reply never
    arrives: a collection settling it must not be handed a token from an earlier commit, and is told
    why there is none (:class:`~threetears.core.exceptions.GenerationNotCommittedError`).

    :param reason: what happened to the commit, for the error a later advance raises
    :ptype reason: str
    :return: nothing
    :rtype: None
    """
    _COMMITTED.set(_Committed(owner=_current_task_ref(), not_committed=reason))


def _take_committed_generation(table_name: str) -> str:
    """the token the broker wrote for ``table_name`` after the commit just made.

    Not removed: every advance of the table for the same commit is handed the same token.

    :param table_name: the table a collection is advancing
    :ptype table_name: str
    :return: the generation token
    :rtype: str
    :raises GenerationNotCommittedError: when the task's last commit landed nothing
    :raises GenerationUnavailableError: when the broker reported it could not advance the table, or
        named no generation for it: the broker is older than generations, or does not hold the
        table's class switched on, or nothing this task committed wrote the table
    """
    held = _COMMITTED.get()
    if held is not None:
        token = held.tokens.get(table_name)
        if token is not None:
            return token
        failure = held.failed.get(table_name)
        if failure is not None:
            raise GenerationUnavailableError(failure)
        if held.not_committed is not None:
            raise GenerationNotCommittedError(
                f"no write generation for {table_name!r}: {held.not_committed}, so nothing it wrote landed"
            )
    raise GenerationUnavailableError(
        f"the L3 broker named no write generation for {table_name!r} for the commit just made: the hub "
        f"is older than write generations, or does not hold the table's collection class switched on, "
        f"or nothing committed here wrote the table"
    )


@runtime_checkable
class GenerationReader(Protocol):
    """reads a table's generation without writing it: what a pod's grant on the epoch bucket allows.

    :class:`threetears.epoch.EpochGenerationReader` is the platform's.
    """

    async def read(self, table_name: str) -> str | None:
        """the table's current generation, or ``None`` when the store holds none for it.

        :param table_name: the table
        :ptype table_name: str
        :return: the token, or ``None``
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation cannot be read
        """
        ...


class BrokerGenerationSource:
    """a pod's generation source: the broker advances, the pod is told what it wrote.

    It reads a generation only with a reader (:attr:`reads_generations`); without one, absence
    caching stays off on its registry, as it was with no source at all.

    :meth:`advance` makes no request. The broker advanced the table after committing the pod's
    write and named the token in its reply; this hands that token to the collection advancing, once.
    :meth:`current` reads the epoch bucket through ``reader``, which a pod is granted; with no
    reader, or for a table that has no generation yet, it raises, so a collection that caches
    absences trusts none rather than one no advance can invalidate.
    """

    def __init__(self, reader: GenerationReader | None = None) -> None:
        """capture the reader; no I/O.

        :param reader: reads a table's generation from the epoch bucket, for :meth:`current`;
            ``None`` when the pod reads none, so :meth:`current` always raises
        :ptype reader: GenerationReader | None
        :return: nothing
        :rtype: None
        """
        self._reader = reader

    @property
    def reads_generations(self) -> bool:
        """whether :meth:`current` can answer: only with a reader.

        :return: ``True`` when a reader was given
        :rtype: bool
        """
        return self._reader is not None

    async def current(self, table_name: str) -> str:
        """the table's current generation, as the epoch bucket holds it.

        A pod cannot mint a table's first generation, so a table with none raises rather than
        answering a token an absence could be stamped with and a later wipe could bring back.

        :param table_name: the table
        :ptype table_name: str
        :return: the token
        :rtype: str
        :raises GenerationUnavailableError: when there is no reader, the bucket cannot be read, or
            it holds no generation for the table yet
        """
        if self._reader is None:
            raise GenerationUnavailableError(
                f"this pod reads no write generation for {table_name!r}: its generation source has no reader"
            )
        token = await self._reader.read(table_name)
        if token is None:
            raise GenerationUnavailableError(
                f"no write generation exists yet for {table_name!r}, and a pod cannot mint one"
            )
        return token

    async def advance(self, table_name: str) -> str:
        """the token the broker wrote for ``table_name`` after the commit this task just made.

        :param table_name: the table
        :ptype table_name: str
        :return: the generation token the broker's advance wrote
        :rtype: str
        :raises GenerationNotCommittedError: when the task's last commit landed nothing
        :raises GenerationUnavailableError: when the broker reported it could not advance the
            table, or named no generation for it
        """
        return _take_committed_generation(table_name)
