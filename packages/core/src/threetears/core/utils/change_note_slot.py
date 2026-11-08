"""a writing process's cached copy of the change-note ring's current slot, so no save reads the ring row.

the cost it removes
-------------------

Every noted table's trigger writes its note into the ring's current note table
(:mod:`threetears.core.ddl.change_notes`). Reading which table that is from the one-row ring table on
every statement made every writer read one row on one tablet: on YugabyteDB the note's cost grew with the
number of writers (p50 +0.8 / +2 / +4 to 5 ms at 1 / 8 / 32 writers, ``docs/runbook-epoch-write-cost.md``).

the rule
--------

A process holds one :class:`ChangeNoteSlot` per database. It reads the ring row (slot, the fence the
writers are told, the database's clock) at most once per :attr:`ChangeNoteSlot.refresh_after` (a quarter
of the fence), and never lets a copy be applied once it is :func:`~threetears.core.ddl.change_notes.slot_max_age`
(half the fence) old: such a copy is read again first. Each connection built by
:func:`change_note_slot_connection_class` carries the copy as the session setting
:data:`~threetears.core.ddl.change_notes.NOTE_SLOT_SETTING` -- ``"<slot>:<valid until>"`` on the database's
clock -- sent before a statement outside a transaction only when it differs from what that connection
holds: once per connection per refresh, never per save.

The database enforces the validity itself: the trigger uses the setting only while its clock is before
``valid until``, and otherwise reads the ring row (the rare path, flagged on the note). So a connection
borrowed by code that does not go through this class, or one idle past its copy's age, is never wrong,
only slower. The safety argument (why a copy up to half the fence old never writes into a table being
emptied) is in :mod:`threetears.core.ddl.change_notes`.

A refresh that fails (the ring not installed, L3 briefly away) is logged and retried after a quarter of
the fence; the connection keeps what it holds, and the database stops honouring it once it expires. A
``COPY`` sent outside a transaction is not preceded by the setting (a ``COPY`` inside one is, at its
``BEGIN``), so its notes read the ring row.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any, Final

import asyncpg

from threetears.core.ddl.change_notes import (
    DEFAULT_NOTE_FENCE,
    NOTE_SLOT_SETTING,
    RING_TABLE,
    slot_max_age,
)
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import contain, get_logger
from threetears.observe.metrics import counter

__all__ = ["ChangeNoteSlot", "change_note_slot_connection_class"]

log = get_logger(__name__)

#: the least pause before a failed refresh is tried again (otherwise a quarter of the fence)
_RETRY_AFTER_FAILURE: Final = timedelta(seconds=5)

#: a statement that may change session settings another way, after which a connection holds nothing known
_FORGETS_RE: Final = re.compile(r"\bRESET\b|\bDISCARD\b|change_note_slot", re.IGNORECASE)

_REFRESHES = counter(
    "threetears_change_note_slot_refreshes_total",
    description="reads of the change-note ring row a process made to refresh its cached slot",
    label_names=("outcome",),
)


class ChangeNoteSlot:
    """the ring's current slot as this process last read it, and when it must be read again.

    One per process and database. Safe to share between connections and tasks: refreshes are serialised,
    and a reader that still holds a usable copy never waits for one.
    """

    def __init__(self, *, notes_schema: str, clock: Callable[[], float] = time.monotonic) -> None:
        """capture where the ring lives; no I/O.

        :param notes_schema: the schema holding the ring (the platform schema), a TRUSTED identifier
        :ptype notes_schema: str
        :param clock: a monotonic clock, in seconds; tests pass their own
        :ptype clock: Callable[[], float]
        :return: nothing
        :rtype: None
        """
        self._ring = f"{quote_identifier(notes_schema)}.{quote_identifier(RING_TABLE)}"
        self._clock = clock
        self._lock = asyncio.Lock()
        #: the setting's value, and the monotonic time the read that produced it began
        self._value: str | None = None
        self._read_at: float | None = None
        self._fence = DEFAULT_NOTE_FENCE
        self._failed_at: float | None = None
        self.reads = 0

    @property
    def max_age(self) -> float:
        """seconds a copy may be applied after its read began: half the fence the ring tells writers.

        :return: the age
        :rtype: float
        """
        return slot_max_age(self._fence).total_seconds()

    @property
    def refresh_after(self) -> float:
        """seconds after which a copy is read again: half its longest use, a quarter of the fence.

        :return: the age
        :rtype: float
        """
        return self.max_age / 2

    def usable(self) -> str | None:
        """the setting's value while its copy is younger than :attr:`max_age`; ``None`` otherwise.

        :return: ``"<slot>:<valid until>"``, or ``None``
        :rtype: str | None
        """
        if self._read_at is None or self._clock() - self._read_at >= self.max_age:
            return None
        return self._value

    def _due(self) -> bool:
        """whether a refresh is due: never read, or older than :attr:`refresh_after`, and not just failed.

        :return: whether to read the ring row now
        :rtype: bool
        """
        now = self._clock()
        if self._failed_at is not None and now - self._failed_at < max(
            self.refresh_after, _RETRY_AFTER_FAILURE.total_seconds()
        ):
            return False
        return self._read_at is None or now - self._read_at >= self.refresh_after

    async def current(self, read: Callable[[str], Any]) -> str | None:
        """the value to apply: refreshed first when due, ``None`` when no usable copy could be had.

        :param read: runs one query and returns its first row (the connection's own, bypassing any hook)
        :ptype read: Callable[[str], Any]
        :return: ``"<slot>:<valid until>"``, or ``None``
        :rtype: str | None
        """
        if not self._due():
            return self.usable()
        if self._lock.locked() and self.usable() is not None:
            # another task is refreshing and this copy is still good: never wait for it
            return self.usable()
        async with self._lock:
            if self._due():
                await self._refresh(read)
        return self.usable()

    async def _refresh(self, read: Callable[[str], Any]) -> None:
        """read the ring row: its slot, the fence it tells writers and the database's clock.

        :param read: runs one query and returns its first row
        :ptype read: Callable[[str], Any]
        :return: nothing
        :rtype: None
        """
        began = self._clock()
        self.reads += 1
        try:
            row = await read(
                f"SELECT current_ring, fence_seconds, extract(epoch FROM now())::float8 AS now FROM {self._ring} "  # noqa: S608 - trusted identifiers
                "WHERE id = 1"
            )
        # prawduct:allow prawduct/broad-except -- a failed refresh leaves the database to read the ring; retried later
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="could not read the change-note ring to refresh this process's cached slot; writes read the "
                "ring row themselves until it can",
                extra={"ring": self._ring, "error": f"{type(exc).__name__}: {exc}"},
            )
            self._failed_at = began
            _REFRESHES.labels(outcome="failed").inc()
            return
        if row is None:
            self._failed_at = began
            _REFRESHES.labels(outcome="failed").inc()
            return
        fence = (
            timedelta(seconds=float(row["fence_seconds"])) if row["fence_seconds"] is not None else DEFAULT_NOTE_FENCE
        )
        self._fence = fence
        valid_until = float(row["now"]) + slot_max_age(fence).total_seconds()
        self._value = f"{int(row['current_ring'])}:{valid_until:.6f}"
        self._read_at = began
        self._failed_at = None
        _REFRESHES.labels(outcome="read").inc()


def change_note_slot_connection_class(
    slot: ChangeNoteSlot, base: type[asyncpg.Connection] = asyncpg.Connection
) -> type[asyncpg.Connection]:
    """a connection class that carries ``slot``'s copy as a session setting, sent only when it changed.

    Before each statement sent outside a transaction (a ``BEGIN`` included), the connection compares the
    process's usable copy with the value it last set and, when they differ, sets it
    (``set_config(..., false)``), refreshing the copy first when it is due. Inside a transaction it sends
    nothing: the value set before ``BEGIN`` stands, and the database checks its validity. A statement that
    may reset session state (``RESET``, ``DISCARD``) makes the connection forget what it holds.

    Pass it as ``connection_class`` to ``asyncpg.create_pool``, or as the base of another connection class
    (:func:`~threetears.core.utils.session_settings.session_settings_connection_class` takes ``note_slot``).

    :param slot: the process's cached slot for this database
    :ptype slot: ChangeNoteSlot
    :param base: the connection class to extend
    :ptype base: type[asyncpg.Connection]
    :return: the connection class
    :rtype: type[asyncpg.Connection]
    """

    class _NoteSlotConnection(base):  # type: ignore[misc]
        """an asyncpg connection that carries the process's cached change-note slot."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._note_slot_held: str | None = None

        async def _read_ring(self, query: str) -> Any:
            return await asyncpg.Connection.fetchrow(self, query)

        async def _carry_note_slot(self, query: str) -> None:
            """set the session's copy before ``query`` when it differs; forget it when ``query`` may reset it.

            :param query: the statement about to be sent
            :ptype query: str
            :return: nothing
            :rtype: None
            """
            if _FORGETS_RE.search(query):
                self._note_slot_held = None
                return
            if self.is_in_transaction():
                return
            value = await slot.current(self._read_ring)
            if value is None or value == self._note_slot_held:
                return
            await asyncpg.Connection.execute(self, "SELECT set_config($1, $2, false)", NOTE_SLOT_SETTING, value)
            self._note_slot_held = value

        async def execute(self, query: str, *args: Any, timeout: float | None = None) -> str:
            await self._carry_note_slot(query)
            status: str = await super().execute(query, *args, timeout=timeout)
            return status

        async def executemany(self, command: str, args: Any, *, timeout: float | None = None) -> None:
            await self._carry_note_slot(command)
            await super().executemany(command, args, timeout=timeout)

        async def fetch(self, query: str, *args: Any, **kwargs: Any) -> list[Any]:
            await self._carry_note_slot(query)
            rows: list[Any] = await super().fetch(query, *args, **kwargs)
            return rows

        async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
            await self._carry_note_slot(query)
            return await super().fetchrow(query, *args, **kwargs)

        async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
            await self._carry_note_slot(query)
            return await super().fetchval(query, *args, **kwargs)

    return _NoteSlotConnection
