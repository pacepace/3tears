"""the L1 fence: one collection's in-process ordering of L1 writes per key, so a read caches a row only while it is
still the newest this process knows.

Moved out of :mod:`threetears.core.collections.base` (2026-11-23), which holds one per collection
(``BaseCollection._l1_fence``), so its own tests reach it by its public name. Its state is immutable values the fence
replaces whole (counts, one packed key batch, a tuple of eras), never a record a key.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from threetears.core import packed_keys

__all__ = ["KeyTicket", "L1Fence", "ScanReadTicket"]


#: the most changes one era of :class:`L1Fence` records, and the most key text its batch holds, before the next era
#: begins. An era's changes and its batch are rebuilt whole at each change it records, so these bound that copy; a
#: ticket's check walks its own era's changes one by one and searches each later era's batch once.
FENCE_ERA_CHANGES: Final = 64
_FENCE_ERA_TEXT: Final = 1 << 16


@dataclass(frozen=True, slots=True)
class KeyTicket:
    """one read's or one write's claim on a key, taken before its first await.

    :ivar key: the key, as the pk values' string forms
    :ivar text: the key's text in a packed batch (:func:`packed_keys.key_text`)
    :ivar seen: the fence's count of changes as this ticket left it (its own write's included)
    :ivar era: the fence era the ticket was taken in; the changes it may be overtaken by are in it or later ones
    :ivar contended: whether a write of the key was already in flight when this ticket was taken
    :ivar writing: whether the ticket is a write's
    """

    key: tuple[str, ...]
    text: str
    seen: int
    era: int
    contended: bool
    writing: bool


@dataclass(frozen=True, slots=True)
class ScanReadTicket:
    """a scan's claim on its collection's L1, taken before the scan's first await (:meth:`BaseCollection.scan_ticket`).

    A scan reads rows it cannot name in advance, so it cannot hold a ticket per key. It holds one
    on the whole collection instead: the rows it read may enter L1 only while no write of any key
    has begun, no eviction has landed, and no write was in flight when it was taken.

    :ivar seen: the fence's count of writes begun and evictions landed, as the ticket left it
    :ivar contended: whether a write of any key was in flight when the ticket was taken
    """

    seen: int
    contended: bool


@dataclass(frozen=True, slots=True)
class _FenceChange:
    """one change :class:`L1Fence` recorded while a ticket was held: a write begun or an eviction landed.

    :ivar seq: the fence's count of changes once this one was counted
    :ivar keys: the keys it names, as one packed batch
    :ivar under: for an eviction by key prefix, each key position and its value; ``None`` otherwise
    :ivar every: whether it names every key (a table's drop)
    """

    seq: int
    keys: str
    under: tuple[tuple[int, str], ...] | None
    every: bool


def _under_names(under: tuple[tuple[int, str], ...], key: tuple[str, ...]) -> bool:
    """whether the key prefix ``under`` (each key position and its value) reaches ``key``.

    :param under: each key position and its value
    :ptype under: tuple[tuple[int, str], ...]
    :param key: the key
    :ptype key: tuple[str, ...]
    :return: ``True`` when it does
    :rtype: bool
    """
    return all(index < len(key) and key[index] == value for index, value in under)


@dataclass(frozen=True, slots=True)
class _FenceEra:
    """a run of :class:`L1Fence`'s changes.

    :ivar changes: the changes recorded in it, in order
    :ivar keys: every key its changes name, as one packed batch (their batches end to end)
    :ivar under: every key prefix its changes name
    :ivar every: whether a change of it names every key
    """

    changes: tuple[_FenceChange, ...]
    keys: str
    under: tuple[tuple[tuple[int, str], ...], ...]
    every: bool

    def names(self, ticket: KeyTicket) -> bool:
        """whether any change of this era reaches ``ticket``'s key (an era begun after the ticket was taken).

        :param ticket: the ticket
        :ptype ticket: KeyTicket
        :return: ``True`` when one does
        :rtype: bool
        """
        if self.every or packed_keys.names(self.keys, ticket.text):
            return True
        return any(_under_names(under, ticket.key) for under in self.under)

    def names_since(self, ticket: KeyTicket) -> bool:
        """whether a change of this era recorded after ``ticket`` was taken reaches its key (the ticket's own era).

        :param ticket: the ticket
        :ptype ticket: KeyTicket
        :return: ``True`` when one does
        :rtype: bool
        """
        for change in self.changes:
            if change.seq <= ticket.seen:
                continue
            if change.every or packed_keys.names(change.keys, ticket.text):
                return True
            if change.under is not None and _under_names(change.under, ticket.key):
                return True
        return False

    def with_change(self, change: _FenceChange) -> _FenceEra:
        """this era with ``change`` recorded.

        :param change: the change
        :ptype change: _FenceChange
        :return: the era
        :rtype: _FenceEra
        """
        keys = self.keys if change.keys == packed_keys.NO_KEYS else self.keys + change.keys
        under = self.under if change.under is None else (*self.under, change.under)
        return _FenceEra(
            changes=(*self.changes, change),
            keys=keys,
            under=under,
            every=self.every or change.every,
        )


#: the era a fence starts in, and returns to whenever no ticket is held
_QUIET_ERA: Final = _FenceEra(changes=(), keys=packed_keys.NO_KEYS, under=(), every=False)


class L1Fence:
    """orders this process's L1 writes of a key against every other write and eviction of it.

    Every L1 write that follows an await -- a pull-through caching what L2 or L3 returned, a save
    caching the row it committed -- is a claim that the row is still the newest, and the await is
    where something newer can land: this process's own write of the key, or a peer's broadcast
    evicting it. Caching after that leaves L1 behind the tier it was read from, with nothing left to
    evict it. The L2 revision fences this between processes; this is the same fence inside one,
    and the only one a collection with no L2 has.

    A read or write takes a ticket before its first await and may cache only while no write of the
    key began, and no eviction of it landed, since -- and, for a write, while no other write of the
    key was in flight when it began. Two overlapping writes can reach L3 in either order, so neither
    knows it is the later one; both drop the key, and the next read takes whichever row L3 kept.
    The check and the L1 write run with no await between them.

    **No state a key** (Pace, 2026-11-23). The fence holds no record a key: one count of changes, the writes in flight
    as one packed batch (:mod:`~threetears.core.packed_keys`), and the changes recorded while any ticket is
    held as a tuple of eras, with the held tickets' keys as one packed batch so a write or eviction of one key no
    ticket holds is only counted, never recorded. Each is an immutable value the fence replaces whole, on the loop, with no await between
    its read and its replacement. A ticket keeps the count as it left it; a change reaches it when the change was
    counted after that and names its key. Its own era's changes are walked one by one (at most
    :data:`FENCE_ERA_CHANGES`); each later era -- begun after the ticket was taken -- is asked once, one substring
    search of its batch. An era ends after :data:`FENCE_ERA_CHANGES` changes or :data:`_FENCE_ERA_TEXT` characters of
    key text, so no change copies more than that, and is dropped whole once every ticket taken in it, or before it, is
    released; with no ticket held nothing is recorded but the count.

    Measured (2026-11-23, M-series, against the record a key it replaced): a read's begin, check and end 0.8 us (0.6);
    a write's begin and end with 1,000 reads held 5.5 us (0.6), its key searched in the held tickets' batch; a held
    read's check after 1,000 writes of other keys 0.2 us (0.03), nothing recorded for keys no ticket holds; a
    33,786-key eviction's batch made on its thread in 3.2 ms (1.0, the keys and their set), recorded on the loop in
    0.02 ms (0.01), and a held read's check after it 0.4 us (0.05).
    """

    __slots__ = ("_changes", "_eras", "_first_era", "_held", "_held_by_era", "_holding", "_writers", "_writing")

    def __init__(self) -> None:
        """start with no ticket held.

        :return: nothing
        :rtype: None
        """
        # every write begun and eviction landed on any key: what a ticket's seen is read against, and what a scan,
        # which cannot name its keys before it reads, is fenced by
        self._changes = 0
        # the writes in flight on any key (a scan's contention), and their keys as one packed batch, a key a write
        self._writers = 0
        self._writing = packed_keys.NO_KEYS
        # tickets held, and the changes they may yet be overtaken by, era by era with the tickets taken in each held;
        # _first_era numbers _eras[0]
        self._held = 0
        # the keys of the tickets held, one entry a ticket: a change of one key no ticket holds is only counted
        self._holding = packed_keys.NO_KEYS
        self._eras: tuple[_FenceEra, ...] = (_QUIET_ERA,)
        self._held_by_era: tuple[int, ...] = (0,)
        self._first_era = 0

    @property
    def watched(self) -> bool:
        """whether any ticket is held, so a change must be recorded rather than only counted.

        :return: ``True`` while a ticket is held
        :rtype: bool
        """
        return self._held > 0

    @property
    def recorded(self) -> tuple[int, ...]:
        """how many changes each era holds, oldest first: what the fence keeps while tickets are held.

        :return: each era's count of changes
        :rtype: tuple[int, ...]
        """
        return tuple(len(era.changes) for era in self._eras)

    def _record(
        self,
        count: int,
        *,
        keys: str = packed_keys.NO_KEYS,
        under: tuple[tuple[int, str], ...] | None = None,
        every: bool = False,
    ) -> None:
        """count ``count`` changes, and record them as one while a ticket is held.

        :param count: how many changes to count (an eviction's keys)
        :ptype count: int
        :param keys: the keys they name, as one packed batch
        :ptype keys: str
        :param under: for an eviction by key prefix, each key position and its value
        :ptype under: tuple[tuple[int, str], ...] | None
        :param every: whether they name every key
        :ptype every: bool
        :return: nothing
        :rtype: None
        """
        self._changes += count
        if self._held == 0:
            return
        change = _FenceChange(seq=self._changes, keys=keys, under=under, every=every)
        eras = self._eras
        era = eras[-1]
        if len(era.changes) < FENCE_ERA_CHANGES and len(era.keys) < _FENCE_ERA_TEXT:
            self._eras = (*eras[:-1], era.with_change(change))
        else:
            self._eras = (*eras, _QUIET_ERA.with_change(change))
            self._held_by_era = (*self._held_by_era, 0)

    def begin(self, key: tuple[str, ...], *, writing: bool) -> KeyTicket:
        """take a ticket on ``key`` before the first await of a read or write of it.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :param writing: whether the ticket is a write's, which every other ticket on the key sees
        :ptype writing: bool
        :return: the ticket
        :rtype: KeyTicket
        """
        text = packed_keys.key_text(key)
        contended = self._writers > 0 and packed_keys.names(self._writing, text)
        if writing:
            self._writers += 1
            self._writing = packed_keys.with_text(self._writing, text)
            # recorded before this ticket is held: its own change is not one it is overtaken by
            self._changed_one(text)
        held = self._held_by_era
        self._held_by_era = (*held[:-1], held[-1] + 1)
        self._held += 1
        self._holding = packed_keys.with_text(self._holding, text)
        return KeyTicket(
            key=key,
            text=text,
            seen=self._changes,
            era=self._first_era + len(held) - 1,
            contended=contended,
            writing=writing,
        )

    def still_newest(self, ticket: KeyTicket) -> bool:
        """whether the row ``ticket``'s holder is about to cache is still the newest this process knows.

        :param ticket: the holder's ticket
        :ptype ticket: KeyTicket
        :return: ``True`` when no write overlapped the ticket and no eviction landed since it was taken
        :rtype: bool
        """
        return not ticket.contended and self.unchanged_since(ticket)

    def unchanged_since(self, ticket: KeyTicket) -> bool:
        """whether no write of ``ticket``'s key began, and no eviction of it landed, since it was taken.

        :meth:`still_newest` without its check that no write was in flight when the ticket was
        taken: what a copy written after its check (:meth:`BaseCollection._cache_read_in_l1`) asks
        once the write has landed.

        :param ticket: the holder's ticket
        :ptype ticket: KeyTicket
        :return: ``True`` when nothing recorded since the ticket was taken names its key
        :rtype: bool
        """
        if self._changes == ticket.seen:
            return True
        eras = self._eras[ticket.era - self._first_era :]
        if eras[0].names_since(ticket):
            return False
        return not any(era.names(ticket) for era in eras[1:])

    def end(self, ticket: KeyTicket) -> None:
        """release ``ticket``; an era no held ticket can be overtaken by any more is dropped whole.

        :param ticket: the ticket to release
        :ptype ticket: KeyTicket
        :return: nothing
        :rtype: None
        """
        if ticket.writing:
            self._writers -= 1
            # one of this ticket's key's entries (two writes of one key each hold one)
            self._writing = packed_keys.without_text(self._writing, ticket.text)
        self._held -= 1
        if self._held == 0:
            self._holding = packed_keys.NO_KEYS
            # nothing held can be overtaken: every era goes, and the numbering moves past them
            self._first_era += len(self._eras)
            self._eras = (_QUIET_ERA,)
            self._held_by_era = (0,)
            return
        self._holding = packed_keys.without_text(self._holding, ticket.text)
        at = ticket.era - self._first_era
        held = self._held_by_era
        held = (*held[:at], held[at] - 1, *held[at + 1 :])
        # the leading eras no ticket is held in: a ticket taken later was taken after all of their changes
        drop = 0
        while drop < len(held) - 1 and held[drop] == 0:
            drop += 1
        self._first_era += drop
        self._eras = self._eras[drop:]
        self._held_by_era = held[drop:]

    def changed(self, key: tuple[str, ...]) -> None:
        """record that ``key`` was evicted or rewritten, so no ticket taken before now may cache it.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        if self._held == 0:
            self._changes += 1
            return
        self._changed_one(packed_keys.key_text(key))

    def _changed_one(self, text: str) -> None:
        """count a change of the key whose text is ``text``, recorded only while a held ticket is on that key.

        A ticket taken later is not overtaken by it, so a change of a key no ticket holds has nothing to record:
        recording every write of 1,000 other keys made each held read's check walk them all (10 us, against 0.03 us
        with a record a key, 2026-11-23); now the checks find nothing to walk.

        :param text: the key's text
        :ptype text: str
        :return: nothing
        :rtype: None
        """
        if packed_keys.names(self._holding, text):
            self._record(1, keys=packed_keys.packed((text,)))
        else:
            self._changes += 1

    def changed_many(self, keys: str, count: int) -> None:
        """:meth:`changed` of every key of the packed batch ``keys``, at once: one change recorded, whatever the
        batch's size, and nothing walked a key.

        A commit's eviction of 33,786 keys called :meth:`changed` a key on the event loop: 33,786 calls, 3-5 ms with
        the loop held (the WD2 bench, 2026-11-16). The batch is made off the loop (:meth:`BaseCollection
        ._packed_fence_keys`), and a ticket's check searches it once.

        :param keys: the keys, as one packed batch (:func:`packed_keys.packed`)
        :ptype keys: str
        :param count: how many keys the batch names
        :ptype count: int
        :return: nothing
        :rtype: None
        """
        self._record(count, keys=keys)

    def changed_under(self, positions: Sequence[tuple[int, str]]) -> None:
        """record that every key whose values at ``positions`` are those values was evicted: a ticket on such a key
        taken before now may not cache what it read, and no scan's either.

        :param positions: each key position and its value's string form (a key prefix, by position)
        :ptype positions: Sequence[tuple[int, str]]
        :return: nothing
        :rtype: None
        """
        self._record(1, under=tuple(positions))

    def changed_all(self) -> None:
        """record that every key was evicted, so no ticket taken before now may cache what it read.

        :return: nothing
        :rtype: None
        """
        self._record(1, every=True)

    def begin_scan(self) -> ScanReadTicket:
        """take a ticket on every key, before the first await of a scan.

        :return: the ticket
        :rtype: ScanReadTicket
        """
        return ScanReadTicket(seen=self._changes, contended=self._writers > 0)

    def scan_still_newest(self, ticket: ScanReadTicket) -> bool:
        """whether the rows a scan read under ``ticket`` are still the newest this process knows.

        :param ticket: the scan's ticket
        :ptype ticket: ScanReadTicket
        :return: ``True`` when no write was in flight when it was taken and no write began, and no
            eviction landed, on any key since
        :rtype: bool
        """
        return not ticket.contended and self._changes == ticket.seen

    @contextmanager
    def watching(self, key: tuple[str, ...], *, writing: bool) -> Iterator[KeyTicket]:
        """hold a ticket on ``key`` for the body, released however the body ends.

        :param key: the key, as the pk values' string forms
        :ptype key: tuple[str, ...]
        :param writing: whether the ticket is a write's
        :ptype writing: bool
        :return: the ticket, for :meth:`still_newest`
        :rtype: Iterator[KeyTicket]
        """
        ticket = self.begin(key, writing=writing)
        try:
            yield ticket
        finally:
            self.end(ticket)
