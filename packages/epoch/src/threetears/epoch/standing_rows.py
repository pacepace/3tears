"""a table's standing rows, held by key and kept by the rows that change: never a whole read after the first.

Many passes want "every row of a table that stands": every active agent, every active customer, every pod serving
an object store. Each read the table whole for itself. Here such a set is read once, cold (its first question, or the
first after the table's reset), and then kept by the rows each write names (:class:`~threetears.epoch.RowFollower`):
a changed row names its whole key, that one row is read by it (:func:`~threetears.core.keyed.read_keyed`), and held
or let go as :attr:`Standing.member` now judges it. The set is never read whole again.

The cold read is the caller's (``cold=``): written where the table is named, so the repository that owns the table
declares it a cold start (``threetears.core.declared.whole_read``) beside its proof and speed test. Nothing here
writes SQL over the table but the keyed read of a changed row.

A caller asks for the keys it holds (:meth:`StandingRows.keys`), a count (:meth:`StandingRows.count`), the held
columns (:meth:`StandingRows.held`), or the rows themselves, read by those keys (:meth:`StandingRows.read`), each read
row judged again so a row that stopped standing since is never answered.

A cold read taken before :meth:`StandingRows.follow` (a process's start sweeps long before it follows) is followed
from the generation read just before it (:func:`~threetears.epoch.cold_read_generations`), so no row written between
the two is missed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.keyed import KeyBatch, read_keyed
from threetears.observe import contain, get_logger

from threetears.epoch.generation import EpochGenerationReader
from threetears.epoch.row_follower import RowFollower, cold_read_generations

__all__ = ["Standing", "StandingRows"]

log = get_logger(__name__)

#: the keys :meth:`StandingRows.read` names in one statement
_READ_PAGE: Final[int] = 500

#: how a change note's key value (always text) is made the key column's value, by the column's SQL type
_PARSE: Final[Mapping[str, Callable[[str], Any]]] = {"uuid": UUID, "text": str, "bigint": int, "integer": int}


@dataclass(frozen=True)
class Standing:
    """which rows of a table stand, and what of each is held.

    :ivar table: the table, a TRUSTED identifier
    :ivar key: its whole primary key, in declared order (the order a change note names a key's values in)
    :ivar key_types: each key column's SQL type (``uuid``, ``text``, ``bigint``, ``integer``)
    :ivar held: the columns held for each standing row besides its key: every column :attr:`member` reads, and any a
        caller narrows the held rows by
    :ivar member: whether a row (its key and :attr:`held` columns) stands; the one judge, of the cold read's rows and
        of every changed row
    """

    table: str
    key: tuple[str, ...]
    key_types: tuple[str, ...]
    held: tuple[str, ...]
    member: Callable[[Mapping[str, Any]], bool]

    def __post_init__(self) -> None:
        if not self.key or len(self.key) != len(self.key_types):
            raise ValueError(f"{self.table}: one SQL type per key column, {self.key} against {self.key_types}")
        unknown = sorted(set(self.key_types) - set(_PARSE))
        if unknown:
            raise ValueError(f"{self.table}: key types {unknown} are not among {sorted(_PARSE)}")
        overlap = sorted(set(self.key) & set(self.held))
        if overlap:
            raise ValueError(f"{self.table}: {overlap} are key columns, held already")

    @property
    def columns(self) -> tuple[str, ...]:
        """every column held of a row: its key, then :attr:`held`.

        :return: the columns
        :rtype: tuple[str, ...]
        """
        return (*self.key, *self.held)

    def key_of(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        """a row's key.

        :param row: the row
        :ptype row: Mapping[str, Any]
        :return: its key's values, in declared order
        :rtype: tuple[Any, ...]
        """
        return tuple(row[column] for column in self.key)

    def parse(self, named: Sequence[str]) -> tuple[Any, ...]:
        """a change note's key (each value as text) as the key's values.

        :param named: the note's key values, in declared order
        :ptype named: Sequence[str]
        :return: the key's values
        :rtype: tuple[Any, ...]
        :raises ValueError: when the note names a key of another length
        """
        if len(named) != len(self.key):
            raise ValueError(f"{self.table}: a change note named {len(named)} key values; the key is {self.key}")
        return tuple(_PARSE[kind](value) for kind, value in zip(self.key_types, named, strict=True))


class StandingRows:
    """a table's standing rows, by key: read cold once, then kept by each row that changes.

    A changed row's read that fails is kept and read again before the next question is answered, which fails if it
    fails again: nothing is answered from a set known to have missed a row.

    :param pool: what the changed rows are read through (anything :func:`~threetears.core.keyed.read_keyed` reads)
    :ptype pool: Any
    :param standing: which rows stand, and what is held of each
    :ptype standing: Standing
    :param cold: the cold read: every standing row's key and :attr:`Standing.held` columns (a row it answers that
        does not stand is let go), declared by the caller
    :ptype cold: Callable[[], Awaitable[Iterable[Mapping[str, Any]]]]
    :param generations: reads the table's write generation just before a cold read taken before :meth:`follow`
    :ptype generations: EpochGenerationReader
    :param name: the set's name, for its follower's tasks and logs
    :ptype name: str
    """

    def __init__(
        self,
        pool: Any,
        standing: Standing,
        *,
        cold: Callable[[], Awaitable[Iterable[Mapping[str, Any]]]],
        generations: EpochGenerationReader,
        name: str,
    ) -> None:
        """bind the reads; no I/O.

        :return: nothing
        :rtype: None
        """
        self._pool = pool
        self._standing = standing
        self._cold = cold
        self._generations = generations
        self._name = name
        self._lock = asyncio.Lock()
        self._loaded = False
        self._following = False
        self._since: dict[str, str | None] | None = None
        self._held: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._missed: set[tuple[Any, ...]] = set()
        self._rows = RowFollower({}, self.on_table_dropped, name=name, key_reads={standing.table: self._on_key})

    @property
    def standing(self) -> Standing:
        """which rows stand.

        :return: the declaration
        :rtype: Standing
        """
        return self._standing

    @property
    def rows(self) -> RowFollower:
        """the follower of the table's rows (each changed row reaches its keyed read through it).

        :return: the row follower
        :rtype: RowFollower
        """
        return self._rows

    def follow(self, registry: CollectionRegistry, reader: Any) -> None:
        """follow the table's rows by key, from the generation read before the cold read when one was taken already.

        Call once ``registry``'s listener runs; :meth:`stop` before it stops.

        :param registry: the registry whose listener hears the rows
        :ptype registry: CollectionRegistry
        :param reader: reads and watches the epoch bucket
        :ptype reader: GenerationWatcher
        :return: nothing
        :rtype: None
        """
        self._rows.follow(registry, reader, since=self._since)
        self._following = True

    async def stop(self) -> None:
        """stop following. Idempotent.

        :return: nothing
        :rtype: None
        """
        await self._rows.stop()
        self._following = False

    async def settled(self) -> None:
        """wait for every changed row's read started so far.

        :return: nothing
        :rtype: None
        """
        await self._rows.settled()

    async def held(self) -> dict[tuple[Any, ...], dict[str, Any]]:
        """every standing row's key and held columns: read cold when nothing is held, a missed row read again first.

        :return: key -> the row's key and held columns (a copy)
        :rtype: dict[tuple[Any, ...], dict[str, Any]]
        :raises Exception: what the cold read, or a missed row's second read, raises
        """
        async with self._lock:
            await self._settle()
            return {key: dict(row) for key, row in self._held.items()}

    async def keys(self, where: Callable[[Mapping[str, Any]], bool] | None = None) -> list[tuple[Any, ...]]:
        """the standing rows' keys, in key order; ``where`` narrows them by their held columns.

        :param where: which held rows to answer; every one when None
        :ptype where: Callable[[Mapping[str, Any]], bool] | None
        :return: the keys
        :rtype: list[tuple[Any, ...]]
        """
        async with self._lock:
            await self._settle()
            if where is None:
                return sorted(self._held)
            return sorted(key for key, row in self._held.items() if where(row))

    async def count(self, where: Callable[[Mapping[str, Any]], bool] | None = None) -> int:
        """how many rows stand; ``where`` narrows them by their held columns.

        :param where: which held rows to count; every one when None
        :ptype where: Callable[[Mapping[str, Any]], bool] | None
        :return: the count
        :rtype: int
        """
        async with self._lock:
            await self._settle()
            if where is None:
                return len(self._held)
            return sum(1 for row in self._held.values() if where(row))

    async def _settle(self) -> None:
        """make what is held answerable: read cold when nothing is held, then each missed row again. Under the lock.

        :return: nothing
        :rtype: None
        """
        if not self._loaded:
            await self._load()
        for key in sorted(self._missed):
            await self._apply(key)
            self._missed.discard(key)

    async def read(
        self,
        columns: Sequence[str],
        *,
        where: Callable[[Mapping[str, Any]], bool] | None = None,
        page: int = _READ_PAGE,
    ) -> list[dict[str, Any]]:
        """the standing rows themselves, read by key ``page`` keys a statement, each judged again as read.

        :param columns: the columns to read (the key's and :attr:`Standing.held`'s are added)
        :ptype columns: Sequence[str]
        :param where: which held rows to read, by their held columns; every one when None
        :ptype where: Callable[[Mapping[str, Any]], bool] | None
        :param page: keys per statement
        :ptype page: int
        :return: the rows that stand as read, in key order
        :rtype: list[dict[str, Any]]
        """
        standing = self._standing
        wanted = tuple(dict.fromkeys((*standing.columns, *columns)))
        keys = await self.keys(where)
        if not keys:
            return []
        rows = await read_keyed(
            self._pool,
            standing.table,
            keys=KeyBatch(standing.key, keys, standing.key_types),
            columns=wanted,
            batch=page,
        )
        found = [row for row in rows if standing.member(row)]
        found.sort(key=standing.key_of)
        return found

    async def _load(self) -> None:
        """the cold start: the caller's cold read, with the table's generation read first when not yet following.

        :return: nothing
        :rtype: None
        """
        standing = self._standing
        if not self._following:
            self._since = await cold_read_generations(self._generations, (standing.table,))
        rows = await self._cold()
        self._held = {standing.key_of(row): dict(row) for row in rows if standing.member(row)}
        self._missed.clear()
        self._loaded = True
        log.info("standing rows read cold: set=%s table=%s rows=%d", self._name, standing.table, len(self._held))

    async def _apply(self, key: tuple[Any, ...]) -> None:
        """read one row by its whole key and hold it, or let it go, as it now stands.

        :param key: the row's key
        :ptype key: tuple[Any, ...]
        :return: nothing
        :rtype: None
        """
        standing = self._standing
        found = await read_keyed(
            self._pool,
            standing.table,
            keys=KeyBatch(standing.key, [key], standing.key_types),
            columns=standing.columns,
        )
        stands = [row for row in found if standing.member(row)]
        if stands:
            self._held[key] = stands[0]
        else:
            self._held.pop(key, None)

    async def _on_key(self, named: tuple[str, ...]) -> None:
        """a row changed: read it by its whole key. Before the cold start nothing is held, and nothing is read (the
        cold read will see it); a failed read is kept for the next question, which reads it again first.

        :param named: the row's whole key, as its change names it
        :ptype named: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        key = self._standing.parse(named)
        async with self._lock:
            if self._loaded:
                try:
                    await self._apply(key)
                except Exception as exc:
                    # the next question reads the row again first and raises to its caller if it fails again
                    self._missed.add(key)
                    contain(
                        exc,
                        log=log,
                        what="standing row not read; the next question reads it first",
                        extra={"set": self._name, "table": self._standing.table, "key": list(named)},
                    )

    def on_table_dropped(self) -> None:
        """the table was dropped whole (an operator's reset): nothing held stands, and the next question reads cold,
        as at a process's start. Nothing is read here.

        :return: nothing
        :rtype: None
        """
        self._loaded = False
        log.info("the %s table was reset; %s reads its standing rows cold next", self._standing.table, self._name)
