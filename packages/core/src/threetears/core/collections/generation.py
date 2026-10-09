"""a table's write generation: the source that moves it, what a collection declares, and a pod's mark.

A collection that caches "this key exists in no tier" has to know when that answer stops being
true. It cannot learn that from the key: a writer in another pod, or another principal, commits
the row somewhere the reader's cache never sees, and every per-key signal can race the marker it
should invalidate. What it can learn instead is whether ANY write to the table committed since the
answer was cached. A reader stamps each absence with the table's generation, read before the L3
lookup that found nothing; a writer advances the generation after its L3 commit. An absence whose
stamp is not the current generation is no longer trusted -- whatever the timing of the writes,
broadcasts or clocks involved.

The same generation answers a second question: "is this pod's cache of the table behind". A pod
that follows a table keeps a :class:`GenerationMarks` mark for it -- the last generation whose
writes it has accounted for -- and every row broadcast names the generation its write advanced to
and how many rows that advance covered. A pod that has heard every row of every advance up to the
table's current generation is not behind; one that has not must drop what it holds of the table,
because it cannot say which row it missed.

The protocol, the declaration and the mark live here, in core, because
:class:`~threetears.core.collections.base.BaseCollection` and
:class:`~threetears.core.collections.registry.CollectionRegistry` consume them and core may not
import the packages that implement the source. ``threetears.epoch`` provides the platform
implementation over its epoch bucket, and the pass that judges a mark against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "WRITE_GENERATION",
    "WRITE_GENERATION_UNDECLARED",
    "GenerationMarks",
    "GenerationSource",
    "GenerationVerdict",
    "NoWriteGeneration",
    "UndeclaredWriteGeneration",
    "WriteGeneration",
    "WriteGenerationDeclaration",
    "split_generation_token",
]


@runtime_checkable
class GenerationSource(Protocol):
    """reads and advances each table's write generation."""

    async def current(self, table_name: str) -> str:
        """the table's current generation, as an opaque token compared for equality only.

        MUST change whenever :meth:`advance` has run for the table since the token was read, and
        MUST also change when the store holding the generation was replaced -- a reset counter
        that returned to an old value would revalidate every absence recorded before it.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be read
        """
        ...

    async def advance(self, table_name: str) -> str:
        """move the table's generation on, after a write to it has committed, and say what it became.

        The token returned is the one this advance wrote, never a later one: the caller stamps it
        on the row broadcasts of the commit it advanced for, and a follower counts those rows
        against exactly that advance. A source whose tokens are ``{incarnation}:{count}``
        (:func:`split_generation_token`) can be followed; any other token only ever compares
        unequal, so a follower drops the table whenever it moves.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token this advance wrote
        :rtype: str
        :raises GenerationUnavailableError: when the generation cannot be advanced
        """
        ...


def split_generation_token(token: str) -> tuple[str, int] | None:
    """split a followable generation token into its incarnation and its count.

    The incarnation names one lifetime of the store holding the generation: it changes when the
    store is replaced, so a count under one incarnation is never compared with a count under
    another. The count moves on by one with every advance.

    :param token: a generation token
    :ptype token: str
    :return: ``(incarnation, count)``, or ``None`` when the token is not ``{incarnation}:{count}``
    :rtype: tuple[str, int] | None
    """
    incarnation, separator, count = token.rpartition(":")
    if not separator or not incarnation or not count.isdigit():
        return None
    return incarnation, int(count)


class WriteGeneration:
    """the type of :data:`WRITE_GENERATION`: every committed write to the table advances its generation."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "WRITE_GENERATION"


@dataclass(frozen=True, slots=True)
class NoWriteGeneration:
    """a collection's declared decision that writes to its table advance no generation.

    For a table that shares a cache across pods and is written too often for an advance per
    commit to be worth it. Nothing can then tell a pod that its cache of the table is behind, so
    the reason is part of the declaration and is never empty.

    :ivar reason: why this table carries no write generation
    """

    reason: str

    def __post_init__(self) -> None:
        """refuse a declaration that gives no reason.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``reason`` is empty or only whitespace
        """
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("NoWriteGeneration needs a reason: say why this table carries no write generation")


class UndeclaredWriteGeneration:
    """the type of :data:`WRITE_GENERATION_UNDECLARED`: the collection has not been switched on.

    What :attr:`BaseCollection.write_generation` is until a class says otherwise, in the releases
    where carrying a write generation is something a table is switched on to do. Such a collection
    behaves exactly as it did before write generations were followed: it advances one only where
    absence caching already made it. Switching a table on is declaring
    ``write_generation = WRITE_GENERATION`` on its collection class; making every collection carry
    one unless it declares :class:`NoWriteGeneration` is changing that one default.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "WRITE_GENERATION_UNDECLARED"


#: declare on a collection class to switch its table on: every committed write advances the
#: table's write generation, once per commit, and every row broadcast of that commit names it.
WRITE_GENERATION: Final = WriteGeneration()

#: the default: the collection has declared nothing, and writes as it did before. See
#: :class:`UndeclaredWriteGeneration`.
WRITE_GENERATION_UNDECLARED: Final = UndeclaredWriteGeneration()

#: everything :attr:`BaseCollection.write_generation` may hold.
WriteGenerationDeclaration = WriteGeneration | NoWriteGeneration | UndeclaredWriteGeneration


class GenerationVerdict(StrEnum):
    """what a pod learns when it compares a table's generation with its own mark.

    :cvar CURRENT: every advance up to the generation was accounted for; nothing to do
    :cvar FIRST_SIGHT: the pod had no mark for the table, so it cannot vouch for what it cached
        before it started following
    :cvar REPLACED: the generation belongs to another incarnation of its store than the mark's, so
        counts cannot be compared and any number of writes may have been missed
    :cvar MISSED: same incarnation, and an advance the pod did not hear every row of
    """

    CURRENT = "current"
    FIRST_SIGHT = "first_sight"
    REPLACED = "replaced"
    MISSED = "missed"

    @property
    def drops(self) -> bool:
        """whether this verdict means the pod must drop what it holds of the table.

        :return: ``True`` for every verdict but :attr:`CURRENT`
        :rtype: bool
        """
        return self is not GenerationVerdict.CURRENT


#: how many advances one table's mark remembers as heard but not yet contiguous with the mark. A
#: follower nobody runs a pass for would otherwise remember every advance after the first one it
#: missed, for ever. Past this the mark stops recording, which a pass reads as an advance unheard.
_MAX_REMEMBERED_ADVANCES: Final = 4096


@dataclass(slots=True)
class _Heard:
    """the rows a pod has heard of one advance.

    :ivar rows: how many row broadcasts of the advance have arrived
    :ivar of: how many rows the advance covered, as its broadcasts state
    """

    rows: int
    of: int


@dataclass(slots=True)
class _TableMark:
    """one followed table's mark.

    :ivar sighted: whether a generation has ever been recorded for the table
    :ivar incarnation: the incarnation of the recorded generation; ``None`` when the store held no
        generation for the table at all
    :ivar accounted: every advance up to and including this count, under :attr:`incarnation`, is
        accounted for
    :ivar heard: advances beyond :attr:`accounted` this pod has heard rows of, by
        ``(incarnation, count)``
    :ivar overflowed: whether :attr:`heard` stopped recording because it was full
    """

    sighted: bool = False
    incarnation: str | None = None
    accounted: int = 0
    heard: dict[tuple[str, int], _Heard] = field(default_factory=dict)
    overflowed: bool = False


class GenerationMarks:
    """per followed table, the last write generation whose writes this pod has accounted for.

    One per :class:`~threetears.core.collections.registry.CollectionRegistry`. Nothing here does
    I/O, awaits or drops anything: the registry feeds it every row broadcast that names a
    generation (:meth:`hear`) and every advance its own writes made (:meth:`account`), and a
    catch-up pass asks it to judge the generation it just read (:meth:`settle`). A table nobody
    asked to follow costs nothing: its broadcasts are not recorded.
    """

    __slots__ = ("_tables",)

    def __init__(self) -> None:
        """start following nothing.

        :return: nothing
        :rtype: None
        """
        self._tables: dict[str, _TableMark] = {}

    def follow(self, table_name: str) -> None:
        """start keeping a mark for ``table_name``; a no-op when one is already kept.

        :param table_name: the table to follow
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        self._tables.setdefault(table_name, _TableMark())

    def unfollow(self, table_name: str) -> None:
        """stop keeping a mark for ``table_name``.

        :param table_name: the table to stop following
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        self._tables.pop(table_name, None)

    def follows(self, table_name: str) -> bool:
        """whether a mark is kept for ``table_name``.

        :param table_name: the table
        :ptype table_name: str
        :return: ``True`` when the table is followed
        :rtype: bool
        """
        return table_name in self._tables

    @property
    def followed(self) -> tuple[str, ...]:
        """every followed table, in the order each was first followed.

        :return: the table names
        :rtype: tuple[str, ...]
        """
        return tuple(self._tables)

    def recorded(self, table_name: str) -> str | None:
        """the generation recorded for ``table_name``, as a token, for a log line or an assertion.

        :param table_name: the table
        :ptype table_name: str
        :return: ``{incarnation}:{accounted}``, or ``None`` when the table is not followed, has no
            mark yet, or its store held no generation when the mark was taken
        :rtype: str | None
        """
        mark = self._tables.get(table_name)
        if mark is None or not mark.sighted or mark.incarnation is None:
            return None
        return f"{mark.incarnation}:{mark.accounted}"

    def hear(self, table_name: str, token: str, bump_rows: int) -> None:
        """record one row broadcast of the advance that wrote ``token``.

        :param table_name: the table the row belongs to
        :ptype table_name: str
        :param token: the generation the row's commit advanced to
        :ptype token: str
        :param bump_rows: how many rows that advance covered
        :ptype bump_rows: int
        :return: nothing
        :rtype: None
        """
        self._record(table_name, token, rows=1, of=bump_rows)

    def account(self, table_name: str, token: str) -> None:
        """record an advance this pod made itself: it needs no broadcast to know every row of it.

        :param table_name: the table advanced
        :ptype table_name: str
        :param token: the generation the advance wrote
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        self._record(table_name, token, rows=1, of=1)

    def _record(self, table_name: str, token: str, *, rows: int, of: int) -> None:
        """count ``rows`` rows of the advance that wrote ``token``, and move the mark on if it can.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation the advance wrote
        :ptype token: str
        :param rows: how many of its rows are being counted
        :ptype rows: int
        :param of: how many rows the advance covered
        :ptype of: int
        :return: nothing
        :rtype: None
        """
        mark = self._tables.get(table_name)
        parts = split_generation_token(token)
        if mark is None or parts is None or of < 1:
            # not followed; or a token that cannot be counted, which the next pass sees as moved.
            return
        incarnation, count = parts
        if mark.sighted and incarnation == mark.incarnation and count <= mark.accounted:
            return
        heard = mark.heard.get(parts)
        if heard is None:
            if len(mark.heard) >= _MAX_REMEMBERED_ADVANCES:
                mark.overflowed = True
                return
            heard = _Heard(rows=0, of=of)
            mark.heard[parts] = heard
        heard.rows += rows
        self._move_on(mark)

    @staticmethod
    def _move_on(mark: _TableMark) -> None:
        """move ``mark`` past every advance heard in full that directly follows it.

        :param mark: the table's mark
        :ptype mark: _TableMark
        :return: nothing
        :rtype: None
        """
        if not mark.sighted or mark.incarnation is None:
            return
        while True:
            following = mark.heard.get((mark.incarnation, mark.accounted + 1))
            if following is None or following.rows < following.of:
                return
            del mark.heard[(mark.incarnation, mark.accounted + 1)]
            mark.accounted += 1

    def judge(self, table_name: str, token: str | None) -> GenerationVerdict:
        """compare a generation just read with the table's mark, changing nothing.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the table's generation as its store holds it now, or ``None`` when the store
            holds none. A generation older than the mark, under the mark's incarnation, is one
            whose advances were all accounted for already: a watcher is handed each generation
            in turn, and the broadcasts of a later one may have arrived first
        :ptype token: str | None
        :return: the verdict
        :rtype: GenerationVerdict
        :raises KeyError: when ``table_name`` is not followed
        """
        mark = self._tables[table_name]
        incarnation, count = self._parts(token)
        verdict = GenerationVerdict.CURRENT
        if not mark.sighted:
            verdict = GenerationVerdict.FIRST_SIGHT
        elif incarnation != mark.incarnation:
            verdict = GenerationVerdict.REPLACED
        elif count > mark.accounted or mark.overflowed:
            verdict = GenerationVerdict.MISSED
        return verdict

    def settle(self, table_name: str, token: str | None) -> GenerationVerdict:
        """judge a generation just read, and record it as the mark when the table must be dropped.

        The caller drops the table when the verdict :attr:`~GenerationVerdict.drops`. A verdict of
        :attr:`~GenerationVerdict.CURRENT` leaves the mark where the broadcasts moved it.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the table's generation as its store holds it now, or ``None`` when the store
            holds none
        :ptype token: str | None
        :return: the verdict
        :rtype: GenerationVerdict
        :raises KeyError: when ``table_name`` is not followed
        """
        verdict = self.judge(table_name, token)
        if verdict.drops:
            mark = self._tables[table_name]
            incarnation, count = self._parts(token)
            mark.sighted = True
            mark.incarnation = incarnation
            mark.accounted = count
            mark.overflowed = False
            # advances already heard beyond the generation read stay: their broadcasts outran it.
            mark.heard = {key: heard for key, heard in mark.heard.items() if key[0] == incarnation and key[1] > count}
            self._move_on(mark)
        return verdict

    @staticmethod
    def _parts(token: str | None) -> tuple[str | None, int]:
        """a generation's incarnation and count, for comparison with a mark.

        :param token: the generation, or ``None`` when its store holds none
        :ptype token: str | None
        :return: ``(incarnation, count)``; a token that is not ``{incarnation}:{count}`` is its own
            incarnation at count zero, so it compares unequal to every mark but itself
        :rtype: tuple[str | None, int]
        """
        if token is None:
            return None, 0
        parts = split_generation_token(token)
        return parts if parts is not None else (token, 0)
