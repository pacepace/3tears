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
table's current generation is not behind; one that has not reads the change record of each advance
it missed (:mod:`threetears.core.collections.epoch_change`) and evicts exactly those rows. It never
drops the table for a gap; a record it cannot read leaves it behind, serving what it holds.

The protocol, the declaration and the mark live here, in core, because
:class:`~threetears.core.collections.base.BaseCollection` and
:class:`~threetears.core.collections.registry.CollectionRegistry` consume them and core may not
import the packages that implement the source. ``threetears.epoch`` provides the platform
implementation over its epoch bucket, and the pass that judges a mark against it.
"""

from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.collections.epoch_change import advance_recording
from threetears.core.exceptions import GenerationUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "L2_WRITE_GENERATION",
    "WRITE_GENERATION",
    "WRITE_GENERATION_UNDECLARED",
    "GenerationMarks",
    "GenerationSource",
    "GenerationVerdict",
    "L2WriteGeneration",
    "NotedTable",
    "NoWriteGeneration",
    "READS_GENERATIONS_ATTRIBUTE",
    "UndeclaredWriteGeneration",
    "WriteGeneration",
    "WriteGenerationDeclaration",
    "announce_unheard_writes",
    "note_generation_class",
    "primary_key_of_table",
    "note_l2_first_class",
    "noted_tables",
    "source_reads",
    "split_generation_token",
    "table_named_by_class",
    "tables_with_write_generation",
    "tables_written_l2_first",
]


@runtime_checkable
class GenerationSource(Protocol):
    """reads and advances each table's write generation.

    A source that can advance and not read declares ``reads_generations = False``
    (:data:`READS_GENERATIONS_ATTRIBUTE`, read through :func:`source_reads`); absence caching is
    then off on its registry.
    """

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

    async def advance(self, table_name: str) -> str | None:
        """move the table's generation on, after a write to it has committed, and say what it became.

        The token returned is the one this advance wrote, never a later one: the caller stamps it
        on the row broadcasts of the commit it advanced for, and a follower counts those rows
        against exactly that advance. A source whose tokens are ``{incarnation}:{count}``
        (:func:`split_generation_token`) can be followed; any other token only ever compares
        unequal, so a follower is behind on the table whenever it moves.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: the generation token this advance wrote, or ``None`` when the source knows that
            nothing was advanced (a pod whose broker advances no generation yet); the write's rows
            then name no generation
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation cannot be advanced
        """
        ...


async def announce_unheard_writes(source: GenerationSource, tables: Iterable[str]) -> dict[str, str]:
    """advance each table once for writes that no collection made and no row broadcast names.

    For a writer that changes rows it cannot list -- a schema migration, a restore, a bulk repair
    in SQL. Nothing is published for the advance, so every process following a table finds a count
    it did not hear and applies its change record, which names the whole table: the reach is unknown,
    so every row of the table, and everything derived from it, is the change. Call it after the writes
    have committed, never before: a follower that applied it first would re-read the old rows.

    Every table is attempted even when one fails, and the failures are raised together after,
    because a table left unadvanced keeps every follower's stale rows with nothing to correct them.

    :param source: the source the writer's process advances through
    :ptype source: GenerationSource
    :param tables: the tables written; each is advanced once
    :ptype tables: Iterable[str]
    :return: each table's token, as :meth:`GenerationSource.advance` returned it
    :rtype: dict[str, str]
    :raises GenerationUnavailableError: when any table could not be advanced, or its advance returned
        no generation, naming each
    """
    tokens: dict[str, str] = {}
    failed: dict[str, str] = {}
    for table in sorted(set(tables)):
        try:
            # a write nobody listed the rows of: its reach is the table
            token = await advance_recording(source, table, None)
        except GenerationUnavailableError as exc:
            failed[table] = str(exc)
            continue
        if not isinstance(token, str) or not token:
            # a source that does not say what it wrote may have moved nothing, and an announcement
            # that reached no follower leaves every one serving the old rows: a failure
            failed[table] = f"the source advanced {table} without returning a generation"
            continue
        tokens[table] = token
    if failed:
        raise GenerationUnavailableError(
            "writes no collection announced could not be put on the epoch system; followers keep what "
            f"they hold of these tables until each moves again: {failed}"
        )
    return tokens


#: the optional attribute a :class:`GenerationSource` sets to ``False`` when it can advance a table's
#: generation and cannot read one, so :meth:`GenerationSource.current` always raises. A source
#: without the attribute reads.
READS_GENERATIONS_ATTRIBUTE: Final = "reads_generations"


def source_reads(source: GenerationSource | None) -> bool:
    """whether ``source`` can read a table's generation, not only advance it.

    Reading and advancing are separate capabilities. A pod's
    :class:`~threetears.core.backends.BrokerGenerationSource` advances through the L3 broker and
    may hold no reader, and a collection that caches absences needs a generation it can READ to
    stamp them with: on a source that cannot, it caches none, exactly as with no source at all.

    :param source: the registry's source, or ``None``
    :ptype source: GenerationSource | None
    :return: ``True`` when there is a source and it does not say it cannot read
    :rtype: bool
    """
    return source is not None and bool(getattr(source, READS_GENERATIONS_ATTRIBUTE, True))


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


class L2WriteGeneration:
    """the type of :data:`L2_WRITE_GENERATION`: a table kept in L2 alone, whose generation is kept there too.

    A table no L3 commit writes has no broker to advance its generation, and a pod may not write the
    epoch bucket. Its writes are compare-and-sets on the owner's own L2 keys, so its generation is one
    more key beside them (``{scope}.{table}-generation``, under the same grant), advanced by every
    write that changes a row -- each won compare-and-set, delete and eviction -- and followed by every
    registry holding the collection (:meth:`CollectionRegistry.catch_up_l2_generations`). Tokens are
    ``{incarnation}:{count}``: the incarnation is minted when the key is first written, so a wiped
    bucket starts another and every follower is behind on the table until its epochs are restored.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "L2_WRITE_GENERATION"


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

#: declare on a collection class kept in L2 alone to keep its write generation in L2 beside its rows
#: (:class:`L2WriteGeneration`), with each advance's change record beside it, so a replica that missed
#: a row broadcast evicts exactly that row at the next catch-up pass.
L2_WRITE_GENERATION: Final = L2WriteGeneration()

#: the default: the collection has declared nothing, and writes as it did before. See
#: :class:`UndeclaredWriteGeneration`.
WRITE_GENERATION_UNDECLARED: Final = UndeclaredWriteGeneration()

#: everything :attr:`BaseCollection.write_generation` may hold.
WriteGenerationDeclaration = WriteGeneration | L2WriteGeneration | NoWriteGeneration | UndeclaredWriteGeneration


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
    catch-up pass asks it to judge the generation it just read (:meth:`judge`). A table nobody
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

    def highest_heard(self, table_name: str) -> str | None:
        """the newest generation this pod has heard any row of or accounted for, as a token.

        At or past :meth:`recorded`: an advance still waiting on a gap has had rows applied here. A
        cache that starts trusting a table part-way through takes this as its floor, since rows up
        to it were applied before it could hear them.

        :param table_name: the table
        :ptype table_name: str
        :return: ``{incarnation}:{count}``, or ``None`` when :meth:`recorded` is
        :rtype: str | None
        """
        mark = self._tables.get(table_name)
        if mark is None or not mark.sighted or mark.incarnation is None:
            return None
        newest = max(
            (count for incarnation, count in mark.heard if incarnation == mark.incarnation),
            default=mark.accounted,
        )
        return f"{mark.incarnation}:{max(newest, mark.accounted)}"

    def adopt(self, table_name: str, source: GenerationMarks) -> None:
        """take ``source``'s mark for ``table_name`` as this one's, and follow the table.

        For a second set of marks fed the same broadcasts from some point on (a cache counting what
        it has itself been told): adopting at that point makes the two agree on everything heard
        before it.

        :param table_name: the table
        :ptype table_name: str
        :param source: the marks to copy from
        :ptype source: GenerationMarks
        :return: nothing
        :rtype: None
        """
        theirs = source._tables.get(table_name)
        if theirs is None:
            self._tables[table_name] = _TableMark()
            return
        self._tables[table_name] = _TableMark(
            sighted=theirs.sighted,
            incarnation=theirs.incarnation,
            accounted=theirs.accounted,
            heard={key: _Heard(rows=heard.rows, of=heard.of) for key, heard in theirs.heard.items()},
            overflowed=theirs.overflowed,
        )

    def pending(self, table_name: str) -> tuple[str, ...]:
        """the advances beyond the mark this pod has heard rows of but cannot yet count, as tokens, for a log line.

        An advance waits here while one before it is unheard: the mark moves past an advance only once every one
        before it is accounted for.

        :param table_name: the followed table
        :ptype table_name: str
        :return: each waiting advance as ``incarnation:count (rows heard/rows)``, in count order; empty when the
            table is not followed or nothing waits
        :rtype: tuple[str, ...]
        """
        mark = self._tables.get(table_name)
        if mark is None:
            return ()
        return tuple(
            f"{incarnation}:{count} ({heard.rows}/{heard.of})"
            for (incarnation, count), heard in sorted(mark.heard.items(), key=lambda item: item[0][1])
        )

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

    def heard_in_full(self, table_name: str, token: str) -> bool:
        """whether every row of the advance that wrote ``token`` was heard or applied here.

        A follower walking a table's epochs skips applying one it heard in full: its broadcasts evicted
        every row already.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the advance
        :ptype token: str
        :return: ``True`` when it was heard in full, or is at or below the mark
        :rtype: bool
        """
        mark = self._tables.get(table_name)
        parts = split_generation_token(token)
        if mark is None or parts is None or not mark.sighted:
            return False
        if parts[0] == mark.incarnation and parts[1] <= mark.accounted:
            return True
        heard = mark.heard.get(parts)
        return heard is not None and heard.rows >= heard.of and not mark.overflowed

    def advance_to(self, table_name: str, token: str) -> None:
        """move the mark to ``token``: every advance of the table up to it is accounted for.

        For a follower that walked the table's epochs up to ``token`` and applied, or heard in full,
        every one. A mark taken over no generation takes ``token``'s incarnation. Advances heard beyond
        ``token`` stay.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the generation accounted up to
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        mark = self._tables.get(table_name)
        parts = split_generation_token(token)
        if mark is None or parts is None or not mark.sighted:
            return
        if mark.incarnation is None:
            mark.incarnation = parts[0]
            mark.accounted = 0
        if parts[0] != mark.incarnation or parts[1] <= mark.accounted:
            return
        mark.accounted = parts[1]
        mark.overflowed = False
        mark.heard = {
            key: heard for key, heard in mark.heard.items() if key[0] == mark.incarnation and key[1] > parts[1]
        }

    def applied(self, table_name: str, token: str) -> None:
        """record that the change of the advance that wrote ``token`` was fetched and applied in full.

        Once every advance up to the generation read is applied or heard, the mark has moved to it
        and an overflow (which hid which advances were heard) is spent.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the advance applied
        :ptype token: str
        :return: nothing
        :rtype: None
        """
        mark = self._tables.get(table_name)
        parts = split_generation_token(token)
        if mark is not None and parts is not None and mark.sighted and mark.incarnation is None:
            mark.incarnation = parts[0]
            mark.accounted = 0
        if mark is None or parts is None or not mark.sighted or parts[0] != mark.incarnation:
            return
        if parts[1] > mark.accounted:
            mark.heard[parts] = _Heard(rows=1, of=1)
            self._move_on(mark)
        if mark.overflowed and not any(heard.rows < heard.of for heard in mark.heard.values()):
            # the overflow only hid which advances were heard; every one up to here is applied now
            mark.overflowed = False

    def continue_under(self, table_name: str, lost: str, restored: str, covered_from: int | None) -> bool:
        """carry the table's mark from a lost log's incarnation to the start of the one restored from it.

        A restore puts back, under ``restored``, what the L3 change notes say changed after ``covered_from``
        of the lost log, in no particular order. A mark under ``lost`` at or past ``covered_from`` missed
        only epochs the notes cover, so it is carried to the restored log's start, with nothing heard: the
        follower walks every restored record and applies each, which evicts some rows it already held
        current, never fewer than it missed. An older mark, or a restore whose notes cover nothing, is left
        as it is.

        :param table_name: the followed table
        :ptype table_name: str
        :param lost: the lost log's incarnation
        :ptype lost: str
        :param restored: the restored log's incarnation
        :ptype restored: str
        :param covered_from: the lost log's sequence from which the notes cover every epoch; ``None`` when
            they cover none
        :ptype covered_from: int | None
        :return: whether the mark was carried over
        :rtype: bool
        """
        mark = self._tables.get(table_name)
        if (
            mark is None
            or not mark.sighted
            or mark.incarnation != lost
            or covered_from is None
            or mark.accounted < covered_from
        ):
            return False
        mark.incarnation = restored
        mark.accounted = 0
        mark.heard = {}
        mark.overflowed = False
        return True

    def adopt_generation(self, table_name: str, token: str | None) -> None:
        """take ``token`` as the table's mark, for a follower that holds nothing older than it.

        A pod starts following before it serves the table, so at its first sight of the generation
        everything it holds was read at or after it: there is nothing to catch up. Advances heard
        beyond ``token`` stay, since their broadcasts outran the read.

        :param table_name: the followed table
        :ptype table_name: str
        :param token: the generation read, or ``None`` when the store holds none
        :ptype token: str | None
        :return: nothing
        :rtype: None
        :raises KeyError: when ``table_name`` is not followed
        """
        mark = self._tables[table_name]
        incarnation, count = self._parts(token)
        mark.sighted = True
        mark.incarnation = incarnation
        mark.accounted = count
        mark.overflowed = False
        mark.heard = {key: heard for key, heard in mark.heard.items() if key[0] == incarnation and key[1] > count}
        self._move_on(mark)

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
        elif mark.incarnation is None and incarnation is not None:
            # the store held no generation when the mark was taken, so the first incarnation minted
            # since is this mark's own: every advance of it is one this pod may have missed
            verdict = GenerationVerdict.MISSED
        elif incarnation != mark.incarnation:
            verdict = GenerationVerdict.REPLACED
        elif count > mark.accounted or mark.overflowed:
            verdict = GenerationVerdict.MISSED
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


#: every collection class defined in this process whose committed writes advance its table's write
#: generation: switched on (``write_generation = WRITE_GENERATION``), or caching absences. Appended
#: by :meth:`BaseCollection.__init_subclass__` (:func:`note_generation_class`), read by :func:`tables_with_write_generation`.
_GENERATION_CLASSES: weakref.WeakSet[type] = weakref.WeakSet()


def note_generation_class(cls: type) -> None:
    """record a collection class whose committed writes advance its table's write generation.

    Called by ``BaseCollection.__init_subclass__`` for each such class it defines; read by
    :func:`tables_with_write_generation`.

    :param cls: the collection class
    :ptype cls: type
    :return: nothing
    :rtype: None
    """
    _GENERATION_CLASSES.add(cls)


def table_named_by_class(cls: type) -> str | None:
    """the table a collection class names without being built, or ``None``.

    The one derivation of "which table does this class name": the hub's broker reads it, through
    :func:`tables_with_write_generation`, to decide which tables to advance, and the
    one-class-per-table census (``threetears.enforcement.collection_census``) reads it to prove
    there is one class per table. Two copies could disagree on a class, and the census would then
    pass a layout the broker reads differently.

    :param cls: a collection class
    :ptype cls: type
    :return: a ``schema``'s ``name``, or what a ``table_name`` property answers with the class
        standing in for an instance; ``None`` when the table is named per instance
    :rtype: str | None
    """
    name = getattr(getattr(cls, "schema", None), "name", None)
    if isinstance(name, str):
        return name
    prop = inspect.getattr_static(cls, "table_name", None)
    if isinstance(prop, property) and prop.fget is not None:
        try:
            answer: Any = prop.fget(cls)
        except AttributeError, TypeError:
            # NOSILENT: the class standing in for an instance has no instance attributes, and that is the answer: a
            # getter that needs an instance names its table per instance. Anything else it raises is a bug, and propagates.
            return None
        return answer if isinstance(answer, str) else None
    return None


def tables_with_write_generation() -> frozenset[str]:
    """every table whose committed writes advance its write generation, by the classes defined here.

    What the hub's L3 broker reads to decide which tables to advance after committing a pod's
    write: the pod's collection and the broker's answer come from the same class, so they agree
    exactly when one class names the table. A table is in the set when a collection class this
    process has imported declares ``write_generation = WRITE_GENERATION`` or caches absences
    (``negative_cache_max_age``), and names its table on the class. A class whose table is named
    per instance is not readable here, and its table is not in the set: a pod switching such a
    table on is told by its own advance, which raises, that the broker advanced nothing.

    :return: the table names
    :rtype: frozenset[str]
    """
    return frozenset(name for name in map(table_named_by_class, list(_GENERATION_CLASSES)) if name is not None)


def primary_key_of_table(table_name: str) -> tuple[str, ...] | None:
    """the primary key columns of ``table_name``, as the collection classes defined here declare it.

    For a writer that must name rows of a table it holds no collection for -- a delete's foreign-key
    action reaching it -- so its epoch is keyed, never the whole table.

    :param table_name: the table
    :ptype table_name: str
    :return: the columns in declared order; ``None`` when no class here names the table, or two disagree
    :rtype: tuple[str, ...] | None
    """
    found: set[tuple[str, ...]] = set()
    for cls in list(_GENERATION_CLASSES):
        if table_named_by_class(cls) != table_name:
            continue
        declared = getattr(cls, "primary_key_column", None)
        if isinstance(declared, str):
            found.add((declared,))
        elif isinstance(declared, tuple) and all(isinstance(column, str) for column in declared):
            found.add(declared)
    return found.pop() if len(found) == 1 else None


#: every collection class defined in this process that commits its writes in NATS first
#: (``BaseCollection.l2_first``), appended by :func:`note_l2_first_class`.
_L2_FIRST_CLASSES: weakref.WeakSet[type] = weakref.WeakSet()


def note_l2_first_class(cls: type) -> None:
    """record a collection class that commits its writes in NATS first.

    :param cls: the collection class
    :ptype cls: type
    :return: nothing
    :rtype: None
    """
    _L2_FIRST_CLASSES.add(cls)


def tables_written_l2_first() -> frozenset[str]:
    """every table whose writes commit in NATS first, by the classes defined here.

    Such a table's epoch is advanced, naming its rows, before its row reaches L3; an L3 write of it
    (the writer's own, or the saver's) is already covered. So the write-mark reconciler deletes a mark
    on one without advancing it again, which it would do as a change to the whole table, and the
    trigger install leaves such a table without one.

    :return: the table names
    :rtype: frozenset[str]
    """
    return frozenset(name for name in map(table_named_by_class, list(_L2_FIRST_CLASSES)) if name is not None)


@dataclass(frozen=True, slots=True)
class NotedTable:
    """a table whose writes leave a change note in L3, and the columns its note names.

    :ivar table: the table
    :ivar keys: its primary-key columns, in declared order
    :ivar columns: its declared invalidation columns, which a note carries as a row broadcast does
    """

    table: str
    keys: tuple[str, ...]
    columns: tuple[str, ...] = ()


def _declared_key(cls: type) -> tuple[str, ...] | None:
    """the primary key a collection class declares, or ``None`` when it declares none readable.

    :param cls: the collection class
    :ptype cls: type
    :return: the columns in declared order
    :rtype: tuple[str, ...] | None
    """
    declared = getattr(cls, "primary_key_column", None)
    if isinstance(declared, str):
        return (declared,)
    if isinstance(declared, tuple) and declared and all(isinstance(column, str) for column in declared):
        return declared
    return None


def noted_tables() -> tuple[NotedTable, ...]:
    """every table whose epochs go into the change log, with the key and columns its change note names.

    The tables of every class this process defined that carries a write generation, caches absences
    (:func:`tables_with_write_generation`) or commits L2-first (:func:`tables_written_l2_first`): each of
    their L3 writes leaves a note in its own transaction
    (:mod:`~threetears.core.collections.change_notes`). A table two classes name with different keys, or
    whose class names it per instance, is left out: no one trigger can name its rows. Two classes naming
    one table with one key carry every invalidation column either declares.

    :return: the tables, in table order
    :rtype: tuple[NotedTable, ...]
    """
    keys_of: dict[str, set[tuple[str, ...]]] = {}
    columns_of: dict[str, dict[str, None]] = {}
    for cls in [*list(_GENERATION_CLASSES), *list(_L2_FIRST_CLASSES)]:
        name = table_named_by_class(cls)
        keys = _declared_key(cls)
        if name is None or keys is None:
            continue
        keys_of.setdefault(name, set()).add(keys)
        declared = getattr(cls, "invalidation_columns", ())
        for column in declared if isinstance(declared, tuple) else ():
            if isinstance(column, str):
                columns_of.setdefault(name, {})[column] = None
    return tuple(
        NotedTable(table=name, keys=next(iter(keys)), columns=tuple(columns_of.get(name, {})))
        for name, keys in sorted(keys_of.items())
        if len(keys) == 1
    )
