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
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from threetears.core.backends.broker_generation import (
    named_commit_s_min,
    named_commit_scopes,
    named_delta_files,
    named_unconfirmed_delta_files,
    note_advanced_generation,
    taken_statement_stamp,
)
from threetears.core.collections.epoch_change import advance_recording, records_changes
from threetears.core.collections.undeclared_tables import report_undeclared_table
from threetears.core.frozen_state import frozen, logged, with_entry, without_entry
from threetears.observe import FrozenWeakSet
from threetears.core.tenancy import TENANCY_UNDECLARED, Tenancy, rows_with_tenants, schema_tenant
from threetears.core.exceptions import GenerationUnavailableError, SchemaTenantUnknownError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from threetears.core.collections.epoch_change import ChangedRow

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
    "advance_for_write",
    "announce_unheard_writes",
    "note_generation_class",
    "primary_key_of_table",
    "note_l2_first_class",
    "note_tenancy_class",
    "has_write_generation",
    "written_l2_first",
    "noted_tables",
    "source_reads",
    "split_generation_token",
    "switched_tables_not_noted",
    "table_named_by_class",
    "tables_with_write_generation",
    "tenancy_of_table",
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


async def announce_unheard_writes(
    source: GenerationSource, tables: Iterable[str], *, schema: str | None = None
) -> dict[str, str]:
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
    :param schema: the schema the writes ran in: an agent-schema table's record goes under it
        (:func:`advance_for_write`); ``None`` for a platform writer, which an agent-schema table refuses
    :ptype schema: str | None
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
            token = await advance_for_write(source, table, None, schema=schema)
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


@dataclass(frozen=True, slots=True)
class _Heard:
    """the rows a pod has heard of one advance.

    :ivar rows: how many row broadcasts of the advance have arrived
    :ivar of: how many rows the advance covered, as its broadcasts state
    """

    rows: int
    of: int


@dataclass(frozen=True, slots=True)
class _TableMark:
    """one followed table's mark.

    :ivar sighted: whether a generation has ever been recorded for the table
    :ivar incarnation: the incarnation of the recorded generation; ``None`` when the store held no
        generation for the table at all
    :ivar accounted: every advance up to and including this count, under :attr:`incarnation`, is
        accounted for
    :ivar heard: advances beyond :attr:`accounted` this pod has heard rows of, by
        ``(incarnation, count)``; a :func:`~threetears.core.frozen_state.logged` map, so a count (one advance set)
        copies a key about log2(n) times over its life, never every waiting advance a count. Replaced whole by
        :func:`~dataclasses.replace` and a frozen ``dict`` copied each count, hearing W waiting advances cost O(W^2):
        20,000 advances heard behind one gap held one turn 0.52 s, and counting each one's last rows 0.88 s more
        (``_MAX_REMEMBERED_ADVANCES`` waiting; macOS arm64, Python 3.14, 2026-11-25)
    :ivar overflowed: whether :attr:`heard` stopped recording because it was full
    """

    sighted: bool = False
    incarnation: str | None = None
    accounted: int = 0
    heard: Mapping[tuple[str, int], _Heard] = field(default_factory=logged)
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
        #: an immutable value replaced whole on change, as each table's mark is
        self._tables: Mapping[str, _TableMark] = frozen()

    def follow(self, table_name: str) -> None:
        """start keeping a mark for ``table_name``; a no-op when one is already kept.

        :param table_name: the table to follow
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        if table_name not in self._tables:
            self._tables = with_entry(self._tables, table_name, _TableMark())

    def unfollow(self, table_name: str) -> None:
        """stop keeping a mark for ``table_name``.

        :param table_name: the table to stop following
        :ptype table_name: str
        :return: nothing
        :rtype: None
        """
        self._tables = without_entry(self._tables, table_name)

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
        # a mark is immutable, so theirs is taken as it is: neither registry's later change reaches the other
        self._tables = with_entry(self._tables, table_name, _TableMark() if theirs is None else theirs)

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

    def hear(self, table_name: str, token: str, bump_rows: int, *, rows: int = 1) -> None:
        """record row broadcasts of the advance that wrote ``token``: one, or a batched message's ``rows``.

        :param table_name: the table the rows belong to
        :ptype table_name: str
        :param token: the generation the rows' commit advanced to
        :ptype token: str
        :param bump_rows: how many rows that advance covered
        :ptype bump_rows: int
        :param rows: how many of its rows were heard
        :ptype rows: int
        :return: nothing
        :rtype: None
        """
        self._record(table_name, token, rows=rows, of=bump_rows)

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
            mark = replace(mark, incarnation=parts[0], accounted=0)
            self._tables = with_entry(self._tables, table_name, mark)
        if parts[0] != mark.incarnation or parts[1] <= mark.accounted:
            return
        incarnation = mark.incarnation
        mark = replace(
            mark,
            accounted=parts[1],
            overflowed=False,
            heard=logged(
                {key: heard for key, heard in mark.heard.items() if key[0] == incarnation and key[1] > parts[1]}
            ),
        )
        self._tables = with_entry(self._tables, table_name, mark)

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
            mark = replace(mark, incarnation=parts[0], accounted=0)
            self._tables = with_entry(self._tables, table_name, mark)
        if mark is None or parts is None or not mark.sighted or parts[0] != mark.incarnation:
            return
        if parts[1] > mark.accounted:
            mark = self._moved_on(replace(mark, heard=with_entry(mark.heard, parts, _Heard(rows=1, of=1))))
        if mark.overflowed and not any(heard.rows < heard.of for heard in mark.heard.values()):
            # the overflow only hid which advances were heard; every one up to here is applied now
            mark = replace(mark, overflowed=False)
        self._tables = with_entry(self._tables, table_name, mark)

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
        self._tables = with_entry(
            self._tables, table_name, replace(mark, incarnation=restored, accounted=0, heard=logged(), overflowed=False)
        )
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
        mark = _TableMark(
            sighted=True,
            incarnation=incarnation,
            accounted=count,
            overflowed=False,
            heard=logged({key: heard for key, heard in mark.heard.items() if key[0] == incarnation and key[1] > count}),
        )
        self._tables = with_entry(self._tables, table_name, self._moved_on(mark))

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
                if not mark.overflowed:
                    # once full, an advance more changes nothing: never the mark rebuilt again a count
                    self._tables = with_entry(self._tables, table_name, replace(mark, overflowed=True))
                return
            heard = _Heard(rows=rows, of=of)
        else:
            heard = _Heard(rows=heard.rows + rows, of=heard.of)
        self._tables = with_entry(
            self._tables, table_name, self._moved_on(replace(mark, heard=with_entry(mark.heard, parts, heard)))
        )

    @staticmethod
    def _moved_on(mark: _TableMark) -> _TableMark:
        """``mark`` moved past every advance heard in full that directly follows it.

        :param mark: the table's mark
        :ptype mark: _TableMark
        :return: the mark moved on (``mark`` itself when none follows)
        :rtype: _TableMark
        """
        incarnation = mark.incarnation
        if not mark.sighted or incarnation is None:
            return mark
        accounted = mark.accounted
        while True:
            following = mark.heard.get((incarnation, accounted + 1))
            if following is None or following.rows < following.of:
                break
            accounted += 1
        if accounted == mark.accounted:
            return mark
        # each passed advance unset alone: a move past a few never rebuilds every advance still waiting
        heard = mark.heard
        for passed in range(mark.accounted + 1, accounted + 1):
            heard = without_entry(heard, (incarnation, passed))
        return replace(mark, accounted=accounted, heard=heard)

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


#: the three class registries below are replaced whole under this lock: a class can be defined on any thread (an
#: import on a worker thread), and two registering at once must both stay
_CLASSES_LOCK: Final = threading.Lock()

#: every collection class defined in this process whose committed writes advance its table's write
#: generation: switched on (``write_generation = WRITE_GENERATION``), or caching absences. Joined
#: by :meth:`BaseCollection.__init_subclass__` (:func:`note_generation_class`), read by :func:`tables_with_write_generation`.
#: Held weakly, so a class defined and dropped (a test's) leaves with it.
_GENERATION_CLASSES: FrozenWeakSet[type] = FrozenWeakSet()


def note_generation_class(cls: type) -> None:
    """record a collection class whose committed writes advance its table's write generation.

    Called by ``BaseCollection.__init_subclass__`` for each such class it defines; read by
    :func:`tables_with_write_generation`.

    :param cls: the collection class
    :ptype cls: type
    :return: nothing
    :rtype: None
    """
    global _GENERATION_CLASSES
    with _CLASSES_LOCK:
        _GENERATION_CLASSES = _GENERATION_CLASSES.with_item(cls)


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


def has_write_generation(table_name: str) -> bool:
    """whether ``table_name``'s committed writes advance its write generation, by the classes defined here.

    The per-table lookup of :func:`tables_with_write_generation`. A table no module listed under the
    ``threetears.collections`` entry points declares answers ``False`` (the fallback) and is reported
    (:func:`~threetears.core.collections.undeclared_tables.report_undeclared_table`). NEXT RELEASE: refused.

    :param table_name: the table
    :ptype table_name: str
    :return: whether it is switched on
    :rtype: bool
    """
    if table_name in tables_with_write_generation():
        return True
    report_undeclared_table(table_name, "write_generation")
    return False


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
#: (``BaseCollection.l2_first``), joined by :func:`note_l2_first_class`.
_L2_FIRST_CLASSES: FrozenWeakSet[type] = FrozenWeakSet()


def note_l2_first_class(cls: type) -> None:
    """record a collection class that commits its writes in NATS first.

    :param cls: the collection class
    :ptype cls: type
    :return: nothing
    :rtype: None
    """
    global _L2_FIRST_CLASSES
    with _CLASSES_LOCK:
        _L2_FIRST_CLASSES = _L2_FIRST_CLASSES.with_item(cls)


def _tables_written_l2_first() -> frozenset[str]:
    """every table whose writes commit in NATS first, by the classes defined here.

    Such a table's epoch is advanced, naming its rows, before its row reaches L3; an L3 write of it
    (the writer's own, or the saver's) is already covered. So a lost write's recovery
    (:mod:`threetears.core.collections.lost_writes`) never advances one again from its notes.

    :return: the table names
    :rtype: frozenset[str]
    """
    return frozenset(name for name in map(table_named_by_class, list(_L2_FIRST_CLASSES)) if name is not None)


def written_l2_first(table_name: str) -> bool:
    """whether ``table_name``'s writes commit in NATS first, by the classes defined here.

    The per-table lookup of :func:`_tables_written_l2_first`. A table no module listed under the
    ``threetears.collections`` entry points declares answers ``False`` (the fallback) and is reported
    (:func:`~threetears.core.collections.undeclared_tables.report_undeclared_table`). NEXT RELEASE: refused.

    :param table_name: the table
    :ptype table_name: str
    :return: whether it commits L2-first
    :rtype: bool
    """
    if table_name in _tables_written_l2_first():
        return True
    report_undeclared_table(table_name, "l2_first")
    return False


#: every collection class defined in this process that declares a tenancy, joined by
#: :func:`note_tenancy_class` and read by :func:`tenancy_of_table`.
_TENANCY_CLASSES: FrozenWeakSet[type] = FrozenWeakSet()


def note_tenancy_class(cls: type) -> None:
    """record a collection class that declares its table's tenancy.

    :param cls: the collection class
    :ptype cls: type
    :return: nothing
    :rtype: None
    """
    global _TENANCY_CLASSES
    with _CLASSES_LOCK:
        _TENANCY_CLASSES = _TENANCY_CLASSES.with_item(cls)


def tenancy_of_table(table_name: str) -> Tenancy:
    """the tenancy the classes defined here declare for ``table_name``.

    What a pod's change-log reader picks a table's keys by (:class:`~threetears.core.tenancy.PodTenants`),
    and what a writer that knows the schema it wrote in names a rowless change's tenant by: the same
    declaration the table's records are keyed by.

    :param table_name: the table
    :ptype table_name: str
    :return: the first declaration found; :data:`~threetears.core.tenancy.TENANCY_UNDECLARED` when no class
        defined here names the table with one. A table no module listed under the ``threetears.collections``
        entry points declares gets that fallback too, and is reported
        (:func:`~threetears.core.collections.undeclared_tables.report_undeclared_table`). NEXT RELEASE: refused.
    :rtype: Tenancy
    """
    for cls in list(_TENANCY_CLASSES):
        if table_named_by_class(cls) == table_name:
            tenancy = getattr(cls, "tenancy", None)
            if isinstance(tenancy, Tenancy):
                return tenancy
    report_undeclared_table(table_name, "tenancy")
    return TENANCY_UNDECLARED


async def advance_for_write(
    source: GenerationSource,
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    schema: str | None,
    keys_unknown: bool = False,
    tenancy: Tenancy | None = None,
    partition: Mapping[str, str] | None = None,
    noted_between: tuple[datetime, datetime] | None = None,
    prefixes: Sequence[Mapping[str, str]] | None = None,
    reason: str | None = None,
    delta_file: str | None = None,
    scopes: Sequence[str] | None = None,
    unconfirmed_delta_file: str | None = None,
    unconfirmed_writer: str | None = None,
    repair: bool = True,
    s_min: int | None = None,
) -> str | None:
    """advance ``table_name`` after a write that ran in ``schema``, each record under the tenant its table declares.

    For a writer that runs other principals' statements and knows the schema they ran in (the hub's broker): a
    pod names no tenant (``ChangedRow.tenant`` is never sent), so the writer names it here, by the one rule. An
    agent-schema table's records, rowless ones too, go under ``schema``'s tenant; a table with a tenant column
    names each row's from the row, and its rowless records go on ``_all``; any other table's go on ``_all``.

    :param source: the writer's generation source
    :ptype source: GenerationSource
    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows the committed write changed; ``None`` when its reach is the whole table
    :ptype rows: Sequence[ChangedRow] | None
    :param schema: the schema the write ran in; ``None`` when it names none
    :ptype schema: str | None
    :param keys_unknown: the writer named no rows and did not change the whole table; requires ``rows=None``
    :ptype keys_unknown: bool
    :param tenancy: the table's tenancy, for a writer that knows it otherwise than by a class it imported: a table
        switched on by its trigger alone, which no class here declares, lives in the schema its write ran in
        (:func:`~threetears.core.collections.lost_writes.tables_switched_by_trigger`), so it is
        :data:`~threetears.core.tenancy.TENANT_BY_SCHEMA`. ``None`` reads the classes (:func:`tenancy_of_table`)
    :ptype tenancy: Tenancy | None
    :param partition: for a whole-table change that is a load from files, the columns its rows hold and their values
        as text (the load's constants): the record names it, so a follower reads those rows again, not the table
    :ptype partition: Mapping[str, str] | None
    :param noted_between: for a keys-unknown change (rows too many to name), the database clock's bracket of the
        commit's change notes on its connection: the record names it, so a follower resolves the keys from the notes
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: for a whole-table change of a load or a clear, the key prefixes it reached
        (:attr:`~threetears.core.collections.epoch_change.EpochChange.prefixes`): a follower reads each as a key range
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: for a whole-table change, ``"reset"`` for an operator's reset; a follower reads the table again as
        on a first start
    :ptype reason: str | None
    :param delta_file: the commit's delta file of the table (``EpochChange.delta_file``), which every record of the
        advance names: the broker passes the one its commit request named (``DELTA_FILES_FIELD``); ``None`` takes the
        one the commit being settled in this context named (``naming_delta_files``), none when it named none
    :ptype delta_file: str | None
    :param scopes: for a keys-unknown change of a commit that saved a delta file, the scopes its transaction touched
        (``EpochChange.scopes``, decision 31): the broker passes the ones its commit request named
        (``COMMIT_SCOPES_FIELD``); ``None`` takes the ones the commit being settled in this context named
        (``naming_commit_scopes``). Named only on a keys-unknown record that names a delta file
    :ptype scopes: Sequence[str] | None
    :param unconfirmed_delta_file: a whole-table change's file of a load request that failed after it may have
        committed some rows (``EpochChange.delta_file_unconfirmed``): the broker passes the one the request named;
        ``None`` takes the one named in this context (``naming_unconfirmed_delta_files``), none when none is
    :ptype unconfirmed_delta_file: str | None
    :param unconfirmed_writer: beside ``unconfirmed_delta_file`` on a load's intent record, written before its first
        commit, the loading writer's beat key (``EpochChange.unconfirmed_writer``); ``None`` names none
    :ptype unconfirmed_writer: str | None
    :param repair: hand a failed advance to the process's writer ledger for retry (``advance_recording``); ``False``
        for an advance whose failure the caller answers itself, so no retry lands after it (a load's intent record,
        whose failure stops the load before its first commit)
    :ptype repair: bool
    :param s_min: the commit's lowest row epoch (``EpochChange.s_min``), named by every record of the advance: the
        broker passes the one its commit request named (``COMMIT_S_MIN_FIELD``); ``None`` takes the one the commit
        being settled in this context named (``naming_commit_s_min``); a write that joined no transaction, the lowest
        stamp its statements carried (``taken_statement_stamp``); none when none is named
    :ptype s_min: int | None
    :return: the token the advance wrote, as :func:`~threetears.core.collections.epoch_change.advance_recording`
    :rtype: str | None
    :raises SchemaTenantUnknownError: when the table is declared per schema, the source records changes itself,
        and the write names no schema: never put on ``_all``, where none of its pods read it
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
    """
    tenancy = tenancy if tenancy is not None else tenancy_of_table(table_name)
    owner = schema_tenant(schema) if tenancy.kind == "schema" and schema else None
    named = rows_with_tenants(tenancy, rows) if rows else rows
    if tenancy.kind == "schema" and owner is None and records_changes(source):
        unnamed = rows is None or any(row.tenant is None for row in rows)
        if unnamed:
            raise SchemaTenantUnknownError(
                f"{table_name!r} is declared per schema and its write names no schema: its record would go on "
                f"_all, where none of its pods read it. Name the writer's schema "
                f"(CollectionRegistry.set_l3_schema, or the broker's schema=)"
            )
    file = delta_file if delta_file is not None else named_delta_files().get(table_name)
    unconfirmed = (
        unconfirmed_delta_file
        if unconfirmed_delta_file is not None
        else named_unconfirmed_delta_files().get(table_name)
    )
    # a one-statement stamped write is its own commit, so its record names its own stamp (taken_statement_stamp); a
    # transaction's commit names its lowest (naming_commit_s_min). Taken either way, so the next advance starts afresh
    statement = taken_statement_stamp(table_name)
    committed = named_commit_s_min()
    lowest = s_min if s_min is not None else committed if committed is not None else statement
    token = await advance_recording(
        source,
        table_name,
        named,
        keys_unknown=keys_unknown,
        tenant=owner,
        partition=partition if rows is None else None,
        noted_between=noted_between if keys_unknown else None,
        prefixes=prefixes if rows is None else None,
        reason=reason if rows is None else None,
        delta_file=file,
        # only a keys-unknown record of a commit with a delta file: its keys are in that file alone (decision 31)
        scopes=(scopes if scopes is not None else named_commit_scopes()) if keys_unknown and file else None,
        unconfirmed_delta_file=unconfirmed,
        unconfirmed_writer=unconfirmed_writer if unconfirmed is not None else None,
        repair=repair,
        s_min=lowest,
    )
    if isinstance(token, str) and token:
        # a commit settling in this context collects it: a writer folds its delta files by the generation it wrote
        # (CallerTransaction.committed_generations)
        note_advanced_generation(table_name, token)
    return token


@dataclass(frozen=True, slots=True)
class NotedTable:
    """a table whose writes leave a change note in L3, and the columns its note names.

    :ivar table: the table
    :ivar keys: its primary-key columns, in declared order
    :ivar columns: its declared invalidation columns, which a note carries as a row broadcast does
    :ivar tenancy: which tenant its rows belong to (:mod:`threetears.core.tenancy`), whose token each note
        carries
    """

    table: str
    keys: tuple[str, ...]
    columns: tuple[str, ...] = ()
    tenancy: Tenancy = TENANCY_UNDECLARED


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
    (:func:`tables_with_write_generation`) or commits L2-first (:func:`written_l2_first`): each of
    their L3 writes leaves a note in its own transaction
    (:mod:`~threetears.core.collections.change_notes`). A table two classes name with different keys, or
    whose class names it per instance, is left out: no one trigger can name its rows. Two classes naming
    one table with one key carry every invalidation column either declares.

    :return: the tables, in table order
    :rtype: tuple[NotedTable, ...]
    """
    keys_of: dict[str, set[tuple[str, ...]]] = {}
    columns_of: dict[str, dict[str, None]] = {}
    tenancy_of: dict[str, Tenancy] = {}
    for cls in [*list(_GENERATION_CLASSES), *list(_L2_FIRST_CLASSES)]:
        name = table_named_by_class(cls)
        keys = _declared_key(cls)
        if name is None or keys is None:
            continue
        keys_of.setdefault(name, set()).add(keys)
        tenancy = getattr(cls, "tenancy", None)
        if isinstance(tenancy, Tenancy) and tenancy.kind != "undeclared":
            tenancy_of.setdefault(name, tenancy)
        declared = getattr(cls, "invalidation_columns", ())
        for column in declared if isinstance(declared, tuple) else ():
            if isinstance(column, str):
                columns_of.setdefault(name, {})[column] = None
    return tuple(
        NotedTable(
            table=name,
            keys=next(iter(keys)),
            columns=tuple(columns_of.get(name, {})),
            tenancy=tenancy_of.get(name, TENANCY_UNDECLARED),
        )
        for name, keys in sorted(keys_of.items())
        if len(keys) == 1
    )


def switched_tables_not_noted() -> dict[str, str]:
    """every table with a write generation that :func:`noted_tables` leaves out, with why: the tables whose missed
    writes the change notes can never name.

    Each switched-on table's lost advance is announced from its notes (:mod:`threetears.core.collections.lost_writes`);
    one with no note trigger leaves nothing to announce from, and the migration runner refuses to create it switched on
    (``switched_tables_without_notes``). The enforcement census (``threetears.enforcement.collection_census``) lists these
    so none ships silently.

    :return: ``table -> why``, in table order: classes naming it with different keys, or no class naming a key readable
        off the class
    :rtype: dict[str, str]
    """
    noted = {entry.table for entry in noted_tables()}
    keys_of: dict[str, dict[str, tuple[str, ...] | None]] = {}
    for cls in [*list(_GENERATION_CLASSES), *list(_L2_FIRST_CLASSES)]:
        name = table_named_by_class(cls)
        if name is not None:
            keys_of.setdefault(name, {})[f"{cls.__module__}.{cls.__qualname__}"] = _declared_key(cls)
    found: dict[str, str] = {}
    for table in sorted(tables_with_write_generation() - noted):
        declared = keys_of.get(table, {})
        keys = {key for key in declared.values() if key is not None}
        if len(keys) > 1:
            found[table] = "its classes declare different keys: " + ", ".join(
                f"{name} {list(key) if key is not None else None}" for name, key in sorted(declared.items())
            )
        else:
            found[table] = "no class declares a primary key readable off the class: " + ", ".join(sorted(declared))
    return found
