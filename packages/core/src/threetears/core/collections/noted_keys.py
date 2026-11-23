"""the plain types of a paged read of a commit's noted keys: its cursor, a page, and the reader that answers one.

They live apart from :mod:`threetears.core.collections.change_notes`, which defines the read
(:func:`~threetears.core.collections.change_notes.noted_keys_page`) and re-exports them, so the registry can hold a
reader without putting the keeper's database client and metrics on the L1 import path.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "NotedKeysCursor",
    "NotedKeysPage",
    "NotedKeysReader",
    "noted_key_pages",
]

#: the most notes one page of a keys-unknown epoch's keys reads
_NOTED_KEYS_PAGE_ROWS: Final = 10_000

#: the most bytes of keys one such page holds; a reader over the bus holds it under the bus's announced payload too
_NOTED_KEYS_PAGE_BYTES: Final = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NotedKeysCursor:
    """where a paged read of a bracket's noted keys goes on: a ring note table, and the last time read in it.

    :ivar slot: the ring note table (``0`` to ``NOTE_RING_SIZE - 1``) the next page reads
    :ivar after: the last ``noted_at`` read in it; ``None`` before its first page
    """

    slot: int
    after: datetime | None = None


@dataclass(frozen=True, slots=True)
class NotedKeysPage:
    """one page of the keys a table's notes name in a bracket.

    :ivar keys: each key's values as text, perhaps with repeats across pages (a set's worth of reads, not a count)
    :ivar columns: each key's declared invalidation columns as its note holds them (``columns[i]`` is ``keys[i]``'s),
        ``None`` for a note that holds none: a cache keyed by them is told the row as a row broadcast would tell it
    :ivar whole_table: a whole-table note of the tenant is in the bracket (a truncate, a load): the keys are not the reach
    :ivar next: where the next page reads from; ``None`` when this was the last
    """

    keys: tuple[tuple[str, ...], ...]
    whole_table: bool
    next: NotedKeysCursor | None
    columns: tuple[dict[str, str | None] | None, ...] = ()


@runtime_checkable
class NotedKeysReader(Protocol):
    """reads a page of the keys a table's notes of this reader's own tenant name in a commit's bracket.

    The pod's :class:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend`: the broker's ``l3.noted_keys``
    door, which names the tenant from the caller's verified schema
    (:func:`~threetears.core.collections.change_notes.noted_keys_page`). A registry that has one resolves a
    keys-unknown epoch record carrying
    :attr:`~threetears.core.collections.epoch_change.EpochChange.noted_between`
    (:meth:`~threetears.core.collections.registry.CollectionRegistry.catch_up_generation`).
    """

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        """a page of the keys, or ``"notes_unavailable"`` / ``"not_covered"``
        (as :func:`~threetears.core.collections.change_notes.noted_keys_page`).

        :param table: the table, as its notes name it
        :ptype table: str
        :param noted_between: the commit's bracket
        :ptype noted_between: tuple[datetime, datetime]
        :param cursor: where the last page ended; ``None`` for the first
        :ptype cursor: NotedKeysCursor | None
        :param limit: the most notes a page reads
        :ptype limit: int
        :param max_bytes: the most bytes of keys a page holds
        :ptype max_bytes: int
        :return: the page, or why the notes cannot say
        :rtype: NotedKeysPage | str
        """
        ...


async def noted_key_pages(
    reader: NotedKeysReader, table: str, noted_between: tuple[datetime, datetime]
) -> AsyncGenerator[NotedKeysPage | str]:
    """a keys-unknown commit's noted keys, a page at a time, each read only once the one before is consumed: the one
    walk of a bracket's notes, for a follower applying the keys (the registry) and for the writer recovering a lost
    delta file's rows (:meth:`~threetears.core.collections.delta_fold.DeltaFolds.recover_table`, decision 31).

    :param reader: what reads the change notes
    :ptype reader: NotedKeysReader
    :param table: the table
    :ptype table: str
    :param noted_between: the commit's notes bracket
    :ptype noted_between: tuple[datetime, datetime]
    :return: each page; or, last, the reader's word for why the notes cannot name the keys (``"not_covered"``: the
        ring no longer holds the bracket; ``"notes_unavailable"``)
    :rtype: AsyncGenerator[NotedKeysPage | str]
    """
    cursor: NotedKeysCursor | None = None
    while True:
        page = await reader.noted_keys(
            table,
            noted_between=noted_between,
            cursor=cursor,
            limit=_NOTED_KEYS_PAGE_ROWS,
            max_bytes=_NOTED_KEYS_PAGE_BYTES,
        )
        if not isinstance(page, NotedKeysPage):
            yield page
            return
        yield page
        if page.next is None:
            return
        cursor = page.next
