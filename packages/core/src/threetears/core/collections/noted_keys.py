"""the plain types of a paged read of a commit's noted keys: its cursor, a page, and the reader that answers one.

They live apart from :mod:`threetears.core.collections.change_notes`, which defines the read
(:func:`~threetears.core.collections.change_notes.noted_keys_page`) and re-exports them, so the registry can hold a
reader without putting the keeper's database client and metrics on the L1 import path.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

from threetears.core.collections.delta_files import text_key_batch

__all__ = [
    "NotedKeysCursor",
    "NotedKeysPage",
    "NotedKeysReader",
    "noted_key_pages",
    "noted_keys_cursor",
    "noted_keys_cursor_wire",
]

#: the most notes one page of a keys-unknown epoch's keys reads
_NOTED_KEYS_PAGE_ROWS: Final = 10_000

#: the most bytes of keys one such page holds; a reader over the bus holds it under the bus's announced payload too
_NOTED_KEYS_PAGE_BYTES: Final = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NotedKeysCursor:
    """where a paged read of a bracket's changed keys goes on (:func:`~threetears.core.collections.epoch_reads
    .epoch_keys_page`): the relation, the index range in it, and the keyset place in that range.

    :ivar slot: a stamped table's bucket (``0`` to ``RESTORE_BUCKETS - 1``), or ``RESTORE_BUCKETS`` past it for that
        bucket's moves (``tenant_moves``); ``0`` for a versioned table's scope
    :ivar schema: the schema of the relation the next page reads
    :ivar scope: a versioned table's scope value the next page reads; ``None`` for a stamped table
    :ivar after: the keyset cursor of the last row read in that range (``"past"`` once it is read to its end);
        ``None`` before its first page
    """

    slot: int = 0
    schema: str | None = None
    scope: str | None = None
    after: str | None = None


def noted_keys_cursor_wire(cursor: NotedKeysCursor) -> dict[str, Any]:
    """a cursor as the ``l3.noted_keys`` request and reply carry it: the one shape the pod and the broker share.

    :param cursor: the cursor
    :ptype cursor: NotedKeysCursor
    :return: ``{slot, schema, scope, after}``
    :rtype: dict[str, Any]
    """
    return {"slot": cursor.slot, "schema": cursor.schema, "scope": cursor.scope, "after": cursor.after}


def noted_keys_cursor(raw: Mapping[str, Any]) -> NotedKeysCursor:
    """a cursor from its wire form (:func:`noted_keys_cursor_wire`).

    :param raw: the wire form
    :ptype raw: Mapping[str, Any]
    :return: the cursor
    :rtype: NotedKeysCursor
    """
    schema, scope, after = raw.get("schema"), raw.get("scope"), raw.get("after")
    return NotedKeysCursor(
        slot=int(raw.get("slot") or 0),
        schema=None if schema is None else str(schema),
        scope=None if scope is None else str(scope),
        after=None if after is None else str(after),
    )


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

    def key_batch(self) -> Any:
        """the page's keys as one Arrow batch, one row a key, its columns named by position (``k0``, ``k1``, ...), each
        as text: what a cache that takes key batches (``on_keys``) is told, and what a reader of many keys works on in
        place of a Python tuple a key. Built here, at the reader, from the keys as the notes' wire and the ring's
        read hand them over (text, a sequence a key); a process without pyarrow, which has no such cache, never
        calls it.

        :return: the keys
        :rtype: pyarrow.Table
        :raises ImportError: without pyarrow
        """
        return text_key_batch(self.keys)


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
