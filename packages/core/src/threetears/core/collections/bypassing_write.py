"""the rows a cache-bypassing write touched, and who settles their caches when it ends.

A targeted UPDATE (or DELETE) that skips :meth:`BaseCollection.save_entity` still changes rows
that ``get`` serves from L1 and L2. Those tiers keep the row the statement replaced until it is
evicted, and a caller that reads the stale row and saves it back writes it over L3. So every such
write must evict every row it may have changed, from every tier, on every replica -- however the
statement ended, since one that raised may still have reached L3, and to completion, since an
eviction a cancellation stopped partway leaves exactly the rows it exists to remove.

:meth:`BaseCollection.bypassing_write` owns that rule; this is the handle its body gets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from threetears.core.collections.base import BaseCollection
    from threetears.core.collections.caller_transaction import CallerTransaction

__all__ = ["BypassingWrite"]


class BypassingWrite:
    """the rows one cache-bypassing write touched, collected for eviction when it ends.

    On the collection's own pool the keys are evicted once the write's body ends. Joined to a
    caller's connection they are enrolled in the enclosing
    :class:`~threetears.core.collections.caller_transaction.CallerTransaction` as they are named,
    and settled when it ends.
    """

    __slots__ = ("_collection", "_keys", "_rows", "_transaction", "_unchanged")

    def __init__(self, collection: BaseCollection[Any], transaction: CallerTransaction | None) -> None:
        """start a write that has touched nothing yet.

        :param collection: the collection whose rows the write changes
        :ptype collection: BaseCollection[Any]
        :param transaction: the caller's transaction the write joins, or ``None`` on the pool
        :ptype transaction: CallerTransaction | None
        """
        self._collection = collection
        self._transaction = transaction
        #: every key named so far, replaced whole on each :meth:`touches`
        self._keys: tuple[Any, ...] = ()
        #: the row each key's write saw, in the order of :attr:`keys` (None where none was named)
        self._rows: tuple[dict[str, Any] | None, ...] = ()
        self._unchanged = False

    @property
    def keys(self) -> tuple[Any, ...]:
        """every key named so far, in the order named.

        :return: the keys, each a pk value or a tuple of pk values in declared column order
        :rtype: tuple[Any, ...]
        """
        return self._keys

    @property
    def rows(self) -> tuple[dict[str, Any] | None, ...]:
        """the row each key's write saw, in the order of :attr:`keys`: what the eviction's broadcast takes the
        collection's declared invalidation columns from (a membership's member, a grant's group), None where none was
        named.

        :return: the rows
        :rtype: tuple[dict[str, Any] | None, ...]
        """
        return self._rows

    @property
    def is_unchanged(self) -> bool:
        """whether the body declared that its write changed no row.

        :return: ``True`` once :meth:`unchanged` was called
        :rtype: bool
        """
        return self._unchanged

    def touches(self, *entity_ids: Any, rows: Sequence[dict[str, Any] | None] | None = None) -> None:
        """name rows the write may have changed; each is evicted when the write, or its transaction, ends.

        Name a row before the statement that may change it when the key is known, so a statement
        that raises is covered; a key only the statement reveals (``RETURNING``) is named after it.

        :param entity_ids: pk values (single-pk) or tuples of pk values in declared column order
        :ptype entity_ids: Any
        :param rows: the row each key's write saw, in the order of ``entity_ids``, for the broadcast's invalidation
            columns (a cache derived from the table evicts what the row reaches); None when no row was seen
        :ptype rows: Sequence[dict[str, Any] | None] | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``rows`` does not name one row a key, before any key is named
        """
        seen: tuple[dict[str, Any] | None, ...] = (None,) * len(entity_ids) if rows is None else tuple(rows)
        if len(seen) != len(entity_ids):
            raise ValueError(f"touches: {len(entity_ids)} key(s) and {len(seen)} row(s); name one row a key")
        named = 0
        try:
            for entity_id, row in zip(entity_ids, seen, strict=True):
                named += 1
                if self._transaction is not None:
                    self._transaction.enroll(self._collection, entity_id, row=row)
        finally:
            # one new tuple per call, holding each key named before an enrolment that raised
            self._keys = (*self._keys, *entity_ids[:named])
            self._rows = (*self._rows, *seen[:named])

    def unchanged(self) -> None:
        """declare that the write is known to have changed no row -- a compare-and-swap that lost.

        Honoured only when the body then ends without raising: a body that raises afterwards left
        the outcome unknown again, and its rows are evicted. Has no effect on a write joined to a
        caller's transaction, whose rows are settled when that ends in any case.

        :return: nothing
        :rtype: None
        """
        self._unchanged = True
