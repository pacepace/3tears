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

    __slots__ = ("_collection", "_keys", "_transaction", "_unchanged")

    def __init__(self, collection: BaseCollection[Any], transaction: CallerTransaction | None) -> None:
        """start a write that has touched nothing yet.

        :param collection: the collection whose rows the write changes
        :ptype collection: BaseCollection[Any]
        :param transaction: the caller's transaction the write joins, or ``None`` on the pool
        :ptype transaction: CallerTransaction | None
        """
        self._collection = collection
        self._transaction = transaction
        self._keys: list[Any] = []
        self._unchanged = False

    @property
    def keys(self) -> tuple[Any, ...]:
        """every key named so far, in the order named.

        :return: the keys, each a pk value or a tuple of pk values in declared column order
        :rtype: tuple[Any, ...]
        """
        return tuple(self._keys)

    @property
    def is_unchanged(self) -> bool:
        """whether the body declared that its write changed no row.

        :return: ``True`` once :meth:`unchanged` was called
        :rtype: bool
        """
        return self._unchanged

    def touches(self, *entity_ids: Any) -> None:
        """name rows the write may have changed; each is evicted when the write, or its transaction, ends.

        Name a row before the statement that may change it when the key is known, so a statement
        that raises is covered; a key only the statement reveals (``RETURNING``) is named after it.

        :param entity_ids: pk values (single-pk) or tuples of pk values in declared column order
        :ptype entity_ids: Any
        :return: nothing
        :rtype: None
        """
        for entity_id in entity_ids:
            self._keys.append(entity_id)
            if self._transaction is not None:
                self._transaction.enroll(self._collection, entity_id)

    def unchanged(self) -> None:
        """declare that the write is known to have changed no row -- a compare-and-swap that lost.

        Honoured only when the body then ends without raising: a body that raises afterwards left
        the outcome unknown again, and its rows are evicted. Has no effect on a write joined to a
        caller's transaction, whose rows are settled when that ends in any case.

        :return: nothing
        :rtype: None
        """
        self._unchanged = True
