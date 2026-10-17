"""commits a collection's process did not make through it drop the rows they changed from its caches.

:func:`bind_collections` connects a repository's changed rows (``GitRepo.on_rows_changed``) to
:meth:`~threetears.core.collections.BaseCollection.invalidate_cache` on the collection serving each
table, which evicts the key from L1 and L2 and broadcasts the eviction. Every commit is covered: a
hand commit the writer noticed, commits fetched from origin by a clone, and this store's own --
including a delete made inside a change set, which no collection write could join. A key a
:class:`~threetears.core.collections.CallerTransaction` already settled is evicted once more, which
costs one more L2 delete and broadcast and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from threetears.core.collections import BaseCollection
from threetears.git_store.repo import RowChange, RowsChanged

__all__ = ["RowChangeSource", "bind_collections"]


class RowChangeSource(Protocol):
    """anything that reports changed rows: a :class:`~threetears.git_store.repo.GitRepo`, or
    :class:`~threetears.git_store.repos.GitRepos` for every repository at once."""

    def on_rows_changed(self, callback: RowsChanged) -> Any:
        """call ``callback`` with the rows each commit changes.

        :param callback: the callback
        :ptype callback: RowsChanged
        :return: whatever the source returns (a remover, or nothing)
        :rtype: Any
        """
        ...


def bind_collections(source: RowChangeSource, collections: Mapping[str, BaseCollection[Any]]) -> None:
    """drop every row a commit changes from the cache of the collection serving its table.

    :param source: the repository, or the set of repositories
    :ptype source: RowChangeSource
    :param collections: the collection serving each table, by table name as the layout names it
    :ptype collections: Mapping[str, BaseCollection[Any]]
    :return: None
    :rtype: None
    """
    served = dict(collections)

    async def evict(rows: list[RowChange]) -> None:
        by_table: dict[str, list[Any]] = {}
        for row in rows:
            collection = served.get(row.table)
            if collection is None:
                continue
            values = tuple(row.pk[column] for column in collection.primary_key_columns)
            by_table.setdefault(row.table, []).append(values[0] if len(values) == 1 else values)
        for table, keys in by_table.items():
            await served[table].invalidate_cache_many(keys)

    source.on_rows_changed(evict)
