"""the write generation a collection's negative cache is stamped with.

A collection that caches "this key exists in no tier" has to know when that answer stops being
true. It cannot learn that from the key: a writer in another pod, or another principal, commits
the row somewhere the reader's cache never sees, and every per-key signal can race the marker it
should invalidate. What it can learn instead is whether ANY write to the table committed since the
answer was cached. A reader stamps each absence with the table's generation, read before the L3
lookup that found nothing; a writer advances the generation after its L3 commit. An absence whose
stamp is not the current generation is no longer trusted -- whatever the timing of the writes,
broadcasts or clocks involved.

The protocol lives here, in core, because :class:`~threetears.core.collections.base.BaseCollection`
consumes it and core may not import the packages that implement it. ``threetears.epoch`` provides
the platform implementation over its epoch counters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["GenerationSource"]


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

    async def advance(self, table_name: str) -> None:
        """move the table's generation on, after a write to it has committed.

        :param table_name: the collection's table
        :ptype table_name: str
        :return: None
        :rtype: None
        :raises GenerationUnavailableError: when the generation cannot be advanced
        """
        ...
