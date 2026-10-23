"""a host predicate every memory read carries in its ``WHERE``.

A host restricts what the memory collections return by handing them a
:class:`ReadFilter`. Each read asks the filter for a SQL boolean over the
memory row's ``tags`` JSONB array and ANDs it into the candidate scan, before
``ORDER BY ... LIMIT``, so a filtered read never under-fills and no arm of a
hybrid search lets an excluded row in.

The package knows nothing about why a row is excluded. The filter is the
host's: what its tags mean, and which rows the current task may read.

Three rules, each structural here rather than left to call sites:

1. **called on every read, never cached.** A host's filter may read per-task
   state (a contextvar), so the same collection answers different tasks
   differently. ``None`` from the filter means "no restriction for this read".
2. **a passage or a file answers to its memory.** A chunk read is filtered on
   its parent memory's tags; a media-content read on the tags of the memory
   its media row is paired with. A row whose memory cannot be found is
   excluded while a filter applies.
3. **writes and bookkeeping are not filtered.** Deleting, decaying or
   re-pointing a memory the current task cannot see must still reach it, or
   a host-side delete leaves rows behind.

The primary-key :meth:`get` the collections inherit is not filtered either: it
answers from the L1/L2 cache tiers, where a per-task SQL predicate cannot run.
A host that hands memory ids to a reader gates them through a filtered read
(``find_for_recall``, ``search_by_ids``), as the memory tools do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["ReadFilter", "ReadFiltered", "filter_clause"]


class ReadFilter(Protocol):
    """a host predicate over a memory row's tags, as SQL.

    :param tags: a SQL expression for the row's ``tags`` JSONB array, e.g. ``m.tags``; it may be NULL
    :ptype tags: str
    :param bind: appends a value to the query being built and returns its placeholder (``$n``)
    :ptype bind: Callable[[Any], str]
    :return: a SQL boolean fragment, or ``None`` for no restriction on this read
    :rtype: str | None
    """

    def __call__(self, tags: str, bind: Callable[[Any], str]) -> str | None:
        """the fragment for this read, or ``None``."""
        ...


def filter_clause(read_filter: ReadFilter | None, tags: str, args: list[Any]) -> str:
    """`` AND (<fragment>)`` for one read, binding the filter's values onto the end of ``args``.

    ``args`` must already hold every positional value the query binds, in order:
    a value the filter binds lands at ``$len(args)`` the moment it is appended,
    so the method's own placeholders keep their numbers and the filter's follow
    them. A filter that binds values and then answers ``None`` leaves ``args``
    as it found it.

    :param read_filter: the host's filter, or ``None``
    :ptype read_filter: ReadFilter | None
    :param tags: SQL expression for the filtered row's tags array
    :ptype tags: str
    :param args: the query's positional values so far; the filter's are appended
    :ptype args: list[Any]
    :return: the clause, or ``""`` when nothing restricts this read
    :rtype: str
    """
    if read_filter is None:
        return ""
    start = len(args)

    def bind(value: Any) -> str:
        args.append(value)
        return f"${len(args)}"

    fragment = read_filter(tags, bind)
    if fragment is None:
        del args[start:]
        return ""
    return f" AND ({fragment})"


class ReadFiltered:
    """the read-filter slot a memory collection carries; ``None`` reads as before.

    :ivar read_filter: the host's filter, consulted on every filtered read
    """

    read_filter: ReadFilter | None = None

    def set_read_filter(self, read_filter: ReadFilter | None) -> None:
        """install ``read_filter`` for every later read; ``None`` removes it.

        :param read_filter: the host's filter, or ``None``
        :ptype read_filter: ReadFilter | None
        :return: nothing
        :rtype: None
        """
        self.read_filter = read_filter
