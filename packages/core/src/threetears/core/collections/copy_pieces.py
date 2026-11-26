"""how much room a copy's change or reload reserves for its result's pieces
(:class:`~threetears.core.collections.shared_copy.SharedCopyBackend`, ``SharedArrowStore.reserve_pieces``): sized to
the largest file its job may write, within a share of the store. Arrow ``nbytes`` alone, no row read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from threetears.core.frozen_state import EMPTY

__all__ = ["APPLY_PIECE_BYTES", "APPLY_PIECE_MINIMUM_BYTES", "apply_piece_bytes", "reload_piece_bytes"]


class _PagedRows(Protocol):
    """a page of a change or a reload as its piece is sized: its table and its rows (a ``shared_copy.ChangePage``)."""

    @property
    def table(self) -> str:
        """the table."""
        ...

    @property
    def rows(self) -> Any:
        """the rows, a ``pyarrow.Table``."""
        ...


#: the most one file of a change's, a merge's or a reload's result may take: the most a piece grows to for a file of
#: its own (``SharedArrowStore.reserve_pieces``' ``grow_to``), and the most one is reserved at. Moved from reports
#: (``shared_backend.SHARED_COPY_RESULT_BYTES``): a fixed 1 GiB slot refused every change of a store with less room.
#: A piece is reserved at what its job is expected to write as one file (:func:`apply_piece_bytes`,
#: :func:`reload_piece_bytes`), never at this: it grows here only for a file that needs it
APPLY_PIECE_BYTES: Final = 1 << 30

#: the least room a change's piece is reserved at: one part's bound, a piece holding each file whole. Moved from
#: reports (``shared_backend.SHARED_COPY_PIECE_MINIMUM_BYTES``): the largest file a change wrote in its 2026-11-03
#: replay's cold start was a few MiB (a 123 MB table in 53 parts)
APPLY_PIECE_MINIMUM_BYTES: Final = 16 << 20

#: a reload's result piece is reserved at its largest table's bytes and this share more (an IPC file's padding and
#: metadata, its locator and packs index), at :data:`APPLY_PIECE_MINIMUM_BYTES` at least
_RELOAD_PIECE_SLACK_DIVISOR: Final = 8

#: a change's result piece is reserved at the largest file it may rewrite and this share more (the IPC file's
#: padding and metadata, a scope's rows cut again), at :data:`APPLY_PIECE_MINIMUM_BYTES` more at least
#: (:func:`apply_piece_bytes`)
_APPLY_PIECE_SLACK_DIVISOR: Final = 8

#: a change's result piece is reserved at most at this share of the store's ``max_bytes`` (a file that needs more
#: grows its piece, up to :data:`APPLY_PIECE_BYTES`, while the store has the room).
#:
#: **Why** (``test_copy_lane``'s second-death test, refused "no room for a hand-off in a 1 GiB store", 2026-11-25).
#: An apply reserved a piece of :data:`APPLY_PIECE_BYTES` (1 GiB), or all the room a smaller store had, for as long
#: as it applied, whatever the change: a change is a few MiB on a busy night (``copy_reload._MAX_REPLAY_BYTES``) and
#: rewrites only the parts it lands in (the largest file the 2026-11-03 replay's cold start wrote was a few MiB, its
#: 123 MB table in 53 parts), so the GiB held was almost all room nobody wrote, and a reload staging or a hand-off
#: beside it found none. Now a piece is reserved at the largest file of the change's tables the version holds (a part,
#: its locator, its packs index) plus the change's own rows of that table and an eighth: the most one rewritten file
#: can hold, unless a hook derives a table or a whole rewrite cuts parts afresh, and then the file grows its piece. It
#: is never past a quarter of the store, beside a reload's quarter (:data:`_RELOAD_STORE_SHARE`): half the store
#: (4 GiB of reports' 8 GiB, ``values.yaml``' ``maxBytes``) stays for versions and hand-offs
_APPLY_STORE_SHARE: Final = 4

#: a reload's result piece takes at most this share of the store's ``max_bytes``.
#:
#: **Why** (inventory ``apply_staged`` / ``reserve_pieces``, 2026-11-25). A reload stages in its own worker for as long
#: as its rows take, holding its open piece's room all the while; at :data:`APPLY_PIECE_BYTES` (1 GiB) every reload
#: took a GiB, or all the room a smaller store had, and a change's apply beside it found less than its 16 MiB minimum
#: and was refused. A reload's largest file is one part of one table, never more than that table's rows, so its piece
#: is sized to its largest table (most reloads are a state's: a few MiB; the largest table the 2026-11-03 replay's
#: cold start wrote was 123 MB, in 53 parts), and never past a quarter of the store, leaving the applies three
#: quarters (2 GiB of reports' 8 GiB store, ``values.yaml``' ``maxBytes``)
_RELOAD_STORE_SHARE: Final = 4


def apply_piece_bytes(max_bytes: int, pages: Sequence[_PagedRows], held: Mapping[str, int] = EMPTY) -> int:
    """the bytes a change's result piece is reserved at: over the tables it changes, the largest of a table's largest
    file the version holds and the change's rows of that table, and an eighth more, at least
    ``APPLY_PIECE_MINIMUM_BYTES`` (16 MiB) more, at most ``APPLY_PIECE_BYTES`` (1 GiB) and a quarter of the store
    (:data:`_APPLY_STORE_SHARE`'s why). A file that needs more grows its piece. Arrow ``nbytes``, no row read.

    :param max_bytes: the store's bound
    :ptype max_bytes: int
    :param pages: the change's pages
    :ptype pages: Sequence[_PagedRows]
    :param held: each changed table's largest file in the version the change applies to; empty for none
    :ptype held: Mapping[str, int]
    :return: bytes
    :rtype: int
    """
    by_table: dict[str, int] = {}
    for page in pages:
        by_table[page.table] = by_table.get(page.table, 0) + page.rows.nbytes
    largest = max((held.get(table, 0) + rows for table, rows in by_table.items()), default=0)
    wanted = largest + largest // _APPLY_PIECE_SLACK_DIVISOR + APPLY_PIECE_MINIMUM_BYTES
    bound = min(APPLY_PIECE_BYTES, max(APPLY_PIECE_MINIMUM_BYTES, max_bytes // _APPLY_STORE_SHARE))
    return max(APPLY_PIECE_MINIMUM_BYTES, min(wanted, bound))


def reload_piece_bytes(max_bytes: int, pages: Sequence[_PagedRows]) -> int:
    """the most one piece of a reload's result may take: its largest table's bytes and an eighth more, at least
    ``APPLY_PIECE_MINIMUM_BYTES`` (16 MiB), at most ``APPLY_PIECE_BYTES`` (1 GiB) and a quarter of the store
    (:data:`_RELOAD_STORE_SHARE`'s why). Arrow ``nbytes``, no row read.

    :param max_bytes: the store's bound
    :ptype max_bytes: int
    :param pages: the reload's pages
    :ptype pages: Sequence[_PagedRows]
    :return: bytes
    :rtype: int
    """
    by_table: dict[str, int] = {}
    for page in pages:
        by_table[page.table] = by_table.get(page.table, 0) + page.rows.nbytes
    largest = max(by_table.values(), default=0)
    wanted = largest + largest // _RELOAD_PIECE_SLACK_DIVISOR + APPLY_PIECE_MINIMUM_BYTES
    bound = min(APPLY_PIECE_BYTES, max(APPLY_PIECE_MINIMUM_BYTES, max_bytes // _RELOAD_STORE_SHARE))
    return max(APPLY_PIECE_MINIMUM_BYTES, min(wanted, bound))
