"""the row epoch: which write left each row of a table, and in what order, carried by the row itself.

Append-only L3 (``design-append-only-l3.md``, agreed 2026-11-25) keeps every version of a key a write leaves, and the
newest wins. "Newest" must be an order every write agrees on, known INSIDE the transaction that writes the row (the
row carries it), and never a clock. The table's write generation cannot be it: the broker advances it only after the
COMMIT (the hub's ``proxy.py`` ``_GenerationReplyFields.advance``), so no row can carry it. The write's version can:
:meth:`~threetears.core.collections.scope_epochs.ScopeEpochs.begin` takes it, fenced, before the first data row, and
every data transaction of the write takes its turn at it (``touch``) before it writes. But one version spans many page
transactions, and two of them may write the same key (a page written again as upserts after a failed copy, an LSF
state rewritten), so the version alone does not order them. The epoch is the version plus a sequence drawn for each
write statement, later draws sorting newer.

**The encoding: one BIGINT, ``version << 31 | sequence``.**

- One column, so the newest-first order is one ``epoch DESC`` term at the end of the primary key, one comparison
  per row, and "every version newer than X" (a restore) is one range on one column: ``epoch > compose(X, max)``.
  Two columns would double the key's tail and every comparison, and a range over them is a row-value comparison an
  index serves less directly.
- 32 bits of version: a version is one write (one upload, one LSF pull), a few thousand a night; 2**32 is far past
  any table's life, and a version above it is refused (:func:`_compose_epoch`), never wrapped.
- 31 bits of sequence, drawn from one counter for the whole process (:func:`next_epoch`): every write statement that
  stamps takes the next number, so a later statement of the same version always sorts newer, whichever transaction it
  is in. 2**31 draws a process lifetime is about a million write statements a night for five years; past it the draw
  is refused, never wrapped.
- WHY one counter for the process, not a database sequence: a version is one writer's (the write's lease, fenced in
  ``ScopeEpochs``), and its writes all run in that writer's process, so a process counter orders every write of a
  version. A database sequence would not: YugabyteDB caches sequence values per connection (``ysql_sequence_cache_minval``,
  at least 100), so two connections' ``nextval`` are not in the order they were taken. Two page transactions of one
  version writing the same key at the same time race either way; the one that drew later wins, which is the writer's
  own order.

**Two kinds of table** (:class:`RowEpoch`):

- **Versions** (:func:`row_versions`), the tables written in pages: every write keeps its own version of the key. The
  primary key is the key and then the epoch, newest first, so a read of a key stops at its first entry. On YugabyteDB
  the first key column stays hashed and ``epoch DESC`` is the last range column (:func:`versioned_primary_key_sql`), so
  every version of a key sits together in one tablet, newest first. On Postgres a primary key takes no sort order; its
  B-tree is read backwards for ``ORDER BY epoch DESC`` and stops at the first entry just the same. The declaration names
  the scope a restore reads "what changed since epoch X" by; its ``(scope, epoch)`` index (the scope hashed on
  YugabyteDB, so the rising end of the epochs spreads across scopes) is declared with that read (append-only chunk 5):
  the no-whole-reads rule refuses an index no keyed read names.
- **Stamped** (:data:`ROW_STAMPED`), the small tables updated in place (``lsf_runs``, ``state_fingerprints``): one row a
  key, carrying its last write's epoch.

**Append-only (versions).** A versioned table is never upserted and never deleted from: a write appends the key's new
version (an ``INSERT`` with no ``ON CONFLICT``, its own epoch, so it never meets an older version's), and a delete
appends a "gone" marker, the key at the delete's epoch with :data:`GONE_COLUMN` true. The marker carries the key, the
scope and the epoch alone: the table's other required columns are required of a live version only (a ``CHECK`` of the
stored table, ``gone OR (c IS NOT NULL AND ...)``), so the database still refuses a live version missing one. A whole
run deleted (a key prefix) is one row of the run-marker table (:mod:`~threetears.core.ddl.run_markers`), never a
marker a row.

**A write that carries no epoch is refused**: by 3tears before it is sent (:class:`UnstampedWriteError`), and by the
database for any write that bypasses 3tears (the column is ``NOT NULL`` with no default).

A leaf module: the collections and backends both import it, and it imports neither.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Final

__all__ = [
    "AppendOnlyWriteError",
    "EPOCH_COLUMN",
    "GONE_COLUMN",
    "ROW_STAMPED",
    "RowEpoch",
    "UnstampedWriteError",
    "next_epoch",
    "row_versions",
    "versioned_primary_key_sql",
]

#: the column every row of a table that declares a :class:`RowEpoch` carries its epoch in
EPOCH_COLUMN: Final[str] = "epoch"

#: the column a versioned table's "gone" marker sets: the key's version at that epoch says the key is gone (a delete,
#: appended rather than run). ``BOOLEAN NOT NULL DEFAULT false``: a write that leaves it out writes a live version
GONE_COLUMN: Final[str] = "gone"

#: the low bits of an epoch, the write statement's sequence
_SEQUENCE_BITS: Final[int] = 31

#: the largest sequence an epoch holds
_MAX_SEQUENCE: Final[int] = (1 << _SEQUENCE_BITS) - 1

#: the largest version an epoch holds: 32 bits, so the epoch stays a positive BIGINT
_MAX_VERSION: Final[int] = (1 << 32) - 1


class AppendOnlyWriteError(ValueError):
    """a ``DELETE`` (or an upsert) of a versioned table: refused before anything is sent. A versioned table is
    append-only: a write appends the key's new version, a delete appends a "gone" marker, a whole run deleted appends
    one run marker (:mod:`~threetears.core.ddl.run_markers`)."""


class UnstampedWriteError(ValueError):
    """a write to a table that declares a row epoch, without one: refused before anything is sent."""


@dataclass(frozen=True, slots=True)
class RowEpoch:
    """a table's row-epoch declaration (``TableSchema(row_epoch=...)``): :func:`row_versions` or :data:`ROW_STAMPED`.

    :ivar versions: whether every write keeps its own version of the key (the stored key the key and then the epoch);
        False for a table of one row a key, updated in place and stamped
    :ivar scope: for versions, the column a restore reads "changed since epoch X" by, a scope at a time (the state a row
        is of, as the writer's seqlock names scopes); its ``(scope, epoch)`` index lands with that read. None for stamped
    """

    versions: bool
    scope: str | None = None

    def __post_init__(self) -> None:
        """refuse a versioned table without a scope, or a stamped one with one.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``scope`` does not fit ``versions``
        """
        if self.versions and not self.scope:
            raise ValueError("a versioned table names the scope column a restore reads it by")
        if not self.versions and self.scope is not None:
            raise ValueError("a stamped table keeps one row a key and is restored by its key; give no scope")


def row_versions(scope: str) -> RowEpoch:
    """a versioned table's declaration: every write keeps its own version of the key, newest first.

    :param scope: the column a restore reads the table by, a scope at a time
    :ptype scope: str
    :return: the declaration
    :rtype: RowEpoch
    """
    return RowEpoch(versions=True, scope=scope)


#: a stamped table's declaration: one row a key, carrying its last write's epoch
ROW_STAMPED: Final = RowEpoch(versions=False)


def _compose_epoch(version: int, sequence: int) -> int:
    """the epoch of a write statement: its write's version and its sequence.

    :param version: the write's version (``ScopeEpochs.begin``), from 0 to :data:`_MAX_VERSION`
    :ptype version: int
    :param sequence: the statement's sequence, from 0 to :data:`_MAX_SEQUENCE`
    :ptype sequence: int
    :return: ``version << 31 | sequence``
    :rtype: int
    :raises ValueError: when either is out of its range: refused, never wrapped, since a wrapped epoch sorts older
    """
    if not 0 <= version <= _MAX_VERSION:
        raise ValueError(f"a write's version must be within 0..{_MAX_VERSION} to be an epoch, got {version}")
    if not 0 <= sequence <= _MAX_SEQUENCE:
        raise ValueError(f"an epoch's sequence must be within 0..{_MAX_SEQUENCE}, got {sequence}")
    return (version << _SEQUENCE_BITS) | sequence


class _Sequence:
    """the process's one sequence of write statements: each draw the next number, from one thread or many.

    The one piece of state here: an int behind a lock, drawn from and never read whole (see the module's "WHY one
    counter for the process").
    """

    __slots__ = ("_last", "_lock")

    def __init__(self) -> None:
        """start the sequence before its first draw."""
        self._lock = threading.Lock()
        self._last = 0

    def draw(self) -> int:
        """the next number.

        :return: it, from 1
        :rtype: int
        :raises ValueError: when the sequence is spent (:data:`_MAX_SEQUENCE` draws this process)
        """
        with self._lock:
            if self._last >= _MAX_SEQUENCE:
                raise ValueError(
                    f"this process has stamped {_MAX_SEQUENCE} write statements; a later one would sort older. "
                    "Restart the process"
                )
            self._last += 1
            return self._last


_SEQUENCE: Final = _Sequence()


def next_epoch(version: int) -> int:
    """a fresh epoch for one write statement of ``version``: newer than every epoch this process drew before.

    :param version: the write's version
    :ptype version: int
    :return: the epoch
    :rtype: int
    :raises ValueError: when the version is out of range, or the process's sequence is spent
    """
    return _compose_epoch(version, _SEQUENCE.draw())


def versioned_primary_key_sql(quoted_key: tuple[str, ...], quoted_epoch: str, *, yugabyte: bool) -> str:
    """a versioned table's primary key clause: the key and then the epoch, newest first.

    YugabyteDB: ``PRIMARY KEY ((k1) HASH, k2 ASC, ..., epoch DESC)``, the first key column hashed as it is today (so
    the table spreads across tablets by key as before) and ``epoch DESC`` the last range column, every version of a key
    in one tablet, newest first. Postgres: ``PRIMARY KEY (k1, ..., epoch)``; a primary key constraint takes no sort
    order there, and its B-tree read backwards answers ``ORDER BY epoch DESC`` stopping at the first entry.

    :param quoted_key: the key's columns, quoted as the DDL quotes them, in order
    :ptype quoted_key: tuple[str, ...]
    :param quoted_epoch: the epoch column, quoted
    :ptype quoted_epoch: str
    :param yugabyte: whether the database is YugabyteDB
    :ptype yugabyte: bool
    :return: the clause
    :rtype: str
    :raises ValueError: when the key is empty
    """
    if not quoted_key:
        raise ValueError("a versioned table's primary key leads with its key; none given")
    if not yugabyte:
        return f"PRIMARY KEY ({', '.join((*quoted_key, quoted_epoch))})"
    first, *rest = quoted_key
    terms = [f"({first}) HASH", *(f"{column} ASC" for column in rest), f"{quoted_epoch} DESC"]
    return f"PRIMARY KEY ({', '.join(terms)})"
