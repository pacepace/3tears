"""the row epoch: which write left each row of a table, and in what order, carried by the row itself.

Append-only L3 (``design-append-only-l3.md``, agreed 2026-11-25) keeps every version of a key a write leaves, and the
newest wins. "Newest" must be an order every write agrees on, known INSIDE the transaction that writes the row (the
row carries it), with no counter every writer shares. The table's write generation cannot be it: the broker advances it
only after the COMMIT (the hub's ``proxy.py`` ``_GenerationReplyFields.advance``), so no row can carry it. The epoch is a
**hybrid logical clock (HLC) kept by the writing process** (Pace 2026-11-25; ``research-shared-clock.md``).

**The encoding: one BIGINT, ``wall_ms << 15 | logical``.**

- One column, so the newest-first order is one ``epoch DESC`` term at the end of the primary key, one comparison per
  row, and "every version since X" (a restore) is one range on one column.
- 48 bits of wall milliseconds (about 8,900 years from 1970) and 15 bits of logical counter: 32,768 stamps a
  millisecond a process before the wall part moves on.
- :func:`next_epoch` answers ``max(last + 1, wall_ms << 15)``: monotone inside the process whatever the wall clock
  does. When the wall clock goes back (an NTP step), the logical part advances from the last stamp; when the logical
  part overflows, ``last + 1`` carries into the wall part, one millisecond on. A stamp past a positive BIGINT is
  refused, never wrapped.
- WHY a process clock, not a shared counter or a database clock: a shared counter is one hot row every write of every
  table takes a turn at; a database sequence is not ordered across connections on YugabyteDB (it caches values per
  connection, ``ysql_sequence_cache_minval``); Postgres xids have no YugabyteDB counterpart, and YugabyteDB's hybrid
  time no Postgres one (``research-shared-clock.md``). The process clock is the same on both engines and costs one
  lock-guarded int.

**Newest-wins across writers** (:class:`~threetears.core.collections.scope_epochs.ScopeEpochs`). One write is one
writer's, in one process, so its own stamps are ordered by its own clock. Across writers, the scope epochs table keeps
``last_epoch``, the highest stamp a transaction committed, on the rows that transaction touched; the next writer's
``begin`` reads it and moves its own clock past it (:func:`observe_epoch`), so its versions always sort above the
previous writer's, whatever the two processes' clocks say.

**Stamped (in-place) tables** never let a key's stamp go back whichever process writes it: their upsert sets
``epoch = GREATEST(epoch + 1, <the new stamp>)`` (:func:`rising_epoch_assignment`).

**A commit older than W is refused** (:data:`MAX_TRANSACTION_AGE_MS`): the caller's transaction checks its lowest
stamp against the clock before its COMMIT, so no committed row is older than W at its commit. That bound, with the
commit-to-log gap A and the clock skew S, is the margin M a restore reads back by (:data:`_RESTORE_MARGIN_MS`): every
log record carries its commit's lowest stamp (``s_min``, ``EpochChange.s_min``), so the scrub can prove M held.

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
import time
from dataclasses import dataclass
from typing import Final

__all__ = [
    "AppendOnlyWriteError",
    "EPOCH_COLUMN",
    "GONE_COLUMN",
    "MAX_TRANSACTION_AGE_MS",
    "ROW_STAMPED",
    "RowEpoch",
    "TransactionTooOldError",
    "UnstampedWriteError",
    "epoch_wall_ms",
    "next_epoch",
    "observe_epoch",
    "rising_epoch_assignment",
    "row_versions",
    "versioned_primary_key_sql",
    "wall_clock_ms",
]

#: the column every row of a table that declares a :class:`RowEpoch` carries its epoch in
EPOCH_COLUMN: Final[str] = "epoch"

#: the column a versioned table's "gone" marker sets: the key's version at that epoch says the key is gone (a delete,
#: appended rather than run). ``BOOLEAN NOT NULL DEFAULT false``: a write that leaves it out writes a live version
GONE_COLUMN: Final[str] = "gone"

#: the low bits of an epoch, the logical counter within one wall millisecond
_LOGICAL_BITS: Final[int] = 15

#: the largest epoch: a positive BIGINT
_MAX_EPOCH: Final[int] = (1 << 63) - 1

#: W, the longest a transaction may run, from its first stamp to its COMMIT: a commit older is refused
#: (:class:`TransactionTooOldError`). WHY 5 minutes: half the 10-minute fence the restore margin starts at (M = W + A +
#: S, :data:`_RESTORE_MARGIN_MS`), far above any page transaction the broker lets idle (60 s,
#: ``tx_idle_timeout_seconds``); a bulk COPY must fit inside it (``research-shared-clock.md``, risk 5). Size it from the
#: burst replay's longest measured transaction before narrowing M
MAX_TRANSACTION_AGE_MS: Final[int] = 5 * 60 * 1000

#: S, the clock skew between two writing processes the margin covers. WHY 5 s: YugabyteDB stops a node past 500 ms
#: (``max_clock_skew_usec``) and EKS nodes keep AWS time sync; the research asks at least 1 s
_CLOCK_SKEW_MS: Final[int] = 5 * 1000

#: A, the longest gap between a commit and the append of its epoch record that the margin covers: the rest of the
#: 10-minute fence today's change-note ring rests on. A writer that stalls longer can be missed by a restore, as today;
#: the scrub reports it exactly from the record's ``s_min``
_COMMIT_TO_LOG_MS: Final[int] = 10 * 60 * 1000 - MAX_TRANSACTION_AGE_MS - _CLOCK_SKEW_MS

#: M = W + A + S, the margin a restore reads back by: given the keeper's mark (log position P at keeper time T), a
#: restore reads every version with ``epoch >= (T - M) << 15`` (append-only chunk 5). WHY it is complete: a record
#: appended after P was appended at T or later, so its commit was at T - A or later (its writer's clock T - A - S), and
#: its stamps at most W older than its commit, so every one is at T - M or later. WHY 10 minutes: the fence today's
#: ring rests on; reading too much only re-evicts keys, which is harmless, and missing one is the only failure
_RESTORE_MARGIN_MS: Final[int] = MAX_TRANSACTION_AGE_MS + _COMMIT_TO_LOG_MS + _CLOCK_SKEW_MS


class AppendOnlyWriteError(ValueError):
    """a ``DELETE`` (or an upsert) of a versioned table: refused before anything is sent. A versioned table is
    append-only: a write appends the key's new version, a delete appends a "gone" marker, a whole run deleted appends
    one run marker (:mod:`~threetears.core.ddl.run_markers`)."""


class UnstampedWriteError(ValueError):
    """a write to a table that declares a row epoch, without one: refused before anything is sent."""


class TransactionTooOldError(RuntimeError):
    """a transaction whose first stamp is older than :data:`MAX_TRANSACTION_AGE_MS` at its COMMIT: refused, rolled
    back, so no committed row is older than W at its commit and a restore's margin holds."""


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


def _wall_ms() -> int:
    """the wall clock, in milliseconds since 1970.

    :return: it
    :rtype: int
    """
    return time.time_ns() // 1_000_000


class _HybridClock:
    """the process's one hybrid logical clock: each stamp above every one before it, from one thread or many.

    The one piece of state here: the last stamp, an int behind a lock, replaced whole (see the module's "The
    encoding").
    """

    __slots__ = ("_last", "_lock")

    def __init__(self) -> None:
        """start the clock before its first stamp."""
        self._lock = threading.Lock()
        self._last = 0

    def now(self) -> int:
        """the next stamp: ``max(last + 1, wall_ms << 15)``.

        :return: it
        :rtype: int
        :raises ValueError: when the stamp would pass a positive BIGINT (refused, never wrapped)
        """
        floor = _wall_ms() << _LOGICAL_BITS
        with self._lock:
            # last + 1 past the logical part's top carries into the wall part: the logical part's overflow moves the
            # wall part on by one millisecond, never back to 0 of the same millisecond
            stamp = max(self._last + 1, floor)
            if stamp > _MAX_EPOCH:
                raise ValueError(f"an epoch past {_MAX_EPOCH} would wrap and sort older; the clock reads {floor}")
            self._last = stamp
        return stamp

    def observe(self, epoch: int) -> None:
        """move the clock past ``epoch``: the next stamp sorts above it.

        :param epoch: a stamp another writer committed
        :ptype epoch: int
        :return: nothing
        :rtype: None
        :raises ValueError: when ``epoch`` is no epoch (negative, or past a positive BIGINT)
        """
        if not 0 <= epoch <= _MAX_EPOCH:
            raise ValueError(f"an epoch is within 0..{_MAX_EPOCH}, got {epoch}")
        with self._lock:
            self._last = max(self._last, epoch)


_CLOCK: Final = _HybridClock()


def next_epoch() -> int:
    """a fresh epoch for one write statement: newer than every epoch this process stamped or observed before.

    :return: the epoch
    :rtype: int
    :raises ValueError: when the stamp would pass a positive BIGINT
    """
    return _CLOCK.now()


def observe_epoch(epoch: int) -> None:
    """move this process's clock past ``epoch``, a stamp another writer committed (the previous writer's
    ``last_epoch``, read by ``ScopeEpochs.begin``): every later :func:`next_epoch` sorts above it.

    :param epoch: the stamp
    :ptype epoch: int
    :return: nothing
    :rtype: None
    :raises ValueError: when ``epoch`` is no epoch
    """
    _CLOCK.observe(epoch)


def epoch_wall_ms(epoch: int) -> int:
    """the wall milliseconds an epoch was stamped at, by the encoding (``epoch >> 15``).

    :param epoch: the epoch
    :ptype epoch: int
    :return: its wall part
    :rtype: int
    """
    return epoch >> _LOGICAL_BITS


def wall_clock_ms() -> int:
    """the wall clock the process's stamps read, in milliseconds: what a transaction's age is measured against.

    :return: it
    :rtype: int
    """
    return _wall_ms()


def rising_epoch_assignment(target: str, quoted_epoch: str = EPOCH_COLUMN) -> str:
    """a stamped table's upsert ``SET`` term for the epoch: ``epoch = GREATEST(<target>.epoch + 1, EXCLUDED.epoch)``.

    A stamped table keeps one row a key, updated in place by whichever process writes it; each process's clock is its
    own, so the new stamp may be behind the row's. The row's stamp never goes back: it rises past the stored one by at
    least one, and takes the new stamp when that is higher.

    :param target: how the statement names the stored row (the table, or its alias)
    :ptype target: str
    :param quoted_epoch: the epoch column, quoted as the statement quotes it
    :ptype quoted_epoch: str
    :return: the term
    :rtype: str
    """
    return f"{quoted_epoch} = GREATEST({target}.{quoted_epoch} + 1, EXCLUDED.{quoted_epoch})"


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
