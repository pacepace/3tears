"""an epoch is its change set: the rows one advance of a table's write generation covered.

A table's write generation (:mod:`threetears.core.collections.generation`) tells a pod THAT it
missed a write. The epoch's change record tells it WHICH rows: the writer that advances the
generation to ``{incarnation}:{N}`` stores, beside it, the rows that commit changed, addressable
by table and token. A pod that finds an advance it did not hear in full fetches that record and
evicts exactly those rows, through the same path a heard row broadcast takes. Catching up after a
missed broadcast, after a pod restart and after a partition is the same operation: fetch epochs
``mine+1 .. head`` and apply them.

A gap never drops a table. A record that cannot be read leaves the pod serving what it holds,
counted and logged as behind, until the record can be read. A record that names the whole table
(:attr:`EpochChange.whole_table`) is the change itself -- a truncate, a file load, a migration
whose rows nobody listed -- and invalidates the table because that is what the write did. Only a
write whose caller changed the whole table records one: a commit of many rows is recorded as
several consecutive keyed records (:func:`changed_row_runs`), each within the record bound, and its
epoch is the last of them.

The records live in L2 (a memory bucket, replicated); the writer that advances writes them, and
core holds only the format and the two protocols, as it does for the generation itself.

**A record names the database it was written in** (:attr:`EpochChange.database`; Pace 2026-11-16). One log is
shared by every database whose writers advance through it -- the hub's platform database and identity-core's -- and
a record names its table by its bare name, so two databases noting a table of one name write records no reader could
tell apart. They do today: 3tears' coordination tables (``coordination_revocations``) are migrated into each one's
schema and noted in each one's ring. Read by name alone, a hub truncate dropped identity's whole cache of it, a hub
key evicted identity's row of that key, a hub keys-unknown record was paged from identity's ring (behind, until a
restart, once its bracket was older than that ring's coverage), and each ring's scrub reported the other's every
write as a missing note. So a source built with its database (its ring's schema, which the keeper checks) names it
on every record, a reader built with it walks only its own database's records and those naming none, and each scrub
judges only its own.

Why a name on the record rather than refusing a second ring's claim on a name: the colliding table is 3tears' own,
declared once and migrated into both databases, so neither database can rename it; and the record's key cannot
change without every older reader losing it. Why it is rolled out safely: the field is new and optional. An older
reader ignores a field it does not know (no model here forbids one), so it reads a new record as before; a record
from an older writer names no database and every reader walks it as before. Once every writer of a shared log names
its database, a contract release may treat a record naming none as another database's.

**A record names its commit's delta file** (:attr:`EpochChange.delta_file`; ``design-pod-delta-files.md``, decision
4). A table that opts in to delta files (:mod:`~threetears.core.collections.delta_files`) has each commit's rows saved
as a file on the pods' shared volume before the commit; the commit's record names that file, so a pod fetches exactly
it and nothing lists the volume. Every record of the commit for the table names it (a commit too large for one record
is several), so a follower that reaches any of them has the whole commit. Rolled out as :attr:`database` was: the
field is optional, an older reader ignores it, and a record from an older writer names none.

**A keys-unknown record names the scopes its commit touched** (:attr:`EpochChange.scopes`; ``design-pod-delta-files
.md``, decision 31). A commit too big to list its rows has its keys only in its delta file (decision 21); should that
file be lost past the change notes' window, nothing else names them. So the record also names the scopes (the states,
``state:CA``, as the writer's seqlock names them: :class:`~threetears.core.collections.scope_epochs.ScopeEpochs`) the
commit's transaction touched, and the writer recovers by re-reading just those states from L3
(:meth:`~threetears.core.collections.delta_fold.DeltaFolds.recover_table`). Bounded (:data:`_EPOCH_SCOPES_MAX`, each
:data:`_SCOPE_MAX_CHARS`): a commit touching more names none, and its recovery then has only the notes. Rolled out as
:attr:`delta_file` was: optional, an older reader ignores it, a record from an older writer names none.

**A record's scopes are known complete only when it says so** (:data:`EVERY_STATE_NAMED`; decision 54 (c)). A commit's
scopes were what its transaction touched (``ScopeEpochs.touch``), which nothing ties to the rows it wrote: a write
that touched fewer states than it wrote would name a strict subset, and a copy reloading just those states from the
full would keep the old rows of the others. So a commit names the mark among its scopes only when its delta files'
own rows say which state each row and delete marker is of
(:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_state_scope`), and then names every one of those
states: complete by construction, from the data. A record without the mark (an older writer, a table that declares no
state scope, a load) is read as naming its states incompletely (:func:`complete_scopes` answers None), so a reader
that must have every state goes by the whole table. The mark is one more scope, so the broker carries it as it carries
the rest (whole, or none past the bound) and an older reader that filters scopes by a table's prefix never sees it.

**A load request that failed after committing some rows names its file unconfirmed** (stored as
``unconfirmed_delta_file``; read as :attr:`EpochChange.delta_file` with :attr:`EpochChange.delta_file_unconfirmed`
set; ruling C of 2026-11-23). A load commits a transaction per 50k rows, so a request that fails part way leaves a
leading part of its file's rows in L3, and its file (saved before the request, holding every row it sent) may hold
rows L3 never took. Its record names the file under a field of its own, never ``delta_file``: an older reader ignores
the unknown field and reads the record as a load naming no file (its prefixes read again, as before), so it never
applies the file. A newer reader never applies it as it stands either: a copy waits, as for a lost file (decision
29), and the writer reads the request's keys (from the file) from L3 by key, once, and writes the file again under its
name holding exactly what L3 holds, marked confirmed
(:attr:`~threetears.core.collections.delta_files.DeltaKeys.confirmed`); the copies then apply it. Named only on a
whole-table record of a load's key prefixes, and never beside ``delta_file``.

**A load names its file unconfirmed BEFORE its first commit, with its writer** (stored as ``unconfirmed_writer``,
read as :attr:`EpochChange.unconfirmed_writer`; Pace's ruling of 2026-11-23, the intent record). A hub that crashed
after a commit and before any record would leave rows no record names, and a pod could not tell a crashed hub from
one still copying. So the hub names the file unconfirmed before the load's first commit, beside the key of its own
liveness: its writer beat in the platform's ``leases`` bucket (:class:`~threetears.core.collections.lost_writes
.WriterBeats`). The writer confirms such a record only once a later record of the table names the same file
(the hub's own confirmed record on success, or its unconfirmed record naming no writer on a failure it answered:
the intent is then superseded and applied by that record alone), or once the named beat is gone: deleted by the
writer stopping cleanly, or by the platform's recovery once it declared the writer dead. Never by elapsed time. Named
only beside ``unconfirmed_delta_file``; a reader built before it ignores it (an older pod reads a load naming no
file, as before ruling C).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from threetears.core.backends.broker_generation import cleanup_delete_floor
from threetears.core.cache.base import stored_instant
from threetears.core.exceptions import GenerationNotCommittedError, GenerationUnavailableError
from threetears.core.utils.write_ledger import process_write_ledger

if TYPE_CHECKING:
    from threetears.core.collections.generation import GenerationSource

__all__ = [
    "EPOCH_CHANGE_MAX_AGE",
    "RESET_REASON",
    "WholeReach",
    "key_text",
    "ChangedRow",
    "EpochChange",
    "EpochChangeReader",
    "EpochLogExtent",
    "EpochQuiet",
    "EpochWalk",
    "epoch_change_body",
    "keyed_record_form",
    "read_epoch_change_body",
    "read_logged_epoch_change",
    "recordable_scopes",
    "EVERY_STATE_NAMED",
    "complete_scopes",
    "walk_by_count",
    "advance_recording",
    "records_changes",
    "changed_row_runs",
    "check_change_form",
    "check_delta_file",
    "check_whole_reach",
    "epoch_change_record",
    "read_epoch_change",
]

#: the one reason a whole-table record names: an operator's reset (``SchemaBackedCollection.reset``), which a follower
#: reads as that table's cold start. A whole-table record naming no reason and no key prefix is an older writer's
RESET_REASON: Final = "reset"


@dataclass(frozen=True, slots=True)
class WholeReach:
    """what a whole-table change reached, as its record names it: the key prefixes of a load or a clear, or an
    operator's reset (:data:`RESET_REASON`). A follower reads each prefix as a key range, and a reset as that table's
    cold start.

    :ivar prefixes: each prefix, the leading key columns and their values as text
    :ivar reason: :data:`RESET_REASON`, or None
    """

    prefixes: tuple[Mapping[str, str], ...] = ()
    reason: str | None = None


#: how long a change record is kept, wherever it is kept: long enough for a partitioned pod to catch
#: up from it. A record gone is a pod behind, never a drop.
EPOCH_CHANGE_MAX_AGE: Final = timedelta(hours=24)

#: the largest record of a commit's rows. A commit whose rows would not fit in one is recorded as
#: several consecutive records, each naming its run of the rows (:func:`changed_row_runs`); never as
#: the whole table. Well under a broker's default message limit.
_EPOCH_CHANGE_MAX_RECORD_BYTES: Final = 256 * 1024

#: what a record holds beside its rows (the table, the token, the field names), kept free under the bound.
_RECORD_OVERHEAD_BYTES: Final = 1024

#: a delta file's name as a record holds it, ``<table>/<write id>.arrow`` under the delta files' directory
#: (``delta_files._delta_file_name``): relative, one directory deep, so a reader never opens anything outside the
#: volume
_DELTA_FILE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*/[A-Za-z0-9_-]+\.arrow$")
#: the most scopes a keys-unknown record names (:attr:`EpochChange.scopes`), and the longest each: 57 states and
#: territories and a national scope are far under it, and at the bound a record is about 33 KB, well under
#: :data:`_EPOCH_CHANGE_MAX_RECORD_BYTES` (a keys-unknown record holds no rows). A commit touching more names none: its
#: recovery then has only the change notes (decision 31)
_EPOCH_SCOPES_MAX: Final = 256
_SCOPE_MAX_CHARS: Final = 128

#: the longest delta file name a record holds, out of the overhead kept free under the bound, so a record of a full
#: run of rows still fits: a 63-character table name and a 32-character write id come to 102
_DELTA_FILE_MAX_CHARS: Final = 256


def key_text(value: Any, *, instant: bool = False) -> str:
    """a key's or an invalidation column's value as a change record, a row message and a change note name it.

    A time zone-aware ``datetime`` (a ``timestamptz``) is named in UTC, to the microsecond, in ISO form
    (``2026-11-09T10:22:10.082658+00:00``): the form a table restore's read spells it in, whatever zone the value
    came in, so a restored record and a written record of one row name it in one spelling, and a key read back from
    the table binds as the same instant. Anything else is its
    ``str``. Every key derived from a row's key values (its L2 key, its fence key) is built from these forms too, so a
    message's text values land on the key the typed values made.

    With ``instant`` (the value is of a ``timestamptz`` column, ``DATETIMETZ_TYPE``), the column's other spellings
    of an instant name it in that one form too: the ISO text the L3 rail answers (``2026-11-03T21:45:00Z``, which a
    delta file and a copy's full load hold), Postgres' own (``2026-11-03 21:45:00+00``), and a naive time, which such a
    column holds as UTC. Why: a delta file named a timestamp key ``...Z`` while the epoch record and the change note
    named it ``....000000+00:00``, so a copy of a table keyed by a timestamp (reports' ``result_snapshots_by_id``)
    never found the keys it heard in the file (2026-11-24). Text that is no instant is kept as it is.

    :param value: the value
    :ptype value: Any
    :param instant: the value is a ``timestamptz`` column's, as a datetime or any ISO text of one
    :ptype instant: bool
    :return: its text
    :rtype: str
    """
    if instant and isinstance(value, str):
        text: str = value
        try:
            value = datetime.fromisoformat(text.strip())
        except ValueError:
            return text  # NOSILENT: not an instant's text, so no instant's canonical text can equal it either way
    if isinstance(value, datetime) and (instant or value.utcoffset() is not None):
        # the L1 keeps a timestamp column in this text too, so an eviction naming the key finds its row; a naive
        # time of a timestamptz column is UTC
        return stored_instant(value if value.utcoffset() is not None else value.replace(tzinfo=UTC))
    return str(value)


class ChangedRow(BaseModel):
    """one row an epoch changed, as a row broadcast names it.

    :ivar ids: the primary key's values as strings, in declared column order
    :ivar l2_current_scope: the L2 key scope whose entry the writer left holding the newest value,
        as :attr:`CacheInvalidationMessage.l2_current_scope`; ``None`` for an unfenced write
    :ivar columns: the declared invalidation columns' values from the row the write saw, as
        :attr:`CacheInvalidationMessage.columns`; ``None`` when none are declared or none was seen
    :ivar l3_pending: the write committed in L2 first (:mod:`threetears.core.collections.l2_first`), so
        L3 is made to match L2 for this row after the commit; ``False`` for a write L3 took first. With
        :attr:`value` ``None`` it is a delete
    :ivar value: the row's stored bytes, as the write put them in L2, for an epoch that is the write's
        commit (an L2-first collection): the row is applied to L2 from here, and a saver puts back any
        row L2 is missing before it saves to L3. ``None`` where the record only names the row
    :ivar l2_revision: the revision of the row's L2 key the committing writer read before its commit. The
        saver applies :attr:`value` only while the key is still at it: a writer that died after its commit
        left it there, and any write applied since moved it on
    :ivar tenant: the tenant the row belongs to (:mod:`threetears.core.tenancy`), whose key the
        record is appended under; ``None`` for ``_all``. Not stored in the record: its key carries it
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    ids: list[str]
    l2_current_scope: str | None = None
    columns: dict[str, str | None] | None = None
    l3_pending: bool = False
    value: bytes | None = None
    l2_revision: int | None = None
    tenant: str | None = Field(default=None, exclude=True)


class EpochChange(BaseModel):
    """the change one advance of a table's write generation made.

    :ivar table: the table
    :ivar token: the generation the advance wrote, ``{incarnation}:{count}``
    :ivar whole_table: the advance covered every row of the table (a truncate, a load, an
        announced migration); :attr:`rows` is then empty
    :ivar keys_unknown: the advance's writer named no rows and did not change the whole table (a
        broker advancing for a write that sent no ``changed_rows``): a follower that missed it cannot
        apply it, and is behind on the table for that epoch -- serving, logged and counted as
        ``keys_unknown``, never a drop. :attr:`rows` is then empty
    :ivar rows: the rows the advance covered, one per row broadcast of its commit
    :ivar partition: for a whole-table record of a load from files, the columns every row it loaded holds and their
        values as text (the load's constants): the rows whose columns equal these are the load's reach, and a follower
        that knows the field reads those rows again rather than the table. A follower built before it reads the
        record as the whole table, which the record also says, so it is never misled. A record of exactly one key
        prefix names it here too, for a follower that knows this field but not :attr:`prefixes`
    :ivar prefixes: for a whole-table record of a load or a clear, the SET of primary-key prefixes it reached, each the
        leading key columns and their values as text (an AP load of a state names each office's key): a follower reads
        each prefix's rows again as a key range, never the table. A follower built before it reads :attr:`partition`,
        or the whole table
    :ivar reason: for a whole-table record, why the whole table: only :data:`RESET_REASON` (an operator's reset). A
        follower built before it reads the record as the whole table
    :ivar noted_between: for a keys-unknown record of a commit whose rows were too many to name, the database clock's
        bracket of the commit's change notes, ``(transaction start, just before COMMIT)`` on the committing
        connection: every note the commit wrote (:mod:`~threetears.core.collections.change_notes`) is in it, so a
        follower resolves the keys from the notes (perhaps with other commits' keys, which it reads again harmlessly).
        A follower built before it reads the record as keys unknown and is behind, never misled
    :ivar database: the database whose write (or whose ring's restore) appended the record, by its ring's schema
        (module docstring); ``None`` from a writer that names none, or one built before the field
    :ivar delta_file: the commit's delta file of the table, its name under the delta files' directory (module
        docstring); ``None`` for a table that does not opt in, and from a writer built before the field
    :ivar delta_file_unconfirmed: the file is a failed load request's (stored as ``unconfirmed_delta_file``, module
        docstring): it may hold rows L3 never took, so it is applied only once the writer has written it again from
        L3 and marked it confirmed; never applied as it stands. Set only on a whole-table record of key prefixes
    :ivar unconfirmed_writer: for a record naming its file unconfirmed BEFORE the load's first commit (the intent
        record, module docstring), the key of the loading writer's beat (:class:`~threetears.core.collections
        .lost_writes.WriterBeats`): the file is confirmed only once a later record names it, or that beat is gone.
        ``None`` on a record written after the load stopped (decision 43), which the writer confirms when asked
    :ivar scopes: for a keys-unknown record of a commit that saved a delta file, the scopes its transaction touched
        (the states, ``state:CA``, as the writer's seqlock names them; module docstring): what the writer re-reads
        from L3 should the file be lost past the notes' window. ``None`` when unknown, past :data:`_EPOCH_SCOPES_MAX`,
        or from a writer built before the field
    :ivar s_min: the lowest row epoch the commit stamped (the row epoch's hybrid logical clock,
        :mod:`~threetears.core.ddl.row_epoch`), on every record of the commit: a scrub of a restore proves its margin
        held by it (a record after the keeper's mark whose ``s_min`` is below the restore's floor is a violation,
        ``design-append-only-l3.md`` decision 3). ``None`` for a commit that stamped no row, and from a writer built
        before the field
    :ivar cleanup_floor: the record is of quiet-time cleanup's own delete (:mod:`~threetears.core.collections
        .quiet_cleanup`), sent marked with the floor F it was worked out against (``CLEANUP_FLOOR_FIELD`` on the
        wire): every row it removed is stamped below F, so no follower or restore reads it, a follower applies it by
        marking it applied, reading nothing (``CollectionRegistry.catch_up_generation``; a cleanup commit carries no
        other write, ``broker_generation.cleanup_mix_refusal``), and a gate fed by the change log
        (``LoadGate.from_change_log``) does not hear it as a load. ``None`` for every other write, and from
        a writer built before the field
    """

    table: str
    token: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []
    partition: dict[str, str] | None = None
    prefixes: list[dict[str, str]] | None = None
    reason: str | None = None
    noted_between: tuple[datetime, datetime] | None = None
    database: str | None = None
    delta_file: str | None = None
    scopes: list[str] | None = None
    delta_file_unconfirmed: bool = False
    unconfirmed_writer: str | None = None
    s_min: int | None = None
    cleanup_floor: int | None = None

    @model_validator(mode="after")
    def _names_its_reach(self) -> EpochChange:
        """refuse a record that is not exactly one of: its rows, the whole table, or keys unknown.

        :return: the record
        :rtype: EpochChange
        :raises ValueError: when the reach is empty or contradictory
        """
        if [bool(self.rows), self.whole_table, self.keys_unknown].count(True) != 1:
            raise ValueError(
                f"an epoch change of {self.table!r} at {self.token!r} names exactly one of its rows, the whole "
                f"table, or keys unknown"
            )
        if self.partition is not None and (not self.whole_table or not self.partition):
            raise ValueError(
                f"an epoch change of {self.table!r} at {self.token!r} names a partition only as a whole-table record, "
                f"and names at least one column"
            )
        check_whole_reach(self.prefixes, self.reason, whole_table=self.whole_table)
        _check_noted_between(self.noted_between, keys_unknown=self.keys_unknown)
        _check_delta_file(self.delta_file)
        _check_scopes(self.scopes, keys_unknown=self.keys_unknown)
        if self.delta_file_unconfirmed:
            _check_unconfirmed(self.delta_file, None, whole_table=self.whole_table, prefixes=self.prefixes)
        _check_unconfirmed_writer(self.unconfirmed_writer, self.delta_file if self.delta_file_unconfirmed else None)
        _check_s_min(self.s_min)
        _check_cleanup_floor(self.cleanup_floor)
        return self


def _check_s_min(s_min: int | None) -> None:
    """refuse an ``s_min`` that is no row epoch: a positive BIGINT.

    :param s_min: the commit's lowest stamp, or None
    :ptype s_min: int | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if s_min is not None and not 0 < s_min < 1 << 63:
        raise ValueError(f"a record's s_min is a row epoch, a positive BIGINT: got {s_min}")


def _check_cleanup_floor(floor: int | None) -> None:
    """refuse a ``cleanup_floor`` that is no row epoch: a positive BIGINT, as the wire's ``CLEANUP_FLOOR_FIELD``.

    :param floor: the floor cleanup's delete was worked out against, or None
    :ptype floor: int | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if floor is not None and not 0 < floor < 1 << 63:
        raise ValueError(f"a record's cleanup_floor is a row epoch, a positive BIGINT: got {floor}")


def _check_unconfirmed(
    unconfirmed: str | None,
    delta_file: str | None,
    *,
    whole_table: bool,
    prefixes: Sequence[Mapping[str, str]] | None,
) -> None:
    """refuse an unconfirmed delta file anywhere but a load's record: a whole-table record of key prefixes naming no
    confirmed file beside it (module docstring).

    :param unconfirmed: the unconfirmed file's name
    :ptype unconfirmed: str | None
    :param delta_file: the confirmed file named beside it, if any
    :ptype delta_file: str | None
    :param whole_table: whether the record is whole-table
    :ptype whole_table: bool
    :param prefixes: the record's key prefixes
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if unconfirmed is None:
        raise ValueError("a record marks a delta file unconfirmed only when it names one")
    _check_delta_file(unconfirmed)
    if delta_file is not None:
        raise ValueError("a record names a delta file confirmed or unconfirmed, never both")
    if not whole_table or not prefixes:
        raise ValueError("an unconfirmed delta file is named only on a whole-table record of a load's key prefixes")


def _check_unconfirmed_writer(writer: str | None, unconfirmed: str | None) -> None:
    """refuse a writer named anywhere but beside an unconfirmed file, or one that is no key.

    :param writer: the loading writer's beat key, or None
    :ptype writer: str | None
    :param unconfirmed: the unconfirmed file named beside it, if any
    :ptype unconfirmed: str | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if writer is None:
        return
    if unconfirmed is None:
        raise ValueError("a record names a loading writer only beside a delta file it names unconfirmed")
    if not writer or len(writer) > _UNCONFIRMED_WRITER_MAX_CHARS or any(ch.isspace() for ch in writer):
        raise ValueError(f"a loading writer is named by its beat key, one KV key: got {writer!r}")


#: the longest beat key a record names as its loading writer (``write-ledger.<schema>.<process>``)
_UNCONFIRMED_WRITER_MAX_CHARS: Final = 255


def _check_scopes(scopes: Sequence[str] | None, *, keys_unknown: bool) -> None:
    """refuse scopes on a record that is not keys unknown, or scopes no record holds (:func:`recordable_scopes`).

    :param scopes: the scopes, or None
    :ptype scopes: Sequence[str] | None
    :param keys_unknown: whether the record is keys unknown
    :ptype keys_unknown: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when they are refused
    """
    if scopes is None:
        return
    if not keys_unknown:
        raise ValueError("a change's scopes are named only on a keys-unknown record: a keyed one names its rows")
    if recordable_scopes(scopes) is None or len(set(scopes)) != len(scopes):
        raise ValueError(
            f"a record names 1 to {_EPOCH_SCOPES_MAX} distinct scopes of 1 to {_SCOPE_MAX_CHARS} characters, "
            f"got {len(scopes)}"
        )


#: the scope a commit's record names beside its states when they are every state its delta files' rows are of
#: (module docstring, decision 54 (c)): no scope of a product's (``state:``, ``race:``) begins ``3tears:``
EVERY_STATE_NAMED: Final = "3tears:every-state-named"


def complete_scopes(scopes: Sequence[str] | None) -> tuple[str, ...] | None:
    """a record's scopes when they are known to name every state its commit wrote (:data:`EVERY_STATE_NAMED` among
    them), the mark left out; None when they are not known complete (no mark: an older writer, a table declaring no
    state scope) or there are none. Why: a reader reloading only these states must have every state the commit wrote,
    or it keeps the old rows of one it missed (module docstring).

    :param scopes: the record's scopes (:attr:`EpochChange.scopes`)
    :ptype scopes: Sequence[str] | None
    :return: the states, the mark left out, or None
    :rtype: tuple[str, ...] | None
    """
    if not scopes or EVERY_STATE_NAMED not in scopes:
        return None
    named = tuple(scope for scope in scopes if scope != EVERY_STATE_NAMED)
    return named or None


def recordable_scopes(scopes: Iterable[str]) -> list[str] | None:
    """the scopes a commit touched as a keys-unknown record names them (:attr:`EpochChange.scopes`): each once,
    sorted; ``None`` when there are none, more than :data:`_EPOCH_SCOPES_MAX`, or one longer than a record holds, so a
    record never grows past its bound and a commit too wide names none (its recovery then has only the notes).

    :param scopes: the scopes, perhaps with repeats
    :ptype scopes: Iterable[str]
    :return: the scopes to name, or None
    :rtype: list[str] | None
    """
    named = sorted(set(scopes))
    if not named or len(named) > _EPOCH_SCOPES_MAX:
        return None
    if any(not isinstance(scope, str) or not scope or len(scope) > _SCOPE_MAX_CHARS for scope in named):
        return None
    return named


def _check_delta_file(delta_file: str | None) -> None:
    """refuse a delta file name that is not one :mod:`~threetears.core.collections.delta_files` makes: a reader opens
    the name on the shared volume, so it never reaches outside it, and a long one would push a full record past its
    bound.

    :param delta_file: the name, or None
    :ptype delta_file: str | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is not ``<table>/<write id>.arrow`` within :data:`_DELTA_FILE_MAX_CHARS`
    """
    if delta_file is not None and (len(delta_file) > _DELTA_FILE_MAX_CHARS or not _DELTA_FILE_NAME.match(delta_file)):
        raise ValueError(f"a record names a delta file as <table>/<write id>.arrow, got {delta_file!r}")


def check_delta_file(delta_file: str | None) -> None:
    """refuse a delta file name an epoch record would refuse: the one check the record makes
    (:meth:`EpochChange` and :func:`epoch_change_body` run it), public so a broker checks a name a pod sent it by the
    record's own rule rather than by building a record to see whether it is refused.

    :param delta_file: the name, or None
    :ptype delta_file: str | None
    :return: nothing
    :rtype: None
    :raises ValueError: when it is not ``<table>/<write id>.arrow`` within the record's bound
    """
    _check_delta_file(delta_file)


def check_whole_reach(prefixes: Sequence[Mapping[str, str]] | None, reason: str | None, *, whole_table: bool) -> None:
    """refuse key prefixes or a reason on a record that is not whole-table, an empty prefix, or a reason not known.

    :param prefixes: the record's key prefixes, or None
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: the record's reason, or None
    :ptype reason: str | None
    :param whole_table: whether the record is whole-table
    :ptype whole_table: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if (prefixes is not None or reason is not None) and not whole_table:
        raise ValueError("a change names key prefixes or a reason only as a whole-table record")
    if prefixes is not None and (not prefixes or not all(prefixes)):
        raise ValueError("a change's key prefixes are at least one, each naming at least one column")
    if reason is not None and reason != RESET_REASON:
        raise ValueError(f"a whole-table change's reason is {RESET_REASON!r} or none, not {reason!r}")
    if reason is not None and prefixes is not None:
        raise ValueError("a reset names the whole table, never key prefixes")


def _check_noted_between(noted_between: tuple[datetime, datetime] | None, *, keys_unknown: bool) -> None:
    """refuse a notes bracket on a record that is not keys unknown, or one that ends before it starts.

    :param noted_between: the bracket, or None
    :ptype noted_between: tuple[datetime, datetime] | None
    :param keys_unknown: whether the record is keys unknown
    :ptype keys_unknown: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when it is refused
    """
    if noted_between is None:
        return
    if not keys_unknown:
        raise ValueError("a change's notes bracket is named only on a keys-unknown record")
    if noted_between[1] < noted_between[0]:
        raise ValueError(f"a change's notes bracket ends before it starts: {noted_between}")


@runtime_checkable
class EpochChangeReader(Protocol):
    """reads the change record of one advance of a table's write generation."""

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        """the record of the advance that wrote ``token``.

        :param table_name: the table
        :ptype table_name: str
        :param token: the generation that advance wrote
        :ptype token: str
        :return: the record, or ``None`` when L2 does not hold it (never written, lost with the
            store, or past its retention): the follower is then behind
        :rtype: EpochChange | None
        :raises GenerationUnavailableError: when the store cannot be read
        """
        ...

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        """every epoch of ``table_name`` after ``after`` and up to ``upto``, in order: what a follower applies.

        :param table_name: the table
        :ptype table_name: str
        :param after: the follower's mark, ``{incarnation}:{count}``
        :ptype after: str
        :param upto: the generation read, under the same incarnation
        :ptype upto: str
        :return: the epochs the log holds in that range, and whether it may be missing any
        :rtype: EpochWalk
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...


@dataclass(frozen=True, slots=True)
class EpochWalk:
    """the epochs of one table a follower reads to catch up, in order.

    :ivar changes: each epoch's record the log holds in the range, oldest first
    :ivar complete: ``False`` when the log may be missing an epoch in the range -- a record lost or
        aged out -- so the follower is behind on the table even after applying :attr:`changes`
    :ivar first_missing: the oldest epoch known or suspected missing, for the log line; ``None`` when
        :attr:`complete`
    :ivar reason: why the walk is not complete: the reason the registry falls behind with, one of
        :data:`threetears.core.collections.registry.BEHIND_REASONS`
    """

    changes: tuple[EpochChange, ...]
    complete: bool
    first_missing: str | None = None
    reason: str = "record_missing"


async def walk_by_count(reader: EpochChangeReader, table_name: str, after: str, upto: str) -> EpochWalk:
    """the walk of a log that numbers one table's epochs one apart: each count in turn.

    For a log that keeps a table's epochs at consecutive counts (a table kept in L2, a test's log). A count
    the log does not hold is an epoch missing; the counts after it are still read.

    :param reader: reads one epoch's record
    :ptype reader: EpochChangeReader
    :param table_name: the table
    :ptype table_name: str
    :param after: the mark
    :ptype after: str
    :param upto: the generation read
    :ptype upto: str
    :return: the walk
    :rtype: EpochWalk
    :raises GenerationUnavailableError: when a record cannot be read
    """
    incarnation, _, low = after.rpartition(":")
    _, _, high = upto.rpartition(":")
    changes: list[EpochChange] = []
    missing: str | None = None
    for count in range(int(low) + 1, int(high) + 1):
        token = f"{incarnation}:{count}"
        change = await reader.change(table_name, token)
        if change is None:
            missing = missing or token
            continue
        changes.append(change)
    return EpochWalk(changes=tuple(changes), complete=missing is None, first_missing=missing)


@dataclass(frozen=True, slots=True)
class EpochQuiet:
    """what a log can vouch for after a follower's mark, when the table's generation reads at or below it.

    :ivar through: the newest epoch through which the log holds nothing of the table after the mark, so
        the mark may move to it; ``None`` when the log holds an epoch of the table (or one it cannot
        name) after the mark, or cannot say
    :ivar aged_out: the log no longer holds the epoch after the mark (retention took it), so it cannot say
        whether the table was written since: the follower is behind (``aged_out``), never current
    """

    through: str | None
    aged_out: bool = False


@runtime_checkable
class EpochLogExtent(Protocol):
    """what a change log can say about its own reach: the lost log a restore continues, and quiet stretches."""

    async def continues(self, incarnation: str) -> tuple[str, int | None] | None:
        """the lost log's incarnation the log ``incarnation`` continues, and the sequence its notes cover it from.

        A follower whose mark under the lost incarnation is at or past that sequence missed nothing the
        restored log does not hold: it walks every record of the restored log. One whose mark is older is
        behind.

        :param incarnation: the log's incarnation, as a generation read names it
        :ptype incarnation: str
        :return: ``(lost incarnation, covered from)``, the second ``None`` when the notes cover none of the
            lost log; ``None`` when the log continues none, or is not ``incarnation`` (any more)
        :rtype: tuple[str, int | None] | None
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...

    async def quiet_after(self, table_name: str, mark: str) -> EpochQuiet:
        """what the log can vouch for of ``table_name`` after ``mark``.

        :param table_name: the table
        :ptype table_name: str
        :param mark: the follower's mark
        :ptype mark: str
        :return: how far the mark may move, and whether the log has aged past it
        :rtype: EpochQuiet
        :raises GenerationUnavailableError: when the log cannot be read
        """
        ...


@runtime_checkable
class _EpochChangeWriter(Protocol):
    """a generation source that records each advance's change as it makes it."""

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
    ) -> str | None:
        """advance the table's generation once and store what the advance covered.

        :param table_name: the table
        :ptype table_name: str
        :param rows: the rows the committed write changed; ``None`` when it changed the whole table
        :ptype rows: Sequence[ChangedRow] | None
        :param keys_unknown: the writer named no rows and did not change the whole table; the record
            says so and a follower that missed it is behind for that epoch. Requires ``rows=None``
        :ptype keys_unknown: bool
        :param tenant: the write's tenant, from a writer that knows it: a rowless record's key, and that of
            any row naming none; ``None`` for ``_all``
        :ptype tenant: str | None
        :return: the generation token the advance wrote, as :meth:`GenerationSource.advance`
        :rtype: str | None
        :raises GenerationUnavailableError: when the generation could not be advanced, or its
            change could not be stored, after a committed write
        :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
        """
        ...


def check_change_form(rows: Sequence[ChangedRow] | None, *, keys_unknown: bool) -> None:
    """refuse ``keys_unknown`` together with rows: a change is its rows, the whole table, or keys unknown.

    :param rows: the rows named, or ``None``
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: whether the writer's keys are unknown
    :ptype keys_unknown: bool
    :return: nothing
    :rtype: None
    :raises ValueError: when both are given
    """
    if keys_unknown and rows is not None:
        raise ValueError("keys_unknown=True names no rows: pass rows=None")


def records_changes(source: object) -> bool:
    """whether ``source`` stores each advance's change itself (an epoch source that keeps the change log).

    A pod's source does not: its broker advances for it and names the change.

    :param source: a generation source
    :ptype source: object
    :return: whether it records changes
    :rtype: bool
    """
    return isinstance(source, _EpochChangeWriter)


async def advance_recording(
    source: GenerationSource,
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    tenant: str | None = None,
    partition: Mapping[str, str] | None = None,
    noted_between: tuple[datetime, datetime] | None = None,
    prefixes: Sequence[Mapping[str, str]] | None = None,
    reason: str | None = None,
    delta_file: str | None = None,
    scopes: Sequence[str] | None = None,
    repair: bool = True,
    unconfirmed_delta_file: str | None = None,
    unconfirmed_writer: str | None = None,
    s_min: int | None = None,
    cleanup_floor: int | None = None,
) -> str | None:
    """advance ``table_name``'s generation through ``source``, storing the change when the source records changes.

    Every advance goes through here, so a source that keeps the epoch change log is never handed an
    advance without its change. A source that keeps none (a pod's, whose broker advances for it)
    advances as it always has.

    **An advance that does not land is never dropped.** One that fails (or is cancelled) after its write committed is
    handed, as it is, to the process's writer ledger (:meth:`~threetears.core.utils.write_ledger.WriteLedger
    .lost_advance`) before the failure is raised: the process's backstop makes it again until it lands, and the ledger's
    watermark, which the backstop publishes, is held at its start meanwhile, so a writer that dies first is announced
    by recovery (:class:`~threetears.core.collections.lost_writes.WriteBackstop`). Not handed over: a commit that did
    not land (:class:`~threetears.core.exceptions.GenerationNotCommittedError`), and a change refused before anything
    was advanced (``ValueError``).

    :param source: the registry's generation source
    :ptype source: GenerationSource
    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows the committed write changed; ``None`` when its reach is the whole table
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows and did not change the whole table; requires ``rows=None``
    :ptype keys_unknown: bool
    :param tenant: the write's tenant, from a writer that knows it (the schema it wrote in): a rowless
        record's key, and that of any row naming none; ``None`` for ``_all``
    :ptype tenant: str | None
    :param partition: a whole-table change's partition (:attr:`EpochChange.partition`), for a source that records it
        (``advance_with_change(partition=)``); one that does not records the whole table, which the partition narrows
    :ptype partition: Mapping[str, str] | None
    :param noted_between: a keys-unknown change's notes bracket (:attr:`EpochChange.noted_between`), for a source that
        records it; one that does not records keys unknown, which the bracket would resolve
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: a whole-table change's key prefixes (:attr:`EpochChange.prefixes`), for a source that records
        them; exactly one is also named as :attr:`EpochChange.partition`, for a follower that knows only that
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: a whole-table change's reason (:data:`RESET_REASON`), for a source that records it
    :ptype reason: str | None
    :param delta_file: the commit's delta file of the table (:attr:`EpochChange.delta_file`), for a source that
        records it; one that does not records the change without it
    :ptype delta_file: str | None
    :param scopes: a keys-unknown change's scopes (:attr:`EpochChange.scopes`), for a source that records them;
        ignored on any other change
    :ptype scopes: Sequence[str] | None
    :param repair: hand a failed advance to the process's ledger for retry; ``False`` for the retry itself
    :ptype repair: bool
    :param unconfirmed_delta_file: a whole-table change's unconfirmed file, a failed load request's
        (:attr:`EpochChange.delta_file_unconfirmed`), for a source that records it; one that does not records the
        change naming no file, never naming it as ``delta_file``
    :ptype unconfirmed_delta_file: str | None
    :param unconfirmed_writer: beside ``unconfirmed_delta_file`` on a load's intent record, the loading writer's beat
        key (:attr:`EpochChange.unconfirmed_writer`), for a source that records it
    :ptype unconfirmed_writer: str | None
    :param s_min: the commit's lowest row epoch (:attr:`EpochChange.s_min`), for a source that records it
    :ptype s_min: int | None
    :param cleanup_floor: the write was quiet-time cleanup's own delete, worked out against this floor
        (:attr:`EpochChange.cleanup_floor`), for a source that records it: the broker passes the floor its request
        named (``CLEANUP_FLOOR_FIELD``); ``None`` takes the one of the :func:`~threetears.core.backends
        .broker_generation.cleanup_deleting` block this advance runs in (a direct pool's cleanup), none outside one
    :ptype cleanup_floor: int | None
    :return: the token the advance wrote, as :meth:`GenerationSource.advance`
    :rtype: str | None
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    :raises ValueError: when ``keys_unknown`` is set with rows, before anything is advanced
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    _check_s_min(s_min)
    _check_delta_file(delta_file)
    named_scopes = list(scopes) if scopes and keys_unknown else None
    _check_scopes(named_scopes, keys_unknown=keys_unknown)
    if unconfirmed_delta_file is not None:
        _check_unconfirmed(
            unconfirmed_delta_file, delta_file, whole_table=rows is None and not keys_unknown, prefixes=prefixes
        )
    _check_unconfirmed_writer(unconfirmed_writer, unconfirmed_delta_file)
    # read here, in the writer's context: a retry from the ledger runs outside it and names the same floor
    cleaning = cleanup_floor if cleanup_floor is not None else cleanup_delete_floor()
    _check_cleanup_floor(cleaning)
    if not repair:
        return await _advance_recording_once(
            source,
            table_name,
            rows,
            keys_unknown=keys_unknown,
            tenant=tenant,
            partition=partition,
            noted_between=noted_between,
            prefixes=prefixes,
            reason=reason,
            delta_file=delta_file,
            scopes=named_scopes,
            unconfirmed_delta_file=unconfirmed_delta_file,
            unconfirmed_writer=unconfirmed_writer,
            s_min=s_min,
            cleanup_floor=cleaning,
        )
    kept_rows = None if rows is None else list(rows)

    async def again() -> str | None:
        return await _advance_recording_once(
            source,
            table_name,
            kept_rows,
            keys_unknown=keys_unknown,
            tenant=tenant,
            partition=partition,
            noted_between=noted_between,
            prefixes=prefixes,
            reason=reason,
            delta_file=delta_file,
            scopes=named_scopes,
            unconfirmed_delta_file=unconfirmed_delta_file,
            unconfirmed_writer=unconfirmed_writer,
            s_min=s_min,
            cleanup_floor=cleaning,
        )

    try:
        return await again()
    except GenerationNotCommittedError, ValueError:
        # nothing committed, or a change refused before anything was advanced: nothing to make again
        raise
    except (Exception, asyncio.CancelledError) as exc:
        process_write_ledger().lost_advance(table_name, again, error=f"{type(exc).__name__}: {exc}")
        raise


async def _advance_recording_once(
    source: GenerationSource,
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool,
    tenant: str | None,
    partition: Mapping[str, str] | None,
    noted_between: tuple[datetime, datetime] | None,
    prefixes: Sequence[Mapping[str, str]] | None,
    reason: str | None,
    delta_file: str | None,
    scopes: list[str] | None,
    unconfirmed_delta_file: str | None = None,
    unconfirmed_writer: str | None = None,
    s_min: int | None = None,
    cleanup_floor: int | None = None,
) -> str | None:
    """one try of :func:`advance_recording`, its change already checked.

    :param source: the registry's generation source
    :ptype source: GenerationSource
    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows, ``None`` for a rowless change
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the change names no keys
    :ptype keys_unknown: bool
    :param tenant: a rowless record's tenant
    :ptype tenant: str | None
    :param partition: a whole-table change's partition
    :ptype partition: Mapping[str, str] | None
    :param noted_between: a keys-unknown change's notes bracket
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: a whole-table change's key prefixes
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: a whole-table change's reason
    :ptype reason: str | None
    :param delta_file: the commit's delta file of the table
    :ptype delta_file: str | None
    :param scopes: a keys-unknown change's scopes
    :ptype scopes: list[str] | None
    :param unconfirmed_delta_file: a failed load request's file, unconfirmed
    :ptype unconfirmed_delta_file: str | None
    :param unconfirmed_writer: the loading writer's beat key beside it, on a load's intent record
    :ptype unconfirmed_writer: str | None
    :param s_min: the commit's lowest row epoch
    :ptype s_min: int | None
    :param cleanup_floor: the floor of quiet-time cleanup's delete the write was, or None
    :ptype cleanup_floor: int | None
    :return: the token the advance wrote
    :rtype: str | None
    :raises GenerationUnavailableError: when the advance, or the record of its change, failed
    """
    if isinstance(source, _EpochChangeWriter):
        extra: dict[str, Any] = {}
        if tenant is not None:
            extra["tenant"] = tenant
        whole = rows is None and not keys_unknown
        check_whole_reach(prefixes, reason, whole_table=whole)
        if prefixes and len(prefixes) == 1 and not partition:
            partition = prefixes[0]
        if partition and whole and _takes(source, "partition"):
            extra["partition"] = dict(partition)
        if prefixes and _takes(source, "prefixes"):
            extra["prefixes"] = [dict(prefix) for prefix in prefixes]
        if reason is not None and _takes(source, "reason"):
            extra["reason"] = reason
        if noted_between is not None and keys_unknown and _takes(source, "noted_between"):
            extra["noted_between"] = noted_between
        if delta_file is not None and _takes(source, "delta_file"):
            extra["delta_file"] = delta_file
        if scopes and keys_unknown and _takes(source, "scopes"):
            extra["scopes"] = scopes
        if unconfirmed_delta_file is not None and _takes(source, "unconfirmed_delta_file"):
            # a source built before the field records the load naming no file: never the file as confirmed
            extra["unconfirmed_delta_file"] = unconfirmed_delta_file
            if unconfirmed_writer is not None and _takes(source, "unconfirmed_writer"):
                # a source built before it records the intent as a stopped load's record: the writer then confirms it
                # when asked, as decision 43 does, which the loading hub's own later record supersedes
                extra["unconfirmed_writer"] = unconfirmed_writer
        if s_min is not None and _takes(source, "s_min"):
            extra["s_min"] = s_min
        if cleanup_floor is not None and _takes(source, "cleanup_floor"):
            # a source built before it records cleanup's delete as any write's: a change-log gate then hears it
            extra["cleanup_floor"] = cleanup_floor
        return await source.advance_with_change(table_name, rows, keys_unknown=keys_unknown, **extra)
    return await source.advance(table_name)


def _takes(source: object, field: str) -> bool:
    """whether ``source``'s ``advance_with_change`` takes ``field``: a source built before it is handed none.

    :param source: a source that records changes
    :ptype source: object
    :param field: the keyword (``partition``, ``noted_between``)
    :ptype field: str
    :return: whether it takes it
    :rtype: bool
    """
    method = getattr(source, "advance_with_change", None)
    return method is not None and field in inspect.signature(method).parameters


def changed_row_runs(table_name: str, rows: Sequence[ChangedRow]) -> list[list[ChangedRow]]:
    """split a commit's rows into consecutive runs whose records each fit the record bound.

    A commit of more rows than one record holds is recorded as one epoch per run, in order, so a
    follower that missed it applies every row by key; nothing becomes a whole-table change for its size.

    :param table_name: the table, for the error
    :ptype table_name: str
    :param rows: the rows the commit changed, at least one
    :ptype rows: Sequence[ChangedRow]
    :return: the runs, in order; one run when every row fits in one record
    :rtype: list[list[ChangedRow]]
    :raises GenerationUnavailableError: when one row alone is past the bound, so no record can name it
    """
    budget = _EPOCH_CHANGE_MAX_RECORD_BYTES - _RECORD_OVERHEAD_BYTES
    runs: list[list[ChangedRow]] = []
    run: list[ChangedRow] = []
    used = 0
    for row in rows:
        # measured as the fullest form any record stores it in, so every record of the run fits
        size = len(row.model_dump_json()) + 1
        if size > budget:
            raise GenerationUnavailableError(
                f"a changed row of {table_name!r} is {size} bytes, past the {budget}-byte record bound: no epoch "
                f"record can name it"
            )
        if run and used + size > budget:
            runs.append(run)
            run, used = [], 0
        run.append(row)
        used += size
    if run:
        runs.append(run)
    return runs


def _within_bound(stored: bytes, table_name: str) -> bytes:
    """``stored``, refused when it is past the record bound.

    :param stored: a record as stored
    :ptype stored: bytes
    :param table_name: the table, for the error
    :ptype table_name: str
    :return: ``stored``
    :rtype: bytes
    :raises GenerationUnavailableError: when it is past the bound; the caller splits with :func:`changed_row_runs`
    """
    if len(stored) > _EPOCH_CHANGE_MAX_RECORD_BYTES:
        raise GenerationUnavailableError(
            f"an epoch record of {table_name!r} is {len(stored)} bytes, past the {_EPOCH_CHANGE_MAX_RECORD_BYTES}-byte "
            f"bound: split its rows with changed_row_runs"
        )
    return stored


def epoch_change_record(
    table_name: str,
    token: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    reach: WholeReach | None = None,
) -> bytes:
    """the stored form of the change one advance made.

    A record past :data:`_EPOCH_CHANGE_MAX_RECORD_BYTES` is refused: the caller records a large
    commit as one advance per run of :func:`changed_row_runs`.

    :param table_name: the table
    :ptype table_name: str
    :param token: the generation the advance wrote
    :ptype token: str
    :param rows: the rows it covered; ``None`` when its reach is the whole table, or its keys unknown
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows and did not change the whole table; requires ``rows=None``
    :ptype keys_unknown: bool
    :param reach: a whole-table change's key prefixes or reason (:attr:`EpochChange.prefixes`,
        :attr:`EpochChange.reason`)
    :ptype reach: WholeReach | None
    :return: the record, as stored
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows, or a reach on a change that is not whole-table
    :raises GenerationUnavailableError: when the record is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    prefixes = [dict(prefix) for prefix in reach.prefixes] if reach is not None and reach.prefixes else None
    record = EpochChange(
        table=table_name,
        token=token,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
        partition=prefixes[0] if prefixes is not None and len(prefixes) == 1 else None,
        prefixes=prefixes,
        reason=reach.reason if reach is not None else None,
    )
    return _within_bound(record.model_dump_json().encode("utf-8"), table_name)


def read_epoch_change(raw: bytes, table_name: str, token: str) -> EpochChange:
    """the change record stored as ``raw``, checked to be the one of ``table_name`` at ``token``.

    :param raw: the stored record
    :ptype raw: bytes
    :param table_name: the table it must be of
    :ptype table_name: str
    :param token: the generation it must be of
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed, or is another epoch's
    """
    try:
        record = EpochChange.model_validate_json(raw)
    except ValidationError as exc:
        raise GenerationUnavailableError(
            f"the change record of {table_name!r} at {token!r} is malformed: {exc}"
        ) from exc
    if record.table != table_name or record.token != token:
        raise GenerationUnavailableError(
            f"the change record stored for {table_name!r} at {token!r} is of {record.table!r} at {record.token!r}"
        )
    return record


class _EpochBody(BaseModel):
    """an epoch's change as a change log stores it: the stream's sequence numbers it, so it names no token."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    table: str
    whole_table: bool = False
    keys_unknown: bool = False
    rows: list[ChangedRow] = []
    partition: dict[str, str] | None = None
    prefixes: list[dict[str, str]] | None = None
    reason: str | None = None
    noted_between: tuple[datetime, datetime] | None = None
    #: last but for the delta file, so a keyed record's form (:func:`keyed_record_form`) carries it in its tail; a
    #: form is cut from a record naming no delta file, which the stored dump leaves out
    database: str | None = None
    delta_file: str | None = None
    #: after the delta file: named only on a keys-unknown record, which no keyed record's form is cut from
    scopes: list[str] | None = None
    #: a failed load request's file (module docstring): named only on a whole-table record, never beside delta_file;
    #: a reader built before it ignores it, and reads the record as a load naming no file
    unconfirmed_delta_file: str | None = None
    #: the loading writer's beat key, beside unconfirmed_delta_file on a load's intent record; a reader built before
    #: it ignores it
    unconfirmed_writer: str | None = None
    #: the commit's lowest row epoch, on every record of it; last, so a keyed record's form (cut from a record naming
    #: none) is unchanged, and a reader built before it ignores it
    s_min: int | None = None
    #: on a record of quiet-time cleanup's own delete, the floor it was worked out against; after s_min, so a keyed
    #: record's form (cut from a record naming none) is unchanged, and a reader built before it ignores it
    cleanup_floor: int | None = None


def epoch_change_body(
    table_name: str,
    rows: Sequence[ChangedRow] | None,
    *,
    keys_unknown: bool = False,
    partition: Mapping[str, str] | None = None,
    noted_between: tuple[datetime, datetime] | None = None,
    prefixes: Sequence[Mapping[str, str]] | None = None,
    reason: str | None = None,
    database: str | None = None,
    delta_file: str | None = None,
    scopes: Sequence[str] | None = None,
    unconfirmed_delta_file: str | None = None,
    unconfirmed_writer: str | None = None,
    s_min: int | None = None,
    cleanup_floor: int | None = None,
) -> bytes:
    """the stored form of a change for a log whose sequence is the epoch, so the record carries no token.

    A record past the record bound is refused, as :func:`epoch_change_record` refuses it: the caller
    splits a large commit's rows with :func:`changed_row_runs`. A record whose rows carry their values is
    the commit itself, so it is stored whole, and the log refuses it if it is past the log's own limit.

    :param table_name: the table
    :ptype table_name: str
    :param rows: the rows changed; ``None`` for the whole table, or with ``keys_unknown``
    :ptype rows: Sequence[ChangedRow] | None
    :param keys_unknown: the writer named no rows; requires ``rows=None``
    :ptype keys_unknown: bool
    :param partition: for a whole-table change that is a load from files, the columns its rows hold and their values
        (:attr:`EpochChange.partition`); requires ``rows=None`` and not ``keys_unknown``
    :ptype partition: Mapping[str, str] | None
    :param noted_between: for a keys-unknown change, its notes bracket (:attr:`EpochChange.noted_between`)
    :ptype noted_between: tuple[datetime, datetime] | None
    :param prefixes: for a whole-table change of a load or a clear, its key prefixes (:attr:`EpochChange.prefixes`)
    :ptype prefixes: Sequence[Mapping[str, str]] | None
    :param reason: for a whole-table change, its reason (:attr:`EpochChange.reason`)
    :ptype reason: str | None
    :param database: the database the change was written in (:attr:`EpochChange.database`); ``None`` names none
    :ptype database: str | None
    :param delta_file: the commit's delta file of the table (:attr:`EpochChange.delta_file`); ``None`` names none
    :ptype delta_file: str | None
    :param scopes: for a keys-unknown change, the scopes its commit touched (:attr:`EpochChange.scopes`); ``None``
        names none
    :ptype scopes: Sequence[str] | None
    :param unconfirmed_delta_file: for a whole-table change of key prefixes, a failed load request's file, stored under
        its own field (:attr:`EpochChange.delta_file_unconfirmed`); ``None`` names none
    :ptype unconfirmed_delta_file: str | None
    :param unconfirmed_writer: beside ``unconfirmed_delta_file`` on a load's intent record, the loading writer's beat
        key (:attr:`EpochChange.unconfirmed_writer`); ``None`` names none
    :ptype unconfirmed_writer: str | None
    :param s_min: the commit's lowest row epoch (:attr:`EpochChange.s_min`); ``None`` names none
    :ptype s_min: int | None
    :param cleanup_floor: for a record of quiet-time cleanup's own delete, its floor (:attr:`EpochChange
        .cleanup_floor`); ``None`` for any other write
    :ptype cleanup_floor: int | None
    :return: the stored body
    :rtype: bytes
    :raises ValueError: when ``keys_unknown`` is set with rows, a partition with either, or a delta file name that is
        not one (:func:`_check_delta_file`)
    :raises GenerationUnavailableError: when a record naming rows without their values is past the bound
    """
    check_change_form(rows, keys_unknown=keys_unknown)
    _check_delta_file(delta_file)
    if partition is not None and (rows is not None or keys_unknown):
        raise ValueError("a partition narrows a whole-table change: pass rows=None without keys_unknown")
    _check_noted_between(noted_between, keys_unknown=keys_unknown)
    _check_scopes(scopes, keys_unknown=keys_unknown)
    check_whole_reach(prefixes, reason, whole_table=rows is None and not keys_unknown)
    if unconfirmed_delta_file is not None:
        _check_unconfirmed(
            unconfirmed_delta_file, delta_file, whole_table=rows is None and not keys_unknown, prefixes=prefixes
        )
    _check_unconfirmed_writer(unconfirmed_writer, unconfirmed_delta_file)
    _check_s_min(s_min)
    _check_cleanup_floor(cleanup_floor)
    body = _EpochBody(
        table=table_name,
        whole_table=rows is None and not keys_unknown,
        keys_unknown=keys_unknown,
        rows=list(rows) if rows is not None else [],
        partition=dict(partition) if partition else None,
        prefixes=[dict(prefix) for prefix in prefixes] if prefixes else None,
        reason=reason,
        noted_between=noted_between,
        database=database,
        delta_file=delta_file,
        scopes=list(scopes) if scopes is not None else None,
        unconfirmed_delta_file=unconfirmed_delta_file,
        unconfirmed_writer=unconfirmed_writer,
        s_min=s_min,
        cleanup_floor=cleanup_floor,
    )
    stored = body.model_dump_json(exclude_none=True).encode("utf-8")
    carries_values = rows is not None and any(row.value is not None for row in rows)
    return stored if carries_values else _within_bound(stored, table_name)


@dataclass(frozen=True, slots=True)
class _KeyedRecordForm:
    """the stored form of a keyed change (:func:`epoch_change_body` over rows) in its fixed pieces, for a record
    assembled from rows already in JSON (a restore's, in a worker process: :mod:`threetears.core.change_note_fold`).

    A record is ``head + <table as a JSON string> + middle + <rows, comma-separated> + tail``; a row is
    ``row_head + <its ids as a JSON array> [+ row_columns + <its columns as a JSON object>] + row_tail``. Taken from
    :func:`epoch_change_body`'s own output (:func:`keyed_record_form`), so the two never differ in form.

    :ivar head: before the table
    :ivar middle: between the table and the rows
    :ivar tail: after the rows
    :ivar row_head: a row's start, before its ids
    :ivar row_columns: between a row's ids and its columns, for a row carrying any
    :ivar row_tail: a row's end
    :ivar max_bytes: the most one record may take, its overhead kept free under the record bound
    """

    head: bytes
    middle: bytes
    tail: bytes
    row_head: bytes
    row_columns: bytes
    row_tail: bytes
    max_bytes: int


def keyed_record_form(database: str | None = None, *, s_min: int | None = None) -> _KeyedRecordForm:
    """the stored form of a keyed change in its fixed pieces, cut out of :func:`epoch_change_body`'s output.

    :param database: the database its records name (:attr:`EpochChange.database`), carried in the form's tail
    :ptype database: str | None
    :param s_min: the lowest row epoch its records name (:attr:`EpochChange.s_min`): a restore from the tables names
        its floor, the lowest stamp any row it read may carry
    :ptype s_min: int | None
    :return: the form
    :rtype: _KeyedRecordForm
    """
    table, ids = "T", ["I"]
    columns: dict[str, str | None] = {"C": "V"}
    bare = ChangedRow(ids=ids).model_dump_json(exclude_none=True).encode("utf-8")
    carried = ChangedRow(ids=ids, columns=columns).model_dump_json(exclude_none=True).encode("utf-8")
    body = epoch_change_body(table, [ChangedRow(ids=ids)], database=database, s_min=s_min)
    ids_json, table_json = json.dumps(ids, separators=(",", ":")).encode(), json.dumps(table).encode()
    columns_json = json.dumps(columns, separators=(",", ":")).encode()
    head, _, rest = body.partition(table_json)
    middle, _, tail = rest.partition(bare)
    row_head, _, row_tail = bare.partition(ids_json)
    row_columns, _, carried_tail = carried.removeprefix(row_head + ids_json).partition(columns_json)
    if carried_tail != row_tail or head + table_json + middle + bare + tail != body:
        raise RuntimeError(f"a keyed record's form could not be cut out of {body!r} and {carried!r}")
    return _KeyedRecordForm(
        head=head,
        middle=middle,
        tail=tail,
        row_head=row_head,
        row_columns=row_columns,
        row_tail=row_tail,
        max_bytes=_EPOCH_CHANGE_MAX_RECORD_BYTES - _RECORD_OVERHEAD_BYTES,
    )


def read_epoch_change_body(raw: bytes, table_name: str, token: str) -> EpochChange:
    """the change a log stored as ``raw`` at the epoch ``token``, checked to be ``table_name``'s.

    :param raw: the stored body
    :ptype raw: bytes
    :param table_name: the table it must be of
    :ptype table_name: str
    :param token: the epoch the log numbered it
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed, or another table's
    """
    try:
        body = _EpochBody.model_validate_json(raw)
        record = EpochChange(
            table=body.table,
            token=token,
            whole_table=body.whole_table,
            keys_unknown=body.keys_unknown,
            rows=body.rows,
            partition=body.partition,
            prefixes=body.prefixes,
            reason=body.reason,
            noted_between=body.noted_between,
            database=body.database,
            # an unconfirmed file is read under the one name every reader of a record's file takes, marked so: each
            # that applies a file checks the mark (module docstring)
            delta_file=body.delta_file if body.unconfirmed_delta_file is None else body.unconfirmed_delta_file,
            scopes=body.scopes,
            delta_file_unconfirmed=body.unconfirmed_delta_file is not None,
            unconfirmed_writer=body.unconfirmed_writer,
            s_min=body.s_min,
            cleanup_floor=body.cleanup_floor,
        )
        if body.unconfirmed_delta_file is not None and body.delta_file is not None:
            raise ValueError("a record names a delta file confirmed or unconfirmed, never both")
    except (ValidationError, ValueError) as exc:
        raise GenerationUnavailableError(
            f"the change record of {table_name!r} at {token!r} is malformed: {exc}"
        ) from exc
    if record.table != table_name:
        raise GenerationUnavailableError(f"the change record at {token!r} is of {record.table!r}, not {table_name!r}")
    return record


def read_logged_epoch_change(raw: bytes, token: str) -> EpochChange:
    """the change a log stored as ``raw`` at the epoch ``token``, of whichever table it names.

    For a reader that walks the log by sequence rather than by table: the change-note scrub.

    :param raw: the stored body
    :ptype raw: bytes
    :param token: the epoch the log numbered it
    :ptype token: str
    :return: the record
    :rtype: EpochChange
    :raises GenerationUnavailableError: when it is malformed
    """
    try:
        body = _EpochBody.model_validate_json(raw)
    except ValidationError as exc:
        raise GenerationUnavailableError(f"the change record at {token!r} is malformed: {exc}") from exc
    return read_epoch_change_body(raw, body.table, token)
