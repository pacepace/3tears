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
  the scope a restore reads "what changed since epoch X" by, and its ``(scope, epoch, key...)`` index answers that
  read by its ``(scope, epoch)`` lead (the scope hashed on YugabyteDB, so the rising end of the epochs spreads across
  scopes; :mod:`~threetears.core.collections.table_restore`) and is what quiet-time cleanup pages a scope by (append-only
  chunk 6, :mod:`~threetears.core.collections.quiet_cleanup`). The index carries the restore's declaration as its
  comment (:func:`restore_declaration`), so the keeper that restores a lost log finds every versioned table, and how to
  read it, from the catalog alone: it runs in the hub, which imports no consumer's table classes.
- **Stamped** (:data:`ROW_STAMPED`), the small tables updated in place (``lsf_runs``, ``state_fingerprints``): one row a
  key, carrying its last write's epoch. A restore reads "what changed since epoch X" from them by a ``(bucket, epoch)``
  index (:func:`bucket_epoch_index_name`): the bucket a stored column the database derives from the key's hash
  (:data:`BUCKET_COLUMN`, :func:`bucket_sql`), one of :data:`RESTORE_BUCKETS`, so the rising end of the epochs is
  spread over that many ranges (one tablet each on YugabyteDB, :func:`bucket_split_sql`) and a restore reads each
  bucket's ``epoch >= X`` as one range. The index carries the restore's declaration as its comment
  (:func:`stamped_restore_declaration`), as a versioned table's does.

**Append-only (versions).** A versioned table is never upserted and never deleted from: a write appends the key's new
version (an ``INSERT`` with no ``ON CONFLICT``, its own epoch, so it never meets an older version's), and a delete
appends a "gone" marker, the key at the delete's epoch with :data:`GONE_COLUMN` true. The marker carries the key, the
scope and the epoch alone: the table's other required columns are required of a live version only (a ``CHECK`` of the
stored table, ``gone OR (c IS NOT NULL AND ...)``), so the database still refuses a live version missing one. A whole
run deleted (a key prefix) is one row of the run-marker table (:mod:`~threetears.core.ddl.run_markers`), never a
marker a row.

**Every scope a stamped write's rows name is touched before its COMMIT, by 3tears itself.** A restore finds the scopes
changed since X by ``last_epoch`` alone, so a version whose scope ``scope_prefix + v`` its transaction did not raise is
never read back, silently, and the scrub cannot see it. So no writer is trusted to name its scopes: the store's one
stamped-write path (``broker_generation.stamped_send``, every write ``SqlL3Backend`` makes of a versioned table) reads the
distinct scope values of the rows it sends, by Arrow on the building thread, and hands them to the transaction the write
joins, which raises each one's ``last_epoch`` to its highest stamp just before its COMMIT
(:func:`touch_last_epoch_sql`): one statement a batch of scopes, one index entry a scope a transaction, never one a row. A
versioned write that joins no transaction to touch them in is refused (:class:`UntouchedScopesError`). A load commits
its own transactions, so it touches its scopes in one of their own before its rows (``SqlL3Backend.load_files``; the
hub's broker for a load it runs).

**A write that carries no epoch is refused**: by 3tears before it is sent (:class:`UnstampedWriteError`), and by the
database for any write that bypasses 3tears (the column is ``NOT NULL`` with no default).

A leaf module: the collections and backends both import it, and it imports neither.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from threetears.interfaces.keyed.filters import Fragment

__all__ = [
    "AppendOnlyWriteError",
    "BUCKET_COLUMN",
    "EPOCH_COLUMN",
    "GONE_COLUMN",
    "MAX_TRANSACTION_AGE_MS",
    "ROW_STAMPED",
    "RowEpoch",
    "TransactionTooOldError",
    "UnstampedWriteError",
    "UntouchedScopesError",
    "cleanup_floor",
    "epoch_wall_ms",
    "next_epoch",
    "RESTORE_BUCKETS",
    "RESTORE_DECLARATION_MARK",
    "invalidation_columns_of",
    "bucket_epoch_index_name",
    "bucket_split_sql",
    "bucket_sql",
    "key_hash_sql",
    "restore_declaration",
    "declared_versions",
    "stamped_restore_declaration",
    "restore_floor",
    "scope_epoch_index_name",
    "touch_last_epoch_sql",
    "observe_epoch",
    "rising_epoch_assignment",
    "rising_stamp",
    "row_versions",
    "stamp_of_values",
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


class UntouchedScopesError(RuntimeError):
    """a stamped write of a versioned table on a connection no transaction holds to touch its scopes in before its
    COMMIT: refused before anything is sent, since a restore would never hear its rows (the module's "Every scope a
    stamped write's rows name is touched")."""


class TransactionTooOldError(RuntimeError):
    """a transaction whose first stamp is older than :data:`MAX_TRANSACTION_AGE_MS` at its COMMIT: refused, rolled
    back, so no committed row is older than W at its commit and a restore's margin holds."""


@dataclass(frozen=True, slots=True)
class RowEpoch:
    """a table's row-epoch declaration (``TableSchema(row_epoch=...)``): :func:`row_versions` or :data:`ROW_STAMPED`.

    :ivar versions: whether every write keeps its own version of the key (the stored key the key and then the epoch);
        False for a table of one row a key, updated in place and stamped
    :ivar scope: for versions, the column a restore reads "changed since epoch X" by, a scope at a time (the state a row
        is of, as the writer's seqlock names scopes) through its ``(scope, epoch)`` index. None for stamped
    :ivar epochs: for versions, the scope epochs table (``ScopeEpochs``) of the table's schema whose scopes name the
        table's: a restore reads the scopes changed since X from it (``last_epoch``), never by scanning this table
    :ivar scope_prefix: how that table names a scope of this one: ``prefix + <the row's scope value>`` (reports:
        ``state:`` before ``join_state_code``). A write of a row of scope value ``v`` touches the scope
        ``scope_prefix + v`` (``ScopeEpochs.touch``), so its stamp raises that scope's ``last_epoch``
    """

    versions: bool
    scope: str | None = None
    epochs: str | None = None
    scope_prefix: str = ""

    def __post_init__(self) -> None:
        """refuse a versioned table without a scope, or a stamped one with one.

        :return: nothing
        :rtype: None
        :raises ValueError: when ``scope`` does not fit ``versions``
        """
        if self.versions and not (self.scope and self.epochs):
            raise ValueError(
                "a versioned table names the scope column a restore reads it by, and its scope epochs table"
            )
        if not self.versions and (self.scope is not None or self.epochs is not None or self.scope_prefix):
            raise ValueError("a stamped table keeps one row a key and is restored by its key; give no scope")


#: the scope epochs table a versioned table's scopes are named in, by default (``ScopeEpochs``' own default name)
_DEFAULT_EPOCHS: Final[str] = "scope_epochs"


def row_versions(scope: str, *, epochs: str = _DEFAULT_EPOCHS, scope_prefix: str = "") -> RowEpoch:
    """a versioned table's declaration: every write keeps its own version of the key, newest first.

    :param scope: the column a restore reads the table by, a scope at a time
    :ptype scope: str
    :param epochs: the scope epochs table of the table's schema whose scopes name this table's
    :ptype epochs: str
    :param scope_prefix: what that table writes before a row's scope value (``state:``)
    :ptype scope_prefix: str
    :return: the declaration
    :rtype: RowEpoch
    """
    return RowEpoch(versions=True, scope=scope, epochs=epochs, scope_prefix=scope_prefix)


#: what a restore declaration's comment opens with: the keeper finds the declared indexes by it
RESTORE_DECLARATION_MARK: Final[str] = "threetears_restore"


def scope_epoch_index_name(table: str) -> str:
    """the name of a versioned table's ``(scope, epoch)`` index: ``<table>_scope_epoch``, a digest past 63 characters.

    :param table: the table
    :ptype table: str
    :return: the name
    :rtype: str
    """
    name = f"{table}_scope_epoch"
    if len(name) <= 63:
        return name
    # an identifier is bounded (NAMEDATALEN): a digest keeps it one per table
    return f"scope_epoch_{hashlib.sha256(table.encode()).hexdigest()[:24]}"


def _declared_columns(declared: dict[str, Any], tenancy: tuple[str, str | None], columns: tuple[str, ...]) -> None:
    """name a table's declared invalidation columns in its restore declaration (``"columns"``), so a restore's records
    carry each one as the change notes did (:func:`invalidation_columns_of`). Named only when they hold a column the
    tenancy does not: a table whose one invalidation column is its tenant column (or that has none) keeps its comment
    byte for byte as before.

    :param declared: the declaration being built
    :ptype declared: dict[str, Any]
    :param tenancy: the table's tenancy kind and its tenant column (or None)
    :ptype tenancy: tuple[str, str | None]
    :param columns: its declared invalidation columns, the tenant column among them or not
    :ptype columns: tuple[str, ...]
    :return: nothing
    :rtype: None
    """
    kind, column = tenancy
    named = tuple(dict.fromkeys((*columns, *((column,) if kind == "column" and column else ()))))
    if named != ((column,) if kind == "column" and column else ()):
        declared["columns"] = list(named)


def invalidation_columns_of(value: dict[str, Any]) -> tuple[str, ...]:
    """the declared invalidation columns a restore declaration names (:func:`restore_declaration`,
    :func:`stamped_restore_declaration`): its ``"columns"``, or a column tenancy's tenant column alone for a declaration
    that names none; the tenant column always among them.

    :param value: the declaration, parsed
    :ptype value: dict[str, Any]
    :return: the columns, in declared order
    :rtype: tuple[str, ...]
    """
    tenant = str(value["tenant_column"]) if value.get("tenancy") == "column" and value.get("tenant_column") else None
    named = tuple(str(each) for each in value.get("columns", ()))
    return tuple(dict.fromkeys((*named, *((tenant,) if tenant else ()))))


def restore_declaration(
    declared: RowEpoch,
    keys: tuple[str, ...],
    instants: tuple[str, ...],
    tenancy: tuple[str, str | None],
    columns: tuple[str, ...] = (),
) -> str:
    """what a versioned table's ``(scope, epoch)`` index says of the table as its comment, for the restore to read it by.

    JSON: the mark, the scope column, the key columns in order (``instants`` those a record spells as their UTC text),
    the scope epochs table and prefix, the tenancy (``schema``, ``column`` with its column, or none), and the declared
    invalidation columns when they name one the tenancy does not (``"columns"``, :func:`_declared_columns`). The keeper
    reads it back from the catalog (:mod:`~threetears.core.collections.table_restore`).

    :param declared: the table's declaration
    :ptype declared: RowEpoch
    :param keys: its primary key's columns, in order (the epoch not among them)
    :ptype keys: tuple[str, ...]
    :param instants: those of them that are ``timestamptz``
    :ptype instants: tuple[str, ...]
    :param tenancy: the table's tenancy kind and its tenant column (or None)
    :ptype tenancy: tuple[str, str | None]
    :param columns: its declared invalidation columns (``TableSchema.invalidation_columns``)
    :ptype columns: tuple[str, ...]
    :return: the comment's text
    :rtype: str
    :raises ValueError: when the declaration is not a versioned table's
    """
    if not declared.versions or declared.scope is None or declared.epochs is None:
        raise ValueError("only a versioned table's (scope, epoch) index carries a restore declaration")
    kind, column = tenancy
    value: dict[str, Any] = {
        RESTORE_DECLARATION_MARK: 1,
        "scope": declared.scope,
        "keys": list(keys),
        "instants": list(instants),
        "epochs": declared.epochs,
        "scope_prefix": declared.scope_prefix,
        "tenancy": kind,
        "tenant_column": column,
    }
    _declared_columns(value, tenancy, columns)
    return json.dumps(value, separators=(",", ":"))


def declared_versions(text: str | None) -> RowEpoch | None:
    """the versioned table's declaration a ``(scope, epoch)`` index's comment carries (:func:`restore_declaration`):
    its scope column, scope epochs table and prefix, for whatever writes the table without its class (the hub's broker,
    which loads files into it and touches the scopes they name). None for a comment that is no restore declaration.

    :param text: the index's comment, or None when it has none
    :ptype text: str | None
    :return: the declaration, or None
    :rtype: RowEpoch | None
    """
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        # NOSILENT: an index comment that is not JSON is no restore declaration; the answer None says so
        return None
    if not isinstance(value, dict) or value.get(RESTORE_DECLARATION_MARK) != 1:
        return None
    return row_versions(str(value["scope"]), epochs=str(value["epochs"]), scope_prefix=str(value["scope_prefix"]))


def restore_floor(mark_ms: int) -> int:
    """X, the lowest epoch a restore from a keeper's mark taken at ``mark_ms`` reads: ``(mark_ms - M) << 15``.

    Every record appended after the mark's log position carries stamps at or above it (:data:`_RESTORE_MARGIN_MS`).

    :param mark_ms: the mark's time, in wall milliseconds
    :ptype mark_ms: int
    :return: the floor
    :rtype: int
    """
    return max(0, mark_ms - _RESTORE_MARGIN_MS) << _LOGICAL_BITS


def touch_last_epoch_sql(quoted_table: str) -> str:
    """the statement that touches scopes of a scope epochs table with a transaction's highest stamp, just before its
    COMMIT: each scope's ``last_epoch`` raised to it (never lowered), the scope's row made when it has none.

    ``$1`` is the stamp, ``$2`` the scopes (``text[]``), unnested in the select list (no ``FROM``, so the broker's
    inspector names no table but the target). A row it makes carries epoch 0 and no pending mark: to a reader exactly a
    scope no write has changed (``EpochSnapshot.epoch`` answers 0 for one), so only the restore and the next writer's
    ``begin``, which read ``last_epoch``, see it. WHY an upsert and not ``UPDATE``: the scope a writer forgot to touch
    may have no row, and an ``UPDATE`` of none would miss it again without a word.

    :param quoted_table: the scope epochs table, quoted
    :ptype quoted_table: str
    :return: the statement
    :rtype: str
    """
    return (
        f"INSERT INTO {quoted_table} (scope, epoch, last_epoch, date_created, date_updated) "  # noqa: S608 - a quoted identifier
        "SELECT unnest($2::text[]), 0, $1::bigint, now(), now() ON CONFLICT (scope) DO UPDATE SET "
        f"last_epoch = GREATEST(COALESCE({quoted_table}.last_epoch, 0), EXCLUDED.last_epoch)"
    )


#: a stamped table's declaration: one row a key, carrying its last write's epoch
ROW_STAMPED: Final = RowEpoch(versions=False)


#: the stored column a stamped table's restore index leads with: the key's hash, cut to one of :data:`RESTORE_BUCKETS`.
#: ``SMALLINT GENERATED ALWAYS AS (...) STORED``, so the database derives it on every write (no writer names it: a write
#: that does is refused) and it is in the L3 table only, never a cache's or an entity's
BUCKET_COLUMN: Final[str] = "epoch_bucket"

#: how many buckets a stamped table's restore index spreads its rows over. WHY 16 (Pace 2026-11-25, research-shared-clock
#: risk 3, 8 to 16): an index led by the epoch alone sends every write to the rising end of one YugabyteDB tablet; 16
#: buckets are 16 rising ends, one tablet each (:func:`bucket_split_sql`), more than the nodes of any of our clusters, so
#: the writes land on every node. A restore pays one short range read a bucket (16 statements a table, a handful of
#: stamped tables), so a few more would buy no spread and cost reads. A power of two: the bucket is the hash's low bits
RESTORE_BUCKETS: Final[int] = 16

if RESTORE_BUCKETS & (RESTORE_BUCKETS - 1):
    raise ValueError(
        f"RESTORE_BUCKETS must be a power of two (the bucket is the hash's low bits), got {RESTORE_BUCKETS}"
    )

#: the immutable hash of a key column's value, by its kind, the same on Postgres and YugabyteDB (the functions each
#: type's hash operator class names). A generated column takes only immutable expressions: a ``timestamptz`` is hashed as
#: its UTC ``timestamp`` (``timezone('UTC', ts)`` is immutable, its text is not), a ``bool`` as an integer
_KEY_HASHES: Final[dict[str, str]] = {
    "text": "hashtext({})",
    "uuid": "uuid_hash({})",
    "int4": "hashint4({})",
    "int8": "hashint8({})",
    "timestamptz": "timestamp_hash({} AT TIME ZONE 'UTC')",
    "numeric": "hash_numeric({})",
    "bool": "hashint4({}::integer)",
    "enum": "hashenum({})",
    "bytea": "hashtext(encode({}, 'hex'))",
    "jsonb": "jsonb_hash({})",
}


def bucket_epoch_index_name(table: str) -> str:
    """the name of a stamped table's ``(bucket, epoch)`` index: ``<table>_bucket_epoch``, a digest past 63 characters.
    A table named with its schema (``platform_audit.audit_events``, the hub's audit table) names its index by the bare
    table: an index lives in its table's schema, and its name is one plain identifier.

    :param table: the table, bare or schema-qualified
    :ptype table: str
    :return: the name
    :rtype: str
    """
    bare = table.rpartition(".")[2]
    name = f"{bare}_bucket_epoch"
    if len(name) <= 63:
        return name
    return f"bucket_epoch_{hashlib.sha256(bare.encode()).hexdigest()[:24]}"


def key_hash_sql(column: str, kind: str) -> str:
    """the immutable hash (an ``integer``) of a key column's value, as a generated column may compute it.

    :param column: the column, quoted as the DDL quotes it
    :ptype column: str
    :param kind: its kind: one of ``text``, ``uuid``, ``int4``, ``int8``, ``timestamptz``, ``numeric``, ``bool``,
        ``enum``, ``bytea``, ``jsonb``
    :ptype kind: str
    :return: the expression
    :rtype: str
    :raises ValueError: when the kind has no hash here
    """
    template = _KEY_HASHES.get(kind)
    if template is None:
        raise ValueError(f"a stamped table's key column of kind {kind!r} has no hash a restore bucket can be made of")
    return template.format(column)


def bucket_sql(hashes: tuple[str, ...]) -> str:
    """a stamped table's :data:`BUCKET_COLUMN` expression: its key columns' hashes folded into one, its low bits.

    One column: its hash. More: each folded in as ``hashint8(<so far>::bigint * 31 + <next>)``, so the order of the
    columns counts (a key ``(a, b)`` and ``(b, a)`` part) and nothing overflows (an ``integer`` times 31 plus an
    ``integer`` fits a ``bigint``). ``& (RESTORE_BUCKETS - 1)`` keeps the low bits of a negative hash too, ``0`` to
    ``RESTORE_BUCKETS - 1``.

    :param hashes: each key column's hash (:func:`key_hash_sql`), in the key's order
    :ptype hashes: tuple[str, ...]
    :return: the expression, a ``smallint``
    :rtype: str
    :raises ValueError: when there is no key column
    """
    if not hashes:
        raise ValueError("a stamped table's restore bucket hashes its key; none given")
    folded = hashes[0]
    for term in hashes[1:]:
        folded = f"hashint8(({folded})::bigint * 31 + {term})"
    return f"(({folded}) & {RESTORE_BUCKETS - 1})::smallint"


def bucket_split_sql() -> str:
    """the clause a stamped table's restore index is split by on YugabyteDB: a tablet a bucket.

    WHY a range split, not the bucket hashed: YugabyteDB hashes a hashed column's value to a 16-bit code and gives each
    tablet a range of codes, and 16 small values' codes fall where they fall (measured on the local YugabyteDB
    2026.1: the codes of 0 to 15 land in 10 of 16 equal ranges, four buckets in one). Range-sharded and split at every
    bucket value, each bucket is its own tablet, whatever the hash.

    :return: ``SPLIT AT VALUES ((1), (2), ...)``
    :rtype: str
    """
    return "SPLIT AT VALUES (" + ", ".join(f"({value})" for value in range(1, RESTORE_BUCKETS)) + ")"


def stamped_restore_declaration(
    keys: tuple[str, ...],
    instants: tuple[str, ...],
    tenancy: tuple[str, str | None],
    moveable: tuple[str, ...] = (),
    columns: tuple[str, ...] = (),
) -> str:
    """what a stamped table's ``(bucket, epoch)`` index says of the table as its comment, for the restore to read it by.

    JSON: the mark, ``"kind": "stamped"``, the bucket column and the number of buckets, the key columns in order
    (``instants`` those a record spells as their UTC text), and the tenancy. A versioned table's declaration
    (:func:`restore_declaration`) names no kind. A table a write may move between tenants (``TableSchema.moveable``,
    :mod:`~threetears.core.ddl.tenant_moves`) names its moveable columns too (``"moveable"``), so a restore reads back
    the tenant each moved row left; a table with none carries no such key, its comment byte for byte as before.

    :param keys: the table's primary key's columns, in order
    :ptype keys: tuple[str, ...]
    :param instants: those of them that are ``timestamptz``
    :ptype instants: tuple[str, ...]
    :param tenancy: the table's tenancy kind and its tenant column (or None)
    :ptype tenancy: tuple[str, str | None]
    :param moveable: the columns a write may move the row's tenant by; none for most tables
    :ptype moveable: tuple[str, ...]
    :param columns: its declared invalidation columns (``TableSchema.invalidation_columns``), named as
        :func:`restore_declaration` names them
    :ptype columns: tuple[str, ...]
    :return: the comment's text
    :rtype: str
    """
    kind, column = tenancy
    declared: dict[str, Any] = {
        RESTORE_DECLARATION_MARK: 1,
        "kind": "stamped",
        "bucket": BUCKET_COLUMN,
        "buckets": RESTORE_BUCKETS,
        "keys": list(keys),
        "instants": list(instants),
        "tenancy": kind,
        "tenant_column": column,
    }
    if moveable:
        declared["moveable"] = list(moveable)
    _declared_columns(declared, tenancy, columns)
    return json.dumps(declared, separators=(",", ":"))


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


def cleanup_floor(s_min: int) -> int:
    """the lowest epoch quiet-time cleanup keeps, given the lowest stamp (``s_min``) of the oldest record the change log
    holds: ``(wall(s_min) - 2M) << 15``. Every version, "gone" marker and run marker at or above it is one a restore
    could read; cleanup removes only what lies below (:mod:`~threetears.core.collections.quiet_cleanup`).

    WHY two margins (append-only chunk 6, ``design-append-only-l3.md`` decision 5): a restore from a keeper's mark P,
    taken at T, reads ``epoch >= (T - M) << 15`` (:data:`_RESTORE_MARGIN_MS`). A mark worth restoring from names a
    position the log still holds, so it was taken after the log's oldest record was appended, and that record's commit
    stamped ``s_min`` at most S (inside M) behind its append on the writer's clock: T >= wall(s_min) - M, so the restore
    reads from at or above ``wall(s_min) - 2M``. A mark older than the log's oldest record restores nothing a follower
    can still use: one that far behind is already behind (``record_lost``) and reloads. Keeping 20 minutes more of
    versions costs nothing a reader sees.

    :param s_min: the oldest held record's lowest stamp
    :ptype s_min: int
    :return: the floor
    :rtype: int
    :raises ValueError: when ``s_min`` is no epoch
    """
    if not 0 <= s_min <= _MAX_EPOCH:
        raise ValueError(f"an epoch is a positive BIGINT; got {s_min}")
    return max(0, epoch_wall_ms(s_min) - 2 * _RESTORE_MARGIN_MS) << _LOGICAL_BITS


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


@dataclass(frozen=True)
class _RisingStamp(Fragment):
    """a keyed ``UPDATE``'s value for a stamped table's epoch, ``GREATEST("epoch" + 1, $1::bigint)`` bound to
    :attr:`stamp` (:func:`rising_stamp`): a :class:`~threetears.interfaces.keyed.Fragment` the update renders as any
    other, and the stamp the update sends on the wire as its write's ``s_min`` (:func:`stamp_of_values`).

    WHY a kind of its own: the keyed layer knows a table by its name alone; the value is how the update learns that it
    stamps, and with which stamp, so the broker records the write's ``s_min`` like every other stamped write.

    :param stamp: the write's stamp, a positive BIGINT
    :ptype stamp: int
    """

    stamp: int = 0


def rising_stamp(stamp: int, quoted_epoch: str = '"epoch"') -> _RisingStamp:
    """a keyed ``UPDATE``'s value for a stamped table's epoch: ``GREATEST(<epoch> + 1, <stamp>)``, so the row's stamp
    never goes back (as :func:`rising_epoch_assignment` keeps an upsert's).

    :param stamp: the write's stamp, a positive BIGINT
    :ptype stamp: int
    :param quoted_epoch: the epoch column, quoted as the statement quotes it
    :ptype quoted_epoch: str
    :return: the value
    :rtype: _RisingStamp
    """
    return _RisingStamp(f"GREATEST({quoted_epoch} + 1, $1::bigint)", (stamp,), stamp=stamp)


def stamp_of_values(values: Mapping[str, Any]) -> int | None:
    """the stamp a keyed ``UPDATE``'s ``values`` carry (a :class:`_RisingStamp` epoch), sent as its write's ``s_min``.

    :param values: column -> its new value, as ``update_keyed`` takes them
    :ptype values: Mapping[str, Any]
    :return: the stamp; None when the update stamps nothing
    :rtype: int | None
    """
    value = values.get(EPOCH_COLUMN)
    return value.stamp if isinstance(value, _RisingStamp) else None


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
