"""the change-note ring's SQL: its three note tables, its ring row, the functions and each table's triggers.

A noted table's row trigger writes one note per changed row -- the table, the row's key and its declared
invalidation columns, never its data -- into the ring's current note table, in the write's own transaction
(:mod:`threetears.core.collections.change_notes`). Below every package that runs it, as the write marks'
SQL is, so the migration that creates the ring and the keeper that installs the triggers import neither each
other.

**No contention by construction.** A note is an insert of a fresh row whose key is a random uuid (the
first primary-key column, so YugabyteDB hash-shards it); no writer touches a row another writer touches,
nothing counts, no index is kept on a rising value, and no constraint spans writers. The notes are not
ordered and need not be: a restore needs every note, not their order.

**Which table is current.** The ring row names it, and only the keeper writes that row, once per switch.
No save reads it. A writing process keeps a cached copy (:class:`threetears.core.utils.change_note_slot.ChangeNoteSlot`)
and hands it to each connection as a session setting, :data:`NOTE_SLOT_SETTING`: ``"<slot>:<valid until>"``,
the slot the ring named when the process last read it and the database clock after which the copy may no
longer be used (the read's ``now()`` plus :func:`slot_max_age` of the fence). At a statement's first noted row
the trigger (:data:`_RING_FUNCTION`) takes the slot from that setting while the database clock is before its
``valid until``, and keeps it for the rest of the statement in a transaction-local setting. Only when the
setting is absent, malformed or expired (raw SQL outside 3tears, a process that could not refresh) does it
read the ring row, and the note it writes then says so (``ring_read``): the keeper counts those at each switch.

**Why a cached slot is safe.** A copy names a table that was current when it was read, and is used only
until ``valid until``: at most :func:`slot_max_age` (half the fence) after. The ring never switches twice
within its wait (at least the fence the writers were told, ``fence_seconds``), and the keeper empties a
table only after the second switch away from it. So a statement that took its slot from a copy writes
into the current or the previous table, never into one being emptied, as long as the statement (its
transaction) ends within the other half of the fence. Clocks: the validity is read and checked on the
database's clock, so no process clock enters it; between YugabyteDB nodes the skew is bounded far below
the margin.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Final

from threetears.core.tenancy import Tenancy
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "DEFAULT_NOTE_FENCE",
    "NOTE_PUT_FUNCTION",
    "NOTE_REACH_SETTING",
    "NOTE_RING_SIZE",
    "NOTE_SLOT_SETTING",
    "NOTE_TRIGGERS",
    "NOTE_TRUNCATE_TRIGGER",
    "NOTE_WHOLE_SETTING",
    "WHOLE_COPY_ROWS",
    "MARKS_TABLE",
    "RING_TABLE",
    "change_notes_ddl",
    "inexact_note_columns",
    "note_function_name",
    "note_reach_text",
    "slot_max_age",
    "note_table",
    "note_trigger_sql",
    "legacy_truncate_trigger_drop_sql",
    "ring_read_sql",
    "ring_slot_sql",
    "ring_switch_sql",
    "ring_update_sql",
    "trigger_function_body",
    "trigger_function_sql",
]

#: the SQL form of :mod:`threetears.core.tenancy`'s token rule, so a note's tenant and a record's agree
_TOKEN_SQL: Final = "^[A-Za-z0-9_-]{1,62}$"
_DIGEST_SQL: Final = 24

#: how many note tables the ring keeps: the current one, the previous one, and one emptied and waiting
NOTE_RING_SIZE: Final = 3

#: how long after a switch the ring may switch again, and so how long before the table two switches back is
#: emptied: well beyond any transaction's run time plus a cached slot's age (:func:`slot_max_age`)
DEFAULT_NOTE_FENCE: Final = timedelta(minutes=10)

#: the session setting a writing connection carries: ``"<slot>:<valid until, database epoch seconds>"``
NOTE_SLOT_SETTING: Final = "threetears.change_note_slot"

#: the ring's note tables are this name and their place in the ring: ``threetears_change_notes_0`` ..
_NOTE_TABLE_PREFIX: Final = "threetears_change_notes_"

#: the one-row table naming the ring's current note table, and what the keeper knows of the coverage
RING_TABLE: Final = "threetears_change_note_ring"

#: the keeper's marks: the change log's position, by the database's clock, every so often. A restore that
#: cannot take the whole coverage takes the notes after a mark, and the mark's position is then its floor
MARKS_TABLE: Final = "threetears_change_note_marks"

#: the keeper's marks table, ``{schema}`` its schema (quoted): written out whole, so it declares its key where it is read
_MARKS_TABLE_DDL: Final = (
    "CREATE TABLE IF NOT EXISTS {schema}.threetears_change_note_marks ("
    "id UUID PRIMARY KEY DEFAULT gen_random_uuid(), log_incarnation TEXT NOT NULL, seq BIGINT NOT NULL, "
    "at TIMESTAMPTZ NOT NULL)"
)

#: the ring's one-row table, ``{schema}`` its schema (quoted): written out whole, so it declares its key where it is read
_RING_TABLE_DDL: Final = (
    "CREATE TABLE IF NOT EXISTS {schema}.threetears_change_note_ring ("
    "id SMALLINT PRIMARY KEY CHECK (id = 1), "
    "current_ring SMALLINT NOT NULL, "
    "switch_no BIGINT NOT NULL, "
    "switched_at TIMESTAMPTZ NOT NULL, "
    # when the previous table became current: the coverage's notes start there (NULL before a first switch)
    "previous_switched_at TIMESTAMPTZ, "
    "switched_seq TEXT, "
    "floor TEXT, "
    "fence_seq TEXT, "
    "emptied BOOLEAN NOT NULL, "
    "log_incarnation TEXT, "
    # the fence the writers were told: a cached slot is used at most slot_max_age(fence) after its read
    "fence_seconds DOUBLE PRECISION NOT NULL, "
    # the least time from the last switch to the next: covers the fence told before it as well
    "wait_seconds DOUBLE PRECISION NOT NULL)"
)

#: the function every table's trigger writes a note through; it reads the ring once per statement
_RING_FUNCTION: Final = "threetears_change_note_put"

#: the same, by its public name: a schema holding it holds the ring
NOTE_PUT_FUNCTION: Final = _RING_FUNCTION

#: the function a whole-table note is written through, once per transaction and table
_WHOLE_FUNCTION: Final = "threetears_change_note_whole"

#: the triggers on every noted table, all running the table's own function, which names the tenant of a
#: whole-table note by the table's declaration: the row trigger, which fires only outside a transaction marked
#: whole (its ``WHEN`` is checked without calling the function, so a whole load's rows cost no call each); the
#: statement trigger for ``TRUNCATE``; and the statement trigger that writes a whole-marked write's one note
NOTE_TRIGGER: Final = "threetears_change_note"
NOTE_TRUNCATE_TRIGGER: Final = "threetears_change_note_truncate_table"
NOTE_WHOLE_TRIGGER: Final = "threetears_change_note_whole_write"
NOTE_TRIGGERS: Final = (NOTE_TRIGGER, NOTE_TRUNCATE_TRIGGER, NOTE_WHOLE_TRIGGER)

#: the ``TRUNCATE`` trigger an install before the tenant of a whole-table note was named gave a table: it ran a
#: shared function that put every truncate on ``_all``. Dropped where the table's own trigger is created
_LEGACY_TRUNCATE_TRIGGER: Final = "threetears_change_note_truncate"

#: each table's own trigger function is this prefix and a digest of its schema and name
_TABLE_FUNCTION_PREFIX: Final = "threetears_note_"

#: the transaction-local setting a whole-table write (a file load) sets, so every row it copies is one
#: whole-table note rather than a note each: the load's epoch names the whole table too
NOTE_WHOLE_SETTING: Final = "threetears.note_whole"

#: the setting a whole-table write names its reach in, as JSON (:func:`note_reach_text`): a load's key prefixes
#: (``{"prefixes": [...]}``) or an operator's reset (``{"reason": "reset"}``), as its epoch record names them. A
#: whole-table note takes it as its ``columns`` (a keyed note's invalidation columns), so a restore puts the record
#: back with that reach rather than as the plain whole table. Transaction-local for a load's transactions; a
#: ``TRUNCATE``, which runs alone, sets it for the session and clears it after. Unset or empty: the note names none
NOTE_REACH_SETTING: Final = "threetears.note_reach"

#: the rows a transaction copies from which it is marked whole (:data:`NOTE_WHOLE_SETTING`): one note for the
#: table rather than one a row. The notes name at most this many rows a table to the write-mark reconciler
#: (more are advanced with their keys unknown), so a note a row past it costs a write each and names nothing.
#: The hub's ``l3.tx.copy`` counts a session's copied rows against it; a direct connection's
#: :meth:`~threetears.core.backends.sql.SqlL3Backend.copy_many` its own call's
WHOLE_COPY_ROWS: Final = 10_000


def note_reach_text(*, prefixes: Sequence[Mapping[str, Any]] | None = None, reason: str | None = None) -> str:
    """a whole-table write's reach as :data:`NOTE_REACH_SETTING` holds it, and its whole-table note keeps it.

    The writer sets it in the write's transaction (``SELECT set_config($1, $2, true)``), the broker as a direct
    pool does, so the note a restore reads names what the epoch record named.

    :param prefixes: a load's or a clear's key prefixes, each the leading key columns and their values
    :ptype prefixes: Sequence[Mapping[str, Any]] | None
    :param reason: an operator's reset (``"reset"``)
    :ptype reason: str | None
    :return: the JSON text, ``""`` for no reach
    :rtype: str
    :raises ValueError: when both are named
    """
    if prefixes and reason is not None:
        raise ValueError("a whole-table write names key prefixes or a reason, never both")
    if prefixes:
        return json.dumps(
            {"prefixes": [{str(column): str(value) for column, value in prefix.items()} for prefix in prefixes]},
            separators=(",", ":"),
        )
    if reason is not None:
        return json.dumps({"reason": reason}, separators=(",", ":"))
    return ""


def slot_max_age(fence: timedelta) -> timedelta:
    """how long a cached copy of the ring's current slot may be used after it was read: half the fence.

    The other half is the margin a writing transaction must end within (the module docstring's argument).

    :param fence: the fence the writers were told (the ring row's ``fence_seconds``)
    :ptype fence: timedelta
    :return: the copy's longest use
    :rtype: timedelta
    """
    return fence / 2


#: the ``information_schema`` name of ``timestamptz``, whose note text is written canonically (:data:`_UTC_MICROS`)
_TIMESTAMPTZ: Final = "timestamp with time zone"

#: a ``timestamptz`` as a note names it: UTC, to the microsecond, in ISO form (``2026-11-09T10:22:10.082658+00:00``),
#: whatever the session's ``TimeZone`` and ``DateStyle``; what :func:`~threetears.core.collections.epoch_change.key_text`
#: names an aware ``datetime`` as, so a note and a record name one instant in one spelling
_UTC_MICROS: Final = '\'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"\''

#: the column types whose text form a note carries exactly as an epoch record names the value: a note's key and
#: invalidation columns are written as ``column::text``, a ``timestamptz`` in its canonical form; anything else can
#: name the same value in another spelling than the record, and stays refused: ``boolean`` (``true`` against Python's
#: ``True``), ``timestamp`` and ``date`` (``DateStyle``, and a naive time names no instant), ``numeric``, ``real`` and
#: ``double precision`` (a value's scale and rounding), ``json``/``jsonb`` (key order, spacing), ``character``
#: (padding), ``bytea``, ``interval``, ``time``, arrays and enums (``USER-DEFINED``)
_EXACT_NOTE_TYPES: Final = frozenset(
    {"uuid", "text", "character varying", "smallint", "integer", "bigint", _TIMESTAMPTZ}
)


def inexact_note_columns(types: dict[str, str], columns: Sequence[str]) -> list[tuple[str, str]]:
    """the columns among ``columns`` whose type's text form does not round-trip exactly, with their types.

    :param types: each column's ``information_schema`` ``data_type``
    :ptype types: dict[str, str]
    :param columns: the key and invalidation columns a note would carry
    :ptype columns: Sequence[str]
    :return: ``(column, type)`` for each refused column; a column the table does not have is refused too
    :rtype: list[tuple[str, str]]
    """
    return [(column, types.get(column, "missing")) for column in columns if types.get(column) not in _EXACT_NOTE_TYPES]


def note_table(index: int) -> str:
    """the name of the ring's note table at ``index``.

    :param index: its place in the ring, ``0`` to ``NOTE_RING_SIZE - 1``
    :ptype index: int
    :return: the table's name
    :rtype: str
    :raises ValueError: when ``index`` is not a place in the ring
    """
    if not 0 <= index < NOTE_RING_SIZE:
        raise ValueError(f"the note ring has {NOTE_RING_SIZE} tables; {index} is not one of them")
    return f"{_NOTE_TABLE_PREFIX}{index}"


def ring_read_sql(schema: str) -> str:
    """the statement reading the ring's one row (its key, ``id = 1``), and the database's clock.

    :param schema: the schema holding the ring, a TRUSTED identifier
    :ptype schema: str
    :return: the ``SELECT``
    :rtype: str
    """
    ring = f"{quote_identifier(schema)}.{quote_identifier(RING_TABLE)}"
    return (
        "SELECT current_ring, switch_no, switched_at, switched_seq, floor, fence_seq, emptied, log_incarnation, "  # noqa: S608 - trusted identifiers
        f"fence_seconds, wait_seconds, now() AS now FROM {ring} WHERE id = 1"
    )


def ring_slot_sql(schema: str) -> str:
    """the statement a writing process reads its cached slot with: the ring row's current table and fence, and the
    database's clock in epoch seconds.

    :param schema: the schema holding the ring, a TRUSTED identifier
    :ptype schema: str
    :return: the ``SELECT``
    :rtype: str
    """
    ring = f"{quote_identifier(schema)}.{quote_identifier(RING_TABLE)}"
    return f"SELECT current_ring, fence_seconds, extract(epoch FROM now())::float8 AS now FROM {ring} WHERE id = 1"  # noqa: S608 - trusted identifiers


def ring_update_sql(schema: str, assignments: str) -> str:
    """the statement writing ``assignments`` to the ring row, only while it is at the switch ``$1``.

    :param schema: the schema holding the ring, a TRUSTED identifier
    :ptype schema: str
    :param assignments: ``column = $n, ...`` over the ring's own columns, numbered from ``$2``
    :ptype assignments: str
    :return: the ``UPDATE``
    :rtype: str
    """
    ring = f"{quote_identifier(schema)}.{quote_identifier(RING_TABLE)}"
    return f"UPDATE {ring} SET {assignments} WHERE id = 1 AND switch_no = $1"  # noqa: S608 - fixed column names


def ring_switch_sql(schema: str) -> str:
    """the statement switching the ring to the emptied table ``$2``, only while it is at the switch ``$1`` and that
    table's emptying finished: the fence's position becomes the floor, ``$3`` the switch's log position and ``$4``
    the fence told the writers.

    :param schema: the schema holding the ring, a TRUSTED identifier
    :ptype schema: str
    :return: the ``UPDATE``
    :rtype: str
    """
    ring = f"{quote_identifier(schema)}.{quote_identifier(RING_TABLE)}"
    return (
        f"UPDATE {ring} SET current_ring = $2, switch_no = switch_no + 1, switched_at = now(), "  # noqa: S608 - trusted identifiers
        "switched_seq = $3, floor = fence_seq, fence_seq = NULL, emptied = FALSE, previous_switched_at = switched_at, "
        "fence_seconds = $4, wait_seconds = greatest($4, fence_seconds) "
        "WHERE id = 1 AND switch_no = $1 AND emptied"
    )


def change_notes_ddl(schema: str, *, fence: timedelta = DEFAULT_NOTE_FENCE) -> tuple[str, ...]:
    """the statements that create the ring in ``schema``: its note tables, its ring row and its functions.

    Each is idempotent, and a ring row already there keeps its state. Run them one at a time (YugabyteDB
    runs no DDL in a transaction).

    :param schema: the schema holding the ring (the platform schema), a TRUSTED identifier
    :ptype schema: str
    :param fence: the fence a new ring row tells the writers; an existing row keeps its own
    :ptype fence: timedelta
    :return: the statements, in order
    :rtype: tuple[str, ...]
    """
    owner = quote_identifier(schema)
    ring = f"{owner}.{quote_identifier(RING_TABLE)}"
    tables = tuple(
        f"CREATE TABLE IF NOT EXISTS {owner}.{quote_identifier(note_table(index))} ("
        # the uuid key is the first primary-key column, so YugabyteDB hash-shards the inserts
        "id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
        "table_name TEXT NOT NULL, "
        # NULL keys: the note is of the whole table (a truncate, a file load)
        "keys TEXT[], "
        # the keys joined by chr(31), the scrub's lookup: led by the (key_text, table_name) index
        "key_text TEXT, "
        # a keyed note's invalidation columns; a whole-table note's reach (NOTE_REACH_SETTING), or NULL
        "columns JSONB, "
        # the tenant the row belongs to (threetears.core.tenancy); NULL for _all
        "tenant TEXT, "
        # the clock at the insert, not the transaction's start: a late note is told by it
        "noted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(), "
        # TRUE when the writer had no valid cached slot and read the ring row; NULL (stored as nothing) otherwise
        "ring_read BOOLEAN)"
        for index in range(NOTE_RING_SIZE)
    )
    indexes = tuple(
        f"CREATE INDEX IF NOT EXISTS {quote_identifier(note_table(index) + '_key')} "
        f"ON {owner}.{quote_identifier(note_table(index))} (key_text, table_name)"
        for index in range(NOTE_RING_SIZE)
    )
    # a table's notes since a time (the write-mark reconciler's read, every few seconds a marked table): led by the
    # table, then in time, so the read touches that table's recent notes and no other's (without it each read
    # scanned the whole note table). It carries the columns the read answers, so the read never visits the table:
    # on YugabyteDB a visit is a second lookup a note, several times the read's cost. On YugabyteDB the first
    # column is hashed and noted_at a range under it. Created IF NOT EXISTS with the rest, so a ring installed
    # before it gains it at the next install.
    by_table = tuple(
        f"CREATE INDEX IF NOT EXISTS {quote_identifier(note_table(index) + '_table')} "
        f"ON {owner}.{quote_identifier(note_table(index))} (table_name, noted_at) "
        "INCLUDE (key_text, keys, columns, tenant)"
        for index in range(NOTE_RING_SIZE)
    )
    # the few notes the keeper's passes look for among the many, each kept in an index of its own so a look reads
    # those and no other: a table's whole-table notes (the scrub's "is this table noted whole", the restore's
    # whole tables and the rows they cover), and the notes written without a cached slot (counted at each switch).
    # Partial, so they hold only those notes and cost the many nothing.
    sparse = tuple(
        statement
        for index in range(NOTE_RING_SIZE)
        for statement in (
            f"CREATE INDEX IF NOT EXISTS {quote_identifier(note_table(index) + '_whole')} "
            f"ON {owner}.{quote_identifier(note_table(index))} (table_name, noted_at) INCLUDE (tenant) "
            "WHERE key_text IS NULL",
            f"CREATE INDEX IF NOT EXISTS {quote_identifier(note_table(index) + '_ring_read')} "
            f"ON {owner}.{quote_identifier(note_table(index))} (table_name) WHERE ring_read",
        )
    )
    marks = _MARKS_TABLE_DDL.format(schema=owner)
    return (
        *tables,
        *indexes,
        *by_table,
        *sparse,
        marks,
        _RING_TABLE_DDL.format(schema=owner),
        f"INSERT INTO {ring} (id, current_ring, switch_no, switched_at, emptied, fence_seconds, wait_seconds) "
        f"VALUES (1, 0, 0, now(), TRUE, {float(fence.total_seconds())!r}, {float(fence.total_seconds())!r}) "
        "ON CONFLICT (id) DO NOTHING",
        _put_function_sql(schema),
        _whole_function_sql(schema),
    )


def _put_function_sql(schema: str) -> str:
    """the function that writes one note into the ring's current table.

    At a statement's first note it takes the slot from the connection's cached copy
    (:data:`NOTE_SLOT_SETTING`) while that copy is valid by the database's clock, and reads the ring row
    only when it is not; either way it keeps the slot for the rest of the statement.

    :param schema: the schema holding the ring
    :ptype schema: str
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    """
    owner = quote_identifier(schema)
    ring = f"{owner}.{quote_identifier(RING_TABLE)}"
    slots = f"[0-{NOTE_RING_SIZE - 1}]"
    inserts = []
    for index in range(NOTE_RING_SIZE):
        table = f"{owner}.{quote_identifier(note_table(index))}"
        keyword = "IF" if index == 0 else "ELSIF"
        inserts.append(
            f"  {keyword} ring = {index} THEN\n"
            f"    INSERT INTO {table} (table_name, keys, key_text, columns, tenant, ring_read) "
            "VALUES (noted_table, noted_keys, array_to_string(noted_keys, chr(31)), noted_columns, noted_tenant, "
            "read_ring);\n"
        )
    return (  # one line: plpgsql source; each interpolated piece is a fixed name or pattern this module builds
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(_RING_FUNCTION)}("
        "noted_table TEXT, noted_keys TEXT[], noted_columns JSONB, noted_tenant TEXT) "
        "RETURNS void LANGUAGE plpgsql AS $put$\n"
        "DECLARE\n"
        "  ring SMALLINT;\n"
        "  cached TEXT;\n"
        "  read_ring BOOLEAN;\n"
        "  kept TEXT;\n"
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        # a whole-table note (no keys) names the write's reach, when the write named one (NOTE_REACH_SETTING)
        "  IF noted_keys IS NULL AND noted_columns IS NULL THEN\n"
        f"    noted_columns := nullif(current_setting('{NOTE_REACH_SETTING}', true), '')::jsonb;\n"
        "  END IF;\n"
        # the slot is decided at a statement's first note and kept for the rest of that statement
        "  IF current_setting('threetears.note_ring_at', true) IS DISTINCT FROM statement_timestamp()::text THEN\n"
        f"    cached := current_setting('{NOTE_SLOT_SETTING}', true);\n"
        f"    IF cached ~ '^{slots}:[0-9]+(\\.[0-9]+)?$' "
        "AND split_part(cached, ':', 2)::float8 > extract(epoch FROM clock_timestamp()) THEN\n"
        "      ring := split_part(cached, ':', 1)::smallint;\n"
        "      read_ring := NULL;\n"
        "    ELSE\n"
        # the rare path: no valid cached copy on this connection
        f"      SELECT current_ring INTO ring FROM {ring} WHERE id = 1;\n"
        "      ring := coalesce(ring, 0);\n"
        "      read_ring := TRUE;\n"
        "    END IF;\n"
        "    PERFORM set_config('threetears.note_ring', ring::text || CASE WHEN read_ring THEN 'r' ELSE '' END, true);\n"
        "    PERFORM set_config('threetears.note_ring_at', statement_timestamp()::text, true);\n"
        "  ELSE\n"
        "    kept := current_setting('threetears.note_ring');\n"
        "    ring := left(kept, 1)::smallint;\n"
        "    read_ring := CASE WHEN length(kept) > 1 THEN TRUE END;\n"
        "  END IF;\n" + "".join(inserts) + "  END IF;\n"
        "END\n"
        "$put$"
    )


def _whole_function_sql(schema: str) -> str:
    """the function that writes a table's whole-table note once per transaction.

    :param schema: the schema holding the ring
    :ptype schema: str
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    """
    owner = quote_identifier(schema)
    return (
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(_WHOLE_FUNCTION)}("
        "noted_relation OID, noted_table TEXT, noted_tenant TEXT) RETURNS void LANGUAGE plpgsql AS $whole$\n"
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        "  IF current_setting('threetears.noted_whole_' || noted_relation::text, true) = '1' THEN\n"
        "    RETURN;\n"
        "  END IF;\n"
        "  PERFORM set_config('threetears.noted_whole_' || noted_relation::text, '1', true);\n"
        f"  PERFORM {owner}.{quote_identifier(_RING_FUNCTION)}(noted_table, NULL, NULL, noted_tenant);\n"
        "END\n"
        "$whole$"
    )


def note_function_name(table_schema: str, table: str) -> str:
    """the name of the trigger function of ``table_schema.table``: a digest, since names are bounded.

    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the function's name
    :rtype: str
    """
    digest = hashlib.sha256(f"{table_schema}.{table}".encode()).hexdigest()[:24]
    return f"{_TABLE_FUNCTION_PREFIX}{digest}"


def _row_terms(prefix: str, columns: Sequence[str], kinds: Mapping[str, str] | None = None) -> list[str]:
    """each column of the trigger's ``NEW`` or ``OLD`` row as text: a ``timestamptz`` in its canonical form.

    :param prefix: ``NEW`` or ``OLD``
    :ptype prefix: str
    :param columns: the columns
    :ptype columns: Sequence[str]
    :param kinds: each column's ``information_schema`` ``data_type``; a column it does not name is cast to text
    :ptype kinds: Mapping[str, str] | None
    :return: the terms
    :rtype: list[str]
    """
    terms: list[str] = []
    for column in columns:
        value = f"{prefix}.{quote_identifier(column)}"
        if kinds is not None and kinds.get(column) == _TIMESTAMPTZ:
            terms.append(f"to_char({value} AT TIME ZONE 'UTC', {_UTC_MICROS})")
        else:
            terms.append(f"{value}::text")
    return terms


def tenant_sql(tenancy: Tenancy | None, prefix: str) -> str:
    """the SQL of the ``NEW`` or ``OLD`` row's tenant token, as :mod:`threetears.core.tenancy`
    derives it: the row's schema, or its tenant column's value; ``NULL`` (``_all``) for a table with none.

    :param tenancy: the table's declaration
    :ptype tenancy: Tenancy | None
    :param prefix: ``NEW`` or ``OLD``
    :ptype prefix: str
    :return: the expression
    :rtype: str
    """
    if tenancy is None or tenancy.kind not in ("schema", "column"):
        return "NULL"
    kind, value = (
        ("s", "TG_TABLE_SCHEMA::text")
        if tenancy.kind == "schema"
        else (
            "v",
            f"{prefix}.{quote_identifier(str(tenancy.column))}::text",
        )
    )
    return (
        f"(CASE WHEN {value} IS NULL THEN NULL WHEN ('{kind}-' || {value}) ~ '{_TOKEN_SQL}' "
        f"THEN '{kind}-' || {value} ELSE 'h-' || left(encode(sha256(convert_to({value}, 'UTF8')), 'hex'), "
        f"{_DIGEST_SQL}) END)"
    )


def _put_statement(
    notes_schema: str,
    prefix: str,
    keys: Sequence[str],
    columns: Sequence[str],
    tenancy: Tenancy | None = None,
    kinds: Mapping[str, str] | None = None,
) -> str:
    """the statement that notes the ``NEW`` or ``OLD`` row.

    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param prefix: ``NEW`` or ``OLD``
    :ptype prefix: str
    :param keys: the primary-key columns, in declared order
    :ptype keys: Sequence[str]
    :param columns: the declared invalidation columns
    :ptype columns: Sequence[str]
    :param tenancy: the table's tenancy, whose token the note carries
    :ptype tenancy: Tenancy | None
    :param kinds: each column's ``data_type`` (:func:`_row_terms`)
    :ptype kinds: Mapping[str, str] | None
    :return: the ``PERFORM`` statement
    :rtype: str
    """
    terms = _row_terms(prefix, columns, kinds)
    named = ", ".join(f"'{column}', {term}" for column, term in zip(columns, terms, strict=True))
    carried = f"jsonb_build_object({named})" if columns else "NULL"
    return (
        f"PERFORM {quote_identifier(notes_schema)}.{quote_identifier(_RING_FUNCTION)}"
        f"(TG_TABLE_NAME, ARRAY[{', '.join(_row_terms(prefix, keys, kinds))}], {carried}, {tenant_sql(tenancy, prefix)});"
    )


def trigger_function_body(
    *,
    notes_schema: str,
    keys: Sequence[str],
    columns: Sequence[str],
    tenancy: Tenancy | None = None,
    kinds: Mapping[str, str] | None = None,
) -> str:
    """the body of one table's trigger function, as the catalogue keeps it (``pg_proc.prosrc``).

    The row's key and invalidation columns are named in the function, so a note reads exactly those
    columns, as text: for a uuid, text or integer key that is the form an epoch record names it in. An
    update notes the new row, and the old one too when its key or an invalidation column moved; a delete
    notes the old row. Inside a transaction marked whole (:data:`NOTE_WHOLE_SETTING`, a file load) one
    whole-table note stands for every row.

    :param notes_schema: the schema holding the ring, a TRUSTED identifier
    :ptype notes_schema: str
    :param keys: the primary-key columns, in declared order; at least one
    :ptype keys: Sequence[str]
    :param columns: the declared invalidation columns
    :ptype columns: Sequence[str]
    :param tenancy: the table's tenancy, whose token each note carries
    :ptype tenancy: Tenancy | None
    :param kinds: each column's ``information_schema`` ``data_type``: a ``timestamptz`` is noted in its canonical
        form (:data:`_UTC_MICROS`); ``None`` casts every column to text
    :ptype kinds: Mapping[str, str] | None
    :return: the body
    :rtype: str
    :raises ValueError: when no key column is named, or a column name holds a quote
    """
    if not keys:
        raise ValueError("a noted table names its key columns")
    if any("'" in column for column in (*keys, *columns)):
        raise ValueError("a noted column name holds a quote")
    owner = quote_identifier(notes_schema)
    watched = [*keys, *(column for column in columns if column not in keys)]
    old_row = ", ".join(_row_terms("OLD", watched, kinds))
    new_row = ", ".join(_row_terms("NEW", watched, kinds))
    # a whole-table note's tenant: the schema, for a schema table; ``_all`` (NULL) for a platform table, whose
    # truncate or load spans customers
    whole_tenant = tenant_sql(tenancy, "NEW") if tenancy is not None and tenancy.kind == "schema" else "NULL"
    return (
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        "  IF TG_OP = 'TRUNCATE' THEN\n"
        f"    PERFORM {owner}.{quote_identifier(_RING_FUNCTION)}(TG_TABLE_NAME, NULL, NULL, {whole_tenant});\n"
        "    RETURN NULL;\n"
        "  END IF;\n"
        f"  IF current_setting('{NOTE_WHOLE_SETTING}', true) = '1' THEN\n"
        f"    PERFORM {owner}.{quote_identifier(_WHOLE_FUNCTION)}(TG_RELID, TG_TABLE_NAME, {whole_tenant});\n"
        "    RETURN NULL;\n"
        "  END IF;\n"
        "  IF TG_OP <> 'DELETE' THEN\n"
        f"    {_put_statement(notes_schema, 'NEW', keys, columns, tenancy, kinds)}\n"
        "  END IF;\n"
        f"  IF TG_OP = 'DELETE' OR (TG_OP = 'UPDATE' AND ROW({old_row}) IS DISTINCT FROM ROW({new_row})) THEN\n"
        f"    {_put_statement(notes_schema, 'OLD', keys, columns, tenancy, kinds)}\n"
        "  END IF;\n"
        "  RETURN NULL;\n"
        "END\n"
    )


def trigger_function_sql(
    *,
    notes_schema: str,
    table_schema: str,
    table: str,
    keys: Sequence[str],
    columns: Sequence[str],
    tenancy: Tenancy | None = None,
    kinds: Mapping[str, str] | None = None,
) -> str:
    """the statement that creates, or replaces, one table's trigger function (:func:`trigger_function_body`).

    :param notes_schema: the schema holding the ring, a TRUSTED identifier
    :ptype notes_schema: str
    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :param keys: the primary-key columns, in declared order; at least one
    :ptype keys: Sequence[str]
    :param columns: the declared invalidation columns
    :ptype columns: Sequence[str]
    :param tenancy: the table's tenancy, whose token each note carries
    :ptype tenancy: Tenancy | None
    :param kinds: each column's ``information_schema`` ``data_type`` (:func:`trigger_function_body`)
    :ptype kinds: Mapping[str, str] | None
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    :raises ValueError: when no key column is named, or a column name holds a quote
    """
    owner = quote_identifier(notes_schema)
    body = trigger_function_body(notes_schema=notes_schema, keys=keys, columns=columns, tenancy=tenancy, kinds=kinds)
    return (
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(note_function_name(table_schema, table))}() "
        f"RETURNS trigger LANGUAGE plpgsql AS $note${body}$note$"
    )


def note_trigger_sql(*, notes_schema: str, table_schema: str, table: str) -> dict[str, str]:
    """the statements that give one table its triggers (:data:`NOTE_TRIGGERS`), by trigger name.

    The row trigger's ``WHEN`` and the whole-write trigger's are each other's complement, so a write is noted
    by exactly one of them: a row each, or one whole-table note for the transaction.

    :param notes_schema: the schema holding the ring and the table's function
    :ptype notes_schema: str
    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: each trigger's ``CREATE TRIGGER`` statement
    :rtype: dict[str, str]
    """
    owner = quote_identifier(notes_schema)
    target = f"{quote_identifier(table_schema)}.{quote_identifier(table)}"
    function = f"{owner}.{quote_identifier(note_function_name(table_schema, table))}"
    whole = f"current_setting('{NOTE_WHOLE_SETTING}', true)"
    return {
        NOTE_TRIGGER: f"CREATE TRIGGER {quote_identifier(NOTE_TRIGGER)} AFTER INSERT OR UPDATE OR DELETE ON {target} "
        f"FOR EACH ROW WHEN ({whole} IS DISTINCT FROM '1') EXECUTE FUNCTION {function}()",
        NOTE_TRUNCATE_TRIGGER: f"CREATE TRIGGER {quote_identifier(NOTE_TRUNCATE_TRIGGER)} AFTER TRUNCATE ON {target} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {function}()",
        NOTE_WHOLE_TRIGGER: f"CREATE TRIGGER {quote_identifier(NOTE_WHOLE_TRIGGER)} AFTER INSERT OR UPDATE OR DELETE "
        f"ON {target} FOR EACH STATEMENT WHEN ({whole} = '1') EXECUTE FUNCTION {function}()",
    }


def legacy_truncate_trigger_drop_sql(*, table_schema: str, table: str) -> str:
    """the statement that drops a table's shared ``TRUNCATE`` trigger an earlier install gave it, run before its own is created.

    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the ``DROP TRIGGER IF EXISTS`` statement
    :rtype: str
    """
    target = f"{quote_identifier(table_schema)}.{quote_identifier(table)}"
    return f"DROP TRIGGER IF EXISTS {quote_identifier(_LEGACY_TRUNCATE_TRIGGER)} ON {target}"
