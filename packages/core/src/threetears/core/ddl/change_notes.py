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
A writer reads it once per statement (:data:`_RING_FUNCTION`), at the statement's first noted row, and
keeps it for the rest of that statement in a transaction-local setting: no writer contends for it, and a
statement that began before a switch finishes writing into the table it started with. That bound -- a
statement's own duration -- is what the keeper's fence waits out before it empties a table.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "NOTE_RING_SIZE",
    "NOTE_TRIGGER",
    "NOTE_TRUNCATE_TRIGGER",
    "NOTE_WHOLE_SETTING",
    "RING_TABLE",
    "change_notes_ddl",
    "note_function_name",
    "note_table",
    "note_trigger_sql",
    "trigger_function_body",
    "trigger_function_sql",
]

#: how many note tables the ring keeps: the current one, the previous one, and one emptied and waiting
NOTE_RING_SIZE: Final = 3

#: the ring's note tables are this name and their place in the ring: ``threetears_change_notes_0`` ..
_NOTE_TABLE_PREFIX: Final = "threetears_change_notes_"

#: the one-row table naming the ring's current note table, and what the keeper knows of the coverage
RING_TABLE: Final = "threetears_change_note_ring"

#: the function every table's trigger writes a note through; it reads the ring once per statement
_RING_FUNCTION: Final = "threetears_change_note_put"

#: the function a whole-table note is written through, once per transaction and table
_WHOLE_FUNCTION: Final = "threetears_change_note_whole"

#: the function a table's ``TRUNCATE`` trigger calls: a truncate changes the whole table
_TRUNCATE_FUNCTION: Final = "threetears_change_note_truncate"

#: the row trigger on every noted table, and its statement trigger for ``TRUNCATE``
NOTE_TRIGGER: Final = "threetears_change_note"
NOTE_TRUNCATE_TRIGGER: Final = "threetears_change_note_truncate"

#: each table's own trigger function is this prefix and a digest of its schema and name
_TABLE_FUNCTION_PREFIX: Final = "threetears_note_"

#: the transaction-local setting a whole-table write (a file load) sets, so every row it copies is one
#: whole-table note rather than a note each: the load's epoch names the whole table too
NOTE_WHOLE_SETTING: Final = "threetears.note_whole"


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


def change_notes_ddl(schema: str) -> tuple[str, ...]:
    """the statements that create the ring in ``schema``: its note tables, its ring row and its functions.

    Each is idempotent, and a ring row already there keeps its state. Run them one at a time (YugabyteDB
    runs no DDL in a transaction).

    :param schema: the schema holding the ring (the platform schema), a TRUSTED identifier
    :ptype schema: str
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
        "columns JSONB, "
        # the clock at the insert, not the transaction's start: a late note is told by it
        "noted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp())"
        for index in range(NOTE_RING_SIZE)
    )
    return (
        *tables,
        f"CREATE TABLE IF NOT EXISTS {ring} ("
        "id SMALLINT PRIMARY KEY CHECK (id = 1), "
        "current_ring SMALLINT NOT NULL, "
        "switch_no BIGINT NOT NULL, "
        "switched_at TIMESTAMPTZ NOT NULL, "
        "switched_seq TEXT, "
        "floor TEXT, "
        "fence_seq TEXT, "
        "emptied BOOLEAN NOT NULL, "
        "log_incarnation TEXT)",
        f"INSERT INTO {ring} (id, current_ring, switch_no, switched_at, emptied) "
        "VALUES (1, 0, 0, now(), TRUE) ON CONFLICT (id) DO NOTHING",
        _put_function_sql(schema),
        _whole_function_sql(schema),
        _truncate_function_sql(schema),
    )


def _put_function_sql(schema: str) -> str:
    """the function that writes one note into the ring's current table, reading the ring once per statement.

    :param schema: the schema holding the ring
    :ptype schema: str
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    """
    owner = quote_identifier(schema)
    ring = f"{owner}.{quote_identifier(RING_TABLE)}"
    inserts = []
    for index in range(NOTE_RING_SIZE):
        table = f"{owner}.{quote_identifier(note_table(index))}"
        keyword = "IF" if index == 0 else "ELSIF"
        inserts.append(
            f"  {keyword} ring = {index} THEN\n"
            f"    INSERT INTO {table} (table_name, keys, columns) VALUES (noted_table, noted_keys, noted_columns);\n"
        )
    return (
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(_RING_FUNCTION)}("
        "noted_table TEXT, noted_keys TEXT[], noted_columns JSONB) RETURNS void LANGUAGE plpgsql AS $put$\n"
        "DECLARE\n"
        "  ring SMALLINT;\n"
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        # the ring is read at a statement's first note and kept for the rest of that statement
        "  IF current_setting('threetears.note_ring_at', true) IS DISTINCT FROM statement_timestamp()::text THEN\n"
        f"    SELECT current_ring INTO ring FROM {ring} WHERE id = 1;\n"
        "    ring := coalesce(ring, 0);\n"
        "    PERFORM set_config('threetears.note_ring', ring::text, true);\n"
        "    PERFORM set_config('threetears.note_ring_at', statement_timestamp()::text, true);\n"
        "  ELSE\n"
        "    ring := current_setting('threetears.note_ring')::smallint;\n"
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
        "noted_relation OID, noted_table TEXT) RETURNS void LANGUAGE plpgsql AS $whole$\n"
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        "  IF current_setting('threetears.noted_whole_' || noted_relation::text, true) = '1' THEN\n"
        "    RETURN;\n"
        "  END IF;\n"
        "  PERFORM set_config('threetears.noted_whole_' || noted_relation::text, '1', true);\n"
        f"  PERFORM {owner}.{quote_identifier(_RING_FUNCTION)}(noted_table, NULL, NULL);\n"
        "END\n"
        "$whole$"
    )


def _truncate_function_sql(schema: str) -> str:
    """the function a table's ``TRUNCATE`` trigger calls: one whole-table note.

    :param schema: the schema holding the ring
    :ptype schema: str
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    """
    owner = quote_identifier(schema)
    return (
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(_TRUNCATE_FUNCTION)}() "
        "RETURNS trigger LANGUAGE plpgsql AS $truncate$\n"
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        f"  PERFORM {owner}.{quote_identifier(_RING_FUNCTION)}(TG_TABLE_NAME, NULL, NULL);\n"
        "  RETURN NULL;\n"
        "END\n"
        "$truncate$"
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


def _row_terms(prefix: str, columns: Sequence[str]) -> list[str]:
    """each column of the trigger's ``NEW`` or ``OLD`` row as text.

    :param prefix: ``NEW`` or ``OLD``
    :ptype prefix: str
    :param columns: the columns
    :ptype columns: Sequence[str]
    :return: the terms
    :rtype: list[str]
    """
    return [f"{prefix}.{quote_identifier(column)}::text" for column in columns]


def _put_statement(notes_schema: str, prefix: str, keys: Sequence[str], columns: Sequence[str]) -> str:
    """the statement that notes the ``NEW`` or ``OLD`` row.

    :param notes_schema: the schema holding the ring
    :ptype notes_schema: str
    :param prefix: ``NEW`` or ``OLD``
    :ptype prefix: str
    :param keys: the primary-key columns, in declared order
    :ptype keys: Sequence[str]
    :param columns: the declared invalidation columns
    :ptype columns: Sequence[str]
    :return: the ``PERFORM`` statement
    :rtype: str
    """
    named = ", ".join(f"'{column}', {term}" for column, term in zip(columns, _row_terms(prefix, columns), strict=True))
    carried = f"jsonb_build_object({named})" if columns else "NULL"
    return (
        f"PERFORM {quote_identifier(notes_schema)}.{quote_identifier(_RING_FUNCTION)}"
        f"(TG_TABLE_NAME, ARRAY[{', '.join(_row_terms(prefix, keys))}], {carried});"
    )


def trigger_function_body(*, notes_schema: str, keys: Sequence[str], columns: Sequence[str]) -> str:
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
    old_row = ", ".join(_row_terms("OLD", watched))
    new_row = ", ".join(_row_terms("NEW", watched))
    return (
        "BEGIN\n"  # one line: plpgsql source, each interpolated piece is one PERFORM statement the module builds
        f"  IF current_setting('{NOTE_WHOLE_SETTING}', true) = '1' THEN\n"
        f"    PERFORM {owner}.{quote_identifier(_WHOLE_FUNCTION)}(TG_RELID, TG_TABLE_NAME);\n"
        "    RETURN NULL;\n"
        "  END IF;\n"
        "  IF TG_OP <> 'DELETE' THEN\n"
        f"    {_put_statement(notes_schema, 'NEW', keys, columns)}\n"
        "  END IF;\n"
        f"  IF TG_OP = 'DELETE' OR (TG_OP = 'UPDATE' AND ROW({old_row}) IS DISTINCT FROM ROW({new_row})) THEN\n"
        f"    {_put_statement(notes_schema, 'OLD', keys, columns)}\n"
        "  END IF;\n"
        "  RETURN NULL;\n"
        "END\n"
    )


def trigger_function_sql(
    *, notes_schema: str, table_schema: str, table: str, keys: Sequence[str], columns: Sequence[str]
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
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    :raises ValueError: when no key column is named, or a column name holds a quote
    """
    owner = quote_identifier(notes_schema)
    body = trigger_function_body(notes_schema=notes_schema, keys=keys, columns=columns)
    return (
        f"CREATE OR REPLACE FUNCTION {owner}.{quote_identifier(note_function_name(table_schema, table))}() "
        f"RETURNS trigger LANGUAGE plpgsql AS $note${body}$note$"
    )


def note_trigger_sql(*, notes_schema: str, table_schema: str, table: str) -> tuple[str, str]:
    """the statements that give one table its row trigger and its ``TRUNCATE`` trigger.

    :param notes_schema: the schema holding the ring and the table's function
    :ptype notes_schema: str
    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the two ``CREATE TRIGGER`` statements
    :rtype: tuple[str, str]
    """
    owner = quote_identifier(notes_schema)
    target = f"{quote_identifier(table_schema)}.{quote_identifier(table)}"
    function = f"{owner}.{quote_identifier(note_function_name(table_schema, table))}"
    return (
        f"CREATE TRIGGER {quote_identifier(NOTE_TRIGGER)} AFTER INSERT OR UPDATE OR DELETE ON {target} "
        f"FOR EACH ROW EXECUTE FUNCTION {function}()",
        f"CREATE TRIGGER {quote_identifier(NOTE_TRUNCATE_TRIGGER)} AFTER TRUNCATE ON {target} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {owner}.{quote_identifier(_TRUNCATE_FUNCTION)}()",
    )
