"""the write-mark queue's SQL: its table, the trigger function and each table's trigger.

Below every package that runs it -- the migration runner gives switched-on tables their trigger, the
reconciler (:mod:`threetears.core.collections.write_marks`) installs and drains them -- so neither imports
the other for it. The reconciler re-exports every name.
"""

from __future__ import annotations

from typing import Final

from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "MARK_FUNCTION",
    "STATEMENT_WRITE_MARK_TRIGGER",
    "WRITE_MARKS_TABLE",
    "WRITE_MARK_TRIGGER",
    "mark_function_sql",
    "replace_statement_trigger_sql",
    "write_mark_trigger_sql",
    "write_marks_ddl",
]

#: the queue of committed writes not yet reconciled
WRITE_MARKS_TABLE: Final[str] = "collection_write_marks"

#: the trigger function every switched-on table's trigger calls
MARK_FUNCTION: Final[str] = "threetears_mark_collection_write"

#: the trigger's name on every switched-on table: a row trigger that marks a transaction's first changed row
WRITE_MARK_TRIGGER: Final[str] = "threetears_collection_row_write_mark"

#: the statement trigger 0.78.0 and before gave each table, which marked a statement that changed no row too;
#: the install drops it as it creates :data:`WRITE_MARK_TRIGGER`, so the two never fire together
STATEMENT_WRITE_MARK_TRIGGER: Final[str] = "threetears_collection_write_mark"

#: the transaction-local setting a table's first marked row sets, so the rest of the transaction's rows of
#: that table mark nothing; named by the table's oid, which is always a valid setting name
_MARKED_SETTING: Final[str] = "'threetears.write_mark.t' || TG_RELID::text"


def write_marks_ddl(schema: str) -> tuple[str, ...]:
    """the statements that create the marks table and the trigger function in ``schema``, idempotently.

    The function names the marks table by its schema, so a trigger on a table in any other schema
    (an agent's) writes the one queue.

    :param schema: the schema that holds the queue (the platform schema)
    :ptype schema: str
    :return: the statements, in order; run them one at a time (YugabyteDB runs no DDL in a transaction)
    :rtype: tuple[str, ...]
    """
    marks = f"{quote_identifier(schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
    return (
        f"CREATE TABLE IF NOT EXISTS {marks} ("
        "id BIGSERIAL PRIMARY KEY, "
        "table_name TEXT NOT NULL, "
        "date_created TIMESTAMPTZ NOT NULL DEFAULT now())",
        mark_function_sql(schema),
    )


def mark_function_sql(schema: str) -> str:
    """the statement that creates, or replaces, the trigger function in ``schema``.

    One mark per table per transaction that changed a row of it: the first changed row inserts the mark and
    sets a transaction-local setting, and every later row of that table in the transaction finds it set and
    marks nothing. A statement that changes no row fires no row trigger, so it marks nothing (an idempotent
    ``INSERT ... WHERE NOT EXISTS`` that found its row, ``UPDATE ... WHERE false``): each mark is an advance,
    and an advance no follower heard drops the table from every follower's caches. ``ROLLBACK TO SAVEPOINT``
    takes back the mark and the setting together, so a later row in the transaction marks again.
    YugabyteDB has no transition tables (``REFERENCING``), which a statement trigger would need to tell a
    statement that changed nothing.

    :param schema: the schema that holds the queue
    :ptype schema: str
    :return: the ``CREATE OR REPLACE FUNCTION`` statement
    :rtype: str
    """
    marks = f"{quote_identifier(schema)}.{quote_identifier(WRITE_MARKS_TABLE)}"
    function = f"{quote_identifier(schema)}.{quote_identifier(MARK_FUNCTION)}"
    return (
        f"CREATE OR REPLACE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $mark$\n"
        "BEGIN\n"
        f"  IF current_setting({_MARKED_SETTING}, true) = '1' THEN\n"
        "    RETURN NULL;\n"
        "  END IF;\n"
        f"  PERFORM set_config({_MARKED_SETTING}, '1', true);\n"
        f"  INSERT INTO {marks} (table_name) VALUES (TG_TABLE_NAME);\n"
        "  RETURN NULL;\n"
        "END\n"
        "$mark$"
    )


def replace_statement_trigger_sql(*, table_schema: str, table: str) -> str:
    """the statement that drops the statement trigger 0.78.0 and before gave a table, if it is there.

    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the ``DROP TRIGGER IF EXISTS`` statement
    :rtype: str
    """
    return (
        f"DROP TRIGGER IF EXISTS {quote_identifier(STATEMENT_WRITE_MARK_TRIGGER)} "
        f"ON {quote_identifier(table_schema)}.{quote_identifier(table)}"
    )


def write_mark_trigger_sql(*, marks_schema: str, table_schema: str, table: str) -> str:
    """the statement that gives one table its write-mark trigger.

    Row-level, behind the function's per-transaction setting (:func:`mark_function_sql`): one mark per
    transaction that changed a row of the table, and none for a statement that changed none.

    :param marks_schema: the schema holding the queue and its function
    :ptype marks_schema: str
    :param table_schema: the table's schema
    :ptype table_schema: str
    :param table: the table
    :ptype table: str
    :return: the ``CREATE TRIGGER`` statement
    :rtype: str
    """
    return (
        f"CREATE TRIGGER {quote_identifier(WRITE_MARK_TRIGGER)} "
        f"AFTER INSERT OR UPDATE OR DELETE ON {quote_identifier(table_schema)}.{quote_identifier(table)} "
        f"FOR EACH ROW EXECUTE FUNCTION {quote_identifier(marks_schema)}.{quote_identifier(MARK_FUNCTION)}()"
    )
