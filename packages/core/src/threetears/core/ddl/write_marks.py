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
    "WRITE_MARKS_TABLE",
    "WRITE_MARK_TRIGGER",
    "write_mark_trigger_sql",
    "write_marks_ddl",
]

#: the queue of committed writes not yet reconciled
WRITE_MARKS_TABLE: Final[str] = "collection_write_marks"

#: the trigger function every switched-on table's trigger calls
MARK_FUNCTION: Final[str] = "threetears_mark_collection_write"

#: the trigger's name on every switched-on table
WRITE_MARK_TRIGGER: Final[str] = "threetears_collection_write_mark"


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
    function = f"{quote_identifier(schema)}.{quote_identifier(MARK_FUNCTION)}"
    return (
        f"CREATE TABLE IF NOT EXISTS {marks} ("
        "id BIGSERIAL PRIMARY KEY, "
        "table_name TEXT NOT NULL, "
        "date_created TIMESTAMPTZ NOT NULL DEFAULT now())",
        f"CREATE OR REPLACE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $mark$\n"
        "BEGIN\n"
        f"  INSERT INTO {marks} (table_name) VALUES (TG_TABLE_NAME);\n"
        "  RETURN NULL;\n"
        "END\n"
        "$mark$",
    )


def write_mark_trigger_sql(*, marks_schema: str, table_schema: str, table: str) -> str:
    """the statement that gives one table its write-mark trigger.

    Statement-level: one mark per write statement, however many rows it touched (a statement that
    touched none marks too, which costs one spurious advance).

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
        f"FOR EACH STATEMENT EXECUTE FUNCTION {quote_identifier(marks_schema)}.{quote_identifier(MARK_FUNCTION)}()"
    )
