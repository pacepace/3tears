"""the rows a statement wrote, read from the warehouse's own record of it."""

from __future__ import annotations

import pytest

from threetears.datasources.drivers.statement_rows import rows_from_command_tag, writes_rows


@pytest.mark.parametrize(
    ("status", "rows"),
    [
        ("INSERT 0 5", 5),
        ("SELECT 120", 120),  # Postgres' tag for a CREATE TABLE AS
        ("UPDATE 3", 3),
        ("DELETE 0", 0),
        ("COPY 7", 7),
        ("CREATE TABLE", None),
        ("CREATE VIEW", None),
        ("", None),
        (None, None),
    ],
)
def test_a_command_tag_reports_its_rows_or_none(status: object, rows: int | None) -> None:
    """Flips if a recorded asyncpg command tag is read as other than the rows it reports, or a DDL tag as rows."""
    assert rows_from_command_tag(status) == rows


@pytest.mark.parametrize(
    ("sql", "writes"),
    [
        ("CREATE TABLE s.t AS SELECT * FROM s.u", True),
        ("create temp table t as select 1", True),
        ("INSERT INTO t SELECT * FROM u", True),
        ("CREATE TABLE t (a int)", False),
        ("CREATE VIEW v AS SELECT 1", False),
        ("GRANT SELECT ON t TO g", False),
    ],
)
def test_a_statement_that_writes_rows_is_told_apart(sql: str, writes: bool) -> None:
    """Flips if a CREATE TABLE AS or an INSERT is not taken as writing rows, or a DDL statement is."""
    assert writes_rows(sql) is writes
