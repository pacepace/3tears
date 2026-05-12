"""
unit tests for migration-driven drift detection.

drift detection answers one question: "does the live DB match the
schema the migrations claim to have produced?" these tests exercise
the three moving pieces that answer it:

- :func:`parse_ddl_to_expected` turns a sequence of DDL strings into a
  structured expected-schema view
- :func:`diff_expected_live` compares expected vs live and produces a
  structured :class:`DriftReport`
- :class:`DriftReport` renders itself to JSON and to human-readable
  lines without changing either shape
"""

from __future__ import annotations

from threetears.core.data.migrations import (
    DriftReport,
    diff_expected_live,
    parse_ddl_to_expected,
)


class TestParseDDL:
    """DDL parser recognizes CREATE TABLE and ALTER TABLE shapes."""

    def test_parse_create_table_with_columns(self) -> None:
        """every column clause becomes an entry in the expected table."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS widgets ("
            "  id UUID PRIMARY KEY,"
            "  name VARCHAR(255) NOT NULL,"
            "  created TIMESTAMP NOT NULL,"
            "  data JSONB"
            ")"
        ]
        expected = parse_ddl_to_expected(stmts)
        assert "widgets" in expected
        table = expected["widgets"]
        assert table.columns["id"] == "uuid"
        assert table.columns["name"] == "character varying"
        assert table.columns["created"] == "timestamp without time zone"
        assert table.columns["data"] == "jsonb"

    def test_parse_alter_add_column_extends_existing_table(self) -> None:
        """ALTER TABLE ADD COLUMN adds columns to the expected table."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)",
            "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS label TEXT",
        ]
        expected = parse_ddl_to_expected(stmts)
        assert expected["widgets"].columns == {"id": "uuid", "label": "text"}

    def test_parse_alter_drop_column_removes_from_expected(self) -> None:
        """ALTER TABLE DROP COLUMN forgets a previously-declared column."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY, legacy TEXT)",
            "ALTER TABLE widgets DROP COLUMN legacy",
        ]
        expected = parse_ddl_to_expected(stmts)
        assert "legacy" not in expected["widgets"].columns

    def test_parse_skips_table_constraints(self) -> None:
        """PRIMARY KEY (col) / FOREIGN KEY / UNIQUE / CHECK are not columns."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS widgets (  id UUID,  name VARCHAR(100),  PRIMARY KEY (id),  UNIQUE (name))"
        ]
        expected = parse_ddl_to_expected(stmts)
        cols = set(expected["widgets"].columns.keys())
        assert cols == {"id", "name"}

    def test_parse_skips_index_statements(self) -> None:
        """CREATE INDEX is not a table-level expectation and is skipped."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)",
            "CREATE INDEX IF NOT EXISTS idx_widgets_id ON widgets (id)",
        ]
        expected = parse_ddl_to_expected(stmts)
        assert set(expected) == {"widgets"}

    def test_parse_schema_qualified_table_name_strips_schema(self) -> None:
        """CREATE TABLE platform.agents is registered as 'agents'."""
        stmts = ["CREATE TABLE IF NOT EXISTS platform.agents (id UUID PRIMARY KEY)"]
        expected = parse_ddl_to_expected(stmts)
        assert "agents" in expected
        assert "platform.agents" not in expected

    def test_parse_quoted_and_qualified_name_strips_both(self) -> None:
        """CREATE TABLE "platform"."agents" strips quotes and schema."""
        stmts = ['CREATE TABLE IF NOT EXISTS "platform"."agents" (id UUID PRIMARY KEY)']
        expected = parse_ddl_to_expected(stmts)
        assert "agents" in expected

    def test_parse_alter_schema_qualified_add_column(self) -> None:
        """ALTER TABLE platform.agents ADD COLUMN applies to 'agents'."""
        stmts = [
            "CREATE TABLE IF NOT EXISTS platform.agents (id UUID PRIMARY KEY)",
            "ALTER TABLE platform.agents ADD COLUMN label TEXT",
        ]
        expected = parse_ddl_to_expected(stmts)
        assert expected["agents"].columns["label"] == "text"


class TestDiff:
    """diff_expected_live produces the expected structured report."""

    def test_clean_when_expected_matches_live(self) -> None:
        """no buckets populated when schemas match."""
        expected = parse_ddl_to_expected(
            ["CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY, name VARCHAR(100))"]
        )
        live = {"widgets": {"id": "uuid", "name": "character varying"}}
        report = diff_expected_live(expected, live)
        assert report.is_clean()

    def test_missing_table_reported(self) -> None:
        """expected table absent from live DB appears in missing_tables."""
        expected = parse_ddl_to_expected(["CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)"])
        live: dict[str, dict[str, str]] = {}
        report = diff_expected_live(expected, live)
        assert report.missing_tables == ["widgets"]
        assert not report.is_clean()

    def test_extra_table_reported(self) -> None:
        """live table with no matching CREATE appears in extra_tables."""
        expected: dict = {}
        live = {"hand_rolled": {"id": "uuid"}}
        report = diff_expected_live(expected, live)
        assert report.extra_tables == ["hand_rolled"]

    def test_extra_column_reported(self) -> None:
        """extra live column on a tracked table appears in extra_columns."""
        expected = parse_ddl_to_expected(["CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)"])
        live = {"widgets": {"id": "uuid", "rogue": "text"}}
        report = diff_expected_live(expected, live)
        assert ("widgets", "rogue", "text") in report.extra_columns

    def test_missing_column_reported(self) -> None:
        """declared column absent from live table appears in missing_columns."""
        expected = parse_ddl_to_expected(["CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY, label TEXT)"])
        live = {"widgets": {"id": "uuid"}}
        report = diff_expected_live(expected, live)
        assert ("widgets", "label") in report.missing_columns

    def test_type_mismatch_reported(self) -> None:
        """column that exists with a different type appears in type_mismatches."""
        expected = parse_ddl_to_expected(["CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY, n INTEGER)"])
        live = {"widgets": {"id": "uuid", "n": "bigint"}}
        report = diff_expected_live(expected, live)
        assert ("widgets", "n", "integer", "bigint") in report.type_mismatches

    def test_cross_schema_table_excluded_when_current_schema_supplied(self) -> None:
        """expected table declared in another schema is skipped at diff time."""
        expected = parse_ddl_to_expected(
            [
                "CREATE TABLE IF NOT EXISTS widgets (id UUID PRIMARY KEY)",
                "CREATE TABLE IF NOT EXISTS other.elsewhere (id UUID PRIMARY KEY)",
            ]
        )
        live = {"widgets": {"id": "uuid"}}
        report = diff_expected_live(expected, live, current_schema="public")
        assert "elsewhere" not in report.missing_tables
        assert report.is_clean()

    def test_cross_schema_table_present_when_current_schema_matches(self) -> None:
        """expected table whose schema matches current_schema is checked."""
        expected = parse_ddl_to_expected(
            [
                "CREATE TABLE IF NOT EXISTS audit.events (id UUID PRIMARY KEY)",
            ]
        )
        live: dict[str, dict[str, str]] = {}
        report = diff_expected_live(expected, live, current_schema="audit")
        assert "events" in report.missing_tables


class TestReportRendering:
    """DriftReport renders to JSON and to human lines consistently."""

    def test_clean_report_as_dict_has_clean_flag(self) -> None:
        """clean report's JSON has ``clean=True``."""
        report = DriftReport()
        data = report.as_dict()
        assert data["clean"] is True

    def test_dirty_report_as_dict_exposes_every_bucket(self) -> None:
        """dirty report's JSON lists every bucket with the expected key names."""
        report = DriftReport(
            missing_tables=["a"],
            extra_tables=["b"],
            missing_columns=[("t", "c")],
            extra_columns=[("t", "c2", "text")],
            type_mismatches=[("t", "c3", "uuid", "text")],
        )
        data = report.as_dict()
        assert data["clean"] is False
        assert data["missing_tables"] == ["a"]
        assert data["extra_tables"] == ["b"]
        assert data["missing_columns"] == [{"table": "t", "column": "c"}]
        assert data["extra_columns"] == [{"table": "t", "column": "c2", "data_type": "text"}]
        assert data["type_mismatches"] == [{"table": "t", "column": "c3", "expected": "uuid", "actual": "text"}]

    def test_human_lines_include_every_drift_category(self) -> None:
        """human-readable lines describe each bucket in turn."""
        report = DriftReport(
            missing_tables=["a"],
            extra_tables=["b"],
            missing_columns=[("t", "c")],
        )
        lines = report.as_human_lines()
        joined = "\n".join(lines)
        assert "missing tables" in joined
        assert "extra tables" in joined
        assert "missing columns" in joined
