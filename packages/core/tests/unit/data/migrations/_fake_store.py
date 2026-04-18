"""
fake DataStore stub for migration runner unit tests.

mirrors the execute/query surface used by MigrationRunner without
requiring a real YugabyteDB connection. tracks applied rows in a
simple in-memory list so idempotent-apply semantics can be asserted.
"""

from __future__ import annotations

from typing import Any


class FakeDataStore:
    """
    in-memory DataStore stub capturing executed SQL and emulating the
    ``_schema_migrations`` bookkeeping contract used by MigrationRunner.

    :ivar executed: list of (sql, params) tuples for every execute call
    :ivar _migrations_rows: list of applied migration row dicts
    :ivar fail_on: sql substring that, when present, triggers RuntimeError
    """

    def __init__(self, fail_on: str | None = None) -> None:
        """
        initialize empty execution log and migration tracker.

        :param fail_on: SQL substring that triggers RuntimeError on match
        :ptype fail_on: str | None
        """
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._migrations_rows: list[dict[str, Any]] = []
        self._migrations_table_created = False
        self._fail_on = fail_on
        self._tables: set[str] = set()
        # monotonically increasing counter stamped as date_applied so
        # history ordering is deterministic in tests.
        self._apply_counter = 0

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record SQL execution and emulate ``_schema_migrations`` side effects.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional statement parameters
        :ptype params: Any
        :return: synthetic status string
        :rtype: str
        :raises RuntimeError: if ``sql`` contains the fail_on substring
        """
        self.executed.append((sql, params))
        if self._fail_on is not None and self._fail_on in sql:
            msg = f"fake store forced failure on sql matching '{self._fail_on}'"
            raise RuntimeError(msg)
        normalized = " ".join(sql.split()).upper()
        result: str
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self._migrations_table_created = True
            result = "CREATE TABLE"
            return result
        if normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self._apply_counter += 1
            self._migrations_rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                    "date_applied": self._apply_counter,
                }
            )
            result = "INSERT 0 1"
            return result
        if normalized.startswith("DELETE FROM _SCHEMA_MIGRATIONS"):
            target_version = params[0]
            target_package = params[1]
            self._migrations_rows = [
                row
                for row in self._migrations_rows
                if not (row["version"] == target_version and row["package"] == target_package)
            ]
            result = "DELETE 1"
            return result
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        emulate DataStore.query for the statements MigrationRunner issues.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional query parameters
        :ptype params: Any
        :return: list of row dictionaries
        :rtype: list[dict[str, Any]]
        """
        normalized = " ".join(sql.split()).upper()
        result: list[dict[str, Any]]
        if "SELECT VERSION, PACKAGE FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [
                {"version": row["version"], "package": row["package"]}
                for row in self._migrations_rows
            ]
            return result
        if "SELECT VERSION, PACKAGE, DESCRIPTION, DATE_APPLIED FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [
                {
                    "version": row["version"],
                    "package": row["package"],
                    "description": row["description"],
                    "date_applied": row["date_applied"],
                }
                for row in sorted(
                    self._migrations_rows,
                    key=lambda r: (r["date_applied"], r["version"], r["package"]),
                )
            ]
            return result
        if "COALESCE(MAX(VERSION)" in normalized:
            max_version = max((row["version"] for row in self._migrations_rows), default=0)
            result = [{"max_version": max_version}]
            return result
        result = []
        return result
