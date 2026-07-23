"""Tests guarding against ScrapeTarget/ScrapeRecipe/ScrapeExtraction <-> DDL drift.

Mirrors ``tests/unit/test_migrations.py``'s pattern exactly (same class of bug:
an entity field persisted in code with no matching DDL column raises
``asyncpg.UndefinedColumnError`` against a real L3 store, invisible to every
in-memory-only test) applied to ``threetears.scrape.migrations`` instead of
``faidh.db.migrations``.
"""

from __future__ import annotations

import re

import pytest
from threetears.core.data.migrations import MigrationRunner

from threetears.scrape.migrations import register

_CREATE_COLUMN_RE = re.compile(
    r"^\s*(\w+)\s+(?:TEXT|INTEGER|FLOAT8|TIMESTAMPTZ|JSONB|BOOLEAN|BIGINT)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ADD_COLUMN_RE = re.compile(r"ADD COLUMN IF NOT EXISTS\s+(\w+)", re.IGNORECASE)


# parity-exempt: hand-rolled subset stub of 3tears' DataStore (execute/query only) -- a real DataStore needs a live registry/pool, defeating the point of a fast, network-free unit test
class _FakeStore:
    """Records every SQL string passed to ``execute()`` -- never touches a database."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, sql: str, *params: object) -> str:
        self.executed.append(sql)
        return "OK"

    async def query(self, sql: str, *params: object) -> list[dict[str, object]]:
        self.executed.append(sql)
        return []


@pytest.fixture(scope="module")
def captured_ddl() -> list[str]:
    """Run every registered 3tears-scrape migration version against a fake
    store and return the full list of executed SQL strings, in registration
    order."""
    import asyncio

    async def _capture() -> list[str]:
        runner = MigrationRunner()
        pkg = register(runner)
        store = _FakeStore()
        for version_num in sorted(pkg.versions.keys()):
            await pkg.versions[version_num](store)
        return store.executed

    return asyncio.run(_capture())


def _ddl_columns(table_name: str, statements: list[str]) -> set[str]:
    """Return every column name defined for *table_name* across every captured statement."""
    columns: set[str] = set()
    for stmt in statements:
        if table_name not in stmt:
            continue
        columns.update(_CREATE_COLUMN_RE.findall(stmt))
        columns.update(_ADD_COLUMN_RE.findall(stmt))
    return columns


def test_registered_versions_are_sequential_starting_at_one():
    runner = MigrationRunner()
    pkg = register(runner)
    versions = sorted(pkg.versions.keys())
    assert versions == list(range(1, len(versions) + 1)), f"non-sequential versions: {versions}"


def test_target_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeTarget exposes must have a matching scrape_targets column."""
    persisted_fields = {
        "target_id",
        "url",
        "driver_backend",
        "rate_limit_key",
        "cadence",
        "multi_row",
        "wait_for",
        "field_schema",
        "nav_steps",
        "extraction_strategy_type",
        "api_results_path",
        "api_fragment_field",
        "timeout_seconds",
    }
    columns = _ddl_columns("scrape_targets", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeTarget fields with no matching scrape_targets DDL column: {missing}"


def test_recipe_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeRecipe exposes must have a matching scrape_recipes column."""
    persisted_fields = {
        "target_id",
        "extraction_strategy",
        "won_at",
        "last_validated_at",
        "consecutive_validation_failures",
    }
    columns = _ddl_columns("scrape_recipes", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeRecipe fields with no matching scrape_recipes DDL column: {missing}"


def test_extraction_fields_covered_by_ddl(captured_ddl: list[str]):
    """Every field ScrapeExtraction exposes must have a matching scrape_extractions column."""
    persisted_fields = {
        "id",
        "target_id",
        "extraction_recipe_id",
        "source_url",
        "retrieved_at",
        "structured_fields",
        "field_confidences",
        "enrichment_notes",
        "validation_status",
    }
    columns = _ddl_columns("scrape_extractions", captured_ddl)
    missing = persisted_fields - columns
    assert not missing, f"ScrapeExtraction fields with no matching scrape_extractions DDL column: {missing}"


def test_every_scrape_table_has_date_created_and_date_updated(captured_ddl: list[str]):
    """BaseCollection.save_entity() unconditionally stamps date_created/date_updated
    on every upsert regardless of what a collection's entity class exposes --
    every scrape table must declare both from the start (the exact failure mode
    faidh's own v018/v019/v022/v023 migrations document and fixed)."""
    for table in ("scrape_targets", "scrape_recipes", "scrape_extractions"):
        columns = _ddl_columns(table, captured_ddl)
        assert "date_created" in columns, f"{table} is missing date_created"
        assert "date_updated" in columns, f"{table} is missing date_updated"
