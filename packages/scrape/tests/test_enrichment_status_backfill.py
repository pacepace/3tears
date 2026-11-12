"""v013's backfill reads ``scrape_extractions`` a page at a time by key and writes each outcome by the rows' keys: never
one ``UPDATE`` of every row (the real-Postgres translation is proved in
``integration/test_enrichment_status_migration.py``)."""

from __future__ import annotations

from typing import Any

from threetears.scrape.migrations import LEGACY_EMPTY_ENRICHMENT_FAILURE, v013_extraction_enrichment_status


class _Session:
    """a migration's session over held rows: every statement recorded, a keyset page answered from the rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append((" ".join(sql.split()), params))
        return "UPDATE 1" if sql.startswith("UPDATE") else "ALTER TABLE"

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((" ".join(sql.split()), params))
        return self.rows if "FROM scrape_extractions" in sql and "> (" not in sql else []


async def test_v013_writes_each_outcome_by_the_rows_keys() -> None:
    """Flips if v013 writes ``scrape_extractions`` by anything but the keys of the rows it read, or writes a row that
    has a status or never ran."""
    session = _Session(
        [
            {"id": "a", "enrichment_status": None, "enrichment_notes": None},
            {"id": "b", "enrichment_status": None, "enrichment_notes": '{"context": "q3"}'},
            {"id": "c", "enrichment_status": None, "enrichment_notes": "{}"},
            {"id": "d", "enrichment_status": None, "enrichment_notes": '"{}"'},
            {"id": "e", "enrichment_status": "enriched", "enrichment_notes": '{"x": 1}'},
        ]
    )

    await v013_extraction_enrichment_status(session)  # type: ignore[arg-type]

    reads = [sql for sql, _ in session.statements if sql.startswith("SELECT") and "scrape_extractions" in sql]
    assert reads == [
        'SELECT "id", "enrichment_status", "enrichment_notes" FROM scrape_extractions ORDER BY "id" LIMIT 501'
    ]
    writes = [(sql, params) for sql, params in session.statements if sql.startswith("UPDATE")]
    assert writes == [
        ('UPDATE "scrape_extractions" SET "enrichment_status" = $1 WHERE "id" = ANY($2::text[])', ("enriched", ["b"])),
        (
            'UPDATE "scrape_extractions" SET "enrichment_status" = $1, "enrichment_notes" = $2, '
            '"enrichment_failure" = $3 WHERE "id" = ANY($4::text[])',
            ("failed", None, LEGACY_EMPTY_ENRICHMENT_FAILURE, ["c", "d"]),
        ),
    ]
