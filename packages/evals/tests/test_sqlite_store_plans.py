"""The SQLite document store's listings read through ``documents_by_type``, never the table whole.

A listing of one scope's documents of one type (``by_doc_type``, ``iter_by_doc_type``) is the shared keyed listing's
statement (:func:`threetears.interfaces.keyed.keyset_rows_statement`, a page at a time in ``seq`` order, or one
bounded first page in insertion order): its pinned ``(scope_id, doc_type)`` is the index's own condition. SQLite's
``EXPLAIN QUERY PLAN`` of each statement, on a file seeded with many scopes and types, shows it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.storage.sqlite import SqliteDocumentStore
from threetears.interfaces.keyed import Sort, keyset_page_statement, keyset_rows_statement, sqlite_numbered

plans = pytest.importorskip("threetears.core.testing.plans")


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    """a store file holding 20 scopes of 5 types, 30 documents each, then analysed."""
    path = tmp_path / "plans.db"
    with SqliteDocumentStore(path) as store:
        for scope in range(20):
            for kind in range(5):
                for number in range(30):
                    store.upsert(
                        {"scope_id": f"s{scope}", "id": f"d{kind}-{number}", "doc_type": f"t{kind}", "rank": number}
                    )
    with sqlite3.connect(path) as db:
        db.execute("ANALYZE")
    return path


def _plan(path: Path, statement: tuple[str, list[Any]]) -> list[Any]:
    sql, params = statement
    with sqlite3.connect(path) as db:
        return db.execute(f"EXPLAIN QUERY PLAN {sqlite_numbered(sql)}", params).fetchall()


def test_a_scopes_documents_of_one_type_are_read_through_documents_by_type(seeded: Path) -> None:
    """Flips if a listing's page (its first, or one past a place, as ``_listing`` reads them), or ``by_doc_type``'s
    bounded first page, stops finding its rows through the index on (scope_id, doc_type)."""
    by = {"scope_id": "s7", "doc_type": "t3"}
    by_seq = (Sort("seq", cast=None),)
    first = keyset_rows_statement("documents", columns=["id", "seq"], by=by, order=by_seq, by_row=True)
    later = keyset_rows_statement("documents", columns=["id", "seq"], by=by, order=by_seq, past=[10], by_row=True)
    bounded = keyset_page_statement("documents", columns=("body", "seq"), by=by, order=by_seq, limit=3)
    for statement in (first, later, bounded):
        assert plans.plan_uses_index(_plan(seeded, statement), "documents_by_type"), statement[0]


def test_the_plan_check_sees_a_read_that_skips_the_index(seeded: Path) -> None:
    """Non-vacuity: a listing pinning neither column walks the table by its rowid, not the index."""
    by_seq = (Sort("seq", cast=None),)
    statement = keyset_rows_statement("documents", columns=["id", "seq"], by={}, order=by_seq)
    assert not plans.plan_uses_index(_plan(seeded, statement), "documents_by_type")
