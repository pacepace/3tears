"""The SQLite store passes the store conformance kit, and keeps what it holds past the connection.

* **It conforms.** Every case of ``STORE_CONFORMANCE_CASES`` runs against a fresh file, parametrised the
  way the kit's docstring tells an adopter to.
* **What is durable about it.** A second connection to the file (another process, in use) reads what the
  first wrote, and a conditional write is refused across connections as it is within one.
* **The file's layout is versioned**, and a file of a later layout is refused rather than misread.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from threetears.evals.schema import StoreConflict
from threetears.evals.storage import SQLITE_STORE_LAYOUT, SqliteDocumentStore
from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase


def _doc(doc_id: str, **fields: object) -> dict[str, object]:
    return {"id": doc_id, "scope_id": "scope-a", "doc_type": "eval_run", **fields}


@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_sqlite_store_conforms(case: StoreConformanceCase, tmp_path: Path) -> None:
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        case.run(store)


@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_sqlite_store_conforms_in_memory(case: StoreConformanceCase) -> None:
    with SqliteDocumentStore(":memory:") as store:
        case.run(store)


def test_a_second_connection_reads_what_the_first_wrote(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as first:
        first.upsert(_doc("r1", status="completed", score=0.5))
    with SqliteDocumentStore(path) as second:
        assert second.get("r1", "scope-a") == _doc("r1", status="completed", score=0.5)
        assert [d["id"] for d in second.by_doc_type("eval_run", "scope-a", status="completed")] == ["r1"]


def test_a_stale_etag_is_refused_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as one, SqliteDocumentStore(path) as other:
        one.upsert(_doc("r1", n=1))
        _, etag = one.get_with_etag("r1", "scope-a")
        other.upsert(_doc("r1", n=2))
        with pytest.raises(StoreConflict):
            one.upsert(_doc("r1", n=3), if_match=etag)
        assert one.get("r1", "scope-a") == _doc("r1", n=2)


def test_a_non_finite_number_round_trips(tmp_path: Path) -> None:
    # Stored models serialise NaN and infinity as JSON constants; the file must hand them back, not refuse them.
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        store.upsert(_doc("r1", value=math.nan, top=math.inf, rank=1))
        read = store.get("r1", "scope-a")
        assert read is not None
        assert math.isnan(read["value"]) and read["top"] == math.inf
        assert [d["id"] for d in store.by_doc_type("eval_run", "scope-a", rank=1, order_by="value")] == ["r1"]


def test_an_object_predicate_is_refused() -> None:
    with SqliteDocumentStore(":memory:") as store, pytest.raises(TypeError, match="scalar"):
        store.by_doc_type("eval_run", "scope-a", payload={"a": 1})


def test_a_file_of_a_later_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    SqliteDocumentStore(path).close()
    with sqlite3.connect(path) as raw:
        raw.execute(f"PRAGMA user_version = {SQLITE_STORE_LAYOUT + 1}")
    with pytest.raises(ValueError, match=f"layout {SQLITE_STORE_LAYOUT + 1}"):
        SqliteDocumentStore(path)
