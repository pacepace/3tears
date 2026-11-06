"""enforcement: a collection that reads its store its own way also reads it on a caller's transaction.

A save that leaves columns for the database to decide (``columns_decided_by_store``) reads its row
back so the handle holds the row L3 holds. Inside a caller's transaction
(``save_entity(entity, conn=conn)``) that read is ``fetch_from_store_in_transaction``, over the
caller's connection. A class that overrides ``fetch_from_store`` reads its own way, which the
framework cannot know, so unless it implements the in-transaction read too, its handle holds the
row as sent and a column the row left out raises ``AttributeError`` (v0.78.0).

**The rule:** a collection class that overrides ``fetch_from_store`` and inherits a
``columns_decided_by_store`` that may name columns defines ``fetch_from_store_in_transaction`` at
or below its ``fetch_from_store``, with the same SELECT bound to ``conn``. A class whose store
decides no columns keeps ``BaseCollection.columns_decided_by_store``, and is never read back.
Enumerated by the collection census (:mod:`threetears.enforcement.collection_census`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from threetears.enforcement.collection_census import find_transaction_read_gaps, run_census

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: every workspace member's source tree, discovered so a new package is covered the day it exists.
_SRC_GLOBS: Final[tuple[str, ...]] = ("packages/*/src", "packages/agent/*/src")

_PLANTED: Final[str] = """
from typing import Any

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.declared_table import DeclaredTableCollection


class ReadsItsOwnWay(DeclaredTableCollection):
    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None


class ReadsInTransactionToo(ReadsItsOwnWay):
    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        return None


class StoreDecidesNothing(BaseCollection):
    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None
"""


def test_every_collection_that_reads_its_store_its_own_way_reads_it_on_a_caller_transaction() -> None:
    roots = sorted({path for pattern in _SRC_GLOBS for path in _REPO_ROOT.glob(pattern) if path.is_dir()})
    census = run_census(roots)
    assert not census["import_failures"], census["import_failures"]
    gaps = find_transaction_read_gaps(census)
    assert not gaps, (
        "a save of these joining a caller's transaction hands back the row as sent; implement "
        "fetch_from_store_in_transaction with the class's own SELECT over conn:\n  " + "\n  ".join(gaps)
    )


def test_the_census_finds_a_class_that_reads_its_own_way_and_nothing_in_a_transaction(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "planted_collections.py").write_text(_PLANTED)
    gaps = find_transaction_read_gaps(run_census([root]))
    assert [gap.split(":", 1)[0] for gap in gaps] == ["planted_collections.ReadsItsOwnWay"]
