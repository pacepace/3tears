"""An index's plan test asks the plan, not the test's word: plan_uses_index, and the purpose an index carries."""

from __future__ import annotations

from threetears.core.collections.schema_backed import Index
from threetears.core.testing.plans import plan_uses_index

_BY_KIND = {
    "Node Type": "Nested Loop",
    "Plans": [
        {"Node Type": "Seq Scan", "Relation Name": "customers"},
        {
            "Node Type": "Bitmap Heap Scan",
            "Relation Name": "datasources",
            "Plans": [
                {"Node Type": "Bitmap Index Scan", "Index Name": "datasources_by_kind", "Index Cond": "(kind = $1)"},
            ],
        },
    ],
}


def test_a_plan_reading_the_index_by_a_condition_uses_it() -> None:
    """Flips if an index found under a nested node by its index condition is not seen, or another index counts."""
    assert plan_uses_index(_BY_KIND, "datasources_by_kind")
    assert not plan_uses_index(_BY_KIND, "datasources_by_status")


def test_an_index_walked_with_no_condition_is_no_keyed_read() -> None:
    """Flips if an index scan with no index condition (every entry, in order) counts as the read using it."""
    walked = {"Node Type": "Index Scan", "Index Name": "datasources_by_kind", "Relation Name": "datasources"}
    assert not plan_uses_index(walked, "datasources_by_kind")


def test_a_keyset_page_walking_the_index_under_its_limit_uses_it() -> None:
    """Flips if an ordered walk of the index that a LIMIT stops (a keyset page's first rows) is not seen as a read
    of it."""
    page = {
        "Node Type": "Limit",
        "Plans": [{"Node Type": "Index Scan", "Index Name": "turn_dedup_date_updated", "Relation Name": "turn_dedup"}],
    }
    assert plan_uses_index(page, "turn_dedup_date_updated")


def test_sqlite_query_plan_rows_name_the_index_they_use() -> None:
    """Flips if SQLite's EXPLAIN QUERY PLAN rows, as mappings or tuples, are not read, or a scan of the table is
    taken for the index."""
    used = [{"detail": "SCAN turn_dedup USING INDEX turn_dedup_date_updated"}]
    assert plan_uses_index(used, "turn_dedup_date_updated")
    assert plan_uses_index([(2, 0, 0, "SEARCH t USING COVERING INDEX t_by_k (k=?)")], "t_by_k")
    assert not plan_uses_index([(2, 0, 0, "SCAN turn_dedup")], "turn_dedup_date_updated")


def test_purpose_plan_test_and_reason_never_make_two_indexes_differ() -> None:
    """Flips if the no-whole-reads notes an index carries change what it is, so a declared index stops equalling
    the one its migration made."""
    noted = Index(
        "datasources_by_kind",
        "kind",
        purpose="src/hub/schema_digest.py::reconcile_schema_digests",
        plan_test="tests/test_plans.py::test_reconcile_uses_datasources_by_kind",
        reason="the reconcile reads each kind's datasources by it",
    )
    assert noted == Index("datasources_by_kind", "kind")
    assert noted.purpose == "src/hub/schema_digest.py::reconcile_schema_digests"


def test_a_partial_index_read_whole_is_its_read_only_when_said_partial() -> None:
    """Flips if a scan of a partial index with no condition (the planner drops the predicate the index implies) is not
    its keyed read when the test says the index is partial, or is one when it does not."""
    counted = {
        "Node Type": "Aggregate",
        "Plans": [{"Node Type": "Index Only Scan", "Index Name": "notes_ring_read", "Relation Name": "notes"}],
    }
    assert plan_uses_index(counted, "notes_ring_read", partial=True)
    assert not plan_uses_index(counted, "notes_ring_read")
