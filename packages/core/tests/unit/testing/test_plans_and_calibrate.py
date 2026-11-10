"""The shared plan check reads rows scanned and looped scans out of a plan; a budget is a multiple of its baseline."""

from __future__ import annotations

import pytest

from threetears.core.testing.calibrate import budget_problem
from threetears.core.testing.plans import StatementPlan, looped_seq_scans, plan_problems, rows_scanned


def _scan(node: str, rows: int, *, removed: int = 0, loops: int = 1, parent: str = "Outer", **extra: object) -> dict:
    return {
        "Node Type": node,
        "Relation Name": "t",
        "Actual Rows": rows,
        "Actual Loops": loops,
        "Rows Removed by Filter": removed,
        "Parent Relationship": parent,
        **extra,
    }


class TestRowsScanned:
    def test_postgres_counts_what_a_scan_answered_and_removed_each_loop(self) -> None:
        """Flips if a filter's removed rows, or a scan's loops, stop counting as rows read."""
        plan = {"Node Type": "Limit", "Actual Rows": 10, "Plans": [_scan("Index Scan", 10, removed=90, loops=2)]}
        assert rows_scanned(plan) == 200

    def test_yugabytedb_counts_its_storage_rows_the_larger_of_table_and_index(self) -> None:
        """Flips if a YugabyteDB scan's storage counts are ignored for the rows its PG layer answered."""
        node = _scan("Index Scan", 10, **{"Storage Table Rows Scanned": 10, "Storage Index Rows Scanned": 5000})
        assert rows_scanned(node) == 5000

    def test_a_node_reading_no_relation_counts_nothing(self) -> None:
        """Flips if a sort or a bitmap's index half is counted as a second read of the same rows."""
        assert rows_scanned({"Node Type": "Bitmap Index Scan", "Index Name": "i", "Actual Rows": 500}) == 0

    def test_a_write_node_counts_nothing_beside_the_scan_under_it(self) -> None:
        """Flips if a write's ModifyTable node (it names the relation it writes, and a RETURNING write answers its
        rows) is counted as a second read of the rows its scan found: DELETE ... RETURNING of 30 rows read 30."""
        plan = {
            "Node Type": "ModifyTable",
            "Operation": "Delete",
            "Relation Name": "results",
            "Actual Rows": 30,
            "Plans": [_scan("Index Scan", 30)],
        }
        assert rows_scanned(plan) == 30


class TestLoopedScans:
    def test_a_large_sequential_scan_inside_a_loop_is_named_and_a_small_one_is_not(self) -> None:
        """Flips if a scan repeated per outer row of a nested loop is missed, or a tiny one is flagged."""
        large = {
            "Node Type": "Nested Loop",
            "Plans": [_scan("Seq Scan", 3), _scan("Seq Scan", 5000, parent="Inner", loops=3)],
        }
        small = {
            "Node Type": "Nested Loop",
            "Plans": [_scan("Seq Scan", 3), _scan("Seq Scan", 20, parent="Inner", loops=3)],
        }
        assert list(looped_seq_scans(large)) == ["t: 5000 rows x 3 loops"]
        assert list(looped_seq_scans(small)) == []


class TestProblems:
    def _plan(self, returned: int, scanned: int) -> StatementPlan:
        return StatementPlan(
            sql="SELECT 1", rows_returned=returned, rows_scanned=scanned, looped=(), seconds=0.0, plan={}
        )

    def test_a_ratio_is_judged_a_statement_or_a_read_at_a_time(self) -> None:
        """Flips if a read scanning past the ratio passes, or one within it fails."""
        plans = [self._plan(1000, 1050), self._plan(1000, 1300)]
        assert plan_problems(plans, scan_ratio=1.1, together=True) == [
            "2 statements scanned 2350 rows for 2000 returned (1.18x, over 1.1x)"
        ]
        assert len(plan_problems(plans, scan_ratio=1.1)) == 1
        assert plan_problems([self._plan(0, 1)], scan_ratio=1.1) == []


@pytest.mark.parametrize(("took", "baseline", "fails"), [(0.019, 0.01, False), (0.021, 0.01, True)])
def test_a_budget_is_twice_its_calibrated_baseline(took: float, baseline: float, fails: bool) -> None:
    """Flips if a path past twice its baseline passes, or one within it fails."""
    assert (budget_problem("path", took, baseline) is not None) is fails
