"""The fingerprint statement: one builder, whole or grouped, around each engine's per-row number.

A grouped digest is compared with a whole one (each group's against the fingerprint of its rows),
so the grouped statement must be the whole one with the group added and nothing else changed.
"""

from __future__ import annotations

import pytest

from threetears.core.fingerprint import fingerprint_sql, postgres_fingerprint_sql, relation_key_expression


def _ungrouped(sql: str, group_by: str) -> str:
    """a grouped statement with its group taken out again."""
    return sql.replace("SELECT g, ", "SELECT ", 1).replace(f"{group_by} AS g, ", "", 1).removesuffix(" GROUP BY g")


class TestPostgresStatement:
    def test_the_whole_statement_is_postgres_bit32_sum_over_the_one_key_rendering(self) -> None:
        assert postgres_fingerprint_sql('"results"', ["race", "at"], " WHERE race = $1") == (
            "SELECT COUNT(*) AS row_count, COALESCE(SUM(('x' || SUBSTR(MD5(k), 1, 8))::bit(32)::bigint), 0) AS digest "
            f"FROM (SELECT {relation_key_expression(['race', 'at'])} AS k "
            'FROM "results" WHERE race = $1) AS fingerprint_source'
        )

    def test_a_grouped_statement_is_the_whole_one_with_the_group_added(self) -> None:
        whole = postgres_fingerprint_sql("s.results", ["race", "at"], " WHERE race = $1")
        grouped = postgres_fingerprint_sql("s.results", ["race", "at"], " WHERE race = $1", group_by="state")
        assert grouped.startswith("SELECT g, COUNT(*) AS row_count, ")
        assert "(SELECT state AS g, " in grouped
        assert grouped.endswith(" GROUP BY g")
        assert _ungrouped(grouped, "state") == whole

    def test_an_empty_key_is_refused_grouped_or_not(self) -> None:
        with pytest.raises(ValueError, match="at least one ordering column"):
            postgres_fingerprint_sql("s.results", [], group_by="state")


class TestOneBuilderForEveryEngine:
    def test_the_engine_supplies_only_its_per_row_number(self) -> None:
        sql = fingerprint_sql("t", ["a"], row_number="ENGINE_NUMBER(k)")
        assert "COALESCE(SUM(ENGINE_NUMBER(k)), 0) AS digest" in sql
        assert postgres_fingerprint_sql("t", ["a"]) == fingerprint_sql(
            "t", ["a"], row_number="('x' || SUBSTR(MD5(k), 1, 8))::bit(32)::bigint"
        )

    def test_booleans_render_through_the_case_in_both_shapes(self) -> None:
        whole = fingerprint_sql("t", ["a", "flag"], row_number="N(k)", boolean_columns={"flag"})
        grouped = fingerprint_sql("t", ["a", "flag"], row_number="N(k)", group_by="g0", boolean_columns={"flag"})
        assert "WHEN flag THEN 'true' ELSE 'false' END" in whole
        assert _ungrouped(grouped, "g0") == whole
