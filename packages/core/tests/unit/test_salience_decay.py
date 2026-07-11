"""Unit tests for the shared salience-decay helper.

Verifies the SQL shape, parameter binding, table interpolation, and
status-tag row-count parsing of :func:`apply_salience_decay` against a
fake L3 pool -- no database. The end-to-end decay math (cadence
independence, floor asymptote) is exercised in the agent-memory
integration suite against real Postgres.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.collections.salience import apply_salience_decay


class _RecordingPool:
    """Minimal L3Backend stand-in capturing the executed SQL + params."""

    def __init__(self, status: str = "UPDATE 0") -> None:
        self.status = status
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any, namespace: str | None = None) -> str:
        self.calls.append((query, params))
        return self.status


class TestApplySalienceDecaySql:
    async def test_emits_clamped_anchored_decay_update(self) -> None:
        pool = _RecordingPool("UPDATE 5")

        count = await apply_salience_decay(
            pool,  # type: ignore[arg-type]
            table="memories",
            half_life_seconds=100.0,
            floor=0.1,
        )

        assert count == 5
        assert len(pool.calls) == 1
        sql, params = pool.calls[0]
        # clamp to [floor, 1.0]
        assert "LEAST(1.0, GREATEST($1" in sql
        # exponential decay factor anchored on last_decayed_at (fallback
        # date_created), scaled by the half-life param
        assert "power(" in sql
        assert "COALESCE(last_decayed_at, date_created)" in sql
        assert "/ $2" in sql
        # stamps the decay anchor and skips evergreen rows
        assert "last_decayed_at = now()" in sql
        assert "WHERE NOT evergreen" in sql
        # floor + half-life ride bound params (never string-formatted)
        assert params == (0.1, 100.0)

    async def test_table_name_is_interpolated(self) -> None:
        pool = _RecordingPool("UPDATE 0")

        await apply_salience_decay(
            pool,  # type: ignore[arg-type]
            table="intentions",
            half_life_seconds=1.0,
            floor=0.2,
        )

        sql, _ = pool.calls[0]
        assert sql.startswith("UPDATE intentions ")

    async def test_skip_evergreen_false_omits_where_clause(self) -> None:
        """tables without an ``evergreen`` column (intentions) drop the guard."""
        pool = _RecordingPool("UPDATE 3")

        count = await apply_salience_decay(
            pool,  # type: ignore[arg-type]
            table="intentions",
            half_life_seconds=100.0,
            floor=0.1,
            skip_evergreen=False,
        )

        assert count == 3
        sql, params = pool.calls[0]
        # no evergreen predicate -- every row is aged
        assert "evergreen" not in sql
        assert "WHERE" not in sql
        # the decay math + anchor are unchanged
        assert "last_decayed_at = now()" in sql
        assert "COALESCE(last_decayed_at, date_created)" in sql
        assert params == (0.1, 100.0)


class TestRowCountParsing:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("UPDATE 42", 42),
            ("UPDATE 0", 0),
            ("UPDATE 1", 1),
        ],
    )
    async def test_parses_asyncpg_status_tag(self, status: str, expected: int) -> None:
        pool = _RecordingPool(status)
        count = await apply_salience_decay(
            pool,  # type: ignore[arg-type]
            table="memories",
            half_life_seconds=1.0,
            floor=0.1,
        )
        assert count == expected

    @pytest.mark.parametrize("status", ["", "UPDATE", "weird"])
    async def test_malformed_status_falls_back_to_zero(self, status: str) -> None:
        pool = _RecordingPool(status)
        count = await apply_salience_decay(
            pool,  # type: ignore[arg-type]
            table="memories",
            half_life_seconds=1.0,
            floor=0.1,
        )
        assert count == 0
