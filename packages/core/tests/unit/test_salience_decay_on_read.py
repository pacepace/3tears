"""Salience decay on read (Pace's ruling 2026-11-11): the effective salience a reader sees is the value the removed
sweep (``apply_salience_decay``) gave at the same instant, it is cadence-free, its SQL computes the same number, and its
ranking expression orders as the effective salience does. The collections' on / off statements are pinned in
agent/memory's tests (off is their default)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from threetears.core.collections import SalienceDecay

_NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1)


def _swept(base: float, stamped_at: datetime, at: datetime, half_life_seconds: float, floor: float) -> float:
    """the removed sweep's formula at ``at``: ``LEAST(1, GREATEST(floor, salience * 0.5^(age / half_life)))``."""
    age = (at - stamped_at).total_seconds()
    return min(1.0, max(floor, base * math.pow(0.5, age / half_life_seconds)))


@pytest.mark.parametrize(
    ("base", "days"), [(1.0, 0.0), (0.9, 30.0), (0.5, 60.0), (0.2, 3650.0), (0.05, 1.0), (0.0, 10.0)]
)
def test_the_effective_salience_is_the_old_sweep_s_value_at_the_same_instant(base: float, days: float) -> None:
    """Flips if a read gives anything but what one sweep at the read's instant stored."""
    stamped = _NOW - timedelta(days=days)
    assert _DECAY.effective(base, stamped, _NOW) == pytest.approx(
        _swept(base, stamped, _NOW, timedelta(days=60).total_seconds(), 0.1)
    )


def test_it_is_cadence_free_however_the_sweeps_were_split() -> None:
    """Flips if a read differs from the sweeps run at any cadence before it (20 days, then 10, then a read at 45),
    floor included."""
    for base in (1.0, 0.6, 0.12):
        created = _NOW - timedelta(days=45)
        swept = base
        stamp = created
        for at in (created + timedelta(days=20), created + timedelta(days=30)):
            swept = _swept(swept, stamp, at, timedelta(days=60).total_seconds(), 0.1)
            stamp = at
        assert _DECAY.effective(swept, stamp, _NOW) == pytest.approx(_DECAY.effective(base, created, _NOW))


def test_a_pinned_row_never_decays() -> None:
    """Flips if an evergreen row's read decays."""
    pinned = SalienceDecay(half_life=timedelta(days=60), floor=0.1, pinned="evergreen")
    assert pinned.effective(0.8, _NOW - timedelta(days=3650), _NOW, pinned=True) == 0.8


def _sql_rows(db: duckdb.DuckDBPyConnection, expression: str, rows: list[tuple[float, datetime]]) -> list[float]:
    db.execute("CREATE OR REPLACE TABLE t (salience DOUBLE, last_decayed_at TIMESTAMPTZ, date_created TIMESTAMPTZ)")
    db.executemany("INSERT INTO t VALUES (?, ?, ?)", [(base, stamped, stamped) for base, stamped in rows])
    sql = expression.replace("now()", f"TIMESTAMPTZ '{_NOW.isoformat()}'").replace("::double precision", "::DOUBLE")
    return [float(value) for (value,) in db.execute(f"SELECT {sql} FROM t ORDER BY rowid").fetchall()]


def test_the_sql_reads_the_same_number_and_ranks_as_the_effective_salience_does() -> None:
    """Flips if the effective-salience SQL computes another number than the formula, or the index-able ranking
    expression orders rows (above the floor) otherwise than their effective salience."""
    rows = [
        (0.9, _NOW - timedelta(days=120)),
        (0.5, _NOW),
        (0.7, _NOW - timedelta(days=10)),
        (1.0, _NOW - timedelta(days=59)),
    ]
    db = duckdb.connect()
    # instants compared in UTC: a local zone's daylight change would put an hour in the age
    db.execute("SET TimeZone = 'UTC'")
    effective = _sql_rows(db, _DECAY.effective_sql(), rows)
    assert effective == pytest.approx([_DECAY.effective(base, stamped, _NOW) for base, stamped in rows])
    ranked = _sql_rows(db, _DECAY.rank_sql(), rows)
    assert sorted(range(len(rows)), key=lambda i: ranked[i]) == sorted(range(len(rows)), key=lambda i: effective[i])


def test_a_half_life_and_floor_out_of_range_are_refused() -> None:
    with pytest.raises(ValueError, match="half-life"):
        SalienceDecay(half_life=timedelta(0), floor=0.1)
    with pytest.raises(ValueError, match="floor"):
        SalienceDecay(half_life=timedelta(days=1), floor=1.5)
