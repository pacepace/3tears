"""Reorder points for the stockroom: when to order a line again, and how much to order.

A line's daily demand is what it sold over its review window divided by the days in that window, kept as a fraction of
a unit: 30 sold in 28 days is about 1.07 a day (:func:`daily_demand`).

Its reorder point covers the demand expected while an order is on its way plus a safety stock, both counted in days of
demand, and is rounded up to a whole unit: ``ceil(daily demand x (lead days + safety days))`` (:func:`reorder_point`).

A line is due when what we hold plus what is already on order is at or below its reorder point. A due line orders
enough to bring it up to the reorder point plus :data:`COVER_DAYS` days of demand, rounded up to whole packs from the
supplier; a line that is not due orders nothing (:func:`order_quantity`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

REVIEW = Path(__file__).with_name("data") / "stock_review.jsonl"
#: Days of demand an order covers beyond the reorder point.
COVER_DAYS = 14


@dataclass(frozen=True)
class Line:
    sku: str
    name: str
    sold: int
    window_days: int
    lead_days: int
    safety_days: int
    on_hand: int
    on_order: int
    pack: int


def daily_demand(sold: int, days: int) -> float:
    """Units a day, from ``sold`` units over ``days`` days."""
    if days <= 0:
        raise ValueError(f"a review window of {days} days")
    if sold < 0:
        raise ValueError(f"{sold} units sold")
    return sold / days


def _units(demand: float, days: int) -> int:
    # Rounded to six places first, so 3/7 of a unit a day over 7 days is 3 and not 3.0000000000000004.
    return math.ceil(round(demand * days, 6))


def reorder_point(line: Line) -> int:
    """The stock level at or below which ``line`` is ordered again."""
    return _units(daily_demand(line.sold, line.window_days), line.lead_days + line.safety_days)


def order_quantity(line: Line) -> int:
    """Units to order for ``line`` today: whole packs, or nothing when the line is not due."""
    point = reorder_point(line)
    position = line.on_hand + line.on_order
    if position > point:
        return 0
    target = point + _units(daily_demand(line.sold, line.window_days), COVER_DAYS)
    packs = math.ceil((target - position) / line.pack)
    return max(packs, 0) * line.pack


def load_lines(path: Path = REVIEW) -> list[Line]:
    """The lines in a stock review file, one JSON object a line; fields beyond :class:`Line`'s are ignored."""
    lines = []
    for text in path.read_text(encoding="utf-8").splitlines():
        if text.strip():
            row = json.loads(text)
            lines.append(Line(**{name: row[name] for name in Line.__dataclass_fields__}))
    return lines


def order_sheet(lines: list[Line]) -> list[tuple[str, int, int]]:
    """``(sku, reorder point, units to order)`` for every line, in the order given."""
    return [(line.sku, reorder_point(line), order_quantity(line)) for line in lines]
