"""Loyalty points for the farm shop's card: what a basket earns at the till.

:func:`points_for` works out a basket's points from its lines and the member's tier:

- **Eligible spend.** Every line counts except those in :data:`NO_POINTS` (tobacco, lottery tickets, gift cards and
  postage stamps). A refunded line has a negative price and takes its amount off.
- **Base points.** One point for every whole pound of eligible spend: £23.99 earns 23. A basket whose eligible spend
  is nothing or less earns nothing at all, bonuses included.
- **Tier bonus**, on top of the base points:

  - ``gold``: the base points again (double points), on every basket;
  - ``silver``: the base points again when the eligible spend is at least £40;
  - ``bronze``: a flat 25 bonus points when the eligible spend is at least £60.

  A basket at exactly the threshold gets the bonus.

:func:`statement` adds up a member's baskets for the month.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TILL_EXPORT = Path(__file__).with_name("data") / "till_export.jsonl"

#: Categories that earn no points.
NO_POINTS = frozenset({"tobacco", "lottery", "gift card", "stamps"})
TIERS = ("bronze", "silver", "gold")
#: The eligible spend, in pence, from which a silver member's points are doubled.
SILVER_DOUBLE_FROM = 4000
#: The eligible spend, in pence, from which a bronze member gets :data:`BRONZE_BONUS`.
BRONZE_BONUS_FROM = 6000
BRONZE_BONUS = 25


@dataclass(frozen=True)
class Line:
    item: str
    category: str
    pence: int


@dataclass(frozen=True)
class Points:
    eligible_pence: int
    base: int
    bonus: int

    @property
    def total(self) -> int:
        return self.base + self.bonus


def eligible_spend(lines: list[Line]) -> int:
    """The pence in ``lines`` that earn points."""
    return sum(line.pence for line in lines if line.category.lower() not in NO_POINTS)


def _tier_bonus(tier: str, base: int, spend: int) -> int:
    if tier == "gold":
        return base
    if tier == "silver":
        return base if spend >= SILVER_DOUBLE_FROM else 0
    if tier == "bronze":
        return BRONZE_BONUS if spend <= BRONZE_BONUS_FROM else 0
    raise ValueError(f"no tier {tier!r}; the tiers are {', '.join(TIERS)}")


def points_for(lines: list[Line], tier: str) -> Points:
    """The points a basket of ``lines`` earns a member of ``tier``."""
    spend = eligible_spend(lines)
    if spend <= 0:
        if tier not in TIERS:
            raise ValueError(f"no tier {tier!r}; the tiers are {', '.join(TIERS)}")
        return Points(eligible_pence=spend, base=0, bonus=0)
    base = spend // 100
    return Points(eligible_pence=spend, base=base, bonus=_tier_bonus(tier, base, spend))


def basket_lines(row: dict[str, object]) -> list[Line]:
    """The lines of a till-export basket, ``{"lines": [{"item", "category", "pence"}, ...]}``."""
    lines = row["lines"]
    if not isinstance(lines, list):
        raise ValueError("a basket's lines are a list")
    return [Line(item=str(line["item"]), category=str(line["category"]), pence=int(line["pence"])) for line in lines]


def statement(baskets: list[dict[str, object]]) -> int:
    """A member's points for the month, from their baskets in the till export."""
    return sum(points_for(basket_lines(row), str(row["tier"])).total for row in baskets)


def load_export(path: Path = TILL_EXPORT) -> list[dict[str, object]]:
    """The baskets in a till export, one JSON object a line."""
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines() if text.strip()]
