"""Mid-cycle plan changes: what a subscriber is credited and charged when they change plan part way through a month.

A subscription bills on its anchor day each month, in advance, for the cycle that runs from that bill up to (not
including) the next one. When a subscriber changes plan on some day of the cycle, the change takes effect from the
start of that day and the next invoice carries two lines:

- a **credit** for the old plan's unused days: the old plan's monthly price times the days left in the cycle (the day
  of the change included) over the days in the cycle;
- a **charge** for the new plan over the same days: the new plan's monthly price times the same fraction.

Each line is rounded to the nearest penny on its own, a half penny rounding up, and the invoice's net is the charge
less the credit (negative for a downgrade, which leaves money on the account). A change on the anchor day itself
credits and charges a whole cycle.

Cycles follow the calendar: an anchor of the 31st falls on the last day of shorter months, so a cycle can be 28 to 31
days long.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

#: Monthly prices in pence.
PLANS = {"starter": 599, "plus": 1199, "team": 2899, "business": 7450}


@dataclass(frozen=True)
class Adjustment:
    credit_pence: int
    charge_pence: int

    @property
    def net_pence(self) -> int:
        return self.charge_pence - self.credit_pence


def bill_day(year: int, month: int, anchor: int) -> date:
    """The day a subscription with ``anchor`` bills in a month: the anchor, or the month's last day if shorter."""
    return date(year, month, min(anchor, calendar.monthrange(year, month)[1]))


def cycle(anchor: int, on: date) -> tuple[date, date]:
    """The cycle ``on`` falls in: its bill day, and the next bill day."""
    this_month = bill_day(on.year, on.month, anchor)
    if on >= this_month:
        start = this_month
    else:
        year, month = (on.year, on.month - 1) if on.month > 1 else (on.year - 1, 12)
        start = bill_day(year, month, anchor)
    year, month = (start.year, start.month + 1) if start.month < 12 else (start.year + 1, 1)
    return start, bill_day(year, month, anchor)


def share(pence: int, days: int, of_days: int) -> int:
    """``pence`` times ``days`` over ``of_days``, to the nearest penny."""
    return pence * days // of_days


def prorate(old_plan: str, new_plan: str, anchor: int, changed_on: date) -> Adjustment:
    """The credit and charge for moving from ``old_plan`` to ``new_plan`` on ``changed_on``."""
    start, end = cycle(anchor, changed_on)
    days, left = (end - start).days, (end - changed_on).days
    return Adjustment(share(PLANS[old_plan], left, days), share(PLANS[new_plan], left, days))
