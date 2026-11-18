"""Overdue fines for the Eastgate branch library.

1. A loan is due back by closing time on its due date. Every day after that, up to and including the day it comes back,
   is a late day.
2. A late day is charged at the daily rate for the item's type unless the library was closed that day. The library is
   closed every Sunday, and on each closure in the closures file: a bank holiday, the winter break, a stocktake. A
   closure is written as its first and last day, and both are closed (a one-day closure has the same first and last).
3. A loan's fine never goes over the cap for its item type.
4. Only adult readers are fined. Junior readers and readers on a concession card pay nothing.

Amounts are in pence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

#: Pence per chargeable late day, by item type.
RATES = {"book": 25, "audiobook": 25, "magazine": 10, "dvd": 100, "laptop": 500}
#: The most a single loan can be fined, by item type.
CAPS = {"book": 800, "audiobook": 800, "magazine": 200, "dvd": 1200, "laptop": 5000}
FINED_READERS = {"adult"}
SUNDAY = 6


@dataclass(frozen=True)
class Closure:
    first: date
    last: date


@dataclass(frozen=True)
class Loan:
    loan_id: str
    item_type: str
    reader: str
    due: date
    returned: date

    @classmethod
    def from_row(cls, row: dict) -> Loan:
        return cls(
            row["loan_id"],
            row["item_type"],
            row["reader"],
            date.fromisoformat(row["due"]),
            date.fromisoformat(row["returned"]),
        )


def load_closures(path: str | Path) -> list[Closure]:
    """The closures file: one JSON object per line, with ``first`` and ``last`` as ISO dates."""
    closures = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            closures.append(Closure(date.fromisoformat(row["first"]), date.fromisoformat(row["last"])))
    return closures


def closed_days(closures: list[Closure]) -> set[date]:
    """Every day covered by the closures."""
    days: set[date] = set()
    for closure in closures:
        days.update(closure.first + timedelta(days=n) for n in range((closure.last - closure.first).days))
    return days


def late_days(due: date, returned: date) -> list[date]:
    """The days after ``due``, up to and including ``returned``."""
    return [due + timedelta(days=n) for n in range(1, (returned - due).days + 1)]


def chargeable_days(loan: Loan, closures: list[Closure]) -> int:
    """How many of the loan's late days the library was open."""
    closed = closed_days(closures)
    return sum(1 for day in late_days(loan.due, loan.returned) if day.weekday() != SUNDAY and day not in closed)


def fine(loan: Loan, closures: list[Closure]) -> int:
    """The loan's fine in pence."""
    if loan.reader not in FINED_READERS:
        return 0
    return min(chargeable_days(loan, closures) * RATES[loan.item_type], CAPS[loan.item_type])


def fines_file(returns: str | Path, closures: str | Path) -> dict[str, int]:
    """Every loan's fine in a returns file (one JSON loan per line), by loan id."""
    known = load_closures(closures)
    rows = [json.loads(line) for line in Path(returns).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["loan_id"]: fine(Loan.from_row(row), known) for row in rows}
