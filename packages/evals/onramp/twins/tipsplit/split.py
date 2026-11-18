"""Splitting a table's bill between the diners.

A bill is what the table ate and drank (the subtotal), the tip the table chose as a whole percentage, and a voucher, if
the table has one, worth a fixed amount off.

1. The tip is on the full subtotal, before the voucher: the staff served the whole meal, whoever pays for part of it.
2. The voucher then comes off the tipped amount. The table's total is that, to the nearest penny (a half penny rounds
   up), and never below zero.
3. The total is shared as evenly as whole pennies allow. When it does not divide exactly, the diners first on the bill
   pay a penny more each, until the pennies left over are used up.

Amounts are in pence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


@dataclass(frozen=True)
class Bill:
    table: str
    subtotal_pence: int
    tip_percent: int
    voucher_pence: int
    diners: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict) -> Bill:
        return cls(
            row["table"],
            row["subtotal_pence"],
            row.get("tip_percent", 0),
            row.get("voucher_pence", 0),
            tuple(row["diners"]),
        )


def total_pence(bill: Bill) -> int:
    """What the table pays altogether."""
    exact = Decimal(bill.subtotal_pence) * (100 + bill.tip_percent) / 100 - bill.voucher_pence
    return max(int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP)), 0)


def split(bill: Bill) -> dict[str, int]:
    """Each diner's share of the total, in the order they are on the bill."""
    if not bill.diners:
        raise ValueError(f"table {bill.table}: a bill needs at least one diner")
    each, left_over = divmod(total_pence(bill), len(bill.diners))
    return {name: each + (1 if place < left_over else 0) for place, name in enumerate(bill.diners)}


def split_file(path: str | Path) -> dict[str, dict[str, int]]:
    """Every bill in a file of bills (one JSON bill per line), split, by table."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["table"]: split(Bill.from_row(row)) for row in rows}
