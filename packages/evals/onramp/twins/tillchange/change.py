"""Change from the till drawer.

The drawer is the count of each note and coin in it, as the cash office's float file writes it: a JSON object from
the denomination, in pounds as printed on the note or coin (``"20"``, ``"0.50"``), to how many there are.

Change is given largest denomination first: as many of the largest as fit in the amount still owed and the drawer
holds, then as many of the next largest, and so on down to the penny. With a full drawer that is the fewest notes and
coins that make the amount. When the drawer can't make the amount, the till says so and the cashier calls a
supervisor rather than give short change.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path


class CannotMakeChange(Exception):
    """The drawer doesn't hold the notes and coins to make the amount."""


def pence(denomination: str) -> int:
    """A denomination's value in pence."""
    return int(Decimal(denomination) * 100)


def load_drawer(path: str | Path) -> dict[str, int]:
    """A float file: denomination to count."""
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def make_change(amount_pence: int, drawer: dict[str, int]) -> dict[str, int]:
    """The notes and coins to give for ``amount_pence``: denomination to how many, leaving out those not given."""
    if amount_pence < 0:
        raise ValueError("change can't be negative")
    owed = amount_pence
    given: dict[str, int] = {}
    for denomination in sorted(drawer, key=pence, reverse=True):
        count = min(owed // pence(denomination), drawer[denomination])
        if count:
            given[denomination] = count
            owed -= count * pence(denomination)
    if owed:
        raise CannotMakeChange(f"the drawer can't make {amount_pence}p ({owed}p short)")
    return given


def change_for(total_pence: int, tendered_pence: int, drawer: dict[str, int]) -> dict[str, int]:
    """The change for a sale of ``total_pence`` paid with ``tendered_pence``."""
    if tendered_pence < total_pence:
        raise ValueError(f"{tendered_pence}p doesn't cover a sale of {total_pence}p")
    return make_change(tendered_pence - total_pence, drawer)
