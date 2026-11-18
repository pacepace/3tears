"""Warranty decisions for the repairs desk.

A claim is a product (its category, the date it was bought, and whether the customer registered it), the date the
claim was made, and the faults the technician logged when the product came in.

1. The warranty period is 12 months from the purchase date, or 24 months for a large appliance. A product registered
   within 30 days of purchase gets 12 months more. A claim made before the period runs out is in warranty; one made
   on the day it runs out (the same date, that many months on) or later is not.
2. Every fault is one of three kinds:
   - covered: a manufacturing fault, which the warranty repairs;
   - cosmetic: a scratch, a dent, faded colour. The warranty doesn't repair it, but it changes nothing else about
     the claim: the covered faults are still repaired;
   - excluding: damage the warranty never covers (liquid, a drop, a repair by someone else). One excluding fault
     refuses the whole claim.
3. A claim is decided as one of:
   - ``out_of_warranty``: made after the period;
   - ``refused``: in warranty, with an excluding fault;
   - ``not_covered``: in warranty, with no excluding fault, and no covered fault either;
   - ``accepted``: anything else. The repair is every covered fault, in the order logged.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

COVERED = frozenset(
    {
        "no_power",
        "battery_failure",
        "screen_failure",
        "motor_failure",
        "pump_failure",
        "door_seal",
        "overheating",
        "noisy_bearing",
        "control_board",
        "heating_element",
        "button_failure",
        "leaking_seal",
    }
)
COSMETIC = frozenset({"scratch", "dent", "discolouration", "chipped_paint"})
EXCLUDING = frozenset({"liquid_damage", "drop_damage", "unauthorised_repair", "power_surge"})
LARGE_APPLIANCES = frozenset({"washing_machine", "dishwasher", "fridge_freezer", "oven", "tumble_dryer"})
BASE_MONTHS = 12
LARGE_APPLIANCE_MONTHS = 24
REGISTRATION_MONTHS = 12
REGISTER_WITHIN_DAYS = 30


@dataclass(frozen=True)
class Claim:
    claim_id: str
    category: str
    bought: date
    registered: date | None
    claimed: date
    faults: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict) -> Claim:
        registered = row.get("registered")
        return cls(
            row["claim_id"],
            row["category"],
            date.fromisoformat(row["bought"]),
            date.fromisoformat(registered) if registered else None,
            date.fromisoformat(row["claimed"]),
            tuple(row["faults"]),
        )


@dataclass(frozen=True)
class Decision:
    outcome: str
    repair: tuple[str, ...] = ()


def add_months(day: date, months: int) -> date:
    """The same day ``months`` later, or the month's last day if that month is shorter."""
    month0 = day.month - 1 + months
    year, month = day.year + month0 // 12, month0 % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def warranty_months(claim: Claim) -> int:
    months = LARGE_APPLIANCE_MONTHS if claim.category in LARGE_APPLIANCES else BASE_MONTHS
    if claim.registered is not None and (claim.registered - claim.bought).days <= REGISTER_WITHIN_DAYS:
        months += REGISTRATION_MONTHS
    return months


def in_warranty(claim: Claim) -> bool:
    return claim.claimed < add_months(claim.bought, warranty_months(claim))


def covered_faults(faults: tuple[str, ...]) -> tuple[str, ...] | None:
    """The faults the warranty repairs, in the order logged, or ``None`` when one of them refuses the claim."""
    repair: list[str] = []
    for fault in faults:
        if fault in EXCLUDING:
            return None
        if fault in COSMETIC:
            return tuple(repair)
        if fault not in COVERED:
            raise ValueError(f"unknown fault code {fault!r}")
        repair.append(fault)
    return tuple(repair)


def decide(claim: Claim) -> Decision:
    """The desk's decision on a claim."""
    if not in_warranty(claim):
        return Decision("out_of_warranty")
    repair = covered_faults(claim.faults)
    if repair is None:
        return Decision("refused")
    if not repair:
        return Decision("not_covered")
    return Decision("accepted", repair)


def decide_file(path: str | Path) -> dict[str, Decision]:
    """Every claim in a file of claims (one JSON claim per line), decided, by claim id."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["claim_id"]: decide(Claim.from_row(row)) for row in rows}
